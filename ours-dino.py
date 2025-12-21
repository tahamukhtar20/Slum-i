import argparse
from copy import deepcopy
import logging
import os
import pprint
import time
import copy

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate, init_seeds, seed_worker, test
from util.utils import ClassAwareThreshold, init_log, AverageMeter
from util.prototype import PrototypeBank

from thop import profile

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation for Slum Detection')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--results-dir', type=str, default='./results')
parser.add_argument('--split', type=str, default='10')

def main():
    init_seeds(42)
    
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    world_size = 1
    local_rank = 0

    all_args = {**cfg, **vars(args), 'ngpus': world_size}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    
    writer = SummaryWriter(args.save_path)
    
    os.makedirs(args.save_path, exist_ok=True)

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
      
    if cfg['lock_backbone']:
        model.lock_backbone()

    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    model.cuda(local_rank)
    
    model.eval()
    C, H, W = 3, cfg['crop_size'], cfg['crop_size']
    dummy_input = torch.randn(1, C, H, W).cuda(local_rank)
    
    temp_model = copy.deepcopy(model)
    flops, params = profile(temp_model, inputs=(dummy_input,), verbose=False)
    model_flops_g = flops / 1e9
    model_params_m = params / 1e6

    del temp_model
    torch.cuda.empty_cache()

    logger.info('Total params (thop): {:.1f}M'.format(model_params_m))
    logger.info('Inference FLOPs (thop): {:.2f}G on size {}x{}'.format(model_flops_g, H, W))
    
    proto_bank = PrototypeBank(num_classes=2, power=2, feat_dim=model.channels[0], queue_size=2000, device='cuda')
    adaptive_thresh = ClassAwareThreshold(num_classes=2, momentum=0.999, min_thresh=0.6, max_thresh=0.95)
    
    model.train()
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)

    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    g = torch.Generator()
    g.manual_seed(42)
    
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True, worker_init_fn=seed_worker, generator=g
    )
    
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True, worker_init_fn=seed_worker, generator=g
    )
    
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False
    )
    
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best_iou, previous_best_ema_iou = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    total_start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                    'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best_iou, best_epoch, previous_best_ema_iou, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_proto_weight = AverageMeter()

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            with torch.no_grad():
                pred_u_w, feats_u_w = model_ema(img_u_w, return_feature=True)
                pred_u_w = pred_u_w.detach()
                feats_u_w = feats_u_w.detach()
                
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
                
                adaptive_thresh.update(conf_u_w, mask_u_w)
                mask_high_conf = adaptive_thresh.get_mask(conf_u_w, mask_u_w)

                update_mask = mask_u_w.clone()
                update_mask[~mask_high_conf] = 255
                
                proto_bank.update(feats_u_w, update_mask)
                proto_weights = proto_bank.get_consistency_weight(feats_u_w, mask_u_w)
            
            total_proto_weight.update(proto_weights.mean().item())
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            proto_weights1 = proto_weights.clone()
            proto_weights2 = proto_weights.clone()
            
            proto_weights1[cutmix_box1 == 1] = proto_weights.flip(0)[cutmix_box1 == 1]
            proto_weights2[cutmix_box2 == 1] = proto_weights.flip(0)[cutmix_box2 == 1]
            
            num_lb = img_x.shape[0]
            preds_all = model(torch.cat((img_x, img_u_w)))
            pred_x, _ = preds_all.split([num_lb, num_lb])
            
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            mask_valid = mask_high_conf.long()
            mask_valid_cutmixed1 = mask_valid.clone()
            mask_valid_cutmixed2 = mask_valid.clone()
            
            mask_valid_cutmixed1[cutmix_box1 == 1] = mask_valid.flip(0)[cutmix_box1 == 1]
            mask_valid_cutmixed2[cutmix_box2 == 1] = mask_valid.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            consistency_weight1 = mask_valid_cutmixed1 * proto_weights1
            loss_u_s1 = loss_u_s1 * (consistency_weight1 * (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / ((ignore_mask_cutmixed1 != 255).sum().item() + 1e-6)
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            consistency_weight2 = mask_valid_cutmixed2 * proto_weights2
            loss_u_s2 = loss_u_s2 * (consistency_weight2 * (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / ((ignore_mask_cutmixed2 != 255).sum().item() + 1e-6)
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss = (loss_x + loss_u_s) / 2.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = (mask_high_conf & (ignore_mask != 255)).sum().item() / ((ignore_mask != 255).sum() + 1e-6)
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            writer.add_scalar('train/proto_weight', total_proto_weight.avg, iters)

            if len(trainloader_u) == 0:
                log_frequency = 1
            else:
                log_frequency = max(1, len(trainloader_u) // 8) 
            
            if i % log_frequency == 0:
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: {:.3f}, Proto W: {:.3f}'.format(
                            i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_mask_ratio.avg, total_proto_weight.avg))
        
        iou_class, macro_iou, overall_acc = evaluate(model, valloader, cfg, multiplier=14)
        ema_iou_class, ema_macro_iou, ema_overall_acc = evaluate(model_ema, valloader, cfg, multiplier=14)
        
        logger.info('***** Evaluation ***** >>>> Background IoU: {:.2f}'.format(iou_class[0]))
        logger.info('***** Evaluation ***** >>>> Foreground IoU: {:.2f}'.format(iou_class[1]))
        logger.info('***** Evaluation ***** >>>> Macro IoU (Main Metric): {:.2f}'.format(macro_iou))
        logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))

        logger.info('***** Evaluation ***** >>>> EMA Background IoU: {:.2f}'.format(ema_iou_class[0]))
        logger.info('***** Evaluation ***** >>>> EMA Foreground IoU: {:.2f}'.format(ema_iou_class[1]))
        logger.info('***** Evaluation ***** >>>> EMA Macro IoU (Main Metric): {:.2f}'.format(ema_macro_iou))
        logger.info('***** Evaluation ***** >>>> EMA Overall Accuracy: {:.2f}\n'.format(ema_overall_acc))

        writer.add_scalar('eval/background_IoU', iou_class[0], epoch)
        writer.add_scalar('eval/foreground_IoU', iou_class[1], epoch)
        writer.add_scalar('eval/macro_IoU', macro_iou, epoch)
        writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)

        writer.add_scalar('eval/ema_background_IoU', ema_iou_class[0], epoch)
        writer.add_scalar('eval/ema_foreground_IoU', ema_iou_class[1], epoch)
        writer.add_scalar('eval/ema_macro_IoU', ema_macro_iou, epoch)
        writer.add_scalar('eval/ema_overall_accuracy', ema_overall_acc, epoch)

        is_best = macro_iou > previous_best_iou
        
        previous_best_iou = max(macro_iou, previous_best_iou)
        previous_best_ema_iou = max(ema_macro_iou, previous_best_ema_iou)
        if macro_iou == previous_best_iou:
            best_epoch = epoch
        if ema_macro_iou == previous_best_ema_iou:
            best_epoch_ema = epoch
        
        checkpoint = {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_iou': previous_best_iou,
            'previous_best_ema_iou': previous_best_ema_iou,
            'best_epoch': best_epoch,
            'best_epoch_ema': best_epoch_ema
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

    total_end_time = time.time()
    total_training_seconds = total_end_time - total_start_time

    test(args, cfg, model, logger, "ours-dino", split=args.split, multiplier=14, calculate_speed=True, total_training_seconds=total_training_seconds, model_flops_g=model_flops_g, model_params_m=model_params_m, H=H, W=W)
    test(args, cfg, model_ema, logger, "ours-dino", split=args.split, suffix='_ema', multiplier=14, calculate_speed=True, total_training_seconds=total_training_seconds, model_flops_g=model_flops_g, model_params_m=model_params_m, H=H, W=W)

if __name__ == '__main__':
    main()