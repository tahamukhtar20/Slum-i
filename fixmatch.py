import argparse
import logging
import os
import pprint
import time

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate, init_seeds, seed_worker, test
from util.utils import init_log, AverageMeter

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

    model = DeepLabV3Plus(cfg)

    C, H, W = 3, cfg['crop_size'], cfg['crop_size']
    dummy_input = torch.randn(1, C, H, W).cuda(local_rank)
    model.cuda(local_rank)
    
    flops, params = profile(model, inputs=(dummy_input,), verbose=False, custom_ops={})
    
    model_flops_g = flops / 1e9
    model_params_m = params / 1e6
    
    logger.info('Total params (thop): {:.1f}M'.format(model_params_m))
    logger.info('Inference FLOPs (thop): {:.2f}G on size {}x{}'.format(model_flops_g, H, W))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)

    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    g = torch.Generator()
    g.manual_seed(42)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    iters = 0
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best_iou = 0.0
    epoch = -1
    
    total_start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best Marco IoU: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best_iou))

        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _),
                (img_u_w_mix, img_u_s_mix, _, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
            img_u_w_mix, img_u_s_mix = img_u_w_mix.cuda(), img_u_s_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            pred_x, pred_u_w = model(torch.cat((img_x, img_u_w))).split([num_lb, num_ulb])
            pred_u_s = model(img_u_s)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w_mix[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            loss_u_s = loss_u_s * ((conf_u_w_cutmixed >= cfg['conf_thresh']) & (ignore_mask_cutmixed != 255))
            loss_u_s = loss_u_s.sum() / (ignore_mask_cutmixed != 255).sum().item()

            loss = (loss_x + loss_u_s) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if len(trainloader_u) == 0:
                log_frequency = 1
            else:
                log_frequency = max(1, len(trainloader_u) // 8) 
            
            if i % log_frequency == 0:
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
        iou_class, macro_iou, overall_acc = evaluate(model, valloader, cfg)

        logger.info('***** Evaluation ***** >>>> Background IoU: {:.2f}'.format(iou_class[0]))
        logger.info('***** Evaluation ***** >>>> Foreground IoU: {:.2f}'.format(iou_class[1]))
        logger.info('***** Evaluation ***** >>>> Macro IoU (Main Metric): {:.2f}'.format(macro_iou))
        logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))

        writer.add_scalar('eval/background_IoU', iou_class[0], epoch)
        writer.add_scalar('eval/foreground_IoU', iou_class[1], epoch)
        writer.add_scalar('eval/macro_IoU', macro_iou, epoch)
        writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)

        is_best = macro_iou > previous_best_iou
        previous_best_iou = max(macro_iou, previous_best_iou)
        if is_best:
            previous_best_acc = overall_acc
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'macro_iou': macro_iou,
            'epoch': epoch,
            'previous_best_iou': previous_best_iou,
            'previous_best_acc': previous_best_acc,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

    total_end_time = time.time()
    total_training_seconds = total_end_time - total_start_time

    test(args, cfg, model, logger, "fixmatch", split=args.split, calculate_speed=True, total_training_seconds=total_training_seconds, model_flops_g=model_flops_g, model_params_m=model_params_m, H=H, W=W)

if __name__ == '__main__':
    main()
