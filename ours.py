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
import wandb

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate, init_seeds, seed_worker, test
from util.utils import init_log, AverageMeter, load_config

from thop import profile

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation for Slum Detection')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--results-dir', type=str, default='./results')
parser.add_argument('--split', type=str, default='10')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--wandb-project', type=str, default='slum-i')
parser.add_argument('--no-wandb', action='store_true')


def main():
    args = parser.parse_args()
    init_seeds(args.seed)

    cfg = load_config(args.config)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    world_size = 1
    local_rank = 0

    all_args = {**cfg, **vars(args), 'ngpus': world_size}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    run_name = 'freematch_{dataset}_split{split}_seed{seed}'.format(
        dataset=cfg['dataset'], split=args.split, seed=args.seed)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=all_args,
            tags=[cfg['dataset'], 'freematch', cfg.get('backbone', 'resnet101'), args.split],
        )

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
    g.manual_seed(args.seed)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)

    iters = 0
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best_iou = 0.0
    previous_best_acc = 0.0
    epoch = -1

    # FreeMatch SAT state (global + per-class EMA thresholds)
    fm_ema_m = 0.999
    fm_time_p = torch.tensor(cfg['conf_thresh']).cuda()          # global threshold EMA
    fm_p_model = torch.ones(cfg['nclass']).cuda() / cfg['nclass']  # per-class prob EMA
    
    total_start_time = time.time()
    
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best IoU: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best_iou))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)

            # FreeMatch SAT: EMA update on the clean weak view (before cutmix)
            with torch.no_grad():
                probs_u_w = pred_u_w.softmax(dim=1)           # (B, C, H, W)
                valid_px = (ignore_mask != 255)
                if valid_px.sum() > 0:
                    fm_time_p = fm_ema_m * fm_time_p + (1 - fm_ema_m) * conf_u_w[valid_px].mean()
                    fm_p_model = fm_ema_m * fm_p_model + (1 - fm_ema_m) * \
                        probs_u_w.permute(0, 2, 3, 1)[valid_px].mean(dim=0)

            # Per-pixel adaptive threshold = global_thresh * (class_conf / max_class_conf)
            fm_mod = (fm_p_model / fm_p_model.max()).clamp(min=1e-6)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            thresh_s1 = (fm_time_p * fm_mod[mask_u_w_cutmixed1]).clamp(max=0.95)
            mask_s1 = (conf_u_w_cutmixed1 >= thresh_s1) & (ignore_mask_cutmixed1 != 255)
            loss_u_s1 = (loss_u_s1 * mask_s1).sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            thresh_s2 = (fm_time_p * fm_mod[mask_u_w_cutmixed2]).clamp(max=0.95)
            mask_s2 = (conf_u_w_cutmixed2 >= thresh_s2) & (ignore_mask_cutmixed2 != 255)
            loss_u_s2 = (loss_u_s2 * mask_s2).sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            thresh_fp = (fm_time_p * fm_mod[mask_u_w]).clamp(max=0.95)
            mask_fp = (conf_u_w >= thresh_fp) & (ignore_mask != 255)
            loss_u_w_fp = (loss_u_w_fp * mask_fp).sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = mask_fp.sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
            writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            writer.add_scalar('train/fm_time_p', fm_time_p.item(), iters)
            if not args.no_wandb:
                wandb.log({'train/loss_all': loss.item(), 'train/loss_x': loss_x.item(),
                           'train/loss_s': (loss_u_s1.item() + loss_u_s2.item()) / 2.0,
                           'train/loss_w_fp': loss_u_w_fp.item(),
                           'train/mask_ratio': mask_ratio,
                           'train/fm_time_p': fm_time_p.item(),
                           'train/lr': lr, 'iter': iters})
    
            if len(trainloader_u) == 0:
                log_frequency = 1
            else:
                log_frequency = max(1, len(trainloader_u) // 8) 
                        
            if i % log_frequency == 0:
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, '
                            'Mask ratio: {:.3f}, FM thresh: {:.4f}'.format(
                            i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                            total_mask_ratio.avg, fm_time_p.item()))
        
        iou_class, macro_iou, overall_acc = evaluate(model, valloader, cfg)

        logger.info('***** Evaluation ***** >>>> Background IoU: {:.2f}'.format(iou_class[0]))
        logger.info('***** Evaluation ***** >>>> Foreground IoU: {:.2f}'.format(iou_class[1]))
        logger.info('***** Evaluation ***** >>>> Macro IoU (Main Metric): {:.2f}'.format(macro_iou))
        logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))

        writer.add_scalar('eval/background_IoU', iou_class[0], epoch)
        writer.add_scalar('eval/foreground_IoU', iou_class[1], epoch)
        writer.add_scalar('eval/macro_IoU', macro_iou, epoch)
        writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)
        if not args.no_wandb:
            wandb.log({
                'eval/background_iou': iou_class[0],
                'eval/foreground_iou': iou_class[1],
                'eval/macro_iou': macro_iou,
                'eval/overall_acc': overall_acc,
                'eval/best_macro_iou': max(macro_iou, previous_best_iou),
                'epoch': epoch,
            })

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

    test(args, cfg, model, logger, "freematch", split=args.split, calculate_speed=True,
         total_training_seconds=total_training_seconds, model_flops_g=model_flops_g,
         model_params_m=model_params_m, H=H, W=W)

    if not args.no_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
