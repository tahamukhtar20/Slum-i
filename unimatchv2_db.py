import argparse
import json
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
import wandb

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate, init_seeds, seed_worker, test
from util.debias import (KeepRateGate, LogitDebias, PLStats, SelfAdaptiveThreshold, SoftClassWeights, labeled_prior,
                         null_image, rank_weights)
from util.utils import init_log, AverageMeter, load_config

from thop import profile
from util.multispectral import widen_first_conv

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

    run_name = 'unimatchv2_db_{dataset}_split{split}_seed{seed}'.format(
        dataset=cfg['dataset'], split=args.split, seed=args.seed)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=all_args,
            tags=[cfg['dataset'], 'unimatchv2', cfg.get('backbone', 'dpt'), args.split],
        )

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
    widen_first_conv(model, cfg.get('in_chans', 3))
      
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
    C, H, W = cfg.get('in_chans', 3), cfg['crop_size'], cfg['crop_size']
    dummy_input = torch.randn(1, C, H, W).cuda(local_rank)
    
    temp_model = copy.deepcopy(model)
    flops, params = profile(temp_model, inputs=(dummy_input,), verbose=False)
    model_flops_g = flops / 1e9
    model_params_m = params / 1e6

    del temp_model
    torch.cuda.empty_cache()

    logger.info('Total params (thop): {:.1f}M'.format(model_params_m))
    logger.info('Inference FLOPs (thop): {:.2f}G on size {}x{}'.format(model_flops_g, H, W))
    model.train()
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)

    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)

    debias = None
    if cfg.get('logit_debias', False):
        target = labeled_prior(cfg['data_root'], args.labeled_id_path, cfg['nclass']) \
            if cfg.get('debias_target') == 'labeled' else None
        debias = LogitDebias(cfg['nclass'], cfg.get('debias_lambda', 0.5), target=target)
        logger.info('Debias target prior: {}'.format(target))
    use_rank = cfg.get('rank_weight', False)
    # pl_scheme: threshold (UniMatch V2), rank (DWL-style), softmatch (class-wise), freematch (SAT),
    # thr_or_rank (weight 1 above conf_thresh, rank weight below), rank_soft (rank until
    # rank_switch_epoch, class-wise softmatch after; the softmatch statistics are tracked throughout)
    scheme = cfg.get('pl_scheme', 'rank' if use_rank else 'threshold')
    soft = SoftClassWeights(cfg['nclass']) if scheme in ('softmatch', 'rank_soft') else None
    sat = SelfAdaptiveThreshold(cfg['nclass'], local=cfg.get('sat_local', 'prob')) if scheme == 'freematch' else None
    # gated_rank: like thr_or_rank, but Rank weights on rejected pixels are scaled by the starvation gate
    gate = KeepRateGate(cfg['nclass'], cfg['conf_thresh']) if scheme == 'gated_rank' else None
    # fixed_class (ablation): fixed per-class thresholds, class_thresh = [bg, slum]
    cls_thr = torch.tensor(cfg['class_thresh'], dtype=torch.float32) if scheme == 'fixed_class' else None
    # cdmad: subtract the teacher's logits on a pattern-free image before pseudo-labelling
    null_img = null_image(cfg['crop_size'], cfg.get('in_chans', 3)) if cfg.get('cdmad', False) else None

    def pl_weight(conf, mask, ignore):
        valid = ignore != 255
        rank = lambda: rank_weights(conf, mask, valid, cfg['nclass'], cfg.get('rank_floor', 0.2), cfg.get('rank_ties', 'scan'))
        if scheme == 'rank' or (scheme == 'rank_soft' and epoch < cfg['rank_switch_epoch']):
            return rank()
        if scheme in ('softmatch', 'rank_soft'):
            return soft(conf, mask, valid)
        if scheme in ('thr_or_rank', 'gated_rank'):
            keep = ((conf >= cfg['conf_thresh']) & valid).float()
            extra = rank() if gate is None else gate.alpha()[mask] * rank()
            return torch.where(keep > 0, keep, extra)
        if scheme == 'freematch':
            return sat(conf, mask, valid)
        if scheme == 'fixed_class':
            return ((conf >= cls_thr.to(conf.device)[mask]) & valid).float()
        return ((conf >= cfg['conf_thresh']) & valid).float()

    logger.info('Pseudo-labelling: scheme={} cdmad={} logit_debias={} (lambda={}) rank floor={} ties={}'.format(
        scheme, null_img is not None, debias is not None, cfg.get('debias_lambda', 0.5), cfg.get('rank_floor', 0.2),
        cfg.get('rank_ties', 'scan')))

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
    g.manual_seed(args.seed)
    
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
        pl_stats = PLStats(cfg['nclass'], cfg['conf_thresh'])
        null_bias = []

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                raw_probs_u_w = pred_u_w.softmax(dim=1)
                if null_img is not None:
                    bias = model_ema(null_img).mean((2, 3))
                    pred_u_w = pred_u_w - bias.view(1, -1, 1, 1)
                    null_bias.append((bias[0, -1] - bias[0, 0]).item())
                probs_u_w = pred_u_w.softmax(dim=1)
                if debias is not None:
                    debias.update(probs_u_w, ignore_mask != 255)
                    probs_u_w = debias.apply(probs_u_w)
                conf_u_w, mask_u_w = probs_u_w.max(dim=1)
                valid_u = ignore_mask != 255
                if soft is not None:
                    soft.update(conf_u_w, mask_u_w, valid_u)
                if sat is not None:
                    sat.update(probs_u_w, valid_u)
                if gate is not None:
                    gate.update(conf_u_w, mask_u_w, valid_u)
                pl_stats.update(conf_u_w, mask_u_w, valid_u, raw_probs_u_w, probs_u_w,
                                weights=pl_weight(conf_u_w, mask_u_w, ignore_mask))
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * pl_weight(conf_u_w_cutmixed1, mask_u_w_cutmixed1, ignore_mask_cutmixed1)
            loss_u_s1 = loss_u_s1.sum() / max((ignore_mask_cutmixed1 != 255).sum().item(), 1)
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * pl_weight(conf_u_w_cutmixed2, mask_u_w_cutmixed2, ignore_mask_cutmixed2)
            loss_u_s2 = loss_u_s2.sum() / max((ignore_mask_cutmixed2 != 255).sum().item(), 1)
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss = (loss_x + loss_u_s) / 2.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum().clamp(min=1)
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
            if not args.no_wandb:
                wandb.log({'train/loss_all': loss.item(), 'train/loss_x': loss_x.item(),
                           'train/loss_s': loss_u_s.item(), 'train/mask_ratio': mask_ratio,
                           'train/lr': lr, 'iter': iters})

            if len(trainloader_u) == 0:
                log_frequency = 1
            else:
                log_frequency = max(1, len(trainloader_u) // 8) 
            
            if i % log_frequency == 0:
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_mask_ratio.avg))
        
        pl_summary = {'epoch': epoch, **pl_stats.summary()}
        if null_bias:
            pl_summary['null_bias_slum_minus_bg'] = sum(null_bias) / len(null_bias)
        if soft is not None:
            pl_summary['soft_mu'], pl_summary['soft_var'] = soft.mu.tolist(), soft.var.tolist()
        if sat is not None:
            pl_summary['sat_thresholds'] = sat.thresholds().tolist()
        if gate is not None:
            pl_summary['gate_keep'], pl_summary['gate_alpha'] = gate.k.tolist(), gate.alpha().tolist()
        logger.info('PL stats: {}'.format(json.dumps(pl_summary)))
        with open(os.path.join(args.save_path, 'pl_stats.jsonl'), 'a') as f:
            f.write(json.dumps(pl_summary) + '\n')

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
        if not args.no_wandb:
            wandb.log({
                'eval/background_iou': iou_class[0],
                'eval/foreground_iou': iou_class[1],
                'eval/macro_iou': macro_iou,
                'eval/overall_acc': overall_acc,
                'eval/best_macro_iou': max(macro_iou, previous_best_iou),
                'eval/ema_background_iou': ema_iou_class[0],
                'eval/ema_foreground_iou': ema_iou_class[1],
                'eval/ema_macro_iou': ema_macro_iou,
                'eval/ema_overall_acc': ema_overall_acc,
                'eval/best_ema_macro_iou': max(ema_macro_iou, previous_best_ema_iou),
                'epoch': epoch,
            })

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

    test(args, cfg, model, logger, "unimatchv2_db", split=args.split, multiplier=14, calculate_speed=True, total_training_seconds=total_training_seconds, model_flops_g=model_flops_g, model_params_m=model_params_m, H=H, W=W)

    if not args.no_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()