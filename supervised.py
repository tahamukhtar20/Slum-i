import argparse
import json
import logging
import os
import pprint
import time

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import wandb

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import init_log, AverageMeter, intersectionAndUnion, load_config

from thop import profile
from util.multispectral import widen_first_conv

parser = argparse.ArgumentParser(description='Supervised Semantic Segmentation for Slum Detection')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--results-dir', type=str, default='./results')
parser.add_argument('--split', type=str, default='full')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--wandb-project', type=str, default='slum-i')
parser.add_argument('--no-wandb', action='store_true')

def init_seeds(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate(model, loader, cfg, calculate_speed=False, multiplier=None):
    model.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()

    if calculate_speed:
        inference_timer = AverageMeter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for img, mask, _ in loader:
            
            img = img.cuda()

            _, _, h, w = img.shape 
            
            new_h, new_w = h, w
            
            if multiplier is not None:
                new_h = math.ceil(h / multiplier) * multiplier
                new_w = math.ceil(w / multiplier) * multiplier
                
                if new_h != h or new_w != w:
                    img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            if calculate_speed:
                start_event.record()

            logits = model(img)
            
            if new_h != h or new_w != w:
                logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            
            pred = logits.argmax(dim=1)
            
            if calculate_speed:
                end_event.record()
                torch.cuda.synchronize()
                inference_timer.update(start_event.elapsed_time(end_event))
         
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            intersection_meter.update(intersection)
            union_meter.update(union)
            
            valid = mask != 255
            correct_pixel.update(((pred.cpu() == mask) & valid).sum().item())
            total_pixel.update(max(valid.sum().item(), 1))
            
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0
    macro_iou = iou_class.mean().item()
    
    if calculate_speed:
        avg_inference_ms = inference_timer.avg
        return iou_class, macro_iou, overall_acc, avg_inference_ms
    return iou_class, macro_iou, overall_acc


def test(args, cfg, model, logger, method, split=None, multiplier=None, calculate_speed=False, total_training_seconds=0, model_flops_g=0.0, model_params_m=0.0, H=0, W=0):
    """Evaluate best.pth and latest.pth on the test set; auto-detect model_ema.

    Writes a single results.json next to the checkpoints (args.save_path).
    """
    testset = SemiDataset(cfg['dataset'], cfg['data_root'], 'test')
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=1,
                            drop_last=False)

    output = {
        'method': method,
        'dataset': cfg['dataset'],
        'split': split,
        'seed': args.seed,
        'backbone': cfg.get('backbone'),
        'total_training_seconds': total_training_seconds,
        'model_flops_g': model_flops_g,
        'model_params_m': model_params_m,
        'input_h': H,
        'input_w': W,
        'avg_inference_ms': 0.0,
    }

    first_eval = True

    for ckpt_name in ['best.pth', 'latest.pth']:
        ckpt_path = os.path.join(args.save_path, ckpt_name)
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        label = ckpt_name.replace('.pth', '')  # 'best' or 'latest'

        ckpt_results = {}
        for state_key in ('model', 'model_ema'):
            if state_key not in ckpt:
                continue
            model.load_state_dict(ckpt[state_key], strict=False)

            do_speed = first_eval and calculate_speed
            if do_speed:
                iou_class, macro_iou, overall_acc, avg_inference_ms = evaluate(
                    model, testloader, cfg, multiplier=multiplier, calculate_speed=True)
                output['avg_inference_ms'] = avg_inference_ms
            else:
                iou_class, macro_iou, overall_acc = evaluate(
                    model, testloader, cfg, multiplier=multiplier)

            ckpt_results[state_key] = {
                'bg_iou': round(float(iou_class[0]), 2),
                'fg_iou': round(float(iou_class[1]), 2),
                'macro_iou': round(macro_iou, 2),
                'accuracy': round(overall_acc, 2),
            }

            if first_eval:
                logger.info('************ Final Test Results ({} / {}) ************'.format(label, state_key))
                logger.info('***** Test ***** >>>> Background IoU: {:.2f}'.format(iou_class[0]))
                logger.info('***** Test ***** >>>> Foreground IoU: {:.2f}'.format(iou_class[1]))
                logger.info('***** Test ***** >>>> Macro IoU (Main Metric): {:.2f}'.format(macro_iou))
                logger.info('***** Test ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))
                first_eval = False

        output[label] = ckpt_results

    results_path = os.path.join(args.save_path, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info('Test results saved to {}'.format(results_path))

    if not getattr(args, 'no_wandb', True) and wandb.run is not None:
        for label in ('best', 'latest'):
            for state_key, metrics in output.get(label, {}).items():
                for metric_name, value in metrics.items():
                    wandb.run.summary[f'test/{label}/{state_key}/{metric_name}'] = value
        headline = output.get('best', {}).get('model') or output.get('latest', {}).get('model') or {}
        for k, v in headline.items():
            wandb.run.summary[f'test/{k}'] = v
        wandb.run.summary['test/avg_inference_ms'] = output['avg_inference_ms']
        wandb.run.summary['training/total_hours'] = total_training_seconds / 3600
        wandb.run.summary['model/params_m'] = model_params_m
        wandb.run.summary['model/flops_g'] = model_flops_g


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

    run_name = 'supervised_{dataset}_split{split}_seed{seed}'.format(
        dataset=cfg['dataset'], split=args.split, seed=args.seed)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=all_args,
            tags=[cfg['dataset'], 'supervised', cfg.get('backbone', 'resnet101'), args.split],
        )

    writer = SummaryWriter(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)

    model = DeepLabV3Plus(cfg)
    widen_first_conv(model, cfg.get('in_chans', 3))
    
    C, H, W = cfg.get('in_chans', 3), cfg['crop_size'], cfg['crop_size']
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

    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best_iou, previous_best_acc = 0.0, 0.0
    epoch = -1
    
    total_start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best Macro IoU: {:.2f}, Overall Accuracy: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc))

        model.train()
        total_loss = AverageMeter()

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion(pred, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            writer.add_scalar('train/loss', loss.item(), iters)
            if not args.no_wandb:
                wandb.log({'train/loss': loss.item(), 'train/lr': lr, 'iter': iters})

            if i % (max(2, len(trainloader) // 8)) == 0:
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

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
            'epoch': epoch,
            'macro_iou': macro_iou,
            'previous_best_iou': previous_best_iou,
            'previous_best_acc': previous_best_acc,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

    total_end_time = time.time()
    total_training_seconds = total_end_time - total_start_time

    test(args, cfg, model, logger, "supervised", calculate_speed=True, split=args.split,
         total_training_seconds=total_training_seconds, model_flops_g=model_flops_g,
         model_params_m=model_params_m, H=H, W=W)

    if not args.no_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
