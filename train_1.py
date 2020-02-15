import torch
import torch.nn.functional as F
from torch.autograd import Variable

import time
import os
import ipdb
import numpy as np
#from apex import amp

from utils import AverageMeter, calculate_mAP_sklearn, calculate_mAP_sklearn_new, calculate_mAP_sklearn_new_noback


def performance(prediction, target):
    # ipdb.set_trace()
    prediction = torch.sigmoid(prediction)
    mAP_new = calculate_mAP_sklearn_new(prediction, target)
    print('sigmoid-sklearn:', mAP_new)

    prediction = F.softmax(prediction, dim=1)
    mAP_new_st = calculate_mAP_sklearn_new(prediction, target)
    print('softmax-sklearn:', mAP_new_st)

    return mAP_new, mAP_new_st


def accuracy(preds: torch.Tensor, truths: torch.Tensor, topk: tuple = (1,)) -> tuple:
    res = []
    for n in topk:
        best_n = np.argsort(preds, axis=1)[:,-n:]
        ts = np.argmax(truths, axis=1)
        successes = 0
        for i in range(ts.shape[0]):
          if ts[i] in best_n[i,:]:
            successes += 1
        res.append(float(successes)/ts.shape[0])
    return tuple(res)


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, writer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    verb_top1 = AverageMeter()
    verb_top5 = AverageMeter()

    classification_results_final = []
    all_targets = []

    end_time = time.time()
    for i, data in enumerate(data_loader):

        data_time.update(time.time() - end_time)

        inputs = data[0]
        targets = data[1].float()

        all_targets.append(targets)

        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)

        optimizer.zero_grad()

        outputs = model(inputs)

        classification_results_final.append(outputs.cpu().data)
        verb_prec1, verb_prec5 = accuracy(outputs.cpu().data, targets.cpu().data, topk=(1, 5))
        verb_top1.update(verb_prec1, inputs.size(0))
        verb_top5.update(verb_prec5, inputs.size(0))

        loss = criterion(outputs, targets)
        loss.backward()
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        writer.add_scalar('train/loss_iter', losses.val, i + 1 + len(data_loader) * (epoch - 1))

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss ({loss.avg:.4f})\t'
				  'Verb Prec@1 {verb_top1.val:.3f} ({verb_top1.avg:.3f})\t'
                  'Verb Prec@5 {verb_top5.val:.3f} ({verb_top5.avg:.3f})\t'
				  .format(
                      epoch, i + 1, len(data_loader),
                      batch_time=batch_time, data_time=data_time,
                      loss=losses, verb_top1=verb_top1, verb_top5=verb_top5))

    classification_results_final = torch.cat(classification_results_final, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    final_mAP_sigmoid, final_mAP_softmax = performance(classification_results_final, all_targets)

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'final_mAP_sigmoid': final_mAP_sigmoid,
        'final_mAP_softmax': final_mAP_softmax,
        'lr': optimizer.param_groups[-1]['lr']
    })

    writer.add_scalar('train/loss_epoch', losses.avg, epoch)
    writer.add_scalar('train/learning_rate_epoch', opt.learning_rate, epoch)
    writer.add_scalar('train/verb_top1', verb_top1.avg, epoch)
    writer.add_scalar('train/verb_top5', verb_top5.avg, epoch)
