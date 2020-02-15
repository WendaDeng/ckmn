import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import time
import numpy as np

from utils import AverageMeter, accuracy, calculate_mAP_sklearn_new, performance


def val_epoch(epoch, data_loader, model, criterion, opt, logger, writer):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    verb_top1 = AverageMeter()
    verb_top5 = AverageMeter()

    classification_results_final = []
    all_targets = []

    end_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs = data[0]
            targets = data[1].float()

            all_targets.append(targets)

            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            
            outputs = model(inputs)
            
            classification_results_final.append(outputs.cpu().data)

            verb_prec1, verb_prec5 = accuracy(outputs.cpu().data, targets.cpu().data, topk=(1, 5))
            verb_top1.update(verb_prec1, inputs.size(0))
            verb_top5.update(verb_prec5, inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            if i % 1 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Verb Prec@1 {verb_top1.avg:.3f} Verb Prec@5 {verb_top5.avg:.3f}\t'
                      'Loss {loss.avg:.5f}'
                    .format(
                        epoch, i + 1, len(data_loader),
                        batch_time=batch_time,
                        verb_top1=verb_top1, verb_top5=verb_top5,
                        loss=losses))

    classification_results_final = torch.cat(classification_results_final, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    final_mAP_sigmoid, final_mAP_softmax = performance(classification_results_final, all_targets)

    logger.log({
        'epoch': epoch,
        'final_mAP_sigmoid': final_mAP_sigmoid,
        'final_mAP_softmax': final_mAP_softmax
    })

    test_metrics = {'val_loss': losses.avg, 'val_verb_acc': verb_top1.avg}
    return test_metrics
