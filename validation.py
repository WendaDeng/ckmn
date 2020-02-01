import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import time
import numpy as np

from utils import AverageMeter, calculate_mAP_sklearn, calculate_mAP_sklearn_new, calculate_mAP_sklearn_new_noback


def performance(prediction, target):
    prediction = F.sigmoid(prediction)
    mAP_new = calculate_mAP_sklearn_new(prediction, target)
    print('sigmoid-sklearn:', mAP_new)

    prediction = F.softmax(prediction, dim=1)
    mAP_new_st = calculate_mAP_sklearn_new(prediction, target)
    print('softmax-sklearn:', mAP_new_st)

    return mAP_new, mAP_new_st


def val_epoch(epoch, data_loader, model, opt, logger, writer):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

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

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if i % 1 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(data_loader),
                    batch_time=batch_time))

    classification_results_final = torch.cat(classification_results_final, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    final_mAP_sigmoid, final_mAP_softmax = performance(classification_results_final, all_targets)

    logger.log({
        'epoch': epoch,
        'final_mAP_sigmoid': final_mAP_sigmoid,
        'final_mAP_softmax': final_mAP_softmax
    })
