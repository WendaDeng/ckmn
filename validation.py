import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import time
import numpy as np

from utils import AverageMeter, accuracy, my_accuracy, performance, multitask_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger, writer):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if opt.dataset == 'EPIC':
        verb_losses = AverageMeter()
        noun_losses = AverageMeter()
        verb_top1 = AverageMeter()
        verb_top5 = AverageMeter()
        noun_top1 = AverageMeter()
        noun_top5 = AverageMeter()

    # classification_results_final = []
    # all_targets = []

    end_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs = data[0]
            targets = data[1].float()

            # all_targets.append(targets)

            inputs = inputs.to(opt.device)
            
            outputs = model(inputs)
            batch_size = inputs.size(0)
            # classification_results_final.append(outputs.cpu().data)
            if opt.dataset == 'EPIC':
                target = {k: v.to(opt.device) for k, v in targets.items()}
                loss_verb = criterion(outputs[0], target['verb'])
                loss_noun = criterion(outputs[1], target['noun'])
                loss = 0.5 * (loss_verb + loss_noun)
                verb_losses.update(loss_verb.item(), batch_size)
                noun_losses.update(loss_noun.item(), batch_size)

                verb_output = outputs[0]
                noun_output = outputs[1]
                verb_prec1, verb_prec5 = my_accuracy(outputs.cpu().data, target['verb'].cpu().data, topk=(1, 5))
                verb_top1.update(verb_prec1, batch_size)
                verb_top5.update(verb_prec5, batch_size)

                noun_prec1, noun_prec5 = my_accuracy(noun_output.cpu().data, target['noun'].cpu().data, topk=(1, 5))
                noun_top1.update(noun_prec1, batch_size)
                noun_top5.update(noun_prec5, batch_size)

                prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                                  (target['verb'], target['noun']),
                                                  topk=(1, 5))
            else:
                targets = targets.to(opt.device)
                loss = criterion(outputs, targets)
                # measure accuracy and record loss
                prec1, prec5 = my_accuracy(outputs.cpu().data, targets.cpu().data, topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)
            top5.update(prec5, batch_size)

            batch_time.update(time.time() - end_time)
            end_time = time.time()


        print("Testing Results: "
             "Verb Prec@1 {verb_top1.avg:.3f} Verb Prec@5 {verb_top5.avg:.3f} "
             "Noun Prec@1 {noun_top1.avg:.3f} Noun Prec@5 {noun_top5.avg:.3f} "
             "Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} "
             "Verb Loss {verb_loss.avg:.5f} "
             "Noun Loss {noun_loss.avg:.5f} "
             "Loss {loss.avg:.5f}"
              ).format(
                verb_top1=verb_top1, verb_top5=verb_top5,
                noun_top1=noun_top1, noun_top5=noun_top5,
                top1=top1, top5=top5,
                verb_loss=verb_losses,
                noun_loss=noun_losses,
                loss=losses)
    # classification_results_final = torch.cat(classification_results_final, dim=0)
    # all_targets = torch.cat(all_targets, dim=0)

    # final_mAP_sigmoid, final_mAP_softmax = performance(classification_results_final, all_targets)

    logger.log({
        'epoch': epoch, 'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg,
        # 'final_mAP_sigmoid': final_mAP_sigmoid,
        # 'final_mAP_softmax': final_mAP_softmax,
        'verb_loss': verb_losses.avg, 'verb_top1': verb_top1.avg, 'verb_top5': verb_top5.avg,
        'noun_loss': noun_losses.avg, 'noun_top1': noun_top1.avg, 'noun_top5': noun_top5.avg
    })

    test_metrics = {'val_loss': losses.avg,
                    'val_noun_loss': noun_losses.avg,
                    'val_verb_loss': verb_losses.avg,
                    'val_acc': top1.avg,
                    'val_verb_acc': verb_top1.avg,
                    'val_noun_acc': noun_top1.avg}
    return test_metrics
