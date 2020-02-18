import torch
import torch.nn.functional as F
from torch.autograd import Variable

import time
import os
import ipdb
import numpy as np
#from apex import amp

from utils import AverageMeter, accuracy, my_accuracy, performance, multitask_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, writer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if opt.dataset == 'EPIC':
        verb_losses = AverageMeter()
        verb_top1 = AverageMeter()
        verb_top5 = AverageMeter()
        noun_losses = AverageMeter()
        noun_top1 = AverageMeter()
        noun_top5 = AverageMeter()

    # classification_results_final = []
    # all_targets = []

    end_time = time.time()
    for i, data in enumerate(data_loader):

        data_time.update(time.time() - end_time)

        inputs = data[0]
        targets = data[1]

        # all_targets.append(targets)

        inputs = inputs.to(opt.device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # classification_results_final.append(outputs.cpu().data)

        batch_size = inputs.size(0)
        if opt.dataset == 'EPIC':
            target = {k: v.to(opt.device) for k, v in targets.items()}
            loss_verb = criterion(outputs[0], target['verb'])
            loss_noun = criterion(outputs[1], target['noun'])
            loss = 0.5 * (loss_verb + loss_noun)
            verb_losses.update(loss_verb.item(), batch_size)
            noun_losses.update(loss_noun.item(), batch_size)

            verb_output = outputs[0]
            noun_output = outputs[1]
            verb_prec1, verb_prec5 = my_accuracy(outputs[0].cpu().data, target['verb'].cpu().data, topk=(1, 5))
            verb_top1.update(verb_prec1, batch_size)
            verb_top5.update(verb_prec5, batch_size)

            noun_prec1, noun_prec5 = my_accuracy(outputs[1].cpu().data, target['noun'].cpu().data, topk=(1, 5))
            noun_top1.update(noun_prec1, batch_size)
            noun_top5.update(noun_prec5, batch_size)

            prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                              (target['verb'], target['noun']),
                                              topk=(1, 5))
        else:
            targets = targets.to(opt.device)
            loss = criterion(outputs, targets)
            prec1, prec5 = my_accuracy(outputs.cpu().data, targets.cpu().data, topk=(1, 5))

        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

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
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' +
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' +
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' +
                 'Verb Loss {verb_loss.val:.4f} ({verb_loss.avg:.4f})\t' +
                 'Noun Loss {noun_loss.val:.4f} ({noun_loss.avg:.4f})\t' +
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' +
                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' +
                 'Verb Prec@1 {verb_top1.val:.3f} ({verb_top1.avg:.3f})\t' +
                 'Verb Prec@5 {verb_top5.val:.3f} ({verb_top5.avg:.3f})\t' +
                 'Noun Prec@1 {noun_top1.val:.3f} ({noun_top1.avg:.3f})\t' +
                 'Noun Prec@5 {noun_top5.val:.3f} ({noun_top5.avg:.3f})'
                 ).format(
                    epoch, i, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, verb_loss=verb_losses,
                    noun_loss=noun_losses, top1=top1, top5=top5,
                    verb_top1=verb_top1, verb_top5=verb_top5,
                    noun_top1=noun_top1, noun_top5=noun_top5, lr=optimizer.param_groups[-1]['lr'])

    # classification_results_final = torch.cat(classification_results_final, dim=0)
    # all_targets = torch.cat(all_targets, dim=0)

    # final_mAP_sigmoid, final_mAP_softmax = performance(classification_results_final, all_targets)

    epoch_logger.log({
        'epoch': epoch, 'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg,
        # 'final_mAP_sigmoid': final_mAP_sigmoid,
        # 'final_mAP_softmax': final_mAP_softmax,
        'lr': optimizer.param_groups[-1]['lr'],
        'verb_loss': verb_losses.avg, 'verb_top1': verb_top1.avg, 'verb_top5': verb_top5.avg,
        'noun_loss': noun_losses.avg, 'noun_top1': noun_top1.avg, 'noun_top5': noun_top5.avg})

    writer.add_scalar('train/loss_epoch', losses.avg, epoch)
    writer.add_scalar('train/learning_rate_epoch', opt.learning_rate, epoch)
    writer.add_scalar('train/verb_top1', verb_top1.avg, epoch)
    writer.add_scalar('train/verb_top5', verb_top5.avg, epoch)
    training_metrics = {'train_loss': losses.avg,
                        'train_noun_loss': noun_losses.avg,
                        'train_verb_loss': verb_losses.avg,
                        'train_acc': top1.avg,
                        'train_verb_acc': verb_top1.avg,
                        'train_noun_acc': noun_top1.avg
                        }
    return training_metrics
