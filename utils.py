import csv
import numpy as np
from sklearn.metrics import average_precision_score
import ipdb
import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> object:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def load_list_file(file_path):
    with open(file_path, 'r') as input_file:
        lists = input_file.readlines()

    return lists


def performance(prediction, target):
    # ipdb.set_trace()
    prediction = torch.sigmoid(prediction)
    mAP_new = calculate_mAP_sklearn_new(prediction, target)
    print('sigmoid-sklearn:', mAP_new)

    prediction = F.softmax(prediction, dim=1)
    mAP_new_st = calculate_mAP_sklearn_new(prediction, target)
    print('softmax-sklearn:', mAP_new_st)

    return mAP_new, mAP_new_st


def my_accuracy(preds: torch.Tensor, truths: torch.Tensor, topk: tuple = (1,)) -> tuple:
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    target = (target == 1.0).nonzero().t()[1]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return tuple(res)


def multitask_accuracy(outputs, labels, topk=(1,)):
    """
    Args:
        outputs: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor has shape [batch_size, class_count]
            but we need shape of [batch_size]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(topk))
    task_count = len(outputs)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(outputs, labels):
        # make label from [batch_size, class_count] to [batch_size]
        label = (label == 1.0).nonzero().t()[1]
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    accuracies = []
    for k in topk:
        all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
        accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
        accuracies.append(accuracy_at_k)
    return tuple(accuracies)


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size


def calculate_mAP(outputs, targets):
    #targets = targets.int().cpu().data.numpy()
    class_num = np.size(targets, 0)
    mAP = []

    for idx in range(class_num):

        temp_outputs = outputs[idx]
        temp_targets = targets[idx]

        temp = zip(temp_outputs, temp_targets)
        temp = sorted(temp, reverse=True)
        temp_outputs_new, temp_targets_new = zip(*temp)

        count = 0
        average_precise = 0

        for i in range(len(temp_targets_new)):
            if temp_targets_new[i] > 0:
                count += 1
                average_precise = average_precise + count / (i + 1)

        mAP.append(average_precise / count)
    mAP = np.mean(np.array(mAP))

    return mAP

def calculate_mAP_new(outputs, targets):
    #targets = targets.int().cpu().data.numpy()
    class_num = np.size(targets, 1)
    mAP = []
    for idx in range(class_num):
        temp_outputs = outputs[:, idx]
        temp_targets = targets[:, idx]

        temp = zip(temp_outputs, temp_targets)
        temp = sorted(temp, reverse=True)
        temp_outputs_new, temp_targets_new = zip(*temp)

        count = 0
        average_precise = 0

        for i in range(len(temp_targets_new)):
            if temp_targets_new[i] > 0:
                count += 1
                average_precise = average_precise + count / (i + 1)

        mAP.append(average_precise / count)
    mAP = np.mean(np.array(mAP))

    return mAP

def calculate_mAP_new_noback(outputs, targets):
    #targets = targets.int().cpu().data.numpy()
    class_num = np.size(targets, 1)
    mAP = []

    for idx in range(class_num-1):

        temp_outputs = outputs[:, idx]
        temp_targets = targets[:, idx]

        temp = zip(temp_outputs, temp_targets)
        temp = sorted(temp, reverse=True)
        temp_outputs_new, temp_targets_new = zip(*temp)

        count = 0
        average_precise = 0

        for i in range(len(temp_targets_new)):
            if temp_targets_new[i] > 0:
                count += 1
                average_precise = average_precise + count / (i + 1)

        mAP.append(average_precise / count)
    mAP = np.mean(np.array(mAP))

    return mAP

def calculate_mAP_sklearn(outputs, targets):
    #ipdb.set_trace()
    class_num = np.size(targets, 0)
    mAP = []

    for idx in range(class_num):
        mAP.append(average_precision_score(targets[idx, :], outputs[idx, :]))

    mAP = np.mean(mAP)

    return mAP

def calculate_mAP_sklearn_new(outputs, targets):
    # ipdb.set_trace()
    class_num = np.size(targets, 1)
    mAP = []
    for idx in range(class_num):
        # if idx in [94, 106, 108, 110, 118, 123]:
        if torch.sum(targets[:, idx]) < 1.0:
            continue
        mAP.append(average_precision_score(targets[:, idx], outputs[:, idx]))

    mAP = np.mean(mAP)

    return mAP

def calculate_mAP_sklearn_new_noback(outputs, targets):
    #ipdb.set_trace()
    class_num = np.size(targets, 1)
    mAP = []

    for idx in range(class_num-1):
        mAP.append(average_precision_score(targets[:, idx], outputs[:, idx]))

    mAP = np.mean(mAP)

    return mAP

def get_fine_tuning_parameters(model, ft_module_names):
    # for k, v in model.named_parameters():
    #     print(k)

    parameters = []
    #parameters_name_one = []
    #parameters_name_two = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                #parameters_name_one.append(k)
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
            #parameters_name_two.append(k)

    return parameters
