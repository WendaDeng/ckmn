import numpy as np
import os
import pickle
import torch
import time

from sklearn.metrics import confusion_matrix, accuracy_score


def test(data_loader, model, opt):

    model.eval()
    results_dict = {}

    with torch.no_grad():
        for split, loader in data_loader.items():
            results = []
            total_num = len(loader.dataset)

            start_time = time.time()
            for i, data in enumerate(loader):
                if opt.use_object_feature:
                    inputs, targets, obj_features = data

                    inputs = inputs.to(opt.device)
                    obj_features = obj_features.to(opt.device)
                    outputs = model(inputs, obj_features)
                else:
                    inputs, targets = data

                    inputs = inputs.to(opt.device)
                    outputs = model(inputs)

                rst = {'verb': outputs[0].cpu().numpy().reshape(opt.event_classes[0]),
                       'noun': outputs[1].cpu().numpy().reshape(opt.event_classes[1])}
                results.append((rst,))

                cnt_time = time.time() - start_time
                print('video {} done, total {}/{}, average {} sec/video'.format(
                    i, i + 1, total_num, float(cnt_time) / (i + 1)))
            results_dict[split] = results

    return results_dict


def print_accuracy(scores, labels):

    video_pred = [np.argmax(np.mean(score, axis=0)) for score in scores]
    cf = confusion_matrix(labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_cnt[cls_hit == 0] = 1  # to avoid divisions by zero
    cls_acc = cls_hit / cls_cnt

    acc = accuracy_score(labels, video_pred)

    print('Accuracy {:.02f}%'.format(acc * 100))
    print('Average Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))


def save_scores(results_dict, scores_dir):

    for split, results in results_dict.items():
        save_dict = {}
        if len(results[0]) == 2:
            keys = results[0][0].keys()
            scores = {k: np.array([result[0][k] for result in results]) for k in keys}
            labels = {k: np.array([result[1][k] for result in results]) for k in keys}
        else:
            keys = results[0][0].keys()
            scores = {k: np.array([result[0][k] for result in results]) for k in keys}
            labels = None

        save_dict[split + '_scores'] = scores
        if labels is not None:
            save_dict[split + '_labels'] = labels

        scores_file = os.path.join(scores_dir, split + '.pkl')
        with open(scores_file, 'wb') as f:
            pickle.dump(save_dict, f)