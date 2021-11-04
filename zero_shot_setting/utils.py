from collections import deque
from torch.nn import init
import torch
import logging
import time
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import collections
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def load(filename, with_rels=True):
    ont = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    ont[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['alt_ids'] = list()
                obj['def'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                if obj is not None:
                    ont[obj['id']] = obj
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'alt_id':
                    obj['alt_ids'].append(l[1])
                elif l[0] == 'namespace':
                    obj['namespace'] = l[1]
                elif l[0] == 'def':
                    obj['def'].append(l[1].split('"')[1])
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif with_rels and l[0] == 'relationship':
                    it = l[1].split()
                    # add all types of relationships
                    obj['is_a'].append(it[1])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
    for term_id in list(ont.keys()):
        for t_id in ont[term_id]['alt_ids']:
            ont[t_id] = ont[term_id]
        if ont[term_id]['is_obsolete']:
            del ont[term_id]
    for term_id, val in ont.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in ont:
                if 'children' not in ont[p_id]:
                    ont[p_id]['children'] = set()
                ont[p_id]['children'].add(term_id)
    return ont


def get_children(go, term_id):
    if term_id not in go:
        return set()
    term_set = set()
    q = deque()
    q.append(term_id)
    while(len(q) > 0):
        t_id = q.popleft()
        if t_id not in term_set:
            term_set.add(t_id)
            for child_id in go[t_id]['children']:
                if child_id in go:
                    q.append(child_id)
    return term_set


def get_anchestors(go, term_id):
    if term_id not in go:
        return []
    term_set = []
    q = deque()
    q.append(term_id)
    while(len(q) > 0):
        t_id = q.popleft()
        if t_id not in term_set:
            term_set.append(t_id)
            for parent_id in go[t_id]['is_a']:
                if parent_id in go:
                    q.append(parent_id)
    return term_set


def save_anchestors(save_list, go):
    a_save = collections.OrderedDict()
    for i in save_list:
        a_save[i] = get_anchestors(go, i)
    save_obj(a_save, '../data/data_2016/anchestors.pkl')
    return a_save


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def vec2classes(go, terms, predictions, th=0.5, is_a=True, a_path='../data/data_2016/anchestors.pkl'):
    terms_list = np.asarray(list(terms.keys()))
    pred_classes = []
    go_term = {i: go[i] for i in list(terms.keys())}
    if is_a == False:
        anchestors = save_anchestors(list(terms.keys()), go_term)
    else:
        anchestors = load_obj(a_path)
    for i in range(np.size(predictions, 0)):
        predI = predictions[i, :]
        classI = terms_list[predI >= th]
        classA = list(terms.keys())
        x = np.asarray(list(anchestors.values()))
        t = anchestors[list(classI)]
        #for j in classI:
        #    classA += anchestors[j]
        pred_classes.append(list(set(classA)))
    debug = 0


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, stop_order='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_order = stop_order
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss
        if self.stop_order == 'min':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        elif self.stop_order == 'max':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Updation change ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


def get_logger(dataset='data-cafa'):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = 'results/training_log/'
    log_name = log_path + dataset + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_prc(labels, preds):
    # Compute PRC curve and PRC area for each class
    pr, rc, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    prc_auc = auc(rc, pr)
    return prc_auc