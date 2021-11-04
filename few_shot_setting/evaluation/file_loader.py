import click as ck
import numpy as np
import pandas as pd
from options import data_loading
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from torch import squeeze
from utils import load
import torch
import random
import utils
import collections
from tqdm import tqdm


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


def one_hot(seq, start=0, max_len=2000):
    AALETTER = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    AAINDEX = dict()
    for i in range(len(AALETTER)):
        AAINDEX[AALETTER[i]] = i + 1
    onehot = np.zeros((21, max_len), dtype=np.int32)
    l = min(max_len, len(seq))
    for i in range(start, start + l):
        onehot[AAINDEX.get(seq[i - start], 0), i] = 1
    onehot[0, 0:start] = 1
    onehot[0, start + l:] = 1
    return onehot


class proteinData(Dataset):
    def __init__(self, data_df, terms, gpu_ids='0'):
        self.pSeq = []
        self.label = []
        self.gpu_ids = gpu_ids
        sequences = list(data_df['sequences'])
        annots = list(data_df['annotations'])
        for i in range(len(annots)):
            annots[i] = list(annots[i])
        for i in range(data_df.shape[0]):
            seqT, annT = sequences[i], annots[i]
            labelT = np.zeros([len(terms), 1])
            for j in range(len(annT)):
                if annT[j] in terms.keys():
                    labelT[terms[annT[j]]] = 1
            self.pSeq.append(one_hot(seqT))
            self.label.append(labelT)

    def __getitem__(self, item):
        in_seq, label = transforms.ToTensor()(self.pSeq[item]), transforms.ToTensor()(self.label[item])
        if len(self.gpu_ids) > 0:
            return {'seq': squeeze(in_seq).float().cuda(), 'label': squeeze(label).float().cuda()}
        else:
            return {'seq': squeeze(in_seq).float(), 'label': squeeze(label).float()}

    def __len__(self):
        return len(self.pSeq)


def list_contain(a, b):
    for j in a:
        if j not in b:
            return False
    return True


def is_limit_count(inf_id, few_shot_count, limit=10):
    for i in inf_id:
        if few_shot_count[i] + 1 > limit:
            return False
    return True


def load_data(train_data_file, test_data_file, train_terms, inference_terms, limit=10):
    train_df, test_df = pd.read_pickle(train_data_file), pd.read_pickle(test_data_file)
    train_zsl_df, inference_zsl_df = collections.OrderedDict(), collections.OrderedDict()
    train_zsl_df['sequences'], train_zsl_df['annotations'] = [], []
    inference_zsl_df['sequences'], inference_zsl_df['annotations'] = [], []
    few_shot_count = collections.OrderedDict()
    for i in range(len(inference_terms)):
        few_shot_count[inference_terms[i]] = 0
    for i in tqdm(list(train_df.index)):
        ann_id = train_df.loc[i]['annotations']
        inf_id = list(set(ann_id).intersection(set(inference_terms)))
        if len(inf_id) == 0:
            train_zsl_df['sequences'].append(train_df.loc[i]['sequences'])
            train_zsl_df['annotations'].append(ann_id)
        if len(inf_id) > 0:
            if is_limit_count(inf_id, few_shot_count, limit=limit):
                train_zsl_df['sequences'].append(train_df.loc[i]['sequences'])
                train_zsl_df['annotations'].append(ann_id)
                for j in range(len(inf_id)): few_shot_count[inf_id[j]] += 1
    print('training size is {}'.format(len(train_zsl_df['sequences'])))
    for i in tqdm(list(test_df.index)):
        ann_id = test_df.loc[i]['annotations']
        inf_id = list(set(ann_id).intersection(set(inference_terms)))
        if len(inf_id) > 0:
            inference_zsl_df['sequences'].append(test_df.loc[i]['sequences'])
            inference_zsl_df['annotations'].append(inf_id)
    print('inference size is {}'.format(len(inference_zsl_df['sequences'])))
    return pd.DataFrame(train_zsl_df), pd.DataFrame(inference_zsl_df)


def emb2tensor(def_embeddings, name_embeddings, terms, text_mode='name'):
    ann_id = list(terms.keys())
    print('Text mode is {}'.format(text_mode))
    if text_mode == 'name':
        embedding_array = np.zeros((len(ann_id), np.size(name_embeddings[ann_id[0]], 1)))
    elif text_mode == 'def':
        embedding_array = np.zeros((len(ann_id), np.size(def_embeddings[ann_id[0]], 1)))
    elif text_mode == 'both':
        embedding_array = np.zeros((len(ann_id), np.size(def_embeddings[ann_id[0]], 1) + np.size(name_embeddings[ann_id[0]], 1)))
    for t_id in ann_id:
        if text_mode == 'name':
            t_name = name_embeddings[t_id].reshape([1, -1])
            t_name = t_name / np.sqrt(np.sum(np.power(t_name, 2), axis=1))
            embedding_array[terms[t_id], :] = t_name
        elif text_mode == 'def':
            t_def = def_embeddings[t_id].reshape([1, -1])
            t_def = t_def / np.sqrt(np.sum(np.power(t_def, 2), axis=1))
            embedding_array[terms[t_id], :] = t_def
        elif text_mode == 'both':
            t_def = def_embeddings[t_id].reshape([1, -1])
            t_name = name_embeddings[t_id].reshape([1, -1])
            t = np.hstack((t_name, t_def))
            t = t / np.sqrt(np.sum(np.power(t, 2), axis=1))
            embedding_array[terms[t_id], :] = t
    rank_e = np.linalg.matrix_rank(embedding_array)
    print('Rank of your embeddings is {}'.format(rank_e))
    embedding_array = torch.from_numpy(embedding_array)
    return embedding_array


class FileLoader:

    def __init__(self, opt):
        terms_df = pd.read_pickle(opt.terms_file)
        self.go_data = load(opt.go_file)
        term_list_T = list(terms_df['terms'])
        self.term_list = []
        for i in range(len(term_list_T)):
            if term_list_T[i] in self.go_data.keys():
                self.term_list.append(term_list_T[i])
        self.training_term_list, self.inference_term_list = self.few_shot_terms()
        self.train_terms = {self.training_term_list[i]: i for i in range(len(self.training_term_list))}
        self.inference_terms = {self.inference_term_list[i]: i for i in range(len(self.inference_term_list))}
        self.train_classes, self.inference_classes = len(self.training_term_list), len(self.inference_term_list)
        train_zsl_df, inference_zsl_df = load_data(opt.train_data_file, opt.test_data_file, self.training_term_list, self.inference_term_list, limit=opt.fsl_limit)
        self.train_data = proteinData(train_zsl_df, self.train_terms, gpu_ids=opt.gpu_ids)
        self.inference_data = proteinData(inference_zsl_df, self.inference_terms, gpu_ids=opt.gpu_ids)
        self.def_embeddings, self.name_embeddings = None, None
        if opt.text_mode in ['both', 'def']:
            self.def_embeddings = pd.read_pickle(opt.def_embedding_file)
        if opt.text_mode in ['both', 'name']:
            self.name_embeddings = pd.read_pickle(opt.name_embedding_file)
        self.emb_tensor_train = emb2tensor(self.def_embeddings, self.name_embeddings, self.train_terms, text_mode=opt.text_mode)
        self.emb_tensor_inference = emb2tensor(self.def_embeddings, self.name_embeddings, self.inference_terms, text_mode=opt.text_mode)
        if len(opt.gpu_ids) > 0:
            self.emb_tensor_train = self.emb_tensor_train.float().cuda()
            self.emb_tensor_inference = self.emb_tensor_inference.float().cuda()
        print('Data Loading Finished!')


    def few_shot_terms(self, r=0.3):
        go_terms = {i : self.go_data[i] for i in self.term_list}
        term_length = len(self.term_list)
        leaf_node = []
        for i in range(term_length):
            term_T = self.term_list[i]
            T_children = utils.get_children(go_terms, term_T)
            if len(T_children) == 1 and term_T not in leaf_node: leaf_node.append(term_T)
        training_term = self.term_list
        inference_num = int(len(leaf_node) * r)
        random.seed(123)
        random.shuffle(leaf_node)
        random.seed()
        return training_term, leaf_node[0 : inference_num]
