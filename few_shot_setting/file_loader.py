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
import pickle


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
    def __init__(self, data_df, terms, prot_vector, prot_description, gpu_ids='0'):
        self.pSeq = []
        self.label = []
        self.vector = []
        self.description = []
        self.gpu_ids = gpu_ids
        sequences = list(data_df['sequences'])
        prot_ids = list(data_df['proteins'])
        annots = list(data_df['annotations'])
        for i in range(len(annots)):
            annots[i] = list(annots[i])
        for i in range(data_df.shape[0]):
            seqT, annT, protT = sequences[i], annots[i], prot_ids[i]
            labelT = np.zeros([len(terms), 1])
            for j in range(len(annT)):
                if annT[j] in terms.keys():
                    labelT[terms[annT[j]]] = 1
            self.pSeq.append(one_hot(seqT))
            self.label.append(labelT)
            self.vector.append(prot_vector[protT])
            self.description.append((prot_description[protT]))

    def __getitem__(self, item):
        in_seq, label = transforms.ToTensor()(self.pSeq[item]), transforms.ToTensor()(self.label[item])
        description = torch.from_numpy(self.description[item])
        vector = torch.from_numpy(self.vector[item])
        if len(self.gpu_ids) > 0:
            return {'seq': squeeze(in_seq).float().cuda(),
                    'description':squeeze(description).float().cuda(),
                    'vector':squeeze(vector).float().cuda(),
                    'label': squeeze(label).float().cuda()}
        else:
            return {'seq': squeeze(in_seq).float(),
                    'description': squeeze(description).float(),
                    'vector': squeeze(vector).float(),
                    'label': squeeze(label).float()}

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


def prop_annots(train_df, test_df, go_data):
    train_df_n, test_df_n = train_df, test_df
    is_prop = 0
    for i in tqdm(train_df_n.index):
        annots = train_df_n.loc[i]['annotations']
        for j in annots:
            j_a = utils.get_anchestors(go_data, j)
            for t in j_a:
                if t not in annots:
                    train_df_n.loc[i]['annotations'].append(t)
                    is_prop = 1
    for i in tqdm(test_df_n.index):
        annots = test_df_n.loc[i]['annotations']
        for j in annots:
            j_a = utils.get_anchestors(go_data, j)
            for t in j_a:
                if t not in annots:
                    test_df_n.loc[i]['annotations'].append(t)
                    is_prop = 1
    if is_prop:
        print('Add anchestors to annotations')
    else:
        print('No need to process anchestors')
    return train_df_n, test_df_n


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


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class FileLoader:

    def __init__(self, opt):
        terms_df = pd.read_pickle(opt.terms_file)
        self.go_data = load(opt.go_file)
        term_list_T = list(terms_df['terms'])
        self.term_list = []
        self.prot_vector = load_obj(opt.prot_vector_file)
        self.prot_description = load_obj(opt.prot_description_file)
        for i in range(len(term_list_T)):
            if term_list_T[i] in self.go_data.keys():
                self.term_list.append(term_list_T[i])
        self.train_terms = collections.OrderedDict()
        for i in range(len(self.term_list)): self.train_terms[self.term_list[i]] = i
        self.fold_training, self.fold_data = self.load_fold_data(opt.k_fold, opt.train_fold_file, opt.validation_fold_file)
        self.fold_few_shot_terms_list = collections.OrderedDict()
        for i in opt.fsl_n:
            self.fold_few_shot_terms_list[i] = self.few_shot_terms(limit=i, k=opt.k_fold)
        self.fold_validation = self.fold_data
        for fold_i in range(opt.k_fold):
            for i in opt.fsl_n:
                print('Fold {} contains {} few shot terms for number {}'.format(fold_i, len(self.fold_few_shot_terms_list[i][fold_i]), i))
            self.fold_training[fold_i] = proteinData(self.fold_training[fold_i], self.train_terms, self.prot_vector, self.prot_description, gpu_ids=opt.gpu_ids)
            self.fold_validation[fold_i] = proteinData(self.fold_validation[fold_i], self.train_terms, self.prot_vector, self.prot_description, gpu_ids=opt.gpu_ids)
        self.def_embeddings, self.name_embeddings = None, None
        if opt.text_mode in ['both', 'def']:
            self.def_embeddings = pd.read_pickle(opt.def_embedding_file)
        if opt.text_mode in ['both', 'name']:
            self.name_embeddings = pd.read_pickle(opt.name_embedding_file)
        self.emb_tensor_train = emb2tensor(self.def_embeddings, self.name_embeddings, self.train_terms, text_mode=opt.text_mode)
        if len(opt.gpu_ids) > 0:
            self.emb_tensor_train = self.emb_tensor_train.float().cuda()
        print('Data Loading Finished!')

    def few_shot_terms(self, limit=30, k=3):
        go_terms = {i : self.go_data[i] for i in self.term_list}
        fold_shot_count = []
        fold_few_shot_terms = []
        for i in tqdm(range(k)):
            validation_set = self.fold_data[i]
            validation_terms = collections.OrderedDict()
            few_shot_terms = collections.OrderedDict()
            for annt_val in list(validation_set['annotations']):
                for a_id in annt_val:
                    validation_terms[a_id] = 0
            few_shot_count = collections.OrderedDict()
            for j in range(len(self.term_list)):
                few_shot_count[self.term_list[j]] = 0
            training_set = self.fold_training[i]
            training_annt = list(training_set['annotations'])
            for annt in training_annt:
                annt = list(set(annt))
                for a_id in annt:
                    if a_id in go_terms.keys(): few_shot_count[a_id] += 1
            fold_shot_count.append(few_shot_count)
            for j in few_shot_count.keys():
                count_j = few_shot_count[j]
                if 0 < count_j <= limit and j in validation_terms.keys():
                    few_shot_terms[j] = count_j
            fold_few_shot_terms.append(few_shot_terms)
        return fold_few_shot_terms

    def load_fold_data(self, k, train_fold_file, validation_fold_file):
        train_fold, val_fold = [], []
        for i in range(k):
            train_fold.append(pd.read_pickle(train_fold_file.format(i)))
            val_fold.append(pd.read_pickle(validation_fold_file.format(i)))
        return train_fold, val_fold

    def k_fold_split(self, train_df, k):
        train_num = train_df.index.size
        idx = np.arange(train_num)
        np.random.seed(123)
        np.random.shuffle(idx)
        np.random.seed()
        fold_size = int(train_num / k)
        fold, fold_idx, fold_training = [], [], []
        for i in range(k):
            fold_idx.append(idx[i*fold_size : (i + 1)*fold_size])
        for i in range(k):
            fold.append(train_df.iloc[fold_idx[i]])
            training_set, is_append = [], 0
            for j in range(k):
                if j != i:
                    training_set.append(train_df.iloc[fold_idx[j]])
            training_set = pd.concat(training_set)
            fold_training.append(training_set)
        return fold, fold_training
