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
from utils import init_weights, EarlyStopping, compute_roc, vec2classes
from model import ProTranslatorModel
from torch.utils.data import DataLoader
from options import model_config, data_loading


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
        self.prot_id = []
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
            self.prot_id.append(protT)

    def __getitem__(self, item):
        in_seq, label = transforms.ToTensor()(self.pSeq[item]), transforms.ToTensor()(self.label[item])
        description = torch.from_numpy(self.description[item])
        vector = torch.from_numpy(self.vector[item])
        if len(self.gpu_ids) > 0:
            return {'seq': squeeze(in_seq).float().cuda(),
                    'description':squeeze(description).float().cuda(),
                    'vector':squeeze(vector).float().cuda(),
                    'label': squeeze(label).float().cuda(),
                    'ids': self.prot_id[item]}
        else:
            return {'seq': squeeze(in_seq).float(),
                    'description': squeeze(description).float(),
                    'vector': squeeze(vector).float(),
                    'label': squeeze(label).float(),
                    'ids': self.prot_id[item]}

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


class ProTranslator:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = ProTranslatorModel.ProTranslatorModel(input_nc=model_config.input_nc,
                                                in_nc=model_config.in_nc,
                                                max_kernels=model_config.max_kernels,
                                                hidden_dim=model_config.hidden_dim,
                                                feature=model_config.features,
                                                seqL=model_config.max_len,
                                                emb_dim=model_config.emb_dim)
        if len(model_config.gpu_ids) > 0:
            self.model = self.model.cuda()
            init_weights(self.model, init_type='xavier')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.lr)

    def backward_model(self, input_seq, input_description, input_vector, emb_tensor, label):
        preds = self.model(input_seq, input_description, input_vector, emb_tensor)
        self.loss = self.loss_func(preds, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


def main():
    data_opt = data_loading(dataset='goa_mouse_cat')
    model_opt = model_config()
    feature_symb = '_'.join(model_opt.features)
    print('features: {}'.format(feature_symb))
    logger = utils.get_logger(data_opt.dataset, data_opt.logger_name.format(data_opt.fsl_n[0], data_opt.fsl_n[-1], feature_symb))
    print('dataset is {}'.format(data_opt.dataset))
    file = FileLoader(data_opt)
    convert_train_terms = []
    save_path = 'results/'
    for i in range(len(file.train_terms.keys())):
        convert_train_terms.append('0')
    for i in file.train_terms.keys():
        convert_train_terms[file.train_terms[i]] = i
    model_opt.emb_dim = file.emb_tensor_train.size(1)
    model_opt.vector_dim = np.size(list(file.prot_vector.values())[0].reshape(-1), 0)
    model_opt.description_dim = np.size(list(file.prot_description.values())[0].reshape(-1), 0)
    for fold_i in range(data_opt.k_fold):
        model_predict = torch.load('results/{}_predict_deepfsl_fold_{}_{}.pth'.format(data_opt.dataset, fold_i, feature_symb))
        inference_dataset = DataLoader(file.fold_validation[fold_i], batch_size=model_opt.batch_size, shuffle=True)
        print('loading data finished')

        desc = 'Inferring Epoch'
        p_bar = tqdm(enumerate(inference_dataset), desc=desc)
        inference_preds, inference_label = [], []
        infer_id = []
        for j, inference_D in p_bar:
            with torch.no_grad():
                preds = model_predict.model(inference_D['seq'], inference_D['description'], inference_D['vector'], file.emb_tensor_train)
                label = inference_D['label']
                infer_id += inference_D['ids']
                loss = model_predict.loss_func(preds, label)
                inference_preds.append(preds)
                inference_label.append(label)
                p_bar.set_postfix(infer_loss=loss.item())
        inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label), dim=0)
        predcitions = pd.DataFrame(inference_preds.cpu().float().numpy(), index=infer_id, columns=convert_train_terms)
        labels = pd.DataFrame(inference_label.cpu().float().numpy(), index=infer_id, columns=convert_train_terms)
        predcitions.to_pickle(save_path + '{}_fold_{}_predictions.pkl'.format(data_opt.dataset, fold_i))
        labels.to_pickle(save_path + '{}_fold_{}_labels.pkl'.format(data_opt.dataset, fold_i))

        for fsl_n in data_opt.fsl_n:
            preds_inf, label_inf = [], []
            for t_id in file.fold_few_shot_terms_list[fsl_n][fold_i]:
                preds_inf.append(inference_preds[:, file.train_terms[t_id]].reshape((-1, 1)))
                label_inf.append(inference_label[:, file.train_terms[t_id]].reshape((-1, 1)))
            preds_inf = torch.cat(preds_inf, dim=1)
            label_inf = torch.cat(label_inf, dim=1)
            preds_inf, label_inf = preds_inf.cpu().float().numpy(), label_inf.cpu().float().numpy()
            random_preds = np.random.rand(np.size(preds_inf, 0), np.size(preds_inf, 1))
            roc_auc = compute_roc(label_inf, preds_inf)
            random_roc = compute_roc(label_inf, random_preds)
            logger.info('fold:{} fsl number:{} inference roc auc:{} random roc:{}'.format(fold_i, fsl_n, roc_auc, random_roc))


if __name__ == '__main__':
    main()