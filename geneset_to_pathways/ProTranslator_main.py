import torch
import torch.nn as nn
from model import ProTranslatorModel
from options import model_config, data_loading
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, vec2classes
import utils
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from torch import squeeze
import pickle
import collections


class ProTranslator:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = ProTranslatorModel.ProTranslatorModel(input_nc=model_config.input_nc,
                                                in_nc=model_config.in_nc,
                                                max_kernels=model_config.max_kernels,
                                                hidden_dim=model_config.hidden_dim,
                                                seqL=model_config.max_len,
                                                emb_dim=model_config.emb_dim)
        if len(model_config.gpu_ids) > 0:
            self.model = self.model.cuda()
            init_weights(self.model, init_type='xavier')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.lr)

    def backward_model(self, input_seq, emb_tensor, label):
        preds = self.model(input_seq, emb_tensor)
        self.loss = self.loss_func(preds, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


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
            self.description.append(prot_description[protT])

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


def emb2tensor(brief_embeddings, full_embeddings, terms, text_mode='brief'):
    ann_id = list(terms.keys())
    print('Text mode is {}'.format(text_mode))
    if text_mode == 'brief':
        embedding_array = np.zeros((len(ann_id), np.size(brief_embeddings[ann_id[0]], 1)))
    elif text_mode == 'full':
        embedding_array = np.zeros((len(ann_id), np.size(full_embeddings[ann_id[0]], 1)))
    elif text_mode == 'both':
        embedding_array = np.zeros((len(ann_id), np.size(full_embeddings[ann_id[0]], 1) + np.size(brief_embeddings[ann_id[0]], 1)))
    for t_id in ann_id:
        if text_mode == 'brief':
            t_brief = brief_embeddings[t_id].reshape([1, -1])
            t_brief = t_brief / np.sqrt(np.sum(np.power(t_brief, 2), axis=1))
            embedding_array[terms[t_id], :] = t_brief
        elif text_mode == 'full':
            t_full = full_embeddings[t_id].reshape([1, -1])
            t_full = t_full / np.sqrt(np.sum(np.power(t_full, 2), axis=1))
            embedding_array[terms[t_id], :] = t_full
        elif text_mode == 'both':
            t_full = full_embeddings[t_id].reshape([1, -1])
            t_brief = brief_embeddings[t_id].reshape([1, -1])
            t = np.hstack((t_brief, t_full))
            t = t / np.sqrt(np.sum(np.power(t, 2), axis=1))
            embedding_array[terms[t_id], :] = t
    rank_e = np.linalg.matrix_rank(embedding_array)
    print('Rank of your embeddings is {}'.format(rank_e))
    embedding_array = torch.from_numpy(embedding_array)
    return embedding_array


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def main():
    model_path = "/home/hwxu/proteinZSL/deepTNPathway/results/goa_human_cat_for_pathway_predict_pathway_deeptnfsl.pth"
    model_predict = torch.load(model_path)
    save_T = 0.95
    dataset = 'kegg_remove_overlap'
    text_mode = 'full'
    gpu_ids = '0'
    batch_size = 32
    logger = utils.get_logger(dataset='{}_{}'.format(dataset, text_mode))

    print('Loading dataset')
    dataset_path = '../data/{}/dataset.pkl'.format(dataset)
    terms_path = '../data/{}/terms.pkl'.format(dataset)
    emb_path = '../data/{}/{}_{}_embeddings_second2last_PubmedFull.pkl'.format(dataset, dataset, text_mode)
    terms_auroc_path = 'results/terms_auroc.pkl'.format()
    infer_data = pd.read_pickle(dataset_path)
    terms = pd.read_pickle(terms_path)
    embeddings = pd.read_pickle(emb_path)
    prot_vector = load_obj('../data/{}/prot_vector.pkl'.format(dataset))
    prot_description = load_obj('../data/{}/prot_description.pkl'.format(dataset))
    term_list = list(terms['terms'])
    infer_terms = {term_list[i]: i for i in range(len(term_list))}
    inference_dataset = proteinData(infer_data, infer_terms, prot_vector=prot_vector, prot_description=prot_description, gpu_ids=gpu_ids)
    inference_dataset = DataLoader(inference_dataset, batch_size=batch_size, shuffle=True)
    if text_mode == 'full':
        brief_emb = None
        full_emb = embeddings
    elif text_mode == 'brief':
        brief_emb = embeddings
        full_emb = None
    emb_pathway_tensor = emb2tensor(brief_embeddings=brief_emb, full_embeddings=full_emb, terms=infer_terms, text_mode=text_mode)
    if len(gpu_ids) > 0: emb_pathway_tensor = emb_pathway_tensor.float().cuda()
    print('loading data finished')
    num = 0
    print('inferring iters')
    inference_preds, inference_label, inference_loss = [], [], 0
    for j, inference_D in tqdm(enumerate(inference_dataset)):
        with torch.no_grad():
            preds = model_predict.model(inference_D['seq'], inference_D['description'], inference_D['vector'], emb_pathway_tensor)
            loss = model_predict.loss_func(preds, inference_D['label'])
            inference_preds.append(preds)
            inference_label.append(inference_D['label'])
            inference_loss += float(loss.item())
            num += 1
    inference_loss = inference_loss/num
    inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label), dim=0)
    inference_preds, inference_label = inference_preds.cpu().float().numpy(), inference_label.cpu().float().numpy()
    random_preds = np.random.rand(np.size(inference_preds, 0), np.size(inference_preds, 1))
    roc_auc = compute_roc(inference_label, inference_preds)
    random_roc = compute_roc(inference_label, random_preds)
    logger.info('inference loss:{} inference roc auc:{} random roc:{}'.format(inference_loss, roc_auc, random_roc))
    auroc_percentage = {0.65: 0, 0.7: 0, 0.75: 0, 0.8: 0, 0.85: 0, 0.9: 0, 0.95: 0}
    save_terms = []
    infer_terms_auroc = collections.OrderedDict()
    for t_id in infer_terms.keys():
        j = infer_terms[t_id]
        preds_id = inference_preds[:, j].reshape([-1, 1])
        label_id = inference_label[:, j].reshape([-1, 1])
        roc_auc_j = compute_roc(label_id, preds_id)
        if roc_auc_j >= save_T:
            save_terms.append(t_id)
        for T in auroc_percentage.keys():
            if T <= roc_auc_j:
                auroc_percentage[T] += 100.0 / np.size(inference_preds, 1)
        infer_terms_auroc[t_id] = roc_auc_j
    for T in auroc_percentage.keys():
        logger.info('roc auc:{} percentage:{}%'.format(T, auroc_percentage[T]))

    save_obj(infer_terms_auroc, terms_auroc_path)




if __name__ == '__main__':
    main()