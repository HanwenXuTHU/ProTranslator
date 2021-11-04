import torch
import torch.nn as nn
from model import deepTNFSLModel, deepZSLModel
from options import model_config, data_loading
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, vec2classes, compute_prc
import utils
from tqdm import tqdm
import numpy as np
import collections
from sklearn.metrics import precision_recall_curve


class DeepTNFSL:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = deepTNFSLModel.deepTNFSLModel(input_nc=model_config.input_nc,
                                                in_nc=model_config.in_nc,
                                                max_kernels=model_config.max_kernels,
                                                hidden_dim=model_config.hidden_dim,
                                                feature=model_config.features,
                                                vector_dim=model_config.gene_vector_dim,
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
    file = FileLoader(data_opt)
    model_opt = model_config()
    feature_symb = '_'.join(model_opt.features)
    logger = utils.get_logger(data_opt.dataset, data_opt.logger_name.format(data_opt.fsl_n[0], data_opt.fsl_n[-1], feature_symb))
    print('dataset is {}'.format(data_opt.dataset))
    model_opt.emb_dim = file.emb_tensor_train.size(1)
    model_opt.vector_dim = np.size(list(file.prot_vector.values())[0].reshape(-1), 0)
    model_opt.description_dim = np.size(list(file.prot_description.values())[0].reshape(-1), 0)
    for fold_i in range(data_opt.k_fold):
        model_predict = DeepTNFSL(model_opt)
        train_dataset = DataLoader(file.fold_training[fold_i], batch_size=model_opt.batch_size, shuffle=True)
        inference_dataset = DataLoader(file.fold_validation[fold_i], batch_size=model_opt.batch_size, shuffle=True)
        print('loading data finished')
        logger.info('few shot learning limit number: {} fold : {}'.format(data_opt.fsl_n, fold_i))
        for i in range(model_opt.epoch):
            train_loss, inference_loss = 0, 0
            num = 0
            print('fsl number is {}'.format(data_opt.fsl_n))
            desc = 'Training Epoch :{} '.format(i)
            p_bar = tqdm(enumerate(train_dataset), desc=desc)
            for j, train_D in p_bar:
                model_predict.backward_model(train_D['seq'], train_D['description'], train_D['vector'], file.emb_tensor_train, train_D['label'])
                train_loss += float(model_predict.loss.item())
                p_bar.set_postfix(train_loss=model_predict.loss.item())
                num += 1
            train_loss = train_loss/num
            num = 0
            if i == model_opt.epoch - 1:
                desc = 'Inferring Epoch :{} '.format(i)
                p_bar = tqdm(enumerate(inference_dataset), desc=desc)
                inference_preds, inference_label = [], []
                for j, inference_D in p_bar:
                    with torch.no_grad():
                        preds = model_predict.model(inference_D['seq'], inference_D['description'], inference_D['vector'], file.emb_tensor_train)
                        label = inference_D['label']
                        loss = model_predict.loss_func(preds, label)
                        inference_preds.append(preds)
                        inference_label.append(label)
                        inference_loss += float(loss.item())
                        p_bar.set_postfix(infer_loss=loss.item())
                        num += 1
                inference_loss = inference_loss/num
                inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label), dim=0)
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
                    logger.info('fold:{} fsl number:{} iter:{} train loss:{} inference loss:{} inference roc auc:{} random roc:{}'.format(fold_i, fsl_n, i, train_loss, inference_loss, roc_auc, random_roc))
        torch.save(model_predict, model_opt.save_path.format(data_opt.dataset, fold_i, feature_symb))


if __name__ == '__main__':
    main()