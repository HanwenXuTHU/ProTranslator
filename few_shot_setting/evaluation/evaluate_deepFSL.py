import torch
import torch.nn as nn
from model import deepFSLModel, deepZSLModel
from options import model_config, data_loading
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, vec2classes
import utils
from tqdm import tqdm
import numpy as np


class DeepFSL:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = deepZSLModel.deepZSLModel(input_nc=model_config.input_nc,
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


def main():
    fsl_number = [10, 4, 3, 2, 1]
    for fsl_n in fsl_number:
        data_opt = data_loading(dataset='data-cafa')
        data_opt.fsl_limit = fsl_n
        print('dataset is {}'.format(data_opt.dataset))
        model_opt = model_config()
        file = FileLoader(data_opt)
        model_predict = DeepFSL(model_opt)
        train_dataset = DataLoader(file.train_data, batch_size=model_opt.batch_size, shuffle=True)
        inference_dataset = DataLoader(file.inference_data, batch_size=model_opt.batch_size, shuffle=True)
        print('loading data finished')
        logger = utils.get_logger(data_opt.dataset, fsl_n)
        logger.info('few shot learning limit number: {}'.format(data_opt.fsl_limit))
        early_stopping = EarlyStopping(patience=1000, verbose=True,
                                       path='results/{}_{}_predict_deepfsl.pth'.format(data_opt.dataset, fsl_n), stop_order='min')
        for i in range(model_opt.epoch):
            train_loss, inference_loss = 0, 0
            num = 0
            print('fsl number is {}'.format(fsl_n))
            print('Training iters')
            for j, train_D in tqdm(enumerate(train_dataset)):
                model_predict.backward_model(train_D['seq'], file.emb_tensor_train, train_D['label'])
                train_loss += float(model_predict.loss.item())
                num += 1
            train_loss = train_loss/num
            num = 0
            print('inferring iters')
            inference_preds, inference_label = [], []
            for j, inference_D in tqdm(enumerate(inference_dataset)):
                with torch.no_grad():
                    preds = model_predict.model(inference_D['seq'], file.emb_tensor_train)
                    preds_inf = []
                    for t_id in file.inference_term_list:
                        preds_inf.append(preds[:, file.train_terms[t_id]].reshape((-1, 1)))
                    preds_inf = torch.cat(preds_inf, dim=1)
                    loss = model_predict.loss_func(preds_inf, inference_D['label'])
                    inference_preds.append(preds_inf)
                    inference_label.append(inference_D['label'])
                    inference_loss += float(loss.item())
                    num += 1
            inference_loss = inference_loss/num
            inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label), dim=0)
            inference_preds, inference_label = inference_preds.cpu().float().numpy(), inference_label.cpu().float().numpy()
            random_preds = np.random.rand(np.size(inference_preds, 0), np.size(inference_preds, 1))
            roc_auc = compute_roc(inference_label, inference_preds)
            random_roc = compute_roc(inference_label, random_preds)
            logger.info('iter:{} train loss:{} inference loss:{} inference roc auc:{} random roc:{}'.format(i, train_loss, inference_loss, roc_auc, random_roc))
            early_stopping(val_loss=inference_loss, model=model_predict)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break


if __name__ == '__main__':
    main()