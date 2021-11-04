import torch
import torch.nn as nn
from model import deepZSLModel, deepSDNModel
from options import model_config, data_loading
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, vec2classes, compute_prc
import utils
from tqdm import tqdm
import numpy as np
import gc


class DeepZSL:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = deepSDNModel.deepSDNModel(input_nc=model_config.input_nc,
                                                in_nc=model_config.in_nc,
                                                max_kernels=model_config.max_kernels,
                                                hidden_dim=model_config.hidden_dim,
                                                feature=model_config.features,
                                                seqL=model_config.max_len,
                                                emb_dim=model_config.emb_dim)
        if len(model_config.gpu_ids) > 0:
            init_weights(self.model, init_type='xavier')
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.lr)

    def backward_model(self, input_seq, input_description, input_vector, emb_tensor, label):
        preds = self.model(input_seq, input_description, input_vector, emb_tensor)
        self.loss = self.loss_func(preds, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


def main():
    data_opt = data_loading(dataset='goa_human_cat')
    print('dataset is {}'.format(data_opt.dataset))
    model_opt = model_config()
    file = FileLoader(data_opt)
    print('loading data finished')
    feature_symb = '_'.join(model_opt.features)
    logger = utils.get_logger(data_opt.dataset +
                              data_opt.logger_name.format(feature_symb))
    for fold_i in [0, 1, 2]:
        model_opt.emb_dim = file.emb_tensor_train.size(1)
        model_opt.vector_dim = np.size(list(file.prot_vector.values())[0].reshape(-1), 0)
        model_opt.description_dim = np.size(list(file.prot_description.values())[0].reshape(-1), 0)
        model_predict = DeepZSL(model_opt)
        train_dataset = DataLoader(file.fold_training[fold_i], batch_size=model_opt.batch_size, shuffle=True)
        inference_dataset = DataLoader(file.fold_validation[fold_i], batch_size=model_opt.batch_size, shuffle=True)
        logger.info('fold : {}'.format(fold_i))
        for i in range(model_opt.epoch):
            train_loss, inference_loss = 0, 0
            num = 0
            print('Training iters')
            for j, train_D in tqdm(enumerate(train_dataset)):
                model_predict.backward_model(train_D['seq'], train_D['description'], train_D['vector'], file.emb_tensor_train, train_D['label'])
                train_loss += float(model_predict.loss.cpu())
                num += 1
            train_loss = train_loss/num
            num = 0
            if i > -1:
                print('inferring iters')
                inference_preds, inference_label = [], []
                for j, inference_D in tqdm(enumerate(inference_dataset)):
                    with torch.no_grad():
                        preds = model_predict.model(inference_D['seq'], inference_D['description'], inference_D['vector'], file.emb_tensor_train)
                        preds_inf, label_inf = [], []
                        label = inference_D['label']
                        for t_id in file.fold_zero_shot_terms_list[fold_i]:
                            preds_inf.append(preds[:, file.train_terms[t_id]].reshape((-1, 1)))
                            label_inf.append(label[:, file.train_terms[t_id]].reshape((-1, 1)))
                        preds_inf = torch.cat(preds_inf, dim=1)
                        label_inf = torch.cat(label_inf, dim=1)
                        loss = model_predict.loss_func(preds_inf, label_inf)
                        inference_preds.append(preds_inf)
                        inference_label.append(label_inf)
                        inference_loss += float(loss.item())
                        num += 1
                inference_loss = inference_loss/num
                inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label), dim=0)
                inference_preds, inference_label = inference_preds.cpu().float().numpy(), inference_label.cpu().float().numpy()
                random_preds = np.random.rand(np.size(inference_preds, 0), np.size(inference_preds, 1))
                roc_auc = compute_roc(inference_label, inference_preds)
                random_roc = compute_roc(inference_label, random_preds)
                prc_auc = compute_prc(inference_label, inference_preds)
                random_prc = compute_prc(inference_label, random_preds)
                logger.info('fold:{} iter:{} train loss:{} inference loss:{} inference roc auc:{} random roc:{} prc auc:{} random prc: {}'.format(fold_i, i, train_loss, inference_loss, roc_auc, random_roc, prc_auc, random_prc))

                #compute auroc percentage:
                auroc_percentage = {0.65: 0, 0.7: 0, 0.75: 0, 0.8: 0, 0.85: 0, 0.95: 0}
                for j in range(len(file.fold_zero_shot_terms_list[fold_i])):
                    preds_id = inference_preds[:, j].reshape([-1, 1])
                    label_id = inference_label[:, j].reshape([-1, 1])
                    roc_auc_j = compute_roc(label_id, preds_id)
                    for T in auroc_percentage.keys():
                        if T <= roc_auc_j:
                            auroc_percentage[T] += 100.0/len(file.fold_zero_shot_terms_list[fold_i])
                for T in auroc_percentage.keys():
                    logger.info('fold:{} iter:{} roc auc:{} percentage:{}%'.format(fold_i, i, T, auroc_percentage[T]))
        torch.save(model_predict, model_opt.save_path.format(data_opt.dataset, fold_i, feature_symb))


if __name__ == '__main__':
    main()