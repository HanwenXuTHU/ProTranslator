import torch
import collections
from model import deepZSLModel, deepSDNModel
from options import model_config, data_loading
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, vec2classes, compute_prc
import utils
from tqdm import tqdm
import numpy as np
import gc
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


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
    dataset = 'goa_human_cat'
    data_opt = data_loading(dataset=dataset)
    print('dataset is {}'.format(data_opt.dataset))
    model_opt = model_config()
    file = FileLoader(data_opt)
    print('loading data finished')
    feature_symb = '_'.join(model_opt.features)
    print('features: {}'.format(feature_symb))
    logger = utils.get_logger(data_opt.dataset +
                              data_opt.logger_name.format(feature_symb))
    name_space = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}

    auroc_ont = collections.OrderedDict()
    auroc_percentage_ont = collections.OrderedDict()
    for ont in ['bp', 'mf', 'cc']:
        auroc_total = collections.OrderedDict()
        auroc_total['mean'] = 0
        auroc_total['err'] = []
        auroc_percentage = {0.65: 0, 0.7: 0, 0.75: 0, 0.8: 0, 0.85: 0, 0.9: 0, 0.95: 0}
        auroc_percentage_err = {0.65: [], 0.7: [], 0.75: [], 0.8: [], 0.85: [], 0.9: [], 0.95: []}
        for T_i in auroc_percentage_err.keys():
            for k in range(file.k_fold):
                auroc_percentage_err[T_i].append(0)
        for fold_i in [0, 1, 2]:
            model_opt.emb_dim = file.emb_tensor_train.size(1)
            model_opt.vector_dim = np.size(list(file.prot_vector.values())[0].reshape(-1), 0)
            model_opt.description_dim = np.size(list(file.prot_description.values())[0].reshape(-1), 0)
            model_predict = torch.load(model_opt.save_path.format(data_opt.dataset, fold_i, feature_symb))
            model_predict.model.eval()
            inference_dataset = DataLoader(file.fold_validation[fold_i], batch_size=model_opt.batch_size, shuffle=True)
            logger.info('Ont: {} fold : {}'.format(ont, fold_i))

            train_loss, inference_loss = 0, 0
            num = 0

            print('inferring iters')
            inference_preds, inference_label = [], []
            zero_shot_ont_terms = []
            for t_id in file.fold_zero_shot_terms_list[fold_i].keys():
                if file.go_data[t_id]['namespace'] == name_space[ont]:
                    zero_shot_ont_terms.append(t_id)
            for j, inference_D in tqdm(enumerate(inference_dataset)):
                with torch.no_grad():
                    preds = model_predict.model(inference_D['seq'], inference_D['description'], inference_D['vector'], file.emb_tensor_train)
                    preds_inf, label_inf = [], []
                    label = inference_D['label']
                    for t_id in zero_shot_ont_terms:
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
            auroc_total['mean'] += roc_auc / file.k_fold
            auroc_total['err'].append(roc_auc)

            #compute auroc percentage:
            for j in range(len(zero_shot_ont_terms)):
                preds_id = inference_preds[:, j].reshape([-1, 1])
                label_id = inference_label[:, j].reshape([-1, 1])
                roc_auc_j = compute_roc(label_id, preds_id)
                for T in auroc_percentage.keys():
                    if T <= roc_auc_j:
                        auroc_percentage[T] += 100.0/len(zero_shot_ont_terms)/file.k_fold
                        auroc_percentage_err[T][fold_i] += 100.0/len(zero_shot_ont_terms)
        auroc_ont[ont] = auroc_total
        auroc_percentage_ont[ont] = collections.OrderedDict()
        auroc_percentage_ont[ont]['mean'] = auroc_percentage
        auroc_percentage_ont[ont]['err'] = auroc_percentage_err

    save_obj(auroc_ont, 'results/training_log/{}_{}_auroc_ont_results.pkl'.format(dataset, feature_symb))
    save_obj(auroc_percentage_ont, 'results/training_log/{}_{}_auroc_percentage_ont_results.pkl'.format(dataset, feature_symb))



if __name__ == '__main__':
    main()