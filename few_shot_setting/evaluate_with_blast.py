import pandas as pd
import numpy as np
from utils import compute_roc
import collections
from tqdm import tqdm
from utils import load
from options import model_config, data_loading
import pickle


class FileLoader:

    def __init__(self, opt):
        terms_df = pd.read_pickle(opt.terms_file)
        self.go_data = load(opt.go_file)
        term_list_T = list(terms_df['terms'])
        self.term_list = []
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


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def compute_blast_preds(diamond_scores,
                        preds,
                        train_data,
                        prot_index,
                        is_load=False,
                        save_path='results/cache/blast_preds.pkl'):
    if is_load ==False:
        blast_preds = collections.OrderedDict()
        #print('Diamond preds')
        annotation_dict = collections.OrderedDict()
        for i in train_data.index:
            prot_id = train_data.loc[i]['proteins']
            annts = set(train_data.loc[i]['annotations'])
            if prot_id not in annotation_dict.keys():
                annotation_dict[prot_id] = collections.OrderedDict()
                for ann_id in annts:
                    annotation_dict[prot_id][ann_id] = None
            else:
                for ann_id in annts:
                    annotation_dict[prot_id][ann_id] = None

        for prot_id in tqdm(preds.index):
            annots = {}

            # BlastKNN
            if prot_id in diamond_scores.keys():
                sim_prots = diamond_scores[prot_id]
                allgos = set()
                total_score = 0.0
                for p_id, score in sim_prots.items():
                    allgos |= set(annotation_dict[p_id].keys())
                    total_score += score
                allgos = set(allgos)
                sim = collections.OrderedDict()
                for go_id in allgos:
                    s = 0.0
                    for p_id in sim_prots.keys():
                        score = sim_prots[p_id]
                        if go_id in annotation_dict[p_id].keys():
                            s += score
                    sim[go_id] = s / total_score
            blast_preds[prot_id] = sim
        save_obj(blast_preds, save_path)
    else:
        blast_preds = load_obj(save_path)
    return blast_preds


def main():
    dataset = 'goa_mouse_cat'
    k_fold = 3
    is_blast = True

    data_opt = data_loading(dataset=dataset)
    print('dataset is {}'.format(data_opt.dataset))
    file = FileLoader(data_opt)
    name_space = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
    alphas = {"mf": 0, "bp": 0, "cc": 0}
    if is_blast:
        alphas = {"mf": 0.68, "bp": 0.63, "cc": 0.46}

    print('alphas: {}'.format(alphas))
    auroc_ont_total = collections.OrderedDict()
    for ont in ['bp', 'mf', 'cc']:
        auroc_total = collections.OrderedDict()
        for i in file.fold_few_shot_terms_list.keys():
            auroc_total[i] = collections.OrderedDict()
            auroc_total[i]['mean'] = 0
            auroc_total[i]['err'] = []

        for fold_i in range(k_fold):
            predictions = pd.read_pickle("/home/hwxu/proteinZSL/deepTNFSL/results/{}_fold_{}_predictions.pkl".format(dataset, fold_i))
            labels = pd.read_pickle("/home/hwxu/proteinZSL/deepTNFSL/results/{}_fold_{}_labels.pkl".format(dataset, fold_i))

            blast_preds_file = "/home/hwxu/proteinZSL/data/{}/blast_preds_fold_{}.pkl".format(dataset, fold_i)
            blast_preds = load_obj(blast_preds_file)

            for fsl_n in auroc_total.keys():
                fsl_name_term = []
                fsl_terms = file.fold_few_shot_terms_list[fsl_n][fold_i]
                for t_id in fsl_terms:
                    if file.go_data[t_id]['namespace'] == name_space[ont]: fsl_name_term.append(t_id)

                prediction_ns = predictions.loc[:][fsl_name_term]
                labels_ns = labels.loc[:][fsl_name_term]
                for prot_id in prediction_ns.index:
                    blast_preds_p_id = blast_preds[prot_id]
                    for go_id in blast_preds_p_id.keys():
                        if go_id in prediction_ns.columns:
                            prediction_ns.loc[prot_id][go_id] = (1 - alphas[ont]) * prediction_ns.loc[prot_id][go_id] + alphas[ont] * blast_preds_p_id[go_id]
                prediction_ns_npy = np.asarray(prediction_ns)
                labels_ns_npy = np.asarray(labels_ns)
                roc_auc = compute_roc(labels_ns_npy, prediction_ns_npy)
                auroc_total[fsl_n]['mean'] += roc_auc / k_fold
                auroc_total[fsl_n]['err'].append(roc_auc)

        for fsl_n in auroc_total.keys():
            auroc_total[fsl_n]['err'] = np.std(auroc_total[fsl_n]['err'])
        auroc_ont_total[ont] = auroc_total

        for fsl_n in auroc_total.keys():
            print('fsl_n: {} ont:{} auroc:{}'.format(fsl_n, ont, auroc_total[fsl_n]))
    if is_blast:
        save_obj(auroc_ont_total, 'results/training_log/{}_ont_blast_dict.pkl'.format(dataset))
    else:
        save_obj(auroc_ont_total, 'results/training_log/{}_ont_dict.pkl'.format(dataset))


if __name__ == '__main__':
    main()