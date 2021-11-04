class data_loading:

    def __init__(self, dataset='data-cafa'):
        root_file = dataset + '/'
        self.dataset = dataset
        self.go_file = '../data/' + root_file + 'go.obo'
        self.train_data_file = '../data/' + root_file + 'train_data.pkl'
        self.test_data_file = '../data/' + root_file + 'test_data.pkl'
        self.train_fold_file = '../data/' + root_file + 'train_data_fold_{}.pkl'
        self.validation_fold_file = '../data/' + root_file + 'validation_data_fold_{}.pkl'
        self.terms_file = '../data/' + root_file + 'terms.pkl'
        self.zero_shot_term_path = '../data/' + root_file + 'zero_shot_terms_fold_{}.pkl'
        self.out_file = '../data/' + root_file + 'predictions_hw.pkl'
        self.def_embedding_file = '../data/' + root_file + '{}_def_embeddings_second2last_PubmedFull.pkl'.format(dataset)
        self.prot_vector_file = '../data/' + root_file + 'prot_vector.pkl'
        self.prot_description_file = '../data/' + root_file + 'prot_description.pkl'
        self.name_embedding_file = '../data/' + root_file + '{}_name_embeddings_second2last_PubmedFull.pkl'.format(dataset)
        self.logger_name = '_emb:second2last_def_fold_{}'
        self.text_mode = 'def'
        self.gpu_ids = '1'
        self.split = 0.9
        self.k_fold = 3

class model_config:

    def __init__(self):
        self.lr = 0.0003
        self.input_nc = 21
        self.in_nc = 512
        self.n_classes = 5101
        self.max_kernels = 129
        self.hidden_dim = [1500]
        self.features = ['network']
        self.max_len = 2000
        self.batch_size = 32
        self.epoch = 30
        self.gpu_ids = '1'
        self.dropout = 0
        self.save_path = 'results/{}_predict_deepsdnzsl_fold_{}_{}.pth'