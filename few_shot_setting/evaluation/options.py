class data_loading:

    def __init__(self, dataset='data_2016'):
        root_file = dataset + '/'
        self.dataset = dataset
        self.go_file = '../../data/' + root_file + 'go.obo'
        self.train_data_file = '../../data/' + root_file + 'train_data.pkl'
        self.test_data_file = '../../data/' + root_file + 'test_data.pkl'
        self.terms_file = '../../data/' + root_file + 'terms.pkl'
        self.out_file = '../../data/' + root_file + 'predictions_hw.pkl'
        self.def_embedding_file = '../../data/' + root_file + 'data-cafa_def_embeddings_PubmedFull.pkl'
        self.name_embedding_file = '../../data/' + root_file + 'data-cafa_name_embeddings_PubmedFull.pkl'
        self.text_mode = 'name'
        self.gpu_ids = '0'
        self.split = 0.9
        self.fsl_limit = 5


class model_config:

    def __init__(self):
        self.lr = 0.0003
        self.input_nc = 21
        self.in_nc = 512
        self.n_classes = 5101
        self.max_kernels = 129
        self.hidden_dim = [5101]
        self.max_len = 2000
        self.batch_size = 128
        self.epoch = 30
        self.gpu_ids = '0'
        self.emb_dim=768*38
        self.dropout=0