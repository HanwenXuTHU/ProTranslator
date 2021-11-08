from torch.autograd import Variable
import numpy as np
import torch
from torch.utils.data import ConcatDataset
import pickle
from torchtext.data import Field, Dataset, Iterator, BucketIterator, Example
from transformers import AutoTokenizer
import random
import spacy


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class protein2def_data(Dataset):

    def __init__(self,
                 train_file='../data/goa_human_terms/train_data.pkl',
                 valid_file='../data/goa_human_terms/valid_data.pkl',
                 tokenize_opt=1,
                 is_train=True,
                 is_gpu=True,
                 aug=True):
        self.train_data, self.valid_data = load_obj(train_file), load_obj(valid_file)
        self.is_train = is_train
        self.is_gpu = is_gpu

        embedding_way = ['bert-base-uncased', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                         'allenai/scibert_scivocab_uncased']
        if tokenize_opt < 3:
            tgt_eT_name = embedding_way[tokenize_opt]
            tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_eT_name)
        elif tokenize_opt == 3:
            spacy_de = spacy.load('de_core_news_sm')

            def tokenize_de(text):
                return [tok.text for tok in spacy_de.tokenizer(text)]

            tgt_tokenizer = tokenize_de

        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"

        self.def_text = Field(tokenize=tgt_tokenizer.tokenize, sequential=True,
                              init_token=BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)
        seqs_field = Field(sequential=False, use_vocab=False)
        fields = [("id", None), ("seqs", seqs_field), ("defs", self.def_text)]
        examples = []

        if self.is_train:
            for i in range(len(self.train_data['seqs'])):
                seq_i = self.train_data['seqs'][i]
                def_i = self.train_data['def'][i]
                if aug:
                    rate = random.random()
                    if rate > 0.5:
                        seq_i = self.dropout(seq_i, 0, p=0.2)
                examples.append(Example.fromlist([None, seq_i, def_i], fields))
        else:
            for i in range(len(self.valid_data['seqs'])):
                seq_i = self.valid_data['seqs'][i]
                def_i = self.valid_data['def'][i]
                examples.append(Example.fromlist([None, seq_i, def_i], fields))
        super(protein2def_data, self).__init__(examples, fields)

    def dropout(self, text, unk, p=0.2):
        len_ = np.size(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = unk
        return text


class file_loader:

    def __init__(self,
                 train_file='../data/goa_human_terms/train_data.pkl',
                 valid_file='../data/goa_human_terms/valid_data.pkl',
                 tokenize_opt=1,
                 train_batch=8,
                 valid_batch=8,
                 is_gpu=True
                 ):
        self.train = protein2def_data(train_file, valid_file, tokenize_opt=tokenize_opt, is_train=True, is_gpu=is_gpu)
        self.valid = protein2def_data(train_file, valid_file, tokenize_opt=tokenize_opt, is_train=False, is_gpu=is_gpu)
        self.train.def_text.build_vocab(self.train, min_freq=5)
        self.vocab = self.train.def_text.vocab
        self.valid.def_text.vocab = self.vocab
        self.pad_idx = self.vocab.stoi["<blank>"]
        self.train_iter = BucketIterator(dataset=self.train, batch_size=train_batch, shuffle=True, sort_within_batch=False,
                                         repeat=False)
        self.val_iter = BucketIterator(dataset=self.valid, batch_size=valid_batch, shuffle=True, sort_within_batch=False,
                                         repeat=False)
        print('Data Loading finished!')



