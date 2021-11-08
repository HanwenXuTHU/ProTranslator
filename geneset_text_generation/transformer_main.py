from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from transformer_model import make_model
from transformer_trainer import trainer, Batch, SimpleLossCompute
from transformer_dataloader import file_loader
import time
import logging
import os


def get_logger(name):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = 'results/training_log/'
    log_name = log_path + name + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main():
    train_file = "/disk1/hwxu/protein2def/data/goa_human_terms/train_data.pkl"
    valid_file = "/disk1/hwxu/protein2def/data/goa_human_terms/valid_data.pkl"
    save_path = 'results/generate_text.csv'
    model_path = 'results/transformer_128_average.pt'
    cut_i = 50
    tokenize_opt = 1
    train_batch = 32
    valid_batch = 32
    d_model=128
    is_gpu = True
    smoothing = 0
    epochs = 2000
    training = False
    valid_best = 0

    logger = get_logger('bleu_score2')

    prot2def_loader = file_loader(train_file, valid_file, tokenize_opt, train_batch, valid_batch, is_gpu)
    prot2def_trainer = trainer(src_channel=21,
                               tgt_channel=len(prot2def_loader.vocab),
                               padding_idx=prot2def_loader.pad_idx,
                               smoothing=smoothing,
                               d_model=d_model,
                               is_gpu=is_gpu)

    if os.path.exists(model_path):
        prot2def_trainer.model = torch.load(model_path)

    for i in range(epochs):
        if training:
            prot2def_trainer.model.train()
            prot2def_trainer.run_epoch(prot2def_loader.train_iter,
                                       SimpleLossCompute(prot2def_trainer.model.generator,
                                                         prot2def_trainer.criterion,
                                                         prot2def_trainer.opt))
        if i%3 == 0:
            prot2def_trainer.model.eval()

            train_bleu_score = prot2def_trainer.generate_text(prot2def_loader.train_iter,
                                                        prot2def_loader.vocab.stoi['<s>'],
                                                        prot2def_loader.vocab,
                                                        cut_i,
                                                        save_path)
            logger.info('epoch: {} training bleu score: {}'.format(i, train_bleu_score))

            valid_bleu_score = prot2def_trainer.generate_text(prot2def_loader.val_iter,
                                           prot2def_loader.vocab.stoi['<s>'],
                                           prot2def_loader.vocab,
                                           cut_i,
                                           save_path)
            logger.info('epoch: {} val bleu score: {}'.format(i, valid_bleu_score))
            if valid_bleu_score > valid_best:
                torch.save(prot2def_trainer.model, model_path)
                logger.info('best valid bleu updates from {} to {}!'.format(valid_best, valid_bleu_score))
                valid_best = valid_bleu_score



if __name__ == '__main__':
    main()
