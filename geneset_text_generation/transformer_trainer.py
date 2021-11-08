from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from transformer_model import make_model
import time
from tqdm import tqdm
import collections
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from onmt.translate.beam_search import BeamSearch
from onmt.translate import GNMTGlobalScorer


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != 0).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    debug = model.src_embed[0]
    return NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        #self.criterion = nn.BCELoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x1 = self.generator(x)
        loss = self.criterion(x1.contiguous().view(-1, x1.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol, is_gpu=True, vocab=None):
    memory = model.encode(src, src_mask)
    bsz = memory.size(0)
    if is_gpu:
        ys = torch.ones(bsz, 1).fill_(start_symbol).int().cuda()
        for i in range(max_len - 1):
            out = model.decode(memory, src_mask,
                               Variable(ys).cuda(),
                               Variable(subsequent_mask(ys.size(1))
                                        .int().cuda()))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys,
                            next_word.reshape([-1, 1]).clone()], dim=1)
        return ys
    else:
        ys = torch.ones(1, 1).fill_(start_symbol).int()
        for i in range(max_len-1):
            out = model.decode(memory, src_mask,
                               Variable(ys),
                               Variable(subsequent_mask(ys.size(1))
                                        .int()))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys,
                            torch.ones(1, 1).int().fill_(next_word)], dim=1)
        return ys


def beam_search(model, src, src_mask, max_len, start_symbol, is_gpu=True, vocab=None):
    beam_size = 5
    pad_id = vocab.stoi['<blank>']
    bos_id = vocab.stoi['<s>']
    eos_id = vocab.stoi['</s>']
    unk_id = vocab.stoi['<unk>']
    beam = BeamSearch(
        beam_size,
        n_best=2,
        batch_size=src.size(0),
        global_scorer=GNMTGlobalScorer(0., 0., "none", "none"),
        pad=pad_id,
        eos=eos_id,
        bos=bos_id,
        unk=unk_id,
        min_length=1,
        ratio=0,
        max_length=max_len,
        return_attention=False,
        stepwise_penalty=False,
        block_ngram_repeat=False,
        exclusion_tokens=None,
        ban_unk_token=None)
    memory = model.encode(src, src_mask).transpose(0, 1).contiguous()

    (fn_map_state, memory_bank, memory_lengths, src_map) = \
        beam.initialize(
        memory, src.size(1)*torch.ones(src.size(0)), src_mask.transpose(0, 1)
    )

    beam.initialize(memory_bank=memory,
                           src_lengths=src.size(1)*torch.ones(src.size(0)))
    with torch.no_grad():
        for i in range(max_len):
            decoder_input = beam.alive_seq
            memory_bank = memory_bank.transpose(0, 1)
            out = model.decode(memory_bank, src_map.transpose(0, 1),
                                     decoder_input, Variable(subsequent_mask(decoder_input.size(1))
                                        .int().cuda()))
            log_probs = model.generator(out)
            log_probs = log_probs[:, -1, :].reshape([out.size(0), -1])
            beam.advance(log_probs, None)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            any_finished = beam.is_finished.any()
            select_indices = beam.select_indices
            memory_bank = memory_bank.transpose(0, 1)
            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(
                        x.index_select(1, select_indices) for x in memory_bank
                    )
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)
                src_map = src_map.index_select(1, select_indices)
    predictions = eos_id*torch.ones([src.size(0), max_len]).int()
    for i in range(src.size(0)):
        predictions[i, 0 : beam.predictions[i][0].size(0)] = beam.predictions[i][0]
    return predictions


class trainer:

    def __init__(self,
                 src_channel=21,
                 tgt_channel=1000,
                 padding_idx=0,
                 smoothing=0.1,
                 d_model=128,
                 is_gpu=True
                 ):
        self.criterion = LabelSmoothing(size=tgt_channel, padding_idx=padding_idx, smoothing=smoothing)
        #self.criterion = nn.CrossEntropyLoss()
        self.model = make_model(src_channel, tgt_channel, d_model=d_model).float()
        self.opt = get_std_opt(self.model)
        self.padding_idx = padding_idx
        self.is_gpu = is_gpu
        if is_gpu:
            self.model = self.model.cuda()

    def run_epoch(self, data_iter, loss_compute):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):
            batch_i = Batch(batch.seqs, batch.defs.transpose(0, 1), self.padding_idx)
            if self.is_gpu:
                batch_i.src = batch_i.src.cuda()
                batch_i.trg, batch_i.trg_y = batch_i.trg.cuda(), batch_i.trg_y.cuda()
                batch_i.src_mask, batch_i.trg_mask = batch_i.src_mask.cuda(), batch_i.trg_mask.cuda()
                batch_i.ntokens = batch_i.ntokens.cuda()
            else:
                batch_i.src = batch_i.src
                batch_i.src_mask, batch_i.trg_mask = batch_i.src_mask, batch_i.trg_mask
            out = self.model.forward(batch_i.src, batch_i.trg,
                                batch_i.src_mask, batch_i.trg_mask)
            loss = loss_compute(out, batch_i.trg_y, batch_i.ntokens)
            #loss = loss_compute(out, batch_i.trg_y, 1)
            total_loss += loss
            total_tokens += batch_i.ntokens
            tokens += batch_i.ntokens
            if i % 50 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                      (i, loss / batch_i.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        return total_loss / total_tokens

    def generate_text(self,
                      data_iter,
                      start_symb,
                      vocab,
                      cut_i=10,
                      save_path='results/generate_text.csv'):
        results = collections.OrderedDict()
        results['predict'], results['truth'] = [], []
        scores, count = 0, 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_iter)):
                src = batch.seqs
                src_mask = (src != 0).unsqueeze(-2)
                if self.is_gpu:
                    src = src.cuda()
                    src_mask = src_mask.cuda()
                    tgt_predict_i = greedy_decode(self.model, src, src_mask, max_len=100, start_symbol=start_symb, vocab=vocab)
                    tgt_predict_i = tgt_predict_i.cpu().numpy()
                    predict_text, label_text = [], []
                    for j in range(np.size(tgt_predict_i, 0)):
                        sents_pre, sents_label = '', ''
                        token_pre, token_label = [], []

                        for k in range(1, np.size(tgt_predict_i, 1), 1):
                            sym = vocab.itos[tgt_predict_i[j, k]]
                            if sym == "</s>":
                                sents_pre = sents_pre[0 : -1]
                                break
                            else:
                                sents_pre += (sym + ' ')
                                token_pre.append(sym)
                        predict_text.append(sents_pre)

                        tgt_label_i = batch.defs.transpose(0, 1).cpu().numpy()
                        for k in range(1, np.size(tgt_label_i, 1), 1):
                            sym = vocab.itos[tgt_label_i[j, k]]
                            if sym == "</s>":
                                sents_label = sents_label[0 : -1]
                                break
                            else:
                                sents_label += (sym + ' ')
                                token_label.append(sym)
                        label_text.append(sents_label)

                        sf = SmoothingFunction()
                        scores += sentence_bleu([token_label], token_pre)
                        count += 1

                    if i > cut_i:
                        break
                    results['predict'] += predict_text
                    results['truth'] += label_text
            results = pd.DataFrame(results)
            results.to_csv(save_path)
            return scores/count


