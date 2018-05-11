import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

# helper funtions to convert sentence in natural language to list of word indexes
def sen2idx(sentence, word2index, EOS_token):
    idxes = [word2index.get(word, 0) for word in sentence]
    idxes.append(EOS_token)
    return idxes # assume that 0 is for <unk>

def sen2tensor(sentence, word2index, EOS_token, device):
    idxes = sen2idx(sentence, word2index, EOS_token)
    #idxes.append(EOS_token)
    return torch.tensor(idxes, dtype=torch.long, device=device)

def resume_order(input, idx):
    # input 
    #   input: Tensor: (batch_size, seq_length)
    #   idx: Tensor or ndarray: (batch_size)
    # output
    #   out: Tensor with reordered sentences in batch: (batch_size, seq_length) 
    
    if isinstance(idx, (np.ndarray)):
        idx = torch.from_numpy(idx).to(input.device)
    out = torch.index_select(input, 0, idx)
    return out

class BatchGenerator():
    def __init__(self, batch_size, sentences_source, sentences_target, word2index_source, word2index_target, EOS_token, device):
        self.device = device
        self.batch_size = batch_size
        self.sentences_source = sentences_source
        self.sentences_target = sentences_target
        self.word2index_source = word2index_source
        self.word2index_target = word2index_target
        self.num_sentence = len(sentences_source)
        self.EOS_token = EOS_token
        self.reset()
    
    def reset(self):
        self.consumed = 0
        self.permutation = np.random.permutation(self.num_sentence)
    
    def get_batch(self):
        # generate id in one batch
        if self.consumed + self.batch_size > self.num_sentence:
            self.reset()
        sample_id = self.permutation[self.consumed:self.consumed + self.batch_size]
        self.consumed += self.batch_size

        #generate a source batch
        sentences_source_tensor = [sen2tensor(self.sentences_source[id], self.word2index_source, self.EOS_token, self.device) for id in sample_id]

        len_array_source = [len(st) for st in sentences_source_tensor]
        reorder_idx_source = np.argsort(len_array_source, kind='mergesort')
        reorder_idx_source = np.argsort(np.flip(reorder_idx_source, 0)) #index to restore unsorted order

        sentences_source_tensor.sort(key=len)
        sentences_source_tensor.reverse()
        sentences_source_packed = pack_sequence(sentences_source_tensor)
        if self.device==torch.device('cuda'):
            sentences_source_packed = sentences_source_packed.cuda()

        #generate a target batch
        sentences_target_tensor = [sen2tensor(self.sentences_target[id], self.word2index_target, self.EOS_token, self.device) for id in sample_id]

        len_array_target = [len(st) for st in sentences_target_tensor]
        reorder_idx_target = np.argsort(len_array_target, kind='mergesort')
        reorder_idx_target = np.argsort(np.flip(reorder_idx_target, 0)) #index to restore unsorted order

        sentences_target_tensor.sort(key=len)
        sentences_target_tensor.reverse()
        sentences_target_packed = pack_sequence(sentences_target_tensor)
        if self.device==torch.device('cuda'):
            sentences_target_packed = sentences_target_packed.cuda()

        return (sentences_source_packed, reorder_idx_source), (sentences_target_packed, reorder_idx_target)