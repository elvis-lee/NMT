import torch
import numpy as np
from .utils import sen2tensor
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

def infer(encoder, decoder, sentence, max_length, device, word2index_source, index2word_target, SOS_token, EOS_token):
    
    encoder.eval()
    decoder.eval()

    sentence_tensor = sen2tensor(sentence, word2index_source, EOS_token, device).to(device)
    #print(sentence_tensor) # to be removed
    
    encoder_hidden_h = encoder.initHidden(1, device)
    encoder_hidden_c = encoder.initHidden(1, device)
    #print(sentence_tensor.size()) # to be removed
    input_tuple = (pack_padded_sequence(sentence_tensor.unsqueeze(0), [len(sentence_tensor)], batch_first=True), np.array([0]))
    #print(input_tuple) # to be removed
    encoder_output, h_n, c_n = encoder(input_tuple, encoder_hidden_h, encoder_hidden_c)
    #print(encoder_output) # to be removed
    # encoder_output is a PackedSequence object
    # encoder_hidden_h: (1, 1, hidden_size)
    # encoder_hidden_c: (1, 1, hidden_size)
    #encoder_output = pad_packed_sequence(encoder_output, batch_first=True) # encoder_output: (1, seq_length, hidden_size)
    #encoder_output = encoder_output[0] # encoder_output: (seq_length, hidden_size)

    decoder_input = decoder.initHidden(1, SOS_token, device)
    decoder_hidden_h = h_n[0]
    decoder_hidden_c = c_n[0]
    
    output = []
    output_idx = []
    
    for di in range(max_length):
        decoder_output, decoder_hidden_h, decoder_hidden_c = decoder(decoder_input, decoder_hidden_h, decoder_hidden_c, encoder_output)
        idx = torch.argmax(decoder_output)
        decoder_input = idx.unsqueeze(0)
        output.append(index2word_target[idx])
        output_idx.append(int(idx))
        if idx == EOS_token:
            break
    return output, output_idx