import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_directions=1, dropout=0, forget_bias=1.0):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # set forget bias initial values
        self.lstm.bias_ih_l0[hidden_size:2*hidden_size].data.fill_(forget_bias) 
        self.lstm.bias_hh_l0[hidden_size:2*hidden_size].data.fill_(forget_bias)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, input_tuple, prev_h, prev_c):
        # input
        # input size: (batch_size, seq_length)
        # prev_h size: (num_layers*num_directions, batch_size, hidden_size)
        # prec_c size: (num_layers*num_directions, batch_size, hidden_size)
        # output
        # h_n size: (num_layers*num_directions, batch_size, hidden_size)
        # c_n size: (num_layers*num_directions, batch_size, hidden_size)
        (sentences_packed, reorder_idx) = input_tuple
        sentences_tensor, sentences_length = pad_packed_sequence(sentences_packed, batch_first=True, padding_value=0)
        #sentences_tensor: (batch_size, seq_length)
        input_embedded = self.embedding(sentences_tensor) # (batch_size, seq_length, hidden_size)
        input_embedded = self.dropout(input_embedded)
        input_embedded_packed = pack_padded_sequence(input_embedded, sentences_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(input_embedded_packed, (prev_h, prev_c))
        return output, h_n, c_n
    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device)


# num_layers and num_directions for encoder must be 0 when this decoder is used
class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0, forget_bias=1.0):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

        # set forget bias initial values
        self.lstm.bias_ih[hidden_size:2*hidden_size].data.fill_(forget_bias) 
        self.lstm.bias_hh[hidden_size:2*hidden_size].data.fill_(forget_bias)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, input, prev_h, prev_c, encoder_output=None):
        input_embedded = self.embedding(input)
        input_embedded = self.dropout(input_embedded)
        h, c = self.lstm(input_embedded, (prev_h, prev_c))
        output = self.softmax(self.out(h))
        return output, h, c
    def initHidden(self, batch_size, SOS_token, device):
        return torch.full((batch_size,), SOS_token, dtype=torch.long, device=device)


class DotAttenDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, attention_vector_size, dropout=0, forget_bias=1.0):
        super(DotAttenDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

        # set forget bias initial values
        self.lstm.bias_ih[hidden_size:2*hidden_size].data.fill_(forget_bias) 
        self.lstm.bias_hh[hidden_size:2*hidden_size].data.fill_(forget_bias)

        self.out = nn.Linear(hidden_size*2, attention_vector_size)
        self.out2 = nn.Linear(attention_vector_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, prev_h, prev_c, encoder_output):
        # encoder_output: PackedSequence to be converted to (batch_size, seq_length, hidden_size*num_directions)
        input_embedded = self.embedding(input)
        input_embedded = self.dropout(input_embedded)
        h, c = self.lstm(input_embedded, (prev_h, prev_c))
        
        # trick: use padding value = -inf to do variable length attention correctly 
        encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True, padding_value=0)
        #print(encoder_output) # to be removed
        scores = torch.matmul(encoder_output, h.unsqueeze(-1)) # (batch_size, seq_length, 1)
        scores[scores==0] = -10e10
        #print(scores) # to be removed
        scores = F.softmax(scores, dim=1)
        context_vector = torch.matmul(torch.transpose(encoder_output, 1, 2), scores).squeeze(-1) # (batch_size, hidden_size)
        attention_vector = F.tanh(self.out(torch.cat((context_vector, h), -1))) # (batch_size, attention_vector_size)
        output = self.softmax(self.out2(attention_vector))
        return output, h, c
        
    def initHidden(self, batch_size, SOS_token, device):
        return torch.full((batch_size,), SOS_token, dtype=torch.long, device=device)