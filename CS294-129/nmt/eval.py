import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
from .utils import resume_order


def eval(source_tuple, target_tuple, encoder, decoder, batch_size, device, SOS_token, PAD_token):
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        encoder_hidden_h = encoder.initHidden(batch_size, device)
        encoder_hidden_c = encoder.initHidden(batch_size, device)

        loss = 0

        criterion = nn.NLLLoss(ignore_index=PAD_token, size_average=False)

        (sentences_source_packed, reorder_idx_source) = source_tuple
        (sentences_target_packed, reorder_idx_target) = target_tuple
        sentences_target_tensor, sentences_target_length = pad_packed_sequence(sentences_target_packed, batch_first=True, padding_value=PAD_token)
        sentences_target_tensor = resume_order(sentences_target_tensor, reorder_idx_target)

        target_length = sentences_target_tensor.size(1)

        # encoder_output size: (batch_size, seq_length, hidden_size*num_directions)
        # encoder_hidden_h size: (num_layers*num_directions, batch_size, hidden_size)
        # encoder_hidden_c size: (num_layers*num_directions, batch_size, hidden_size)
        encoder_output, encoder_hidden_h, encoder_hidden_c = encoder(source_tuple, encoder_hidden_h, encoder_hidden_c)


        decoder_input = decoder.initHidden(batch_size, SOS_token, device)
        #decoder_input = torch.full((batch_size,), SOS_token, dtype=torch.long, device=device) # to be removed
        
        decoder_hidden_c = resume_order(encoder_hidden_c[0], reorder_idx_source)
        decoder_hidden_h = resume_order(encoder_hidden_h[0], reorder_idx_source)


        for di in range(target_length):
            decoder_output, decoder_hidden_h, decoder_hidden_c = decoder(decoder_input, decoder_hidden_h, decoder_hidden_c, encoder_output)
            loss += criterion(decoder_output, sentences_target_tensor[:,di])
            decoder_input = sentences_target_tensor[:,di]

        denominator = torch.sum(sentences_target_length).float()
        if device == torch.device("cuda"):
            denominator = denominator.cuda()
        loss = loss / denominator

        return loss.item()