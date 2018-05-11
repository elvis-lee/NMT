import numpy as np
import torch 
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
from .utils import resume_order
from .eval import eval
from .bleu import getBLEU


def train(source_tuple, target_tuple, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, device, SOS_token, PAD_token):
    
    encoder.train()
    decoder.train()
    
    encoder_hidden_h = encoder.initHidden(batch_size, device)
    encoder_hidden_c = encoder.initHidden(batch_size, device)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
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
    
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()


def trainIters(batch_generator_train, batch_generator_test, encoder, decoder, n_iters, batch_size, device, SOS_token, PAD_token, print_every=1000, plot_every=100, step_every_epoch=1000, learning_rate=0.1, bleu_params=None):
    
    epoch = 0

    plot_losses_train = []
    plot_losses_test = []
    plot_loss_total = 0
    print_loss_total = 0
    plot_bleu = []
    
    encoder_optimizer = optim.Adam(encoder.parameters(), learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), learning_rate)
    
    for iter in range(1, n_iters+1):
        source_tuple, target_tuple = batch_generator_train.get_batch()
        loss = train(source_tuple, target_tuple, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, device, SOS_token, PAD_token)
        print_loss_total += loss
        plot_loss_total += loss
        
        if iter%print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # evaluate on testing set
            source_tuple, target_tuple = batch_generator_test.get_batch()
            print_loss_test = eval(source_tuple, target_tuple, encoder, decoder, batch_size, device, SOS_token, PAD_token)
            print('(step:%d %d%%) loss_train:%.4f, loss_test:%.4f' % (iter, iter / n_iters * 100, print_loss_avg, print_loss_test))

        if iter%plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses_train.append(plot_loss_avg)
            plot_loss_total = 0
            if print_every == plot_every:
                plot_loss_test = print_loss_test
            else:
                source_tuple, target_tuple = batch_generator_test.get_batch()
                plot_loss_test = eval(source_tuple, target_tuple, encoder, decoder, batch_size, device, SOS_token, PAD_token)
            plot_losses_test.append(plot_loss_test)

         
        if iter%step_every_epoch == 0:
            epoch += 1
            print("epoch: {}".format(epoch))
            if bleu_params:   
                bleu_score = getBLEU(encoder, decoder, bleu_params['sentences_source'], bleu_params['sentences_ref'], bleu_params['max_length'], device, bleu_params['word2index_source'], bleu_params['word2index_target'], bleu_params['index2word_target'], SOS_token, bleu_params['EOS_token'])
                bleu_score = bleu_score[0]
                plot_bleu.append(bleu_score)
                print("bleu_test:{}".format(bleu_score))

    return plot_losses_train, plot_losses_test, plot_bleu
