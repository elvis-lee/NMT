import numpy as np
from .bleu_utils import compute_bleu
from .utils import sen2idx
from .infer import infer

def getBLEU(encoder, decoder, sentences_source, sentences_ref, max_length, device, word2index_source, word2index_target, index2word_target, SOS_token, EOS_token):
	ref_all = []
	translated_all = []
	for i in range(len(sentences_source)):
		_, translated = infer(encoder, decoder, sentences_source[i], max_length, device, word2index_source, index2word_target, SOS_token, EOS_token)
		translated_all.append(translated)

	for i in range(len(sentences_ref)):
		sentence_index = sen2idx(sentences_ref[i], word2index_target, EOS_token)
		ref_all.append([sentence_index])

	bleu = compute_bleu(ref_all, translated_all)
	return bleu