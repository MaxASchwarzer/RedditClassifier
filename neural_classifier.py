from __future__ import absolute_import
#from __future__ import print_function

import theano
import theano.tensor as T
import numpy
import matplotlib.pyplot as plt
import os
import cPickle as pkl
import copy
import time


from os import listdir
from os.path import isfile, join

from theano import tensor, function

from data_iterator import TextIterator


import logging
import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt


from keras.models import Graph, Sequential, load_model
from keras.layers import Embedding, Reshape, Dense
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Dropout, Activation, Flatten, Masking, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalMaxPooling1D
from keras.utils import np_utils, generic_utils
from keras.optimizers import Nadam
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from seya.layers.ntm import NeuralTuringMachine as NTM


#From NMT code
def prepare_data(seqs_x, seqs_y, n_words=30000, maxlen = None):
	# x: a list of sentences
	lengths_x = [len(s) for s in seqs_x]
	
	if maxlen is not None:
		new_seqs_x = []
		new_lengths_x = []
		for l_x, s_x in zip(lengths_x, seqs_x):
			if l_x < maxlen:
				new_seqs_x.append(s_x)
				new_lengths_x.append(l_x)
		lengths_x = new_lengths_x
		seqs_x = new_seqs_x
		if len(lengths_x) < 1:
			return None, None

	n_samples = len(seqs_x)
	maxlen_x = numpy.max(lengths_x) + 1

	x = numpy.zeros((n_samples, maxlen_x)).astype('int32')
	for idx, s_x in enumerate(seqs_x):
		x[idx, :lengths_x[idx]] = s_x
		
	return x, numpy.asarray(seqs_y)

def build_model(dim=1024, word_dim = 512, vocab_size = 30000, maxlen = None, use_dropout = False):
	model = Sequential()
	model.add(Embedding(vocab_size, word_dim, mask_zero = False))
	if use_dropout:
		model.add(TimeDistributed(Dropout(0.2)))
	model.add(Bidirectional(LSTM(dim, return_sequences = True), merge_mode = "concat"))
	if use_dropout:
		model.add(TimeDistributed(Dropout(0.2)))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(int(dim/2), activation = "tanh"))
	if use_dropout:
		model.add(Dropout(0.2))
	model.add(Dense(2, activation = "softmax"))
	sgd = Nadam()
	model.compile(loss='binary_crossentropy', optimizer=sgd)
	return model
	
def train(word_dim=512,  # word vector dimensionality
		  dim=1024,  # the number of LSTM units
		  encoder='gru',
		  decoder='gru_cond',
		  patience=10,  # early stopping patience
		  max_epochs=5000,
		  finish_after=10000000,  # finish after this many updates
		  dispFreq=100,
		  decay_c=0.,  # L2 regularization penalty
		  alpha_c=0.,  # alignment regularization
		  clip_c=-1.,  # gradient clipping threshold
		  lrate=0.01,  # learning rate
		  vocab_size=30000,  # vocabulary size
		  #maxlen=64,  # maximum length of the description
		  batch_size=32,
		  valid_batch_size=32,
		  savedir="E:/Users/Max/NeuralNetworkModels/large_wiki_NTM_dropout/",
		  validFreq=100,
		  saveFreq=2500,   # save the parameters after every saveFreq updates
		  sampleFreq=1000,   # generate some samples after every sampleFreq
		  dataset="./reddit_comment_training.tsv",
		  valid_dataset="./reddit_comment_valid.tsv",
		  dictionary="./reddit_comment_training.tsv.pkl",
		  use_dropout=True,
		  reload=False,
		  overwrite=False):
	
	worddict = None
	worddict_r = None
	with open(dictionary, 'r') as f:
		worddict = pkl.load(f)
	worddict_r = dict()
	for kk, vv in worddict.iteritems():
		worddict_r[vv] = kk

	# The generator to sample examples from
	train = TextIterator(dataset, dictionary, n_words_source=vocab_size, batch_size=batch_size, shuffle = True)
	valid = TextIterator(valid_dataset, dictionary, n_words_source=vocab_size, batch_size=batch_size, shuffle = False)
	
	
	print "Compiling training functions"
	# The model (1-layer Neural Turing Machine)
	model = build_model(dim = dim, word_dim  = word_dim, vocab_size = vocab_size, use_dropout = use_dropout)
	print "Training functions compiled"
	
	
	uidx = 0
	if (reload):
		modelfiles = [(join(savedir, f), int(f.split(".")[-2].replace("iter", ""))) for f in listdir(savedir) if isfile(join(savedir, f)) and "model" in f and ".h5" in f]
		most_recent_model = ("", 0)
		for modelfile in modelfiles:
			if modelfile[1] >= most_recent_model[1]:
				most_recent_model = modelfile
			
		if os.path.isfile(most_recent_model[0]):
			print "Loading from model", most_recent_model[0]
			model.load_weights(most_recent_model[0])
			uidx = most_recent_model[1] + 1
		else:
			print "Failed to load model -- no acceptable models found"

	scores = []
	history_errs = []
	ud_start = time.time()
	estop = False
	for eidx in xrange(max_epochs):
		n_samples = 0
		for x, y in train:
			n_samples += len(x)
			x, y = prepare_data(x, y, vocab_size)

			if x is None:
				print 'Minibatch with zero samples under length ', maxlen
				uidx -= 1
				continue
				
			score = model.train_on_batch(x, y)
			scores.append(score)
			
			# check for bad numbers; if one is encountered, just reload the model from the most recent save.  Dropout's randomness should ensure that this will
			# eventually progress past the NaN, although it may take reloaded attempts in the worst case.
			if numpy.isnan(score) or numpy.isinf(score):
				print 'NaN detected'
				print "Attempting to reload model to reset to pre-NaN state"
				
				modelfiles = [(join(savedir, f), int(f.split(".")[-2].replace("iter", ""))) for f in listdir(savedir) if isfile(join(savedir, f)) and "model" in f and ".h5" in f]

						
				for modelfile in modelfiles:
					if modelfile[1] >= most_recent_model[1]:
						most_recent_model = modelfile
					
				if os.path.isfile(most_recent_model[0]):
					print "Loading from model", most_recent_model[0]
					del model
					model = build_model(emb, dec, batch_size=batch_size, num_units = dim, memory_shape=(maxlen, word_dim), vocab_size = vocab_size, word_dim  = word_dim, maxlen = maxlen, use_dropout = use_dropout)
					model.load_weights(most_recent_model[0])
					uidx = most_recent_model[1] + 1
					continue
				else:
					print "Failed to load model -- no acceptable models found"
					return 1., 1., 1.

			# verbose
			if numpy.mod(uidx, dispFreq) == 0:
				ud = (time.time() - ud_start)/dispFreq
				try:
					reportString = 'Epoch ' + str(eidx).strip() + '  Update ' + str(uidx).strip() + " Loss: " + str(numpy.mean(scores)) + "  Average time taken: " + str(ud) 
					print reportString
				except:
					print "Exception encountered while printing report.  Continuing training."
				ud_start = time.time()
				scores = []

			# save the best model so far, in addition, save the latest model
			# into a separate file with the iteration number for external eval
			if numpy.mod(uidx, saveFreq) == 0:

				# save with uidx
				
				print 'Saving the model at iteration {}...'.format(uidx),
				saveto_uidx = join(savedir, 'model.iter{}.h5'.format(uidx))
				model.save_weights(saveto_uidx)
				print 'Done'


			# generate some samples with the model and display them
			# if numpy.mod(uidx, sampleFreq) == 0:
				# num_samples = numpy.minimum(5, x.shape[0])
				# sample_in = x[:num_samples]
				# sample_out = model.predict_classes(sample_in, batch_size=num_samples)
				# for index, sample in enumerate(sample_out):
				
					# print 'Source ', index, ': ',
					# for vv in x[index, :]:
						# if vv == 0:
							# break
						# if vv in worddict_r:
							# print worddict_r[vv],
						# else:
							# print 'UNK',
					# print
					# print 'Truth ', index, ' : ',
					# for vv in y[index, :, 0]:
						# if vv == 0:
							# break
						# if vv in worddict_r:
							# print worddict_r[vv],
						# else:
							# print 'UNK',
					# print
					# print 'Sample ', index, ': ',
					# for vv in sample:
						# if vv == 0:
							# break
						# if vv in worddict_r:
							# print worddict_r[vv],
						# else:
							# print 'UNK',
					# print

			# validate model on validation set and early stop if necessary
			if numpy.mod(uidx, validFreq) == 0:
				
				valid_errs = []
				for x, y in valid:
					x, y = prepare_data(x, y, maxlen, vocab_size)
					v_err = model.evaluate(x, y)
					valid_errs.append(v_err)
				valid_err = numpy.mean(valid_errs)
				history_errs.append(valid_err)

				if uidx == 0 or valid_err <= numpy.array(history_errs).min():
					print "New best valid error!"
				print 'Valid ', valid_err

			# finish after this many updates
			if uidx >= finish_after:
				print 'Finishing after %d iterations!' % uidx
				estop = True
				break
			uidx += 1
			
		print 'Seen %d samples' % n_samples

		if estop:
			break

	#use_noise.set_value(0.)
	valid_errs = []
	for x, y in valid:
		v_err = model.evaluate(x, y)
		valid_errs.append(v_err)
	valid_err = valid_errs.mean()
	print 'Valid ', valid_err

	model.save(savedir + "final.h5")

	return valid_err

if __name__ == '__main__':
	train()