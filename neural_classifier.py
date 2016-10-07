from __future__ import absolute_import

import theano
import numpy
import os
import cPickle as pkl
import time
import numpy as np
from os import listdir
from os.path import isfile, join

from data_iterator import TextIterator

np.random.seed(12345)

from keras.models import Graph, Sequential, load_model
from keras.layers import Embedding, Dense
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import Nadam
from keras import backend as K

# TODO: Add evaluation on the testing set.
# WARNING: This code will barely run on CPU.  GPU use advised.

def prepare_data(seqs_x, seqs_y, n_words=30000, maxlen = None):
	# x: a list of sentences
	lengths_x = [len(s) for s in seqs_x]
	
	# Currently will never happen, but may be reinstated
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
	
def get_class_weights(inputfile):

	total = 0
	class1 = 0
	class2 = 0
	with open(inputfile, "rb") as f:
		for example in f:
			if example.strip().split("\t")[-1] == "removecomment":
				class2 += 1
			else:
				class1 += 1
	return {0 : 1, 1 : class1/class2}

def build_model(dim=1024, word_dim = 512, vocab_size = 30000, maxlen = None, use_dropout = False):
	""" This network structure is based on Recurrent Convolutional Neural Networks for Text Classification, Lai et al., 2015, """

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
		  patience=10,  # early stopping patience
		  max_epochs=5000,
		  finish_after=10000000,  # finish after this many updates
		  dispFreq=100,
		  vocab_size=30000,  # vocabulary size
		  #maxlen=64,  # maximum length of the description, currently deprecated unless experimental results show it is necessary
		  batch_size=32,
		  valid_batch_size=32,
		  savedir="./",
		  validFreq=2500,
		  saveFreq=25000,   # save the parameters after every saveFreq updates
		  dataset="./reddit_comment_training.tsv",
		  valid_dataset="./reddit_comment_valid.tsv",
		  dictionary="./reddit_comment_training.tsv.pkl",
		  use_dropout=True,
		  reload=False,
		  overwrite=False):


	
	# The dataset this model was built for is heavily unbalanced, so we generate weightings to equalize the importance of the classes.
	class_weights = get_class_weights(dataset)
	print "Class weights being used", class_weights
	train = TextIterator(dataset, dictionary, n_words_source=vocab_size, batch_size=batch_size, shuffle = True)
	valid = TextIterator(valid_dataset, dictionary, n_words_source=vocab_size, batch_size=batch_size, shuffle = False)
	
	print "Building the model"
	model = build_model(dim = dim, word_dim  = word_dim, vocab_size = vocab_size, use_dropout = use_dropout)
	print "Model built"
	
	# Initializaton
	uidx = 0
	scores = []
	history_errs = []
	ud_start = time.time()
	estop = False
	
	if (reload):
		print "Attempting to reload"
		modelfiles = [(join(savedir, f), int(f.split(".")[-2].replace("iter", ""))) for f in listdir(savedir) if isfile(join(savedir, f)) and "model" in f and ".h5" in f]
		most_recent_model = ("", 0)
		for modelfile in modelfiles:
			if modelfile[1] >= most_recent_model[1]:
				most_recent_model = modelfile
			
		if os.path.isfile(most_recent_model[0]):
			print "Loading from model", most_recent_model[0]
			model.load_weights(most_recent_model[0])
			uidx = most_recent_model[1] + 1  #Adding one avoids repeating a validation error calculation for many reloads.
		else:
			print "Failed to load model -- no acceptable models found"

	for eidx in xrange(max_epochs):
	
		n_samples = 0
		
		for x, y in train:
			n_samples += len(x)
			x, y = prepare_data(x, y, vocab_size)

			if x is None:
				print 'Minibatch with zero samples under length ', maxlen
				uidx -= 1
				continue
				
			score = model.train_on_batch(x, y, class_weight = class_weights)
			scores.append(score)
			
			# check for bad numbers; if one is encountered, just reload the model from the most recent save.  Dropout's randomness should ensure that this will
			# eventually progress past the NaN, if it is enabled, although it may take several reloaded attempts in some cases.
			if numpy.isnan(score) or numpy.isinf(score):
				print 'NaN detected'
				
				if use_dropout:
					print "Attempting to reload model to reset to pre-NaN state"
				
					# this is a slightly modified version of the reload code, above.
					modelfiles = [(join(savedir, f), int(f.split(".")[-2].replace("iter", ""))) for f in listdir(savedir) if isfile(join(savedir, f)) and "model" in f and ".h5" in f]
					for modelfile in modelfiles:
						if modelfile[1] >= most_recent_model[1]:
							most_recent_model = modelfile
						
					if os.path.isfile(most_recent_model[0]):
						print "Loading from model", most_recent_model[0]
						
						# We need to rebuild the model to clean out its gradients (as they are likely NaN in some position).
						model = build_model(emb, dec, batch_size=batch_size, num_units = dim, memory_shape=(maxlen, word_dim), vocab_size = vocab_size, word_dim  = word_dim, maxlen = maxlen, use_dropout = use_dropout)
						
						model.load_weights(most_recent_model[0])
						uidx = most_recent_model[1] + 1
						continue
					else:
						print "Failed to find a valid model to reload."
						return 1., 1., 1.
				else:
					print "Dropout not enabled, so model is deterministic.  Terminating."
					return 1., 1., 1.

			# display a brief status update
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
					history_errs = [valid_err]
				print 'Valid ', valid_err
				
				if len(history_errs) > patience:
					estop = True
					print "Halting training: early stopping patience exceeded!"

			# finish after this many updates
			if uidx >= finish_after:
				print 'Finishing after %d iterations!' % uidx
				estop = True
				break
			uidx += 1
			
		
		print 'Seen %d samples' % n_samples

		if estop:
			break

	#If stopping training, get the validation error one more time and then save to a special file
	valid_errs = []
	for x, y in valid:
		x, y = prepare_data(x, y, maxlen, vocab_size)
		v_err = model.evaluate(x, y)
		valid_errs.append(v_err)
	valid_err = valid_errs.mean()
	print 'Valid ', valid_err
	model.save_weights(savedir + "final.h5")
	return valid_err

if __name__ == '__main__':
	train()