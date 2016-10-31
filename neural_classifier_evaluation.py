from __future__ import absolute_import

import theano
import numpy
import os
import cPickle as pkl
import time
from os import listdir
from os.path import isfile, join

from data_iterator import TextIterator

from keras.models import Graph, Sequential, load_model, Model
from keras.layers import Embedding, Dense, MaxoutDense, Input, merge, MaxoutDense, Flatten
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import Nadam
from keras import backend as K
from keras.layers.advanced_activations import PReLU, LeakyReLU

# TODO: Add evaluation on the testing set.
# WARNING: This code will barely run on CPU.  GPU use advised.

def prepare_data(seqs_x, seqs_y, maxlen = None):
	# x: a list of sentences
	lengths_x = [len(s[0]) for s in seqs_x]
	
	# Currently will never happen, but may be reinstated
	if maxlen is not None:
		new_seqs_x = []
		new_lengths_x = []
		for l_x, s_x in zip(lengths_x, seqs_x):
			if l_x < maxlen:
				new_seqs_x.append(s_x)
				new_lengths_x.append(l_x)
			else:
				new_seqs_x.append((s_x[0][:maxlen], s_x[1]))
				new_lengths_x.append(maxlen)
				
		lengths_x = new_lengths_x
		seqs_x = new_seqs_x
		if len(lengths_x) < 1:
			return None, None

	n_samples = len(seqs_x)
	maxlen_x = numpy.max(lengths_x) + 1
	
	seqs_y = numpy.asarray(seqs_y)
	
	x_text = numpy.zeros((n_samples, maxlen_x)).astype('int32')
	x_sr = numpy.zeros((n_samples)).astype("int32")
	
	for idx, s_x in enumerate(seqs_x):
		x_text[idx, :lengths_x[idx]] = s_x[0]
		x_sr[idx] = s_x[1]
	return [x_text, x_sr], seqs_y
	
def get_class_weights(inputfile):

	total = 0
	class1 = 0
	class2 = 0
	with open(inputfile, "rb") as f:
		for example in f:
			if "removecomment" in example.strip().split("\t")[-1]:
				class2 += 1
			else:
				class1 += 1
	return {0 : 1, 1 : class1/class2}
	
	
def build_model(dim=256, word_dim = 256, subreddit_dim = 64, vocab_size = 30000, n_subreddits = 1000, maxlen = None, use_dropout = False):
	""" This network structure is based on Recurrent Convolutional Neural Networks for Text Classification, Lai et al., 2015, """	
	
	input_text = Input(shape=(None,), dtype='int32', name='text_input')
	model = Embedding(vocab_size, word_dim, mask_zero = False)(input_text)
	model = Bidirectional(LSTM(dim, return_sequences = True), merge_mode = "concat")(model)
	model = LeakyReLU(0.2)(model)
	if use_dropout:
		model = Dropout(0.2)(model)
	model = LSTM(dim, return_sequences = True)(model)
	model = LeakyReLU(0.2)(model)
	if use_dropout:
		model = Dropout(0.2)(model)
	model = LSTM(dim, return_sequences = False)(model)
	model = LeakyReLU(0.2)(model)
	
	input_subreddit = Input(shape=(1,), dtype='int32', name='subreddit_input')
	sr_embedding = Embedding(n_subreddits, subreddit_dim, mask_zero = False)(input_subreddit)
	sr_flattened = Flatten()(sr_embedding)
	model = merge([sr_flattened, model], mode="concat", concat_axis = 1)
	if use_dropout:
		model = Dropout(0.2)(model)
	model = Dense(dim)(model)
	if use_dropout:
		model = Dropout(0.5)(model)
	model = LeakyReLU(0.2)(model)
	model = Dense(dim/2)(model)
	model = LeakyReLU(0.2)(model)
	if use_dropout:
		model = Dropout(0.5)(model)
	modelout = MaxoutDense(2, nb_feature = 5)(model)
	modelout = Activation("softmax")(modelout)
	model = Model(input = [input_text, input_subreddit], output = [modelout])
	model.compile(loss='binary_crossentropy',
				  optimizer='adam')
				  
	return model
	
	
def test(word_dim=256,  # word vector dimensionality
		  dim=768,  # the number of LSTM units
		  patience=10,  # early stopping patience
		  max_epochs=5000,
		  finish_after=10000000,  # finish after this many updates
		  dispFreq=100,
		  vocab_size=30000,  # vocabulary size
		  n_subreddits = 512, # number of subreddits to track specifically
		  subreddit_dim = 64, # subreddit vector dimensionality
		  maxlen=200,  # maximum length of the description
		  batch_size=64,
		  valid_batch_size=64,
		  savedir="./",
		  validFreq=100000,
		  saveFreq=25000,   # save the parameters after every saveFreq updates
		  dataset="./reddit_comment_training.tsv",
		  valid_dataset="./reddit_comment_valid.tsv",
		  test_dataset = "./reddit_comment_testing.tsv",
		  dictionary="./reddit_comment_training.tsv_worddict.pkl",
		  sr_dictionary="./reddit_comment_training.tsv_srdict.pkl",
		  use_dropout=True,
		  reload=True,
		  overwrite=False):


	test = TextIterator(test_dataset, dictionary, sr_dictionary, n_words_source=vocab_size, n_subreddits = n_subreddits, batch_size=batch_size, shuffle = False)
	
	print "Building the model"
	model = build_model(dim = dim, word_dim  = word_dim, vocab_size = vocab_size, n_subreddits = n_subreddits, subreddit_dim = subreddit_dim, use_dropout = use_dropout)
	print "Model built"
	
	# Initializaton
	uidx = 1
	scores = []
	history_errs = []
	ud_start = time.time()
	estop = False
	
	if (reload):
		print "Attempting to load most recent model"
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

	num_correct = 0
	num_incorrect = 0
	for x, y in test:
		x, y = prepare_data(x, y, maxlen = maxlen)
		predictions = model.predict_on_batch(x, y, batch_size = len(x))
		for pred, truth in zip(predictions, y):
			if pred == truth:
				num_correct += 1
			else:
				num_incorrect += 1
	
	print (str((num_correct + 0.0) / (num_incorrect + num_correct + 0.0)) + "% correct")
		
		
if __name__ == '__main__':
	test()	