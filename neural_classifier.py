from __future__ import absolute_import

import theano
import numpy
import os
import cPickle as pkl
import time
from os import listdir
from os.path import isfile, join

from postmunge import PostmungedTextIterator

from keras.models import Sequential, load_model, Model
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

def prepare_data(seqs_x, seqs_y, maxlen = None, class_weights = [1, 1]):
	# x: a list of sentences
	lengths_x = [(len(s[0]), len(s[2])) for s in seqs_x]
	
	if maxlen is not None:
		new_seqs_x = []
		new_lengths_x = []
		for (l_x1, l_x2), s_x in zip(lengths_x, seqs_x):
			if l_x1 < maxlen/2 and l_x2 < maxlen/2:
				new_seqs_x.append(s_x)
				new_lengths_x.append((l_x1, l_x2))
			else:
				new_text_length = min(maxlen/2, l_x1)
				new_parent_length = min(maxlen/2, l_x2)
				new_seqs_x.append((s_x[0][:new_text_length], s_x[1], s_x[2][:new_parent_length]))
				new_lengths_x.append((new_text_length, new_parent_length))
				
		lengths_x = new_lengths_x
		seqs_x = new_seqs_x
		if len(lengths_x) < 1:
			return None, None

	n_samples = len(seqs_x)
	post_lengths = map(lambda x : x[0], lengths_x)
	parent_lengths = map(lambda x : x[1], lengths_x)
	
	maxlen_post = numpy.max(post_lengths) + 1
	maxlen_parent = numpy.max(parent_lengths) + 1
	
	seqs_y = numpy.asarray(seqs_y)
	
	x_text = numpy.zeros((n_samples, maxlen_post)).astype('int32')
	x_parent = numpy.zeros((n_samples, maxlen_parent)).astype('int32')
	x_sr = numpy.zeros((n_samples)).astype("int32")
	
	for idx, s_x in enumerate(seqs_x):
		x_text[idx, :lengths_x[idx][0]] = s_x[0]
		x_parent[idx, :lengths_x[idx][1]] = s_x[2]
		x_sr[idx] = s_x[1]
	return [x_text, x_sr, x_parent], seqs_y
	
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
	text_model = Embedding(vocab_size, word_dim, mask_zero = False)(input_text)
	text_model = Bidirectional(LSTM(dim, return_sequences = True), merge_mode = "concat")(text_model)
	text_model = LeakyReLU(0.2)(text_model)
	if use_dropout:
		text_model = Dropout(0.2)(text_model)
	text_model = Bidirectional(LSTM(dim, return_sequences = True), merge_mode = "concat")(text_model)
	text_model = LeakyReLU(0.2)(text_model)
	if use_dropout:
		text_model = Dropout(0.2)(text_model)
	text_model = GlobalMaxPooling1D()(text_model)
	text_model = LeakyReLU(0.2)(text_model)
	
	input_parent = Input(shape=(None,), dtype='int32', name='parent_input')
	parent_model = Embedding(vocab_size, word_dim, mask_zero = False)(input_parent)
	parent_model = Bidirectional(LSTM(dim, return_sequences = True), merge_mode = "concat")(parent_model)
	parent_model = LeakyReLU(0.2)(parent_model)
	if use_dropout:
		parent_model = Dropout(0.2)(parent_model)
	parent_model = Bidirectional(LSTM(dim, return_sequences = True), merge_mode = "concat")(parent_model)
	parent_model = LeakyReLU(0.2)(parent_model)
	if use_dropout:
		parent_model = Dropout(0.2)(parent_model)
	parent_model = GlobalMaxPooling1D()(parent_model)
	parent_model = LeakyReLU(0.2)(parent_model)
	
	input_subreddit = Input(shape=(1,), dtype='int32', name='subreddit_input')
	sr_embedding = Embedding(n_subreddits, subreddit_dim, mask_zero = False)(input_subreddit)
	sr_flattened = Flatten()(sr_embedding)
	model = merge([sr_flattened, text_model, parent_model], mode="concat", concat_axis = 1)
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
	modelout = MaxoutDense(1, nb_feature = 5)(model)
	modelout = Activation("sigmoid")(modelout)
	model = Model(input = [input_text, input_subreddit, input_parent], output = [modelout])
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
	
	
def train(word_dim=256,  # word vector dimensionality
		  dim=512,  # the number of LSTM units
		  patience=2,  # early stopping patience
		  max_epochs=5000,
		  finish_after=10000000,  # finish after this many updates
		  dispFreq=100,
		  vocab_size=30000,  # vocabulary size
		  n_subreddits = 8, # number of subreddits to track specifically
		  subreddit_dim = 128, # subreddit vector dimensionality
		  maxlen=200,  # maximum length of the description
		  batch_size=96,
		  valid_batch_size=96,
		  savedir="./",
		  validFreq=100000,
		  saveFreq=25000,   # save the parameters after every saveFreq updates
		  dataset="./reddit_comment_training.tsv",
		  test_dataset = "./reddit_comment_testing.tsv",
		  valid_dataset="./reddit_comment_valid.tsv",
		  dictionary="./reddit_comment_training.tsv_worddict.pkl",
		  sr_dictionary="./reddit_comment_training.tsv_srdict.pkl",
		  legal_subreddits = None,#["science"],
		  use_dropout=True,
		  reload=False,
		  overwrite=False):


	#class_weights = get_class_weights(dataset)
	#print class_weights
	# The dataset this model was built for is heavily unbalanced, so we generate weightings to equalize the importance of the classes.
	train = PostmungedTextIterator(dataset, dictionary, sr_dictionary, n_words_source=vocab_size, n_subreddits = n_subreddits, batch_size=batch_size, shuffle = False, legal_subreddits = legal_subreddits)
	valid = PostmungedTextIterator(valid_dataset, dictionary, sr_dictionary, n_words_source=vocab_size, n_subreddits = n_subreddits, batch_size=batch_size, shuffle = False, legal_subreddits = legal_subreddits)
	
	print "Building the model"
	model = build_model(dim = dim, word_dim  = word_dim, vocab_size = vocab_size, n_subreddits = n_subreddits, subreddit_dim = subreddit_dim, use_dropout = use_dropout)
	print "Model built"
	
	
	# Initializaton
	uidx = 0
	scores = []
	history_errs = []
	ud_start = time.time()
	estop = False
	
	if (reload):
		print "Attempting to reload"
		modelfiles = [(join(savedir, f), int(f.split(".")[-2].replace("iter", ""))) for f in listdir(savedir) if isfile(join(savedir, f)) and "model" in f and ".h5" in f and not ".png" in f]
		most_recent_model = ("", 0)
		for modelfile in modelfiles:
			if modelfile[1] >= most_recent_model[1]:
				most_recent_model = modelfile
			
		if os.path.isfile(most_recent_model[0]):
			print "Loading from model", most_recent_model[0]
			model = load_model(most_recent_model[0])
			uidx = most_recent_model[1] + 1  #Adding one avoids repeating a validation error calculation for many reloads.
		else:
			print "Failed to load model -- no acceptable models found"

	for eidx in xrange(max_epochs):
	
		n_samples = 0
		
		for x, y in train:
			n_samples += len(x)
			x, y = prepare_data(x, y, maxlen = maxlen)

			if x is None:
				print 'Minibatch with zero samples under length ', maxlen
				uidx -= 1
				continue
				
			score = model.train_on_batch(x, y)#, class_weight = class_weights)
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
						model = build_model(batch_size=batch_size, num_units = dim, vocab_size = vocab_size, n_subreddits = n_subreddits, subreddit_dim = subreddit_dim, word_dim  = word_dim, use_dropout = use_dropout)
						
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
				model.save(saveto_uidx)
				print 'Done'

			# validate model on validation set and early stop if necessary
			if numpy.mod(uidx, validFreq) == 0:
				
				valid_errs = []
				for x, y in valid:
					x, y = prepare_data(x, y, maxlen = maxlen)
					v_err = model.evaluate(x, y, batch_size = batch_size)
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
		x, y = prepare_data(x, y)
		v_err = model.evaluate(x, y)
		valid_errs.append(v_err)
	valid_err = valid_errs.mean()
	print 'Valid ', valid_err
	model.save(savedir + "final.h5")
	return valid_err

if __name__ == '__main__':
	train()	