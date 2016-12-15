from __future__ import absolute_import

import theano
import numpy
import os
import cPickle as pkl
import time
from os import listdir
from os.path import isfile, join

from postmunge import PostmungedTextIterator
from neural_classifier import build_model, prepare_data
from keras.models import Sequential, load_model, Model

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from nltk.tokenize import WordPunctTokenizer


def preprocess_data(input, sr_dictionary, word_dictionary, vocab_size = 30000, n_subreddits = 500):
	with open(sr_dictionary, "rb") as s, open(word_dictionary, "rb") as w:
		sr_dictionary = pkl.load(s)
		word_dictionary = pkl.load(w)

	tokenizer = WordPunctTokenizer()
	with open(input, "rb") as f:
		raw_input = f.readlines()
	x = []
	y = []
	originals = []
	
	for ss in raw_input:

		ss = ss.split("\t")
		subreddit = ss[0]
		
		print ss
		original_text = ss[1]
		originals.append(original_text)
		tt = ss[2]
		parent = ss[3]
		
		text = [word_dictionary[w] if w in word_dictionary else 1 for w in tokenizer.tokenize(original_text)]
		parent = [word_dictionary[w] if w in word_dictionary else 1 for w in tokenizer.tokenize(parent)]
		if subreddit in sr_dictionary:
			subreddit = sr_dictionary[subreddit]
		else:
			subreddit = 0
		if not vocab_size is None:
			text = [w if w < vocab_size else 0 for w in text]
			parent = [w if w < vocab_size else 0 for w in parent]

		if n_subreddits > 0 and (subreddit + 1) > n_subreddits:
			subreddit = 0
			

		if "removecomment" in tt:
			tt = 1
		else:
			tt = 0	
			
		print text
		x.append([text, subreddit, parent])
		y.append(tt)
		
	return x, y, originals
	
	
	
def predict(word_dim=256,  # word vector dimensionality
		  dim=512,  # the number of LSTM units
		  patience=2,  # early stopping patience
		  max_epochs=5000,
		  finish_after=10000000,  # finish after this many updates
		  dispFreq=100,
		  vocab_size=30000,  # vocabulary size
		  n_subreddits = 8, # number of subreddits to track specifically
		  subreddit_dim = 128, # subreddit vector dimensionality
		  maxlen=300,  # maximum length of the description
		  batch_size=96,
		  valid_batch_size=96,
		  savedir="./",
		  validFreq=100000,
		  saveFreq=25000,   # save the parameters after every saveFreq updates
		  dataset="./reddit_comment_samples.tsv",
		  dictionary="./reddit_comment_training.tsv_worddict.pkl",
		  sr_dictionary="./reddit_comment_training.tsv_srdict.pkl",
		  use_dropout=True,
		  reload=True,
		  overwrite=False,
		  model_directory = "./"):
		  
	x, y, originals = preprocess_data(dataset, sr_dictionary, dictionary, vocab_size, n_subreddits)
		  
	print originals
	print "Attempting to load most recent model"
	modelfiles = [(join(model_directory, f), int(f.split(".")[1].replace("iter", ""))) for f in listdir(model_directory) if (isfile(join(model_directory, f)) and ("model" in str(f)) and (("npz" in str(f)) or ("h5" in str(f))) and (not "validout" in str(f)) and (not "testout" in str(f)) and (not ".pkl" in str(f)) and (not ".png" in str(f)))]
	modelfiles = sorted(modelfiles, key = lambda x: x[1])
	most_recent_model = modelfiles[-1]
		
	if os.path.isfile(most_recent_model[0]):
		print "Loading from model", most_recent_model[0]
		model = load_model(most_recent_model[0])
	else:
		print "Failed to load model -- no acceptable models found"
	
	
	processed_x, processed_y = prepare_data(x, y)
	predictions = model.predict(processed_x, batch_size = 1)
	
	for sentence, prediction in zip(originals, predictions):
		print sentence, "\t", prediction[0]
	
		
if __name__ == '__main__':
	predict()