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

def generate_progress_graph(model_directory, valid_dataset, dictionary, sr_dictionary, test_dataset):
	modelfiles = [(join(model_directory, f), int(f.split(".")[1].replace("iter", ""))) for f in listdir(model_directory) if (isfile(join(model_directory, f)) and ("model" in str(f)) and (("npz" in str(f)) or ("h5" in str(f))) and (not "validout" in str(f)) and (not "testout" in str(f)) and (not ".pkl" in str(f)) and (not ".png" in str(f)))]
	modelfiles = sorted(modelfiles, key = lambda file: file[1])
	print modelfiles
	modelfiles, iters = zip(*modelfiles)
	accs = []
	precs = []
	recs = []
	
	for model in modelfiles:
		acc, prec, rec = test(modelfile = model, valid_dataset = valid_dataset, dictionary = dictionary, sr_dictionary = sr_dictionary, test_dataset = test_dataset)
		accs.append(acc)
		precs.append(prec)
		recs.append(rec)
		
		
	with open(join(model_directory, "_training_results.csv"), "wb") as f:
		f.write("iteration, accuracy, precision, recall \n")
		for line in zip(*[iters, accs, precs, recs]):
			line = ", ".join(map(str, line))
			f.write(line + "\n")
	fig, ax1 = plt.subplots()
	t = numpy.asarray(iters)
	ax1.plot(t, numpy.asarray(accs), 'b-', label="Accuracy")
	ax1.plot(t, numpy.asarray(precs), 'g-', label="Precision")
	ax1.plot(t, numpy.asarray(recs), 'k-', label="Recall")
	ax1.set_xlabel('Number of Iterations')
	# Make the y-axis label and tick labels match the line color.
	ax1.set_ylabel('Percent score', color='b')
	for tl in ax1.get_yticklabels():
		tl.set_color('b')

	plt.savefig("iterations_progress.png", bbox_inches='tight')
	plt.clf()
	
def test(word_dim=256,  # word vector dimensionality
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
		  use_dropout=True,
		  reload=True,
		  overwrite=False,
		  legal_subreddits = None, #["science"],
		  modelfile = None):



	test = PostmungedTextIterator(test_dataset, dictionary, sr_dictionary, n_words_source=vocab_size, n_subreddits = n_subreddits, batch_size=batch_size, shuffle = False, legal_subreddits = legal_subreddits)
	
	
	if modelfile == None:
		print "Attempting to load most recent model"
		modelfiles = [(join(savedir, f), int(f.split(".")[-2].replace("iter", ""))) for f in listdir(savedir) if isfile(join(savedir, f)) and "model" in f and ".h5" in f]
		most_recent_model = ("", 0)
		for modelfile in modelfiles:
			if modelfile[1] >= most_recent_model[1]:
				most_recent_model = modelfile
			
		if os.path.isfile(most_recent_model[0]):
			print "Loading from model", most_recent_model[0]
			model = load_model(most_recent_model[0])
		else:
			print "Failed to load model -- no acceptable models found"
	else:
		print "Loading from model "+ modelfile
		model = load_model(modelfile)
	
	true_negative = 0.
	true_positive = 0.
	false_negative = 0.
	false_positive = 0.
	y_true_score = []
	y_pred_score = []
	for x, y in test:
		x, y = prepare_data(x, y, maxlen = maxlen)
		predictions = model.predict(x)
		for pred, truth in zip(predictions, y):
			pred = pred[0]
			y_pred_score.append(pred)
			y_true_score.append(truth)
			if pred > 0.5:
				pred = 1
			else:
				pred = 0
			if pred == 0 and truth == 0:
				true_negative += 1
			elif pred == 0 and truth == 1:
				false_negative += 1
			elif pred == 1 and truth == 1:
				true_positive += 1
			elif pred == 1 and truth == 0:
				false_positive += 1
			else:
				print "Illegal values: ", "Predicted: " + str(pred), "True: " + str(truth)
	
	num_correct = true_negative + true_positive
	num_incorrect = false_negative + false_positive
	
	# Avoid a crash in some edge cases where nothing was marked as positive
	if (true_positive + false_positive) == 0:
		false_positive += 1
	if (true_positive + false_negative) == 0:
		false_negative += 1
		
	percent_correct = 100.0*(num_correct + 0.0) / (num_incorrect + num_correct + 0.0)
	precision = 100.0*(true_positive + 0.0) / (true_positive + false_positive + 0.0)
	recall = 100.0*(true_positive + 0.0) / (true_positive + false_negative + 0.0)
	print modelfile
	print (str(percent_correct) + "% correct")
	print (str(precision) + "% precision")
	print (str(recall) + "% recall")
	print (str(100 * (true_positive + false_positive) / (num_correct + num_incorrect)) + "% predicted to be removed")
	
	y_true_score = numpy.asarray(y_true_score)
	y_pred_score = numpy.asarray(y_pred_score)
	fpr, tpr, _ = roc_curve(y_true_score, y_pred_score)
	roc_auc = auc(fpr, tpr)

	
	plt.clf()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.savefig(modelfile + "-ROC.png", bbox_inches = "tight")

	new_precision, new_recall, _ = precision_recall_curve(y_true_score, y_pred_score)
	pr_auc = auc(new_recall, new_precision)

	plt.clf()
	plt.plot(new_recall, new_precision, label='PR curve (area = %0.2f)' % pr_auc)
	plt.plot([0, 1], [float(false_positive + true_negative) / (true_positive + false_negative + false_positive + true_negative), float(false_positive + true_negative) / (true_positive + false_negative + false_positive + true_negative)], linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.legend(loc="upper right")
	plt.savefig(modelfile + "-Precision-Recall.png", bbox_inches = "tight")
	plt.clf()
	
	return percent_correct, precision, recall
	
	
	
		
if __name__ == '__main__':
	generate_progress_graph("./", "./reddit_comment_valid.tsv", "./reddit_comment_training.tsv_worddict.pkl", "./reddit_comment_training.tsv_srdict.pkl",  "./reddit_comment_testing.tsv")