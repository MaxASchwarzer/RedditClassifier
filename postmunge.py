import numpy
import random
import json

import cPickle as pkl
import gzip
from nltk.tokenize import WordPunctTokenizer
from collections import OrderedDict



def fopen(filename, mode='r'):
	if filename.endswith('.gz'):
		return gzip.open(filename, mode)
	return open(filename, mode)

def postmunge(source, source_dict, sr_dict, n_words_source=30000, n_subreddits = 1000, legal_subreddits = None, character_level = False):
	tokenizer = WordPunctTokenizer()
	with open(source_dict, 'rb') as f:
		source_dict = pkl.load(f)
	with open(sr_dict, "rb") as f:
		sr_dict = pkl.load(f)
	for key in source_dict.keys():
		if source_dict[key] > n_words_source and n_words_source > 0:
			del source_dict[key]		
			
			
	with open(source, "rb") as f:
		with open(source + ".postmunged", "wb") as g:
			for ss in f:
				ss = ss.split("\t")
				subreddit = ss[0]
				if legal_subreddits != None and not subreddit.strip() in legal_subreddits:
					continue
				text = ss[1]
				tt = ss[2]
				parent = ss[3]

				text = tokenizer.tokenize(text)
				parent = tokenizer.tokenize(parent)
				if (character_level):
					text = " ".join(text)
					parent = " ".join(parent)
				text = [source_dict[w] if w in source_dict else 1 for w in text]
				parent = [source_dict[w] if w in source_dict else 1 for w in parent]
				if n_words_source > 0:
					text = [w if w < n_words_source else 0 for w in text]
					parent = [w if w < n_words_source else 0 for w in parent]
					
				if ss[1] in sr_dict:
					subreddit = sr_dict[ss[1]]
				else:
					subreddit = 0

				if n_subreddits > 0 and (subreddit + 1) > n_subreddits:
					subreddit = 0
					
				if "removecomment" in tt:
					tt = 1
				else:
					tt = 0
				
				printable_text = " ".join(map(str, text))
				printable_parent = " ".join(map(str, parent))
				#text = str(text).replace(",", "")[1:-1] #ditch the brackets and commas
				g.write(str(subreddit) + "\t" + printable_text + "\t" + str(tt) + "\t" + printable_parent + "\n")
				
class PostmungedTextIterator:
	"""Simple text iterator.  IMPORTANT: do not set shuffle=True if the dataset is too large to load into memory"""
	def __init__(self, source, source_dict, sr_dict,
				 batch_size=128, #maxlen=100, maxlen is currently deprecated unless experimental data shows it is necessary
				 n_words_source=-1, n_subreddits = 1000, shuffle = False, k = 100, legal_subreddits = None, character_level = False):
		
		postmunge(source, source_dict, sr_dict, n_words_source, n_subreddits, legal_subreddits, character_level)
		self.source = fopen(source + ".postmunged", 'r')
		self.source_name = source
		
		self.shuffling = shuffle
		if self.shuffling:
			self.shuffle()


		self.batch_size = batch_size
		#self.maxlen = maxlen

		self.n_words_source = n_words_source
		self.n_subreddits = n_subreddits

		self.source_buffer = []
		self.target_buffer = []
		self.k = batch_size * k

		self.end_of_data = False

	def __iter__(self):
		return self
		
	def shuffle(self):
		self.source.seek(0)
		sinput = self.source.readlines()
		random.shuffle(sinput)
		
		self.source.close()
		
		with open(self.source_name, "wb") as s:
			for sentence in sinput:
				s.write(sentence)

		self.source = fopen(self.source_name, 'r')
		
		
	def reset(self):
		if self.shuffling:
			self.shuffle()
		else:
			self.source.seek(0)

	def next(self):
		if self.end_of_data:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		source = []
		target = []

		
		# if more entries in one buffer, except -- an IO error has occurred or the data was incorrect.
		assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
		
		# fill buffer, if it's empty
		if len(self.source_buffer) == 0:
			for k_ in xrange(self.k):
				ss = self.source.readline()
				if ss == "":
					break
				
				ss = ss.split("\t")
				subreddit = ss[0]
				text = ss[1]
				tt = ss[2]
				parent = ss[3]

				self.source_buffer.append([text, subreddit, parent])
				self.target_buffer.append(tt)

			# sort source buffer on length
			slen = numpy.array([len(t[0]) + len(t[2]) for t in self.source_buffer])
			sidx = slen.argsort()

			_sbuf = [self.source_buffer[i] for i in sidx]
			_tbuf = [self.target_buffer[i] for i in sidx]

			self.source_buffer = _sbuf
			self.target_buffer = _tbuf

		if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		try:

			# actual work here
			while True:

				# read from source buffer and map to word index
				try:
					ss = self.source_buffer.pop()
					tt = self.target_buffer.pop()
				except IndexError:
					break
				text = [int(x) for x in ss[0].strip().split()]
				parent = [int(x) for x in ss[2].strip().split()]
				subreddit = int(ss[1])
				tt = int(tt.strip())
				source.append([text, subreddit, parent])
				target.append(tt)

				if len(source) >= self.batch_size:
					break
					
		except IOError:
			self.end_of_data = True

		if len(source) <= 0 or len(target) <= 0:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		return source, target
