import numpy
import random

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
	if filename.endswith('.gz'):
		return gzip.open(filename, mode)
	return open(filename, mode)


class TextIterator:
	"""Simple text iterator."""
	def __init__(self, source, source_dict,
				 batch_size=128, #maxlen=100, maxlen is currently deprecated unless experimental data shows it is necessary
				 n_words_source=-1, shuffle = False, k = 100):
		self.source = fopen(source, 'r')
		self.source_name = source
		with open(source_dict, 'r') as f:
			self.source_dict = pkl.load(f)
		
		
		self.shuffling = shuffle
		if self.shuffling:
			self.shuffle()


		self.batch_size = batch_size
		#self.maxlen = maxlen

		self.n_words_source = n_words_source

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
				tt = ss[1]
				ss = ss[0]
				
				if tt == "":
					break

				self.source_buffer.append(ss.strip().split())
				self.target_buffer.append(tt.strip().split())

			# sort source buffer on length
			slen = numpy.array([len(t) for t in self.source_buffer])
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

				# read from source file and map to word index
				try:
					ss = self.source_buffer.pop()
				except IndexError:
					break
				ss = [self.source_dict[w] if w in self.source_dict else 1
					  for w in ss]
				if self.n_words_source > 0:
					ss = [w if w < self.n_words_source else 1 for w in ss]

				# read from source file and map to word index
				tt = self.target_buffer.pop()
				if tt == ["removecomment"]:
					tt = [0., 1.]
				else:
					tt = [1., 0.]
				
				source.append(ss)
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
