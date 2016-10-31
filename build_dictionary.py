import numpy
import cPickle as pkl

from nltk.tokenize import WordPunctTokenizer
import sys
import fileinput

from collections import OrderedDict


def main():
	tokenizer = WordPunctTokenizer()
	for filename in sys.argv[1:]:
		print ('Processing' + "  " + filename)
		word_freqs = OrderedDict()
		sr_freqs = OrderedDict()
		with open(filename, 'r') as f:
			for line in f:
			
				#By convention, text is the first column of the .tsv
				line = line.split("\t")
				subreddit = line[0]
				text = line[1]
				
				words_in = tokenizer.tokenize(text)
				for w in words_in:
					if w not in word_freqs:
						word_freqs[w] = 0
					word_freqs[w] += 1
				if subreddit not in sr_freqs:
					sr_freqs[subreddit] = 0
				sr_freqs[subreddit] += 1
		
		words = word_freqs.keys()
		freqs = word_freqs.values()

		sorted_idx = numpy.argsort(freqs)
		sorted_words = [words[ii] for ii in sorted_idx[::-1]]

		worddict = OrderedDict()
		worddict['eos'] = 0
		worddict['UNK'] = 1
		for ii, ww in enumerate(sorted_words):
			worddict[ww] = ii+2

		with open('%s_worddict.pkl'%filename, 'wb') as f:
			pkl.dump(worddict, f)

			
		subreddits = sr_freqs.keys()
		freqs = sr_freqs.values()
		
		sorted_idx = numpy.argsort(freqs)
		sorted_subreddits = [subreddits[ii] for ii in sorted_idx[::-1]]
		srdict = OrderedDict()
		srdict["UNK"] = 0
		for ii, sr in enumerate(sorted_subreddits):
			srdict[sr] = ii+1
			
		with open('%s_srdict.pkl'%filename, 'wb') as f:
			pkl.dump(srdict, f)
		
		print ('Done')


if __name__ == '__main__':
	main()
