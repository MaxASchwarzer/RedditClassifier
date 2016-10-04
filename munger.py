import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer


df1 = pd.read_csv("./comments_partial.tsv", "\t", header = None, names = ["uid1", "subreddit", "uid2", "postid", "parentid", "threadid", "text"])
df2 = pd.read_csv("./actions.tsv", "\t", header = None, names = ["action", "timestamp", "postid", "subreddit", "is_automod"])

del df1 ["uid1"]
del df1 ["uid2"]
del df1 ["subreddit"]
del df1 ["parentid"]
del df1 ["threadid"]
del df2 ["timestamp"]
del df2 ["subreddit"]
del df2 ["is_automod"]

df3 = df1.merge(df2, on = "postid", how = "outer")
df3 = df3.reindex(np.random.permutation(df3.index))

del df3["postid"]
df3 = df3.fillna("no_action")

tokenizer = WordPunctTokenizer()
lowercase = lambda x: (" ").join(tokenizer.tokenize(x.lower()))
df3 = df3.applymap(lowercase)

testing_df = df3.iloc[:int(len(df3)/10)]
valid_df = df3.iloc[int((len(df3)/10)):int((3*len(df3)/20))]
training_df = df3.iloc[int((3*len(df3)/20)):]

testing_df.to_csv("reddit_comment_testing.tsv", "\t", header = False, index = False)
valid_df.to_csv("reddit_comment_valid.tsv", "\t", header = False, index = False)
training_df.to_csv("reddit_comment_training.tsv", "\t", header = False, index = False)