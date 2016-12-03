""" This script processes the Reddit data from the original format, removes extraneous information, and makes an 80-10-10 train/valid/test split. """
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import numpy as np
import copy

df1 = pd.read_csv("./comments.tsv", "\t", header = None, names = ["uid1", "subreddit", "uid2", "postid", "parentid", "threadid", "text"])
df2 = pd.read_csv("./actions.tsv", "\t", header = None, names = ["action", "timestamp", "postid", "subreddit", "is_automod"])

# delete unused data
del df1 ["uid1"]
del df1 ["uid2"]
del df1 ["threadid"]
del df2 ["timestamp"]
del df2 ["subreddit"]
del df2 ["is_automod"]

# merge comments and actions and randomize their order
df2 = df2.sort_values("action")
df2 = df2.drop_duplicates(subset = ["postid"])
df3 = df1.merge(df2, on = "postid", how = "outer")
unique_sorted_joined_table = df3.reindex(np.random.permutation(df3.index))

processed_table = unique_sorted_joined_table.fillna(value={"action":"approvecomment", "text":"<EMPTY>"})
processed_table_copy_with_only_body_and_comment_fullname = copy.deepcopy(processed_table)[["postid","text"]]
processed_table = processed_table.merge(processed_table_copy_with_only_body_and_comment_fullname, how="left", left_on="parentid", right_on="postid")
processed_table.rename(columns={'text_x': 'text', "postid_x": "postid"}, inplace=True)
del processed_table ["postid_y"]
del processed_table ["postid"]
del processed_table ["parentid"]
processed_table = processed_table.fillna(value={"text_y":"No Parent"})
print processed_table


# tokenize and lowercase the data
#tokenizer = WordPunctTokenizer()
#lowercase = lambda x: (" ").join(tokenizer.tokenize(x.lower()))
#processed_table = processed_table.applymap(lowercase)

# split 80-10-10
testing_df = processed_table.iloc[:int(len(processed_table)/10)]
valid_df = processed_table.iloc[int((len(processed_table)/10)):int((3*len(processed_table)/20))]
training_df = processed_table.iloc[int((3*len(processed_table)/20)):]

# print data to .tsv files
testing_df.to_csv("reddit_comment_testing.tsv", "\t", header = False, index = False)
valid_df.to_csv("reddit_comment_valid.tsv", "\t", header = False, index = False)
training_df.to_csv("reddit_comment_training.tsv", "\t", header = False, index = False)