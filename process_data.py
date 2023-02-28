import os
import string
import pdb
import re
import ast
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

corpus = pd.read_csv("HYPO.tsv", delimiter="\t")
# Drop missing values
corpus = corpus.dropna().reset_index(drop=True) # 698 items left

hypo = corpus.iloc[:, 0]
paraphrase = corpus.iloc[:, 1]
minimal = corpus.iloc[:, 2]

## Transform to labelled dataset of hyperboles, paraphrases, and minimal units

pos_labels = np.ones(709, hypo); neg_labels = np.zeros(709, hypo)
## Get hyperbolic *token( spans)
# Helper function to check that numbers consist of strictly increasing integers

def groupSequence(x):
    it = iter(x)
    prev, res = next(it), []
    while prev is not None:
        start = next(it, None)
        if prev + 1 == start:
            res.append(prev)
        elif res:
            yield list(res + [prev])
            res = []
        prev = start

def get_target(x, y, z):
    #pdb.set_trace()

    f = lambda s: np.array(word_tokenize(re.sub(r"\.+", ".", s.lower()).translate(dict.fromkeys(string.punctuation))))
    #f = lambda s: np.array(word_tokenize(s.lower()))
    #y = y.replace(".", "PUNCT")

    x = f(x); y = f(y); z = f(z)
    filtered_words = stopwords.words('english')+['!', '.', ',']
    x_filtered = np.array(['xSTOP' if w in filtered_words else w for w in x])
    y_filtered = np.array(['ySTOP' if w in filtered_words else w for w in y])

    #x_indices = np.where(~np.in1d(x, z))[0]
    x_indices = np.where(np.in1d(x_filtered, y_filtered))[0]
    y_indices = np.where(np.in1d(y_filtered, x_filtered))[0]

    try:
        if len(x_indices) >= 3:
            x_indices = sorted(list(groupSequence(x_indices.tolist())), key=len, reverse=True)[0]
    except IndexError:
        pass

    try:
        if len(y_indices) >= 3:
            y_indices = sorted(list(groupSequence(y_indices.tolist())), key=len, reverse=True)[0]
    except IndexError:
        pass

    try:
        x_start = x_indices[0]; x_end = x_indices[-1]+1; y_start = y_indices[0]; y_end = y_indices[-1]+1
    except IndexError:
     x_start, x_end, y_start, y_end = 0, 1, 0, 1

    x_span1 = [x_start, x_end]; x_span_word = " ".join(x[x_indices].tolist()); y_span1 = [y_start, y_end]; y_span_word = " ".join(y[y_indices].tolist())
    x_targets, y_targets = make_target_dict(x_span1, x_span_word, 'nonliteral'), make_target_dict(y_span1, y_span_word, 'literal')
    return [x_targets], [y_targets]

def make_target_dict(span1, span_word, label):
    target_dict = dict()
    target_dict['span1'] = span1
    target_dict['span_word'] = span_word
    target_dict['label'] = label
    return target_dict

targets = corpus.apply(lambda X: get_target(X['HYPO'], X['MINIMAL UNITS CORPUS'], X['PARAPHRASES']), axis=1)
target_df = pd.DataFrame(list(zip(*targets)),index=['hyp', 'min']).T

## Re-write data

column_names = ['text', 'targets']
hypo_df = pd.DataFrame(columns=column_names)

hypo_df['text'] = pd.concat([hypo, minimal])
hypo_df['targets'] = pd.concat([target_df['hyp'], target_df['min']], ignore_index=True)
#hypo_df['label'] = np.array([pos_labels, neg_labels, neg_labels]).reshape(-1)
#label_names = ['pos', 'neg']

# Word tokenize text
hypo_df['text'] = hypo_df.apply(lambda X: " ".join(word_tokenize(X['text'])), axis=1)

## Write to CSV file

current_dir = os.getcwd()
new_dir = os.path.join(current_dir, "preprocessed_hypo_dataset")
if not os.path.exists(new_dir):
    os.mkdir(new_dir)   

#hypo_df.to_csv(os.path.join(new_dir, "preproc_hypo_all.csv"), index=True)

### Load manually edited data

import ast
import pandas as pd

hypo_df = pd.read_csv(os.path.join("data", "preproc_hypo_all_edited.csv")).drop('index_col', axis=1)
hypo_df['targets'] = hypo_df['targets'].apply(ast.literal_eval)

#import csv
#with open(os.path.join('data', 'preproc_hypo_all_edited.csv'), newline='') as csvfile:
#    reader = csv.DictReader(csvfile)
#    for i, row in enumerate(reader):
#        try:
#            targets = ast.literal_eval(row['targets'])
#        except SyntaxError as e:
#            print(f"Error in row {i+1}: {e}")

## Split data (70/30, in accordance with Biddle et Al.)
hypo_df_train = hypo_df.sample(frac=0.7, random_state=42)
hypo_df_remains = hypo_df.drop(hypo_df_train.index)
# Split test data into development and test sets
hypo_df_test = hypo_df_remains.sample(frac=0.66, random_state=42)
hypo_df_dev = hypo_df_remains.drop(hypo_df_test.index)

## Write to json files

hypo_df_train.to_json(os.path.join(new_dir, "train.json"), 'records', lines=True)
hypo_df_test.to_json(os.path.join(new_dir, "test.json"), orient='records', lines=True)
hypo_df_dev.to_json(os.path.join(new_dir, "devjson"), orient='records', lines=True)

### Write new file with dev set samples for annotation purposes

annotation_indices = hypo_df_dev.index[hypo_df_dev.index < 698]
annotation_df = corpus.loc[annotation_indices]
annotation_df.to_csv(os.path.join(new_dir, "annotations.tsv"), sep="\t")

### Hyperprobe

#probe_df = pd.read_csv("hyperprobe.csv")
#hyp_probe = probe_df[probe_df.label == 1]
#par_probe = probe_df[probe_df.label == 0]

## Extract span from lexical unit via the keyword column

#new_probe_df = pd.DataFrame(column_names)
#new_probe_df['text'] = probe_df.text
#new_probe_df['label'] = probe_df.label