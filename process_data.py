import os
import string
import pdb
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

corpus = pd.read_csv("HYPO.tsv", delimiter="\t")
# Drop missing values
corpus.dropna(inplace=True)

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
    y = y.replace(".", "!")

    x = f(x); y = f(y); z = f(z)
    #y = np.array(['STOP' if w in stopwords.words('english') else w for w in y])

    x_indices = np.where(~np.in1d(x, z))[0]
    y_indices = np.where(np.in1d(y, x))[0]

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

hypo_df['text'] = pd.concat([hypo, minimal], ignore_index=True)
hypo_df['targets'] = pd.concat([target_df['hyp'], target_df['min']], ignore_index=True)
#hypo_df['label'] = np.array([pos_labels, neg_labels, neg_labels]).reshape(-1)
#label_names = ['pos', 'neg']

# Word tokenize text
hypo_df['text'] = hypo_df.apply(lambda X: " ".join(word_tokenize(X['text'])), axis=1)

## Split data (70/30, in accordance with Biddle et Al.)
hypo_df_train = hypo_df.sample(frac=0.7, random_state=1)
hypo_df_test = hypo_df.drop(hypo_df_train.index)

## Write to json files
current_dir = os.getcwd()
new_dir = os.path.join(current_dir, "preprocessed_hypo_dataset")
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

hypo_df.to_csv(os.path.join(new_dir, "preproc_hypo_all.csv"), index=False)
hypo_df_train.to_json(os.path.join(new_dir, "train.json"), 'records', lines=True)
hypo_df_test.to_json(os.path.join(new_dir, "test.json"), orient='records', lines=True)

#hypo_df_train.to_csv(os.path.join(new_dir, "train.csv"), index=False)
#hypo_df_test.to_csv(os.path.join(new_dir, "test.csv"), index=False)
#hypo_df.to_csv(os.path.join(new_dir, "preprocessed_hypo.csv"), index=False)

### Hyperprobe

#probe_df = pd.read_csv("hyperprobe.csv")
#hyp_probe = probe_df[probe_df.label == 1]
#par_probe = probe_df[probe_df.label == 0]

## Extract span from lexical unit via the keyword column

#new_probe_df = pd.DataFrame(column_names)
#new_probe_df['text'] = probe_df.text
#new_probe_df['label'] = probe_df.label