"""Regenerate DCNet's GloVe caption pickles with the author's own `preprocess/process_cap.py` recipe.

Recipe source: workbench/third_party/DCNet/preprocess/process_cap.py, run against
`<dataset>/captions`. Two recorded deviations, both logged in the report:
  1. the author script samples OOV vectors with an UNSEDED `np.random.normal(0, 0.3, (300,))`; this
     run seeds NumPy so the artifact is reproducible (the generated vectors for OOV words differ
     from the author's bytes either way — they were never published);
  2. the spaCy model is loaded from an explicit path instead of the `en_vectors_web_lg` package name.
Nothing else changes: the vocabulary is built from `cap*train.json`, hyphen/dot handling, NLTK
`word_tokenize`, occurrence threshold `cnt > 2`, and the output filenames all follow the author code.
"""
import glob, json, os, pickle, sys
from collections import Counter

import nltk
import numpy as np
import spacy

dataset_root = sys.argv[1]
vectors_path = sys.argv[2]
np.random.seed(0)

glove = spacy.load(vectors_path).vocab

cap_train_files = sorted(glob.glob(os.path.join(dataset_root, 'captions', 'cap*train.json')))
counter = Counter()
for cap_file in cap_train_files:
    with open(cap_file) as f:
        for data in json.load(f):
            for caption in data['captions']:
                caption = caption.replace('-', ' ').replace('.', '')
                counter.update(nltk.tokenize.word_tokenize(caption.lower()))
print('vocab tokens', len(counter), flush=True)

vocab_words, no_word = {}, []
for word, cnt in counter.items():
    if glove.has_vector(word):
        vocab_words[word] = glove.get_vector(word)
    elif cnt > 2:
        no_word.append(word)
        vocab_words[word] = np.random.normal(0, 0.3, (300,))
print('vocab_words', len(vocab_words), 'oov_filled', len(no_word), flush=True)

with open(os.path.join(dataset_root, 'captions', 'glove_vecs.pkl'), 'wb') as f:
    pickle.dump(vocab_words, f)

for cap_file in sorted(glob.glob(os.path.join(dataset_root, 'captions', 'cap*.json'))):
    with open(cap_file) as f:
        cap_data = json.load(f)
    sn = os.path.basename(cap_file).split('.')
    sn = '.'.join(sn[:2] + ['glove'] + sn[2:]).replace('.json', '.pkl')
    save_file = os.path.join(os.path.dirname(cap_file), sn)
    for data in cap_data:
        w2v = []
        for caption in data['captions']:
            caption = caption.replace('-', ' ').replace('.', '')
            for word in nltk.tokenize.word_tokenize(caption.lower()):
                if word in vocab_words:
                    w2v.append(vocab_words[word])
        data['wv'] = np.stack(w2v)
    with open(save_file, 'wb') as f:
        pickle.dump(cap_data, f)
    print('wrote', save_file, flush=True)
