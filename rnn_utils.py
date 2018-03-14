import numpy as np
import os
import re
from collections import defaultdict
import operator

def load_matrix_imdb(path='imdb.npz', num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):
    """
    Modified code from Keras
    Loads data matrixes from npz file, crops and pads seqs and returns
    shuffled (x_train, y_train), (x_test, y_test)
    """
    if not os.path.exists(path):
        print("Downloading matrix data into current folder")
        os.system("wget https://s3.amazonaws.com/text-datasets/imdb.npz")
        
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if not num_words:
        num_words = max([max(x) for x in xs])
    if not maxlen:
        maxlen = max([len(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    xs_new = []
    for x in xs:
        x = x[:maxlen] # crop long sequences
        if oov_char is not None: # replace rare or frequent symbols 
            x = [w if (skip_top <= w < num_words) else oov_char for w in x]
        else: # or filter rare and frequent symbols
            x = [w for w in x if skip_top <= w < num_words]
        x_padded = np.zeros(maxlen)#, dtype = 'int32')
        x_padded[-len(x):] = x
        xs_new.append(x_padded)    
            
    idx = len(x_train)
    x_train, y_train = np.array(xs_new[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs_new[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)