import sys
import gzip
import numpy as np
from gensim.models import KeyedVectors
import io


def read_word_vectors(filename):
    word_vecs = {}
    with (gzip.open(filename, 'rt', encoding='utf-8') if filename.endswith('.gz') else open(filename, 'r',
                                                                                            encoding='utf-8')) as file_object:
        for line_num, line in enumerate(file_object):
            line = line.strip().lower()
            parts = line.split()

            if len(parts) < 2:
                continue

            word = parts[0]
            vec_values = np.array([float(val) for val in parts[1:]])

            norm = np.linalg.norm(vec_values)
            word_vecs[word] = vec_values / (norm + 1e-6)

    sys.stderr.write("Vectors read from: " + filename + "\n")
    return word_vecs


def load_fasttext_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def load_word_vectors(path, wv_type):
    x_train_ori = []
    x_train_names = []
    print("Loading vectors.")
    if wv_type == "Glove":
        word_vecs = read_word_vectors(path)
        glove = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                glove[word] = coefs
            f.close()
        for x in glove:
            x_train_ori.append(glove[x])
            x_train_names.append(x)

    elif wv_type == "Word2vec":
        word_vecs = KeyedVectors.load_word2vec_format(path, binary=True)
        for word in word_vecs.key_to_index:
            x_train_ori.append(word_vecs[word])
            x_train_names.append(word)

    elif wv_type == "Fasttext":
        word_vecs = load_fasttext_vectors(path)
        for word, vector in word_vecs.items():
            x_train_ori.append(vector)
            x_train_names.append(word)

    x_train_ori = np.array(x_train_ori)

    print("x_train_ori:", np.asarray(x_train_ori).shape)
    print("Done.")

    return word_vecs, x_train_ori, x_train_names

