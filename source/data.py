import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import pickle
import sys, getopt
from scipy.sparse import csr_matrix

def onehot(data, min_length):
    """
    Convert data (document) to onehot vectors
    """
    return np.bincount(data, minlength=min_length)

def convert_to_onehot(rawdata, vocab_size):
    """
    Convert a raw data set to a matrix of one-hot vectors
    This is used for prodLDA and LDA-vae
    """
    print ('Converting data to one-hot representation')
    data = np.array([onehot(doc.astype('int'),vocab_size) for doc in rawdata if np.sum(doc)!=0])
    return(data)

def count_words(train_data, test_data):
    """
    Create a vector of word counts
    """
    print("Creating word count vector")
    vocab_size = train_data.shape[1]
    word_count = np.zeros((vocab_size,1))
    for ii in range(vocab_size):
        word_count[ii] = np.sum(train_data[:,ii]) + np.sum(test_data[:,ii])
    return(word_count)

def count_doc_with_word(train_data, test_data):
    """
    Count the number of documents that contain each words
    """
    print("Counting the number of documents that contain word i")
    vocab_size = train_data.shape[1]
    doc_count = np.zeros((vocab_size,1))
    for ii in range(vocab_size):
        doc_count[ii] = np.sum(train_data[:,ii]>0) + np.sum(test_data[:,ii]>0)
    return(doc_count)

def count_word_cooccurance(train_data, test_data):
    """
    Count the number of word co-occurances within the same document
    """
    print("Creating word co-occurance count matrix")
    vocab_size = train_data.shape[1]
    coword_count = np.zeros((vocab_size,vocab_size))
    for ii in range(vocab_size):
        for jj in range(ii+1, vocab_size):
            coword_count[ii,jj] = np.sum(np.logical_and(train_data[:,ii] > 0, train_data[:,jj])) \
            + np.sum(np.logical_and(test_data[:,ii] > 0, test_data[:,jj]))
            coword_count[jj,ii] = coword_count[ii,jj]
    return(coword_count)

def rawdata_to_text(rawdata, vocab, text_dir):
    """
    Convert raw data to text strings
    """
    # remove & create directory for text files
    if os.path.isdir(text_dir):
        os.rmdir(text_dir)
    os.mkdir(text_dir)
    # begin conversion
    ndocs = rawdata.shape[0]
    inv_vocab = {v: k for k, v in vocab.items()}
    for dd in range(ndocs):
        doc_fn = os.path.join(text_dir, 'doc_' + str(dd) + '.txt')
        txt_idx = [inv_vocab[x] for x in rawdata[dd]]
        txt = " ".join(txt_idx)
        with open(doc_fn, "w") as text_file:
            text_file.write(txt)
