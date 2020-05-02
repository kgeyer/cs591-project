import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import sys, getopt

def calcPerp(model, docs_te):
    """
    Caclulate model perlexity for prodLDA or nvlda
    """
    cost=[]
    for doc in docs_te:
        doc = doc.astype('float32')
        n_d = np.sum(doc)
        c=model.test(doc)
        cost.append(c/n_d)
    perp = np.exp(np.mean(np.array(cost)))
    print ('The approximated perplexity is: ',(perp))
    return(perp)

def print_top_words(beta, vocab, n_top_words=10):
    """
    print the top words for each topic of the prodlda/ldavae models
    """
    inv_vocab = {v: k for k, v in vocab.items()}
    K = len(beta)
    topic_keywords = []
    for kk in range(K):        # for each topic
        this_topic = []
        top_keyword_locs = (-1.*beta[kk]).argsort()[:n_top_words]
        for ww in range(n_top_words):
            this_topic.append(inv_vocab[top_keyword_locs[ww]])
        topic_keywords.append(this_topic)
    return(topic_keywords)

def show_topics(vectorizer, lda_model, n_words=10):
    """
    print the top words of each topic from dmflda
    """
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

def calculate_lda_coherence(doc_count_fn, word_cocount_fn, vocab, topic_keywords, total_docs):
    """
    Calculate the coherence for each topic, for dmflda
    """
    # Load count arrays
    print('Loading word count vector')
    word_count = np.load(doc_count_fn)
    total_words = np.sum(word_count)
    print('Loading word co-occurance count matrix')
    coword_count = np.load(word_cocount_fn)
    # Calcualte coherence
    K = len(topic_keywords)
    print("K:", K)
    nwords = len(topic_keywords[0])
    print("nwords: ", nwords)
    coher = np.zeros((K,1))
    for kk in range(K):
        for ii in range(nwords):
            wordii = vocab[topic_keywords[kk][ii]]   # idx
            for jj in range(ii+1,nwords):
                wordjj = vocab[topic_keywords[kk][jj]] #
                coher[kk] += np.log((coword_count[wordjj,wordii]/total_docs) + 1.) - np.log(word_count[wordii][0]/total_words)
    print("The coherence is ", coher)
    return(coher)
