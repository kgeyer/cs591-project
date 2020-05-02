#!/usr/bin/python

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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from source import prodlda, nvlda, data, results

def main():
    # Load results
    results_dir = "results/"
    result_fn1 = "results/prodlda_results.pkl"
    infile = open(result_fn1, 'rb')
    prodlda_result = pickle.load(infile)
    infile.close()
    result_fn2 = "results/nvlda_results.pkl"
    infile = open(result_fn2, 'rb')
    nvlda_result = pickle.load(infile)
    infile.close()
    result_fn3 = "results/dmfvi_results.pkl"
    infile = open(result_fn3, 'rb')
    dmfvi_result = pickle.load(infile)
    infile.close()

    # Print perplexity scores
    print("perp of prodlda: ", prodlda_result['perplexity'])
    print("perp of nvlda: ", nvlda_result['perplexity'])
    print("perp of dmfvi: ", dmfvi_result['perplexity'])

    # Create coherence historgram
    plt.hist(prodlda_result['coherence'], alpha=0.5, label='prodLDA', color="blue")
    plt.hist(nvlda_result['coherence'], alpha=0.5, label='AVITM', color="red")
    plt.hist(dmfvi_result['coher'], alpha=0.5, label='LDA VI', color="green")
    plt.axvline(x=np.mean(prodlda_result['coherence']), color="blue")
    plt.axvline(x=np.mean(nvlda_result['coherence']), color="red")
    plt.axvline(x=np.mean(dmfvi_result['coher']), color="green")
    plt.xlabel('Coherence')
    plt.ylabel('Frequency')
    plt.title("Coherence for each Topic (50 per Model)")
    plt.legend(loc='upper right', shadow=True)
    plt.savefig(os.path.join(results_dir, "coherence.pdf"))
    plt.close()
    plt.hist(prodlda_result['coherence'], facecolor='blue', alpha=0.5)
    plt.axvline(x=np.mean(prodlda_result['coherence']), color="r")
    plt.xlabel('Coherence')
    plt.ylabel('Frequency')
    plt.title("prodLDA Coherence")
    plt.savefig(os.path.join(results_dir, "prodlda_coherence.pdf"))
    plt.close()
    plt.hist(nvlda_result['coherence'], facecolor='blue', alpha=0.5)
    plt.axvline(x=np.mean(nvlda_result['coherence']), color="r")
    plt.xlabel('Coherence')
    plt.ylabel('Frequency')
    plt.title("LDA-VAE Coherence")
    plt.savefig(os.path.join(results_dir, "ldavae_coherence.pdf"))
    plt.close()
    plt.hist(dmfvi_result['coher'], facecolor='blue', alpha=0.5)
    plt.axvline(x=np.mean(dmfvi_result['coher']), color="r")
    plt.xlabel('Coherence')
    plt.ylabel('Frequency')
    plt.title("LDA-DMFVI Coherence")
    plt.savefig(os.path.join(results_dir, "dmfvi_coherence.pdf"))
    plt.close()

    # Make t-sne plots
    #n_samples = 300
    train_txt_dir = "data/20news_clean/train_txt_dir"
    if not os.path.isdir(train_txt_dir):
        data.rawdata_to_text(rawdata_tr, vocab, train_txt_dir)
    txt_fns = filter(lambda x: x.endswith('.txt'), os.listdir(train_txt_dir))
    txt_fns = list(map(lambda x: os.path.join(train_txt_dir, x), txt_fns))
    cvz = dmfvi_result["vectorizer"].fit_transform(txt_fns)
    #X_topics = dmfvi['lda'].fit_transform(cvz)
    n_components = 2
    (fig, subplots) = plt.subplots(1, 5, figsize=(15, 8))
    perplexities = [5, 30, 50, 100]



        #mod = dmfvi_result['lda']
        #print(mod)
        #out = mod.transform()
        #print(out)



    # Make histogram of perplexity scores (1 per model)


if __name__ == "__main__":
   main()
