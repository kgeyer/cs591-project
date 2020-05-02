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

'''-----------Data--------------'''
dataset_tr = 'data/20news_clean/train.txt.npy'
rawdata_tr = np.load(dataset_tr, allow_pickle=True, encoding="bytes")
dataset_te = 'data/20news_clean/test.txt.npy'
rawdata_te = np.load(dataset_te, allow_pickle=True, encoding="bytes")
vocab = 'data/20news_clean/vocab.pkl'
vocab = pickle.load(open(vocab,'rb'))
vocab_size=len(vocab)
print("vocab size: ", vocab_size)
#--------------convert to one-hot representation------------------
data_tr = data.convert_to_onehot(rawdata_tr, vocab_size)
data_te = data.convert_to_onehot(rawdata_te, vocab_size)
#data_tr = np.array([onehot(doc.astype('int'),vocab_size) for doc in rawdata_tr if np.sum(doc)!=0])
#data_te = np.array([onehot(doc.astype('int'),vocab_size) for doc in rawdata_te if np.sum(doc)!=0])
#--------------print the data dimentions--------------------------
print ('Data Loaded')
print ('Dim Training Data',data_tr.shape)
print ('Dim Test Data',data_te.shape)
total_docs = data_te.shape[0] + data_tr.shape[0]
total_words = np.sum(data_te) + np.sum(data_tr)
'''-----------------------------'''

'''--------------Global Params---------------'''
n_samples_tr = data_tr.shape[0]
n_samples_te = data_te.shape[0]
#docs_tr = data_tr
#docs_te = data_te
batch_size=200
learning_rate=0.002
network_architecture = \
    dict(n_hidden_recog_1=100, # 1st layer encoder neurons
         n_hidden_recog_2=100, # 2nd layer encoder neurons
         n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
         n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
         n_z=50)  # dimensionality of latent space
'''-----------------------------'''

'''--------------Network Architecture and settings---------------'''
def make_network(layer1=100,layer2=100,num_topics=50,bs=200,eta=0.002):
    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1, # 1st layer encoder neurons
             n_hidden_recog_2=layer2, # 2nd layer encoder neurons
             n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
             n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
             n_z=num_topics)  # dimensionality of latent space
    batch_size=bs
    learning_rate=eta
    return network_architecture,batch_size,learning_rate
'''-----------------------------'''

'''--------------Methods--------------'''
def create_minibatch(data):
    rng = np.random.RandomState(10)
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]

def train(network_architecture, minibatches, type='prodlda',learning_rate=0.001,batch_size=200, training_epochs=100, display_step=5):
    tf.reset_default_graph()
    vae=''
    if type=='prodlda':
        vae = prodlda.VAE(network_architecture,learning_rate=learning_rate,batch_size=batch_size)
    elif type=='nvlda':
        vae = nvlda.VAE(network_architecture,learning_rate=learning_rate,batch_size=batch_size)
    emb=0
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples_tr / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = next(minibatches)
            # Fit training using batch data
            cost,emb = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples_tr * batch_size
            if np.isnan(avg_cost):
                print (epoch,i,np.sum(batch_xs,1).astype(np.int),batch_xs.shape)
                print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
    return vae,emb
'''-----------------------------'''

def main():
    # Format data
    print("Format data")
    minibatches = create_minibatch(data_tr.astype('float32'))
    t = 50          # number of topics
    e = 100         # number of epochs

    # Calculate data for coherence scores
    #--------------total word count--------------------------
    doc_count_fn = 'data/20news_clean/doc_count.npy'
    if os.path.isfile(doc_count_fn):
        print('Loading doc count vector')
        doc_count = np.load(doc_count_fn)
    else:
        doc_count = data.count_doc_with_word(data_tr, data_te)
        np.save(doc_count_fn, doc_count)
    #--------------word co-occurance in document count--------------------------
    coword_count_fn = 'data/20news_clean/coword_count.npy'
    if os.path.isfile(coword_count_fn):
        print('Loading word co-occurance count matrix')
        np.load(coword_count_fn)
    else:
        coword_count = data.count_word_cooccurance(data_tr, data_te)
        np.save(coword_count_fn, coword_count)

    # Reference
    # "hpnm:f:s:t:b:r:,e:",["default=","model=","layer1=","layer2=","num_topics=","batch_size=","learning_rate=","training_epochs"])

    # Get results for prodLDA
    result_fn1 = "results/prodlda_results.pkl"
    if not os.path.isfile(result_fn1):
        print("RUNNING prodLDA")
        m='prodlda'
        f=100
        s=100
        b=200
        r=0.002
        network_architecture1,batch_size1,learning_rate1 = make_network(f,s,t,b,r)
        vae1,emb1 = train(network_architecture1, minibatches,m, training_epochs=e,batch_size=batch_size1,learning_rate=learning_rate1)
        perp1 = results.calcPerp(vae1, data_te)
        kw1 = results.print_top_words(emb1, vocab)
        coher1 = results.calculate_lda_coherence(doc_count_fn, coword_count_fn, vocab, kw1, total_docs)
        prodlda_result = {"perplexity": perp1, "emb": emb1, 'topic_keywords': kw1, 'coherence': coher1}
        print(prodlda_result)
        outfile = open(result_fn1, 'wb')
        pickle.dump(prodlda_result, outfile)
        outfile.close()
    else:
        infile = open(result_fn1, 'rb')
        prodlda_result = pickle.load(infile)
        infile.close()
    print("RESULTS FOR PRODLDA")
    #print(prodlda_result["topic_keywords"])

    # Get results for NVLDA
    result_fn2 = "results/nvlda_results.pkl"
    if not os.path.isfile(result_fn2):
        print("RUNNING NVLDA")
        m='nvlda'
        f=100
        s=100
        b=200
        r=0.01
        network_architecture2,batch_size2,learning_rate2=make_network(f,s,t,b,r)
        vae2,emb2 = train(network_architecture2, minibatches,m, training_epochs=e,batch_size=batch_size2,learning_rate=learning_rate2)
        perp2 = results.calcPerp(vae2, data_te)
        kw2 = results.print_top_words(emb2, vocab)
        coher2 = results.calculate_lda_coherence(doc_count_fn, coword_count_fn, vocab, kw2, total_docs)
        nvlda_result = {"perplexity": perp2, "emb": emb2, 'topic_keywords': kw2, 'coherence': coher2}
        print(nvlda_result)
        outfile = open(result_fn2, 'wb')
        pickle.dump(nvlda_result, outfile)
        outfile.close()
    else:
        infile = open(result_fn2, 'rb')
        nvlda_result = pickle.load(infile)
        infile.close()
    print("RESULTS FOR NVLDA")
    #print(nvlda_result["topic_keywords"])

    # Get results for DMFVI - use implementation from sklearn
    # Convert data to sklearn Format
    train_txt_dir = "data/20news_clean/train_txt_dir"
    if not os.path.isdir(train_txt_dir):
        data.rawdata_to_text(rawdata_tr, vocab, train_txt_dir)
    txt_fns = filter(lambda x: x.endswith('.txt'), os.listdir(train_txt_dir))
    txt_fns = list(map(lambda x: os.path.join(train_txt_dir, x), txt_fns))
    test_txt_dir = "data/20news_clean/test_txt_dir"
    if not os.path.isdir(test_txt_dir):
        data.rawdata_to_text(rawdata_te, vocab, test_txt_dir)
    result_fn3 = "results/dmfvi_results.pkl"
    if not os.path.isfile(result_fn3):
        print("RUNNING DMFVI")
        # format data for LatentDirichletAllocation
        print("Extracting tf features for LDA")
        tf_vectorizer = CountVectorizer(input='filename', max_features=vocab_size, stop_words='english')
        tf = tf_vectorizer.fit_transform(txt_fns)
        print("Fitting DMFVI model")
        lda = LatentDirichletAllocation(n_components=t, learning_method='batch')
        lda.fit(tf)
        perp3 = lda.perplexity(tf)
        topic_keywords3 = results.show_topics(tf_vectorizer, lda) #list of 50
        coher3 = results.calculate_lda_coherence(doc_count_fn, coword_count_fn, vocab, topic_keywords3, total_docs)
        print(coher3)
        dmfvi_result = {"perplexity": perp3, "lda": lda, "vectorizer": tf_vectorizer, 'topic_keywords': topic_keywords3, 'coher': coher3}
        print(dmfvi_result)
        outfile = open(result_fn3, 'wb')
        pickle.dump(dmfvi_result, outfile)
        outfile.close()
    else:
        infile = open(result_fn3, 'rb')
        dmfvi_result = pickle.load(infile)
        infile.close()
    print("RESULTS FOR DMFLDA")
    print(dmfvi_result["topic_keywords"])


if __name__ == "__main__":
   main()
