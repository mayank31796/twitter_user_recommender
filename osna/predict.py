import pickle
import nltk
import pandas as pd
import numpy as np
import tweepy
import re
import json
from tweepy.models import Status
import pandas as pd
import string
from nltk.corpus import stopwords
import spacy
import gensim 
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity
from gensim.corpora.dictionary import Dictionary
import time
from scipy.spatial.distance import jensenshannon
import en_core_web_sm
nlp = en_core_web_sm.load()
#import import_ipynb
import dict_df_helper as dfo
import matplotlib.pyplot as plt
nltk_stopwords = stopwords.words("english")+["rt", "via","-»","--»","--","---","-->","<--","->","<-","«--","«","«-","»","«»"]


def get_lda_scores(directory,data):
    #path = input("Enter path to LDA model: ")
    lda = LdaModel.load(str(directory)+"\\data\\models\\LDA\\lda_model")
    corpus = MmCorpus(directory)+"\\data\\models\\LDA\\lda_corpus.mm")
    new_dictionary = Dictionary(data['tokens'])
    new_corpus = [new_dictionary.doc2bow(doc) for doc in data['tokens']]
    new_corpus = (lda[new_corpus][0])
    new_doc_distribution= []
    for i in range(len(new_corpus)):
        new_doc_distribution.append(new_corpus[i][1])
    new_doc_distribution = np.array(new_doc_distribution)
    
    doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
    doc_topic_dist.shape
    
    scores = []
    for i in range(len(doc_topic_dist)):
        scores.append(jensenshannon(new_doc_distribution,doc_topic_dist[i]))
    #data['scores'] = scores
    #similar = data.sort_values(by='scores', ascending=False)
    return scores

def predict_lda(directory,new_data,original_data):
    scores = get_lda_scores(directory,new_data)
    original_data['scores'] = scores
    similar = original_data['scores'].sort_values(ascending=False)
    return similar
    
def predict_tfid(directory,pred_data,data):
    path = input("Enter path to LDA model: ")
    tfid = gensim.models.TfidfModel.load(str(directory)+"\\data\\models\\TFIDF\\tfid_model")
    corpus = MmCorpus(str(directory)+"\\data\\models\\TFIDF\\tfid_corpus.mm")
    tfid_corpus = tfid[corpus]
    new_dictionary = Dictionary(data['tokens'])
    new_corpus = [new_dictionary.doc2bow(doc) for doc in data['tokens']]
    index_sparse = SparseMatrixSimilarity(tfid_corpus, num_features=corpus.num_terms)
    index_sparse.num_best = 500
    idx =(index_sparse[new_corpus])
    print("Most Similar users are as follows: ")
    print("Name\t\t\tscore ")
    m=1
    for i in idx[0]:
        print("{}. {}     {}".format(m,data.iloc[i[0]]['handles'],i[1]))
        m+=1
    return


    