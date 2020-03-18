# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""

# add whatever imports you need.
# be sure to also add to requirements.txt so I can install them.
import click
import json
import glob
import pickle
import sys
from .tweepy_wrapper import *
from .preprocessing import*
from .train_LDA import *
from .train_TFID import *
from .predict import*
import numpy as np
import os
import pandas as pd
import re
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
from gensim.models.coherencemodel import CoherenceModel
from collections import Counter
from gensim.corpora.dictionary import Dictionary
import en_core_web_sm
import time
nlp = en_core_web_sm.load()
#import import_ipynb
import dict_df_helper as dfo
from scipy.stats import entropy
import matplotlib.pyplot as plt
from textblob import TextBlob
nltk_stopwords = stopwords.words("english")+["rt", "via","-»","--»","--","---","-->","<--","->","<-","«--","«","«-","»","«»"]
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from . import credentials_path, clf_path
from .mytwitter import Twitter

@click.group()
def main(args=None):
    """Console script for osna."""
    return 0

@main.command('collect')
@click.argument('directory', type=click.Path(exists=True))
def collect(directory):
    """
    Collect data and store in given directory.

    This should collect any data needed to train and evaluate your approach.
    This may be a long-running job (e.g., maybe you run this command and let it go for a week).
    """
    '''
    twitter = Twitter(credentials_path)
    limit = 100
    fname = directory + os.path.sep + 'data.json'
    outf = open(fname, 'wt')
    ncollected = 0
    for tw in twitter.request('statuses\filter', {'track': 'chicago'}):
        print(tw['text'])
        outf.write(json.dumps(tw, ensure_ascii=False) + '\n')
        ncollected += 1
        if ncollected >= limit:
            break
    print('collected %d tweets, stored to %s' % (ncollected, fname))
    '''


    #path = input("Enter path to twitter credentials as a .cfg file: ")
    auth, api = get_twitter()

    main_user = input("Enter the screen name of the main user for whom you want predictions: ")

    print("\n\n****Warning this step involves a lot of data being pulled from Twitter and may take up to days for completion******\n\n")

    print("Getting friends of ",main_user)
    friends = get_friends(api, main_user)
    print("Getting all tweets of ",main_user)
    main_tweets = main_tweets(api, main_user)

    print("\nThis step may take several days\n")
    print("\nGetting second degree users\n")

    deg2_count = input("Enter count of deg2 users to extract per friend: ")
    deg2_users = get_deg2_users(api, friends , deg2_count)
    print("\nFinished getting deg2 users\n")

    print("Removing friends from deg2 users\n")
    a = set(friends)
    b = set(deg2_users)
    not_following = list(set(a - b))
    friends = list(a)
    deg2_users = list(b)

    print("\n\nNow gettting 200 tweets each for all the deg2 users\n\n")
    u_tweets, protected, not_found= fetch_tweets(api, not_following)
    print("\n\nSuccessfully finished collecting all data!!\n\n")

    print("\n\nBacking up the data\n\n")
    print("Saving main user tweets")
    save_json(main_tweets ,str(directory+'\\data\\'),'main_tweets')
    save_pkl(main_tweets ,str(directory+'\\data\\'),'main_tweets')
    print("Success\n\n")

    print("Saving main user friends")
    save_pkl(friends ,str(directory+'\\data\\'),'friends')
    print("Success\n\n")

    print("Saving deg2 users")
    save_pkl(deg2_users ,str(directory+'\\data\\'),'deg2_users')
    print("Success\n\n")

    print("Saving deg2 users not being followed")
    save_pkl(not_following ,str(directory+'\\data\\'),'not_following')
    print("Success\n\n")    

    print("Saving deg2 user tweets")
    save_json(u_tweets ,str(directory+'\\data\\'),'u_tweets')
    save_pkl(u_tweets ,str(directory+'\\data\\'),'u_tweets')

    print("Success\n\n\n")

    print("Proceeding with data cleaning......\n\n")

    clean_main = get_tokens(main_tweets)
    clean_utweets = get_tokens(u_tweets)

    print("Saving cleaned main user tweets")
    save_pkl(clean_main ,str(directory+'\\data\\'),'clean_main')
    print("Success\n\n")   

    print("Saving cleaned tweets of deg2 users not being followed")
    save_pkl(clean_utweets ,'directory\\data\\','clean_utweets')
    print("Success\n\n")   

    print("Processed all the data successfully\n\n\nReturning....\n")
    return
    outf.close()

@main.command('evaluate')
def evaluate():
    """
    Report accuracy and other metrics of your approach.
    For example, compare classification accuracy for different
    methods.
    """
    pass
    test_data = pd.read_pickle(str(directory)+'\\data\\clean_main.pkl')
    train_data = pd.read_pickle(str(directory)+'\\data\\clean_utweets.pkl')
    opt = input("Which model do you want to evaluate?\n\nEnter 1 for LDA\nEnter 2 for TFIDF\n\n")
    if(opt == 1):
        print("Now we evaluate LDA\n\n")
        print("Loading the model")
        results = predict_lda(directory,test_data,train_data)
        print(results)
        return
    elif (opt == 2):
        print("Now we evaluate TFIDF\n\n")
        predict_tfid(directory,test_data)
        return
    else:
        print("WARNING\t invalid entry\n")
        train(directory)
    

@main.command('network')
def network():
    """
    Perform the network analysis component of your project.
    E.g., compute network statistics, perform clustering
    or link prediction, etc.
    """
    pass

@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
    """
    Read all data and print statistics.
    E.g., how many messages\users, time range, number of terms\tokens, etc.
    """
    print('reading from %s' % directory)
    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.


@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier on all of your labeled data and save it for later
    use in the web app. You should use the pickle library to read\write
    Python objects to files. You should also reference the `clf_path`
    variable, defined in __init__.py, to locate the file.
    """
    print('reading from %s' % directory)
    train_data = pd.read_pickle('directory\\data\\clean_utweets.pkl')
    opt = input("Which model do you want to train?\n\nEnter 1 for LDA\nEnter 2 for TFIDF\n\n")
    if(opt == 1):
        print("Now we train LDA\n\n")
        topics_opt = input("Enter y if you know the number of topic you want to train for or enter n to determine using coherence analysis\n")
        if topics_opt == 'y':
            num_topics = input("Enter the number of topics you want: ")
            dictionary,corpus,lda = train_lda(train_data,num_topics)
        elif topics_opt == 'n':
            print("Performing coherence analysis and producing plot for you\n\n")
            print("Look at the plot and see the number of topics after which the plot flatlines\n\n")
            coherence_analysis(train_data)
            print("Now choose the number of topics as you determined earlier in the option\n")
            train(directory)
        else:
            print("WARNING\t invalid entry\n")
            train(directory)

        print("Saving out LDA and corpus")
        lda_save(str(directory+'\\data\\models\\LDA\\'),lda,corpus)
    elif (opt == 2):
        print("Now we train TFIDF\n\n")
        tfidf_model, tfid_corpus = train_tfid(train_data)
        print("Saving TFIDF model\n")
        tfid_save(str(directory+'\\data\\models\\TFIDF\\'),tfidf_model, tfid_corpus )

    else:
        print("WARNING\t invalid entry\n")
        train(directory)
    return

@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True, default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=9999, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    """
    Launch a web app for your project demo.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
    


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover