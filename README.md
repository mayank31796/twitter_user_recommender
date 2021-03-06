# Topic Modeling based Twiter User recomendation system
The project is a user recommendation model for twitter users.

# Introduction
Online Social Networking platforms are a great place to make new connections and meet people of similar tastes. The growing popularity of such social networks has raised new areas of applications for recommender systems. The main aim of such recommender systems is to identify and suggest new friends for a user such that they have similar topics of interest.

In this project we try to build such a recommender system which profiles the user and generates the topics of interest of the user, Then compare his topics of interest with that of other users in the network whom the user is not friends with. Based on the similarity of the topic distribution, users are ranked and recommended to the main user.

# Data
Data for building the recommendation system is generated by extracting tweets using Twitter API's search and streaming API.
The data collected is of two phases:
## 1. Extraction of prospective twitter users
The prospective users are the ones who are are in the main users ego network whom the main user has no direct edge with. These users are first extracted.

## 2. Extraction of tweets
Tweets of the main user and the prospective users are all extracted and stores as key-value pairs. These are later used to generate topic models for each user.

# Method
## 1. Generating Topic Model 
Tweets extracted are cleaned and tokenized. The tokens are converted to Gensim dictonaries which are then used to generate bag of words corpus.
Once we have the corpus we can formulate the The bow vectors are used to create a TF-IDF vectors. Then we use LDA Topic Modeling method to create a topic model for each user. 

Latent Dirichlet Allocation (LDA) is an unsupervised generative model that assigns topic distributions to documents. LDA assumes that each document can be explained by the distribution of topics and each topic can be explained by the distribution of words. Once we specify the number of topics it assumes that the document is explained only by those number of topics and generates two variables:
➢ A distribution over topics for each document
➢ A distribution over words for each topic

To select the number of topic we use coherence measure to determine the ideal number of topics which can explain the almost all the users tweets. 
  - To do this we make use of coherence measures. Topic coherence measures each topic by scoring it based on calculating the degree of    semantic similarity between words in the topic. It is often considered as a metric to evaluate the quality of a topic. 
  - We calcualte coherence measure for number of topics ranging from 5 to 200 and plot the coherence values.
  - Then select the number of topics which has the best coherence to number of topics ratio.
 Then create the topic models for each user.
 
 ## 2. Ranking users with the most similar topics of interest
Once we have generated the topics of interest for the main user we use Jensen-Shanon divergence measure with the topic models of other users. Cosine similarity measure works well for two vectors but LDA generates distributions. Thus using Jensen-Shannon we can determine which documents are statistically “closer” (and therefore more similar), by comparing the divergence of their distributions. 

Based on the divergence score we rank users with least divergence (most similar) and recommend them.

# Validation
User recommendation methods can't be evaluated normally. Instead for evaluation we first remove n number of user's friends. Then use our model to predict the same number of n users. Now we see how many of our initially removed users are now predicted for evaluation.
