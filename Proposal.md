## Overview

Recommendation of friends on twitter using clustering of interests analysis of user.



In this project we aim to design a system to suggest user with most similar members who are within n hops from the user.



In this we mine for user's personality traits using his tweets/retweets/likes data.

Once we have the user's personality information we check for traits for the members who are under n hops from the user.



Then cluster based on each personality trait and rank users with highest likelihood of link using link prediction methods. Once we have ranked users for each trait we can aggregate the values and obtain top x users who are similar to user in all aspects and recommend them.

## Data

To acquire data we want to make use of Twitter streaming API and pick a user and graph of users within n hops from the user.



Since we need to acquire tweets/retweets and other related data of all users it may be a large task and especially with twitter rate limits can take long time to accumulate.



As an alternative an twitter circles data set can be used. https://snap.stanford.edu/data/egonets-Twitter.html



Another problem we can anticipate is the word segmentation of tweets of users. Since words in twitter are not accurately spelled and are not appropriately separated using white spaces it becomes difficult to segment them.

## Method

For predict link prediction we aim to use and compare the methods discussed in class to see which method provides accurate results.



For user interests extraction we plan on using tf id vectorization scores to obtain most prominent topics of user's interest.

## Related Work

https://ieeexplore-ieee-org.ezproxy.gl.iit.edu/document/8405708

https://ieeexplore-ieee-org.ezproxy.gl.iit.edu/document/7489465

https://ieeexplore.ieee.org/document/7823266

https://ieeexplore.ieee.org/document/7403692

https://ieeexplore-ieee-org.ezproxy.gl.iit.edu/document/6779068

https://ieeexplore-ieee-org.ezproxy.gl.iit.edu/document/8673242

https://ieeexplore-ieee-org.ezproxy.gl.iit.edu/document/7474353

## Evaluation

For evaluating the results of the implementation we could follow the evaluation method discussed in class where we remove edges from the users actual graph G to obtain G' and predict users in G' and compare how many predicted users are actually in G. 



In some papers other evaluation metrics such as Precision, F1 and AUC characteristics have been used.

