This is my implementation of sentiment analysis based on tweets considering _The Social Dilemma_ movie. The dataset can be downloaded from: https://www.kaggle.com/datasets/kaushiksuresh147/the-social-dilemma-tweets

There are 3 classes of tweets: positive, neutral and negative opinions about the movie. The goal is to create a machine learning model, which would classify tweets whether they are positive, neutral or negative.

I've tested 2 approaches of NLP transformation. The first one is the concept of _Bag Of Words_ (BOW), where the order of the words in the sentence doesn't matter. In this approach I've created my own vocabulary based on the words in tweets in the dataset. The number of dimensions is the size of vocabulary and for every example (tweet) the more a particular word occurs, the bigger relative value there is in the corresponding dimension. The second approach is based on transforming whole tweets as sentences IN PROGRESS
