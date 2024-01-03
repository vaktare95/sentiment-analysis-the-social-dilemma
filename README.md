This code is similar to my Kaggle notebook: https://www.kaggle.com/code/vaktare95/accuracy-90-1-f1-score-88-2

This is my implementation of sentiment analysis based on tweets considering _The Social Dilemma_ movie. The dataset can be downloaded from: https://www.kaggle.com/datasets/kaushiksuresh147/the-social-dilemma-tweets

There are 3 classes of tweets: positive, neutral and negative opinions about the movie. The goal is to create a machine learning model, which would classify tweets whether they are positive, neutral or negative.

I've tested 2 approaches of NLP transformation. The first one is the concept of _Bag Of Words_ (BOW), where the order of the words in the sentence doesn't matter. In this approach I've created my own vocabulary based on the words in tweets in the dataset. The number of dimensions is the size of vocabulary and for every example (tweet) the more a particular word occurs, the bigger relative value there is in the corresponding dimension. The second approach is based on transforming whole tweets with sentence transformer (ST) from BERT all-mpnet-base-v2 to vectors with 768 dimensions: https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/

Multi Layer Perceptron with data transformed by BOW transformer turned out to have the best performance in 5-fold cross validation based on ~5k examples, as it can be seen in the `boxplots` folder. Finally, using all the examples from the dataset (~20k) with BOW transformation, I've tested Multi Layer Perceptron with train/test ratio equal to 80%/20%. **The accuracy of the model was ~90.0% and the F1-score was ~88.1%.**

The code is adjusted for python3.10.

To run validations and test on your own, just run:
```
python3 -m pip install -r requirements.txt
```
and then:
```
python3 test_sentiment_analysis.py
```
