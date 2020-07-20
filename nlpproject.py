# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:57:12 2020

@author: VISHESH
"""
 
import random
import pandas as pd
from collections import Counter
from nltk import word_tokenize , WordNetLemmatizer
from nltk . corpus import stopwords
from nltk import NaiveBayesClassifier , classify
#store all the stopwords onto a list
stoplist = stopwords . words ("english")
#function for lemmatizing the non stopwords in the headlines
def preprocess ( sentence ) :
 lemmatizer = WordNetLemmatizer ()
 return [ lemmatizer . lemmatize ( word . lower () ) for word in word_tokenize (sentence ) ]
#function returning tuples with the word and true is the word is not a stopword
def get_features (text , setting ) :
#if setting =bow then we are giving our own headline to check if the headlne is for  clickbait or info"    
 if setting == "bow ":
  return { word : count for word , count in Counter (preprocess ( text ) ) . items () if not word in stoplist }
 else :
    return { word : True for word in preprocess ( text ) if not word in stoplist }
def train ( features , samples_proportion ) :
 train_size = int(len( features ) * samples_proportion )
 # initialise the training and test sets
 train_set , test_set = features [: train_size ] , features [train_size :]
 print ("Training set size = " + str(len( train_set ) ))
 print ("Test set size = " + str(len( test_set ) ))
 # train the classifier
 classifier = NaiveBayesClassifier . train ( train_set )
 return train_set , test_set , classifier
def evaluate ( train_set , test_set , classifier ) :
    # show how the classifier performs on the training and test sets
    print ("Accuracy on the training set = " + str( classify .accuracy ( classifier , train_set ) ) )
    print ("Accuracy of the test set = " + str( classify .accuracy ( classifier , test_set ) ) )
    # show which words are most informative for the classifier
    classifier . show_most_informative_features (20)
#save the dataset into a dataframe    
data=pd.read_csv("clickbait_data.csv")
#make list of tuples of clickbait and info headlines
hls=[(hl,"clickbait") for hl in data.query("clickbait == 1").headline.tolist()  ]
hls+=[(hl,"Info") for hl in data.query("clickbait == 0").headline.tolist()  ]
#shuffle randomly clickbait and info headlines
random.shuffle(hls)
print ('Corpus size = ' + str(len( hls ) ) )
#extract the features
features= [( get_features (hl , '') , label ) for (hl ,label ) in hls ]
print ('Collected '  + str(len( features ) ) + ' feature sets')
#train the classifier
train_set , test_set , classifier = train ( features , 0.8)
#evaluate performance
evaluate ( train_set , test_set , classifier )
print ( classifier . classify ( get_features (" Here ’s what happens when three senators disagree on gun policy", "bow ") ) )
