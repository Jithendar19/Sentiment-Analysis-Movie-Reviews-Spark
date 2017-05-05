# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 17:35:47 2017

@author: Jeethu
"""
#importing modules
import os, sys
import numpy as np
import collections
import re, string
os.environ['SPARK_HOME'] = "C:\Spark"
sys.path.append("C:\Spark\python")
sys.path.append("C:\Spark\python\lib")

from pyspark import SparkContext

#Creating Spark context
sc = SparkContext()
   
#Reading the positive training dataset from IMDB
train_pos = sc.textFile("C:\Users\Alpha\Desktop\IMDB\\train\pos_sample\*")

#Method to print the contents of the RDD
def p(doc):
    for x in doc.collect():
        print x

#Function removes punctuation and converts to lower case
def removePunctuation(text):
    output = re.sub('[%s]' % re.escape(string.punctuation), "", text)
    output = output.lower().strip()
    return output

#Converting to lowerCase and removing punctuation from the reviews
train_pos_punc = train_pos.map(removePunctuation)

#Viewing the training dataset after removing punctuation
#p(train_pos_punc)

#Counting the frequency of each word
train_pos_counts =  train_pos_punc.flatMap(lambda x: x.split()).map(lambda x : (x,1)).reduceByKey(lambda x,y : x+y)

#Caching the RDD
train_pos_counts.cache()

#Viewing the count of each word
#p(train_pos_counts)

#Counting the number of words in the dataset
count_pos_total = train_pos_counts.map(lambda (x, y): y).reduce(lambda x, y: x + y)

print "No. of words in the dataset :" ,count_pos_total

#Counting the number of unique words
unique_count_pos = train_pos_counts.count()

print "No. of unique words in the dataset", unique_count_pos

#Adding total count and unique count to the tuple
def add_counts(tuple, count_class, unique_count_pos_class):
    w, count = tuple
    return (w, count, count_class, unique_count_pos_class)

#Calculating the probability of each word using "Laplace smoothing"
#Refer : Link : https://gist.github.com/ttezel/4138642

prob_words_pos = train_pos_counts.map(lambda x: add_counts(x, count_pos_total, unique_count_pos)) \
    .map(lambda (w, count, count_pos_total, unique_count_pos): (w, float((count + 1))/(count_pos_total + unique_count_pos + 1))) \
    .collectAsMap()

#Viewing the probability of each word in positive dataset    
print prob_words_pos     

######Train negative dataset#####

#Reading the positive training dataset from IMDB
train_neg = sc.textFile("C:\Users\Alpha\Desktop\IMDB\\train\\neg_sample\*")

#Removing punctuation from the reviews
train_neg_punc = train_neg.map(removePunctuation)

#Counting the frequency of each word
train_neg_counts =  train_neg_punc.flatMap(lambda x: x.split()).map(lambda x : (x,1)).reduceByKey(lambda x,y : x+y)

#Caching the RDD
train_neg_counts.cache()

#Counting the number of words in the dataset
count_neg_total = train_neg_counts.map(lambda (x, y): y).reduce(lambda x, y: x + y)

print "No. of words in the dataset :" ,count_neg_total

#Counting the number of unique words
unique_count_neg = train_neg_counts.count()

#Calculating the probability of each word using "Laplace smoothing"

prob_words_neg = train_neg_counts.map(lambda x: add_counts(x, count_neg_total, unique_count_pos)) \
    .map(lambda (w, count, count_neg_total, unique_count_neg): (w, float((count + 1))/(count_pos_total + unique_count_pos + 1))) \
    .collectAsMap()
    
#Viewing the probability of each word in negative dataset    
print prob_words_neg    



 






