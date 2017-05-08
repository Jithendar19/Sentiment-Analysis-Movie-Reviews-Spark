# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 17:35:47 2017

@author: Jeethu
"""
#importing modules
from __future__ import division
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

print "No. of words in the positive training dataset :" ,count_pos_total

#Counting the number of unique words
unique_count_pos = train_pos_counts.count()

print "No. of unique words in the positive training dataset", unique_count_pos

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
#print prob_words_pos     

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

print "No. of words in the negative training dataset :" ,count_neg_total

#Counting the number of unique words
unique_count_neg = train_neg_counts.count()

print "No. of unique words in the negative training dataset", unique_count_neg

#Calculating the probability of each word using "Laplace smoothing"

prob_words_neg = train_neg_counts.map(lambda x: add_counts(x, count_neg_total, unique_count_pos)) \
    .map(lambda (w, count, count_neg_total, unique_count_neg): (w, float((count + 1))/(count_pos_total + unique_count_pos + 1))) \
    .collectAsMap()
    
#Viewing the probability of each word in negative dataset    
#print prob_words_neg    


#Calculating the probability of each class #POS and #NEG

train_pos_count = train_pos.count()
train_neg_count = train_neg.count()

PROB_POS_CLASS = float(train_pos_count/(train_pos_count + train_neg_count))

PROB_NEG_CLASS = float(train_neg_count/(train_pos_count + train_neg_count))

#Viewing the probabilities of class
print PROB_POS_CLASS
print PROB_NEG_CLASS

######### Test POS ########

#Reading the positive test dataset from IMDB
test_pos = sc.textFile("C:\Users\Alpha\Desktop\IMDB\\test\pos_sample\*")

#Removing punctuations and converting to lower case
test_pos_punc = test_pos.map(removePunctuation)

#Counting the frequency of each word using counter and appending the sentiment to each document 
test_pos_counts = test_pos_punc.map(lambda x: x.split()) \
    .map(lambda x: collections.Counter(x))\
    .map(lambda x: x.items()).map(lambda x: ("POS", x))

#p(test_pos_counts) 

### Combinging Job ####

#Finds the probability of each word in the test document from the model (prob_words_pos, prob_words_neg).
#If the probability is not found in the model, calculate the probability based on the text size and no. of unique words
def combineWithModel(test_pos_counts, prob_words_pos, prob_words_neg, unique_count_pos, unique_count_neg, count_pos_total, count_neg_total):
    #Returns (word, count, prob_pos, prob_neg)
    senti = test_pos_counts[0]
    word_list = test_pos_counts[1]
    result_list = []
    for item in word_list:
        word, counts = item
        if word in prob_words_pos:
            prob_words_pos_n = prob_words_pos[word]
        else:
            prob_words_pos_n = 1.0/(count_pos_total + unique_count_pos + 1)
        if word in prob_words_neg:
            prob_words_neg_n = prob_words_neg[word]
        else:
            prob_words_neg_n = 1.0/(count_neg_total + unique_count_neg + 1)
        result_list.append((word, counts, prob_words_pos_n, prob_words_neg_n))
    result = (senti, result_list)
    return result


#Combining the model with the test data
classify_data_pos = test_pos_counts.map(lambda x: combineWithModel(x, prob_words_pos, prob_words_neg, unique_count_pos, unique_count_neg,
                                                count_pos_total, count_neg_total))

#Multipying the probability of each word with it's frequency 

def multiply_prob(classify_data):
    #classify_data : (senti, (word, counts, prob_words_pos_n, prob_words_neg_n))
    #returns (senti, (log_pos_words_pos_n, log_pos_words_neg_n))
    
    result_list = []
    for item in classify_data[1]:
        word, counts, prob_words_pos_n, prob_words_neg_n = item
        log_pos_words_pos_n = np.log(prob_words_pos_n) * counts
        log_pos_words_neg_n = np.log(prob_words_neg_n) * counts
        result_list.append((log_pos_words_pos_n, log_pos_words_neg_n))
    result = (classify_data[0], result_list)
    return result

##### CLASSIFY JOB #####
#Add all the probabilities of each class and decide the classification
def classifier(data, PROB_POS_CLASS, PROB_NEG_CLASS):
    #data : (senti, (log_pos_words_pos_n, log_pos_words_neg_n))
    #returns (senti, classificaion)
    total_pos = 0
    total_neg = 0
    for item in data[1]:
        log_pos_words_pos_n, log_pos_words_neg_n = item
        total_pos += log_pos_words_pos_n
        total_neg += log_pos_words_neg_n
    #total_pos = total_pos + PROB_POS_CLASS
    #total_neg = total_neg + PROB_NEG_CLASS
    if total_pos > total_neg:
        classificaion = "POS" #1.0 indicates positive sentiment
    else:
        classificaion = "NEG" #0.0 indicates negative sentiment
    result = (data[0], classificaion)
    return result

#Multiplying probabilities and classifying each test review
classification_result_pos = classify_data_pos.map(lambda x: multiply_prob(x))\
    .map(lambda x: classifier(x, PROB_POS_CLASS, PROB_NEG_CLASS))

#Viewing the result
p(classification_result_pos)

#Count the total no. of test positive reviews
total_pos = classification_result_pos.count()
#Count number of positive test reviews predicted wrong
wrong_pos = classification_result_pos.filter(lambda (x, y): (x != y)).count()

#Viewing the result
print "Total number of test positive reviews:" , total_pos
print "Number of positive test reviews predicted wrong:", wrong_pos

######### Test NEG ########

#Reading the negative test dataset from IMDB
test_neg = sc.textFile("C:\Users\Alpha\Desktop\IMDB\\test\neg_sample\*")

#Removing punctuations and converting to lower case
test_neg_punc = test_pos.map(removePunctuation)

#Counting the frequency of each word using counter and appending the sentiment to each document 
test_neg_counts = test_neg_punc.map(lambda x: x.split()) \
    .map(lambda x: collections.Counter(x))\
    .map(lambda x: x.items()).map(lambda x: ("NEG", x))
    
#Combining the model with the test data
classify_data_neg = test_neg_counts.map(lambda x: combineWithModel(x, prob_words_pos, prob_words_neg, unique_count_pos, unique_count_neg,
                                                count_pos_total, count_neg_total))
 
#Multiplying probabilities and classifying each test review
classification_result_neg = classify_data_neg.map(lambda x: multiply_prob(x))\
    .map(lambda x: classifier(x, PROB_POS_CLASS, PROB_NEG_CLASS)) 
    

#Viewing the result
p(classification_result_neg)

#Count the total no. of test positive reviews
total_neg = classification_result_neg.count()
#Count number of positive test reviews predicted wrong
wrong_neg = classification_result_neg.filter(lambda (x, y): (x != y)).count()

#Viewing the result
print "Total number of test positive reviews:" , total_neg
print "Number of positive test reviews predicted wrong:", wrong_neg


#### Calculating Accuracy ####

accuracy = (total_pos + total_neg - wrong_pos - wrong_neg)/(total_pos + total_neg)

print "Accuracy: ", accuracy   


