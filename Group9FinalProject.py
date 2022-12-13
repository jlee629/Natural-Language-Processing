# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:29:15 2022

@author:Group9

"""
##1. Load the data into a pandas data frame.
import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

filename = 'Youtube03-LMFAO.csv'
path = 'C:/Users/User/Documents/AI/finalProject/'
fullpath = os.path.join(path,filename)

LMFAO_group9 = pd.read_csv(fullpath)
LMFAO_group9.shape
LMFAO_group9.head(5)
LMFAO_group9.info

##2. Carry out some basic data exploration and present your results. (Note: You only need two columns for this project, make sure you identify them correctly, if any doubts ask your professor)
data=LMFAO_group9.drop(columns=['COMMENT_ID','AUTHOR','DATE'])
x_data = LMFAO_group9['CONTENT']
type(x_data)
y_data = LMFAO_group9['CLASS']
type(y_data)

data.shape
data.head(5)
data.info
##3.Using nltk toolkit classes and methods prepare the data for model building, refer to the third lab tutorial in module 11 (Building a Category text predictor ). Use count_vectorizer.fit_transform().
#xdata=x_data.values.tolist()
count_vectorizer = CountVectorizer()
cv_x_data = count_vectorizer.fit_transform(x_data)

##4.Present highlights of the output (initial features) such as the new shape of the data and any other useful information before proceeding.
print("\nDimensions of training data:", cv_x_data.shape)

##5.Downscale the transformed data using tf-idf and again present highlights of the output (final features) such as the new shape of the data and any other useful information before proceeding.
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(cv_x_data)
type(train_tfidf)
print(train_tfidf.shape)
##6.Use pandas.sample to shuffle the dataset, set frac =1 
data_shuffle=data.sample(frac=1)

##7.Using pandas split your dataset into 75% for training and 25% for testing, make sure to separate the class from the feature(s). (Do not use test_train_ split)
df_training = data.sample(frac=0.75)
df_testing = data.drop(df_training.index)

##8.Fit the training data into a Naive Bayes classifier. 

Xtrain = tfidf.fit_transform(count_vectorizer.fit_transform(df_training['CONTENT']))
Ytrain = df_training.CLASS
Xtest = tfidf.fit_transform(count_vectorizer.fit_transform(df_testing['CONTENT']))
Ytest = df_testing.CLASS
classifier = MultinomialNB()
classifier.fit(Xtrain, Ytrain)

##9.Cross validate the model on the training data using 5-fold and print the mean results of model accuracy. 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score as ascore

num_folds = 5
accuracy_values = cross_val_score(classifier, Xtrain, Ytrain, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")
##10.Test the model on the test data, print the confusion matrix and the accuracy of the model.
from sklearn.metrics import confusion_matrix

classifier.fit(Xtest, Ytest)
y_test_pred = classifier.predict(Xtest)
print(y_test_pred)
print(Ytest)
print(ascore(y_test_pred,Ytest))
print(confusion_matrix(y_test_pred,Ytest))

##11.As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam) and pass them to the classifier and check the results. You can be very creative and even do more happy with light skin tone emoticon.
input_data = [
    'It is a great song!',
    'Check out this www.funny.com',
    'If you r sad, come and chat with me, my number: 8943667788',
    'Love the beat!',
    'They have so many great songs!',
    'They are great singers!']

input_tc = count_vectorizer.transform(input_data)
input_tfidf = tfidf.transform(input_tc)
predictions = classifier.predict(input_tfidf)

for sent, category in zip(input_data, predictions):
    print('\nInput:', sent,'\nPredicted category:', \
          predictions[category])

















