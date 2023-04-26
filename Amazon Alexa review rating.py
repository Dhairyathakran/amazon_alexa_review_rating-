# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 10:08:40 2023

@author: dhair
"""

#     *************   Importing libraries first   ***************

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer






# *****************  Importing Data Set  ***************

df_alexa = pd.read_csv('/Users/dhair/OneDrive/Desktop/amazon_alexa.tsv',  sep = '\t' )

print("Alexa Reviews : /n" , df_alexa.head(10))

print ("Showing the Verified Reviews Column :  \n" ,df_alexa['verified_reviews'])

# ********** Defining the reviews positive or negative with Feedback *********

positive = df_alexa [df_alexa['feedback'] == 1]
negative = df_alexa [df_alexa['feedback'] == 0]

print('Positive Reviews : \n' , positive)
print('Negative Reviews : \n' , negative)

#n ******** Visualizing the feedback with count plot **********

sns.countplot(df_alexa['feedback'] , label = 'count')
plt.show()

#****** Visualizing rating in count plot *******

sns.countplot(df_alexa ['rating'] , label = 'count')

plt.show()

# ****** Histogram **** 

df_alexa['rating'].hist(bins = 5)
plt.show()

#********* Showing the barplot of rating and variation ****** 

plt.figure(figsize = (30 , 10))
sns.barplot(x = 'variation' , y = 'rating' , data = df_alexa , palette =('deep'))
plt.show()

# ********** Drop the columns *************

drop = df_alexa.drop(['rating' , 'date'] , axis = 1 , inplace = True)

#****** Conveting variation column into the number with pd.get_dummies func******

variation_dummies = pd.get_dummies(df_alexa['variation'] , drop_first = True)

print ('Variation  : \n ' , variation_dummies)

drop = df_alexa.drop(['variation'] , axis =1 , inplace = True)

#****** Concatenate variation_Dummies into the data frame******

df_alexa =pd.concat([df_alexa , variation_dummies ] , axis = 1)

#******* using Countvectorizer  ******* 

vectorizer = CountVectorizer()
alexa_countvectorizer = vectorizer.fit_transform(df_alexa["verified_reviews"])

print(alexa_countvectorizer.shape)

#print(vectorizer.get_feature_names())

print(alexa_countvectorizer.toarray())

drop = df_alexa.drop(['verified_reviews'] , axis = 1, inplace = True)

encoded_reviews = pd.DataFrame(alexa_countvectorizer.toarray())

df_alexa = pd.concat([df_alexa , encoded_reviews] , axis = 1)

#*********** Defining the X and Y ****** 

X = df_alexa.drop (['feedback'] , axis = 1)
Y = df_alexa ['feedback']

print("X axis :/n" ,X)
print("Yaxis :/n" ,Y)

#*********** Model Training *******

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.3 , random_state = 20)
#print (X_train)

#******Finding the confusion matrix , classification report , random forest*****

randomforest_classifier = RandomForestClassifier(n_estimators = 100 , criterion = 'entropy' )
randomforest_classifier.fit(X_train ,Y_train)

#***** Evaluating the model ******

Y_predict_train = randomforest_classifier.predict(X_train)

#****** confusion Matrix******

cm = confusion_matrix(Y_train, Y_predict_train)
print ("Confusion Matrix : /n" , cm)

#***** showing in heat map******

sns.heatmap(cm , annot = True)
plt.show()

#***** Printing the classification report *****

print (classification_report(Y_train, Y_predict_train))

#****** Prediction on the Test data set ******

Y_predict = randomforest_classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict)
print (" Confiusion Matrix on Test data :" , cm )
sns.heatmap(cm, annot = True)
print (classification_report(Y_test, Y_predict))

#*********** Imporve the Model ******************

df_alexa = pd.read_csv('/Users/dhair/OneDrive/Desktop/amazon_alexa.tsv',  sep = '\t' )
df_alexa = pd.concat([df_alexa, pd.DataFrame(alexa_countvectorizer.toarray())], axis = 1)
print(df_alexa)

df_alexa['length'] = df_alexa['verified_reviews'].apply(len)
X = df_alexa.drop(['rating', 'date', 'variation', 'verified_reviews', 'feedback'], axis = 1)

y = df_alexa['feedback']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

randomforest_classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
randomforest_classifier.fit(X_train, y_train)
y_predict = randomforest_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(y_test, y_predict))

















