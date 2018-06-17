#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:34:09 2018

@author: Binderiya
"""

from build_vocab import load_doc, clean_doc
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import SGDClassifier
def doc_to_line(file_name):
    doc = load_doc(file_name)
    tokens = clean_doc(doc)
    return ' '.join(tokens)

def process_docs(directory, is_train = True):
    lines = list()
    for file_name  in listdir(directory):
        if is_train and file_name.startswith('cv9'):
            continue
        if not is_train and not file_name.startswith('cv9'):
            continue
        path = directory + '/' + file_name
        line = doc_to_line(path)
        lines.append(line)
    return lines
    

if __name__ == '__main__':
    positive_dir = './data/review_polarity/txt_sentoken/pos'
    negative_dir = './data/review_polarity/txt_sentoken/neg'
    # Training data prep
    positive_lines = process_docs(positive_dir)
    negative_lines = process_docs(negative_dir)
    train_lines = positive_lines + negative_lines
    train_target = [0 for _ in range(900)] + [1 for _ in range(900)]
    # Test data prep
    test_positive_line = process_docs(positive_dir, is_train = False)
    test_negative_line = process_docs(negative_dir, is_train = False)
    test_lines = test_positive_line + test_negative_line
    test_target = [0 for _ in range(100)] + [1 for _ in range(100)]
    # Feature extraction
#    txt_clf_nbayes = Pipeline([('vect', CountVectorizer()),
#                                ('tfidf',TfidfTransformer()),
#                                ('clf', MultinomialNB())])
#    txt_clf_nbayes.fit(train_lines, train_target)
#    predicted = txt_clf_nbayes.predict(test_lines)
#    print('Naive Bayes prediction accuracy is', accuracy_score(test_target, predicted))
    
    txt_clf_svm = Pipeline([('vect', CountVectorizer()), 
                            ('tfidf', TfidfTransformer()), 
                            ('clf', SGDClassifier())])
    txt_clf_svm.fit(train_lines, train_target)
#    predicted = txt_clf_svm.predict(test_lines)
#    print('SVM prediction accuracy is', accuracy_score(test_target, predicted))
    user_input = True
    while user_input == True:
        reviews = list()
        review = input('Please enter your movie review in english: \n')
        reviews.append(review)
        predicted = txt_clf_svm.predict(reviews)
        for i in range(len(predicted)):
            if predicted[i] == 0:
                print('Your review is positive :)')
            else:
                print('Your review is negative :/')
        another_review = input('Would you like to submit another review? (Type Yes or No) \n')
        while not another_review.lower() in ['no', 'yes']:
            another_review = input('Please type Yes or No: \n')
        if another_review.lower() == 'no':
            user_input = False