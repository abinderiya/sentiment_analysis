#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:52:21 2018

@author: Binderiya
"""
import nltk
import string 
from collections import Counter
from os import listdir 
#nltk.download('stopwords')

def load_doc(file_name):
    file = open(file_name, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_word = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_word]
    return tokens 

def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)
    
def process_docs(directory, vocab):
    for filename in listdir(directory):
        if filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)
        
def process_tokens(vocab, min_occurance = 2):
    tokens = [k for k, c in vocab.items() if c >= 2]
    return tokens
    

def save_vocab(tokens, filename):
    data = '\n'.join(tokens)
    file = open(filename, 'w')
    file.write(data)
    file.close()

if __name__ == '__main__':
    
    positive_dir = './data/review_polarity/txt_sentoken/pos'
    negative_dir = './data/review_polarity/txt_sentoken/neg'
    file_name = 'vocab.txt'
    vocab = Counter()
    process_docs(positive_dir, vocab)
    process_docs(negative_dir, vocab)
    vocab = process_tokens(vocab)
    save_vocab(vocab, file_name)