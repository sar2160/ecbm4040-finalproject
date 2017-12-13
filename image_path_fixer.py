# coding: utf-8
import pandas as pd
train = pd.read_csv('data/train.csv')
train.head()
train.filename.str.lstrip(15)
train.filename.str.lstrip(12)
train.filename = train.filename.str.replace('/Users/simonrimmele/Documents/urban-environments/data/','')
train.to_csv('data/train_relpath')
test = pd.read_csv('data/test.csv')
test.filename = test.filename.str.replace('/Users/simonrimmele/Documents/urban-environments/data/','')
test.head()
test.to_csv('data/test_relpath.csv')
