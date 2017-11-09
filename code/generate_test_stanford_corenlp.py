
# coding: utf-8


import pandas as pd
from pycorenlp import StanfordCoreNLP
from math import ceil
import pickle
import json, os
import gzip
import sys


def read_data(path_to_file):
    df = pd.read_csv(path_to_file)
    print ("Shape of base training File = ", df.shape)
    df.dropna(inplace=True)
    print("Shape of base training data after cleaning = ", df.shape)
    return df

def _getNLPToks_(rawSentence):
    try:
        output = nlp.annotate(rawSentence, properties={
            'annotators': 'tokenize,ssplit,pos,parse,ner,depparse',
            'outputFormat': 'json'
        })
    except:
        print("Stanford NLP crash on row")
        return

    if (isinstance(output, str)):
        # output = json.loads(output) # Convert str output to dict
        print("Error processing row. Attempt to strip % and quotes")
        return _getNLPToks_(rawSentence.replace("%","").replace('"','').replace("'",''))

    dependencies = output['sentences'][0]['basicDependencies']
    tokens = output['sentences'][0]['tokens']
    parse = output['sentences'][0]['parse'].split("\n")

    return {'deps':dependencies,
            'toks':tokens,
            'parse':parse}


nlp = StanfordCoreNLP('http://localhost:9000')
dataframe = pd.read_csv('../../input/test.csv')

dataframe.question1 = dataframe.question1.fillna('Null')
dataframe.question2 = dataframe.question2.fillna('Null')


count = 0

fout = gzip.open('stanford_corenlp_test.nlp', 'wb')

for row in dataframe.iterrows():
    try:
        q1_stanford = _getNLPToks_(row[1]['question1'])
        q2_stanford = _getNLPToks_(row[1]['question2'])

        tmp = {'q1': {
                'raw': row[1]['question1'],
                'toks': q1_stanford['toks'],
                'deps': q1_stanford['deps'],
                },
               'q2': {
                'raw': row[1]['question2'],
                'toks': q2_stanford['toks'],
                'deps': q2_stanford['deps'],
                },
               'id':row[1]['test_id']
               }

        pickle.dump(tmp, fout, protocol=pickle.HIGHEST_PROTOCOL)

    except:
        print("Failure on row: %d" % count)

    count+=1

print("NLP Generation completed!")
fout.close()

