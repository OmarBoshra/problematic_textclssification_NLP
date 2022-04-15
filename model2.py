
# Load libraries
from pandas import read_csv
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

import random, codecs
from nltk.tokenize import word_tokenize
import re
import numpy as np
import gensim.models.word2vec
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import sentiwordnet as swn
from nltk.tree import Tree
from nltk.wsd import lesk
import nltk
nltk.download('sentiwordnet')
import time
import os, re
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from nltk.tree import Tree
# from lxml import etreepy
import xml.etree.ElementTree as ET
#  try to understand 


import nltk
from nltk.book import *
nltk.download('sentiwordnet')

import time

import os, re

import matplotlib.pyplot as plt
import textdistance
from yellowbrick.text import DispersionPlot

# import rpy2.robjects as robjects

from rpy2.robjects.packages import importr
from nltk.tree import Tree
# from lxml import etreepy

import xml.etree.ElementTree as ET

from nltk.corpus import stopwords

# os.environ['R_HOME'] = 'C:/Program Files/R/R-4.1.2'
# os.environ['R_USER'] = 'C:/Users/osari/AppData/Local/Programs/Python/Python310/Lib/site-packages/rpy2'

import rpy2
import string
# print(rpy2.__version__)

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import regexp_tokenize
import inspect 
import glob 
import spacy
import en_core_web_sm
tm = importr("tm")
base = importr("base")

from nltk.corpus import webtext

# nltk.download('book')
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

stop_words = set(stopwords.words("english"))


rawtext_ACL = open('data_upload/acl_corpus.txt','r', errors='ignore').read().lower()

lemmatizer = WordNetLemmatizer()
rawtext_ACL = lemmatizer.lemmatize(rawtext_ACL)

rawtext_ACL = ''.join([i for i in rawtext_ACL if not i.isdigit()])

# rawtext_ACL= rawtext_ACL.translate(str.maketrans('', '', string.punctuation))


sentences_acl_nonproblematic= sent_tokenize(' '.join(chapters_paragraphs[1:2]))

sentences_acl_problematic= sent_tokenize(' '.join(chapters_paragraphs[4:5]))

  

# the stanford parser

# tree = []
# for s in sentences_acl_nonproblematic:
#     output = nlp.annotate(s, properties={
#       'timeout': '100000',
#       'annotators': 'parse',
#       'outputFormat': 'json'
#     })
#     tree.append(output['sentences'][0]['parse'])

    

problem_lemmas=[" ".join([l.get("lem") for l in s.findall(".//lemma")]) for s in ET.parse("data_upload/problem_strings_rasp.xml").getroot().findall(".//sentence")]
non_problem_lemmas=[" ".join([l.get("lem") for l in s.findall(".//lemma")]) for s in ET.parse("data_upload/non_problem_strings_rasp.xml").getroot().findall(".//sentence")]

# for non_problematic paragraph
problem_string_paragraph=[l.strip() for l in codecs.open("data_upload/problem_strings_paragraph.txt","r","utf-8").readlines()]
non_problem_string_paragraph=[l.strip() for l in codecs.open("data_upload/non_problem_strings_paragraph.txt","r","utf-8").readlines()]
problem_string_paragraph_method2=[l.strip() for l in codecs.open("data_upload/problem_strings_paragraph_method2.txt","r","utf-8").readlines()]
non_problem_string_paragraph_method2=[l.strip() for l in codecs.open("data_upload/non_problem_strings_paragraph_method2.txt","r","utf-8").readlines()]

# for problematic paragraph
problem_string_paragraph2=[l.strip() for l in codecs.open("data_upload/problem_strings_paragraph2.txt","r","utf-8").readlines()]
non_problem_string_paragraph2=[l.strip() for l in codecs.open("data_upload/non_problem_strings_paragraph2.txt","r","utf-8").readlines()]
problem_string_paragraph2_method2=[l.strip() for l in codecs.open("data_upload/problem_strings_paragraph2_method2.txt","r","utf-8").readlines()]
non_problem_string_paragraph2_method2=[l.strip() for l in codecs.open("data_upload/non_problem_strings_paragraph2_method2.txt","r","utf-8").readlines()]



robjects.r("""source('writeArff.R')""")
waikatoWriteArff = robjects.globalenv['waikato.write.arff']
       


import gensim.downloader as api
from gensim.models import KeyedVectors


from sklearn.decomposition import PCA

dtm = ''



problem_class_labels = ["problem"]*15 + ["non_problem"]*15
pragraph_class_labels = ["problematic_paragraph"]*30 +  ["nonproblematic_paragraph "]*30



print ("bag of words for leemas in corpora")
unigramControl = robjects.r('list(removePunctuation=TRUE,stopwords=TRUE,removeNumbers=TRUE,stripWhiteSpace=TRUE)')
dtm = base.as_matrix(tm.DocumentTermMatrix(tm.Corpus(tm.VectorSource(problem_string_paragraph2_method2[0:15]+non_problem_string_paragraph2_method2[0:15]+problem_string_paragraph2[0:15]+non_problem_string_paragraph2[0:15])),control=unigramControl))



# dtm = base.cbind(dtm,class_label=problem_class_labels)
# dtm_baseline = base.cbind(dtm,class_label=pragraph_class_labels)

# waikatoWriteArff(base.data_frame(dtm_baseline),file="baseline_method2.arff",class_col="class_label")



print ("WordPositioning_method2")


def index_method2(words,dtm,sentence):
    
    indexies = []
    sentence_group =' '
    sentence_indexies = []
    i=0
    split = 2
    dsplit = split
    j=0
    while i < (len(words)): 
       
        if i < 5 and i>=0 :
            sentence_group = sentence[j]+sentence[j+1]
            j+=split
            if i==4 :
                split = split+2
                dsplit = split
                j=0
            
        elif i < 9 and i>=5 :
          
            sentence_group =' '
            while j<dsplit:
                
                sentence_group = sentence_group +(sentence[j])
                j+=1
                
            
            j = j - (split - 2)
            dsplit= j+split
            if i == 8:
                j=0
                split = split+2
                dsplit = split
        elif i < 12 and i>=9 :
            sentence_group =' '
            
            while j<dsplit:
                
                sentence_group = sentence_group +(sentence[j])
                j+=1
                
            j = j - (split - 2)
            dsplit= j+split
            if i == 11:
                j=0
                split = split+2
                dsplit = split
        elif i < 14 and i>=12 :
            sentence_group =' '
            
            
            while j<dsplit:
                
                sentence_group = sentence_group +(sentence[j])   
                j+=1
                
            j = j - (split - 2)
            dsplit= j+split
            if i == 13:
              j=0
              split = split+2 
              dsplit = split
        else:
            
            sentence_group =' '
          
            
            while j<dsplit:
                
                sentence_group = sentence_group +(sentence[j])
                j+=1
                
            j = j - (split - 2)
            dsplit= j+split
        for w in words[i]:
        
                try:
                    sentence_indexies.append(sentence_group.index(''.join(w)))
                except ValueError:
                    continue
    
        while (largest_length > len(sentence_indexies)):
                sentence_indexies.append(-1)
                
        indexies.append(sentence_indexies)

        sentence_indexies = []
        
        i+=1
        
    return indexies



# for parg problematic
total_indexies = []
total_length = 0

problem_words2 = []
largest_length = 0
for i,s in enumerate(problem_string_paragraph2_method2[0:15]):
    problem_words2.append(word_tokenize(s))
    
    
    if largest_length < len(problem_words2[i]):
        largest_length = len(problem_words2[i])
      
total_length = largest_length


nonproblem_words2 = []
largest_length = 0
for i,s in enumerate(non_problem_string_paragraph2_method2[0:15]):
    nonproblem_words2.append(word_tokenize(s))
    
    if largest_length < len(nonproblem_words2[i]):
        largest_length = len(nonproblem_words2[i])


total_length = total_length + largest_length

# for parg non problematic
problem_words = []
largest_length = 0
for i,s in enumerate(problem_string_paragraph_method2[0:15]):
    problem_words.append(word_tokenize(s))
    
    if largest_length < len(problem_words[i]):
        largest_length = len(problem_words[i])


total_length = total_length + largest_length


nonproblem_words = []
largest_length = 0
for i,s in enumerate(non_problem_string_paragraph_method2[0:15]):
    nonproblem_words.append(word_tokenize(s))
    
    if largest_length < len(nonproblem_words[i]):
        largest_length = len(nonproblem_words[i])


total_length = total_length + largest_length


problem_indexies2 = (index_method2(problem_words2,total_length,sentences_acl_problematic[0:10])) 
nonproblem_indexies2 = (index_method2(nonproblem_words2,total_length,sentences_acl_nonproblematic[0:10])) 
problem_indexies = (index_method2(problem_words,total_length,sentences_acl_nonproblematic[0:10])) 
nonproblem_indexies = (index_method2(nonproblem_words,total_length,sentences_acl_nonproblematic[0:10])) 



problem_indexies2 = np.array(problem_indexies2).T.tolist() 
nonproblem_indexies2 = np.array(nonproblem_indexies2).T.tolist() 
problem_indexies = np.array(problem_indexies).T.tolist() 
nonproblem_indexies = np.array(nonproblem_indexies).T.tolist() 



for i , s in enumerate(problem_indexies):
    dtm = base.cbind(dtm,ind=problem_indexies2[i] + nonproblem_indexies2 [i] + problem_indexies [i] + nonproblem_indexies [i])

# dtm = base.cbind(dtm,class_label=problem_class_labels)
# dtm_index = base.cbind(dtm,class_label=pragraph_class_labels)

# waikatoWriteArff(base.data_frame(dtm_index),file="dtm_index_method2.arff",class_col="class_label")



def doc2vec(dtm):
    print ("doc2vec")
    doc2vecVectors=[]
    doc2vecModel = Doc2Vec.load("acl_sent_doc2vec.model")
    for s in problem_string_paragraph2_method2[0:15]+non_problem_string_paragraph2_method2[0:15]+problem_string_paragraph2[0:15]+non_problem_string_paragraph2[0:15]:
        doc2vecVectors.append(doc2vecModel.infer_vector(s.split()))
    for i in range(0,len(doc2vecVectors[0])):
        dtm = base.cbind(dtm,doc2vec=list(float(docVec[i]) for docVec in doc2vecVectors))
    
    return dtm



dtm = doc2vec(dtm)

# dtm = base.cbind(dtm,class_label=problem_class_labels)
# dtm_doc2vec = base.cbind(dtm,class_label=pragraph_class_labels)

# waikatoWriteArff(base.data_frame(dtm_doc2vec),file="dtm_doc2vec_method2.arff",class_col="class_label")

    
print ("syntax")
vb_count=0
total_count=0
vb_count_total=0
total_count_total=0

sent_posInfo = []

posInfo = []
conc = []

total_posInfo = []

tree = [s.strip() for s in codecs.open("data_upload/paragraph_problematic_trees.txt","r","ISO-8859-1").readlines()]
tree = tree+tree
tree =[s.strip() for s in codecs.open("data_upload/paragraph_nonproblematic_trees.txt","r","ISO-8859-1").readlines()]

tree = tree + tree
tree = tree+tree
for i,line in enumerate(tree):
    t = Tree.fromstring(line.strip())

    sent_posInfo.append([pos for word,pos in t.pos() if not pos in string.punctuation])
    
    
    
sentence_group =[]
i=0
split = 2
dsplit = split
j=0
for s in (problem_words2+nonproblem_words2+problem_words+nonproblem_words): 
   
    if i < 5 and i>=0 :
        sentence_group = (sent_posInfo[j]+sent_posInfo[j+1])
        j+=split
        if i==4 :
            split = split+2
            dsplit = split
            j=0
        
    elif i < 9 and i>=5 :
      
        sentence_group =[]
        while j<dsplit:
            
            sentence_group = sentence_group + (sent_posInfo[j])
            j+=1
            
        
        j = j - (split - 2)
        dsplit= j+split
        if i == 8:
            j=0
            split = split+2
            dsplit = split
    elif i < 12 and i>=9 :
        sentence_group =[]
        
        while j<dsplit:
            
            sentence_group = sentence_group + (sent_posInfo[j])
            j+=1
            
        j = j - (split - 2)
        dsplit= j+split
        if i == 11:
            j=0
            split = split+2
            dsplit = split
    elif i < 14 and i>=12 :
        sentence_group =[]
        
        
        while j<dsplit:
            
            sentence_group = sentence_group + (sent_posInfo[j])  
            j+=1
            
        j = j - (split - 2)
        dsplit= j+split
        if i == 13:
          j=0
          split = split+2 
          dsplit = split
    else:
      
        sentence_group =[]
       
      
        
        while j<dsplit:
            
            sentence_group = sentence_group + (sent_posInfo[j])
            j+=1
            
        j = j - (split - 2)
        dsplit= j+split  
        
    
    
    posInfo.append(sentence_group)
    i+=1    
    
    if i == 15:
        posInfo.sort()
        total_posInfo.append(posInfo)
        posInfo = []
        split = 2
        dsplit = split
        j = 0
        i = 0
            

    
for word,pos in t.pos():
        if pos[0:3]=="VB" and i<500: vb_count+=1
        elif i < 500: vb_count_total+=1
        elif pos[0:3]=="VB" and i>=500: total_count+=1
        elif i >= 500: total_count_total+=1

total_pos_tags = list(set([pos for sent in sent_posInfo for pos in sent]))
print ([pos+"."+str(i) for i,pos in enumerate(total_pos_tags)])

pos_tag_vector = []
for posInfo in total_posInfo:
    for pos in total_pos_tags:
        pos_tag_vector.append([1 if p.count(pos)>0 else 0 for p in posInfo])
for i,pos in enumerate(total_pos_tags):
   
    dtm = base.cbind(dtm,pos=pos_tag_vector[i])
    print (vb_count,vb_count_total, total_count, total_count_total)

# dtm = base.cbind(dtm,class_label=problem_class_labels)
# dtm_syntax = base.cbind(dtm,class_label=pragraph_class_labels)

# waikatoWriteArff(base.data_frame(dtm_syntax),file="universal_method2.arff",class_col="class_label")



