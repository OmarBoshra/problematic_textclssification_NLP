
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

nltk.download('stopwords')


# frequency_distribution = FreqDist(meaningful_words)
# print(frequency_distribution.most_common(100))


# frequency_distribution.plot(100, cumulative=True)
    

rawtext_MobyDick = gutenberg.raw('melville-moby_dick.txt')

rawtext_ACL = open('data_upload/acl_predeiction_corpus.txt','r', errors='ignore').read().lower()

lemmatizer = WordNetLemmatizer()
rawtext_ACL = lemmatizer.lemmatize(rawtext_ACL)

rawtext_ACL = ''.join([i for i in rawtext_ACL if not i.isdigit()])

# rawtext_ACL= rawtext_ACL.translate(str.maketrans('', '', string.punctuation))

# paper--end
chapters_paragraphs = re.split('CHAPTER ',rawtext_MobyDick)
# 12 45
# 110:120 0:10

sentences_acl_nonproblematic= sent_tokenize(' '.join(chapters_paragraphs[17:18]))

sentences_acl_problematic= sent_tokenize(' '.join(chapters_paragraphs[135:136]))



def check(word, sent):
    if word in sent:
        print("The word is in the list!")
        return word
    else:
        print("The word is not in the list!")
        return False


keywords = ['categories' ,'given' ,'predict' ,'texts' ,'classes' ,'classification' ,'hierarchical' , 'problem' , 'relations' ,'prediction' ,'information' , 'multiscale' ,'losses' , 'accuracy' , 'product' ,'ranking' , 'abstract' , 'ecommerce' ,'system' ,'different' ,'reckoned', 'address' , 'paper' ,'incorporates' , 'introduces' , 'networks' , 'neural' , 'representation' , 'sharing' , 'strategy' , 'also' ,'combined' ,'define' , 'function.' ,'loss', 'novel' , 'approach' ,'approaches' , 'existing' ,'outperforms' , 'proposed' , 'fundamental' , 'learning' , 'machine','one' , 'regarded' ,'tasks' ,'category' ,'influence' , 'predicted' , 'will' , 'acyclic' ,'directed' , 'graph' ,'organized' ,'tree' , 'websites' ,'figure' , 'achieve' , 'automatic' , 'correlation' , 'evaluators' , 'human' ,'judgement' ,'moderate' ,'robust' ,'build' ,'evaluator' , 'referencefree' ,'conversational' , 'evaluation' , 'systems' ,'automated' , 'correlate' , 'metrics' , 'poorly' ,'expensive' ,'slow' , 'alternative' , 'dialogue' , 'response' ,'exploit' , 'language','masked' ,'models' ,'power' , 'pretrained' ,'semisupervised' , 'training' , 'demonstrate' , 'experimental' ,'results' ,'code' , 'data']


sentwords = " "
actualwords = []




for sent in  sentences_acl_problematic[0:10]+sentences_acl_nonproblematic[110:120]:
  
    for w in keywords:   
        if w in sent:
            sentwords = sentwords + " " + w

            
    
    actualwords.append(sentwords)
          
    sentwords = " "
    
actualwords.append(" ")
actualwords.append(" ")
actualwords.append(" ")
actualwords.append(" ")
actualwords.append(" ")

actualwords.append(" ")
actualwords.append(" ")
actualwords.append(" ")
actualwords.append(" ")
actualwords.append(" ")     

# actualwords = list(dict.fromkeys(actualwords))

print (actualwords)






# text1.similar("problem",100)
sentences= sent_tokenize(rawtext_MobyDick)
#  get random chapters 

s7 = ' '.join(sentences[68:70])

rp = ' '.join(chapters_paragraphs[1:2])
  


    
    
# print(output[0]) # tagged output sentence
# for token in nlp(s7):
#  print ("{:<15} | {:<8} | {:<15} | {:<15} | {:<20}"
#          .format(str(token.text), str(token.dep_), str(token.head.text), str(token.head.pos),str([child for child in token.children])))
#   # pos = nltk.pos_tag(s7)

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






problem_string_paragraph_rawtext_ACL = open('data_upload/problem_strings_paragraph.txt','r', errors='ignore').read().lower()

problem_string_paragraph_rawtext_ACL = ''.join([i for i in problem_string_paragraph_rawtext_ACL if not i.isdigit()])



# words_tok = set(word_tokenize(problem_string_paragraph_rawtext_ACL))
# lenghts = len(words_tok)



# # the preprocessing of predictions
# words_list = []

# for p in sentences_acl_problematic:
#     p = ''.join([i for i in p if not i.isdigit()])
#     p = ''.join([i for i in p if not i in string.punctuation])
#     p = ''.join([i for i in p if not i in string.punctuation])
#     p =' '.join([word for word in word_tokenize(p) if word not in stopwords.words("english")])
    
#     if p:
#         words_list.append(p)
    
    

# sentences_acl_problematic = words_list

# tree = []
# for s in sentences_acl_nonproblematic[110:120]:
#     output = nlp.annotate(s, properties={
#       'timeout': '100000',
#       'annotators': 'parse',
#       'outputFormat': 'json'
#     })
#     tree.append(output['sentences'][0]['parse'])


# problems = ' '.join(problem_lemmas)

# problems = ''.join([i for i in problems if not i.isdigit()])

# problems = ''.join([i for i in problems if not i in string.punctuation])
 
 


# words = set(
#     word for word in words if word not in stop_words and word not in string.punctuation
# )


# meaningful_problemWords = [gensim.utils.simple_preprocess(s) for s in meaningful_problemWords]



td=[]
# for p in chapters_paragraphs:
    
#     try:
#         td.append(textdistance.Cosine(qval=2).distance(p, problems))
#     except ZeroDivisionError:
#         continue


 
 
# Create the visualizer and draw the plot
# visualizer = DispersionPlot(target_words)
# visualizer.fit(target_wordss)
# visualizer.show()


# words_list = []

# for p in problem_lemmas:
#     p = ''.join([i for i in p if not i.isdigit()])
#     p = ''.join([i for i in p if not i in string.punctuation])
#     p =' '.join([word for word in word_tokenize(p) if word not in stopwords.words("english")])
    
#     if p:
#         words_list.append(p)
    
# problem_lemmas = words_list
# text1.dispersion_plot(problem_lemmas)

# for w in meaningful_problemWords:
#     print(w)
#     sim = text1.similar(w,num =20)

# frequency_distribution = FreqDist(sim)
# print(frequency_distribution.most_common(100))


robjects.r("""source('writeArff.R')""")
waikatoWriteArff = robjects.globalenv['waikato.write.arff']






# print ("bag of words baseline")
# unigramControl = robjects.r('list(removePunctuation=TRUE,stopwords=TRUE,removeNumbers=TRUE,stripWhiteSpace=TRUE)')
# dtm = base.as_matrix(tm.DocumentTermMatrix(tm.Corpus(tm.VectorSource(actualwords)),control=unigramControl))
# # dtm_baseline = base.cbind(dtm,class_label=problem_class_labels)
# # waikatoWriteArff(base.data_frame(dtm_baseline),file="problem_baseline.arff",class_col="class_label")



# for row in rpy2.situation.iter_info():
#     print(row)



# print(a)

# word2vec_model = gensim.models.word2vec.Word2Vec.load("data_upload/fuse_word2vec.model")
         
problem_strings=[l.strip() for l in codecs.open("data_upload/problem_strings.txt","r","utf-8").readlines()]
non_problem_strings=[l.strip() for l in codecs.open("data_upload/non_problem_strings.txt","r","utf-8").readlines()]

           
problem_heads = [(p.get("HEAD"),p.get("HEAD-POS")) for i,s in enumerate(ET.parse("data_upload/problem_heads.xml").getroot().findall(".//SENT"))for p in s.findall(".//PROBLEM") if p.text==problem_strings[i]]
non_problem_heads = [(p.get("HEAD"),p.get("HEAD-POS")) for i,s in enumerate(ET.parse("data_upload/non_problem_heads.xml").getroot().findall(".//SENT"))for p in s.findall(".//NON-PROBLEM") if p.text==non_problem_strings[i]]






# from scipy import sparse

# print ("bag of words baseline")
# vect = CountVectorizer()
# vect.fit(problem_lemmas+non_problem_lemmas)
# bag_of_words = vect.transform(problem_lemmas+non_problem_lemmas)

# print (bag_of_words)
# sparse.save_npz('bag',bag_of_words)
# ok = pd.DataFrame.sparse.from_spmatrix(bag_of_words)



# a = sparse.load_npz("bag.npz")
# print(a)

# count_tokens = vect.get_feature_names()

# df_count_vect = pd.DataFrame(data=bag_of_words.toarray(),columns=count_tokens)


# arff.dump('problem_baseline.arff'
#       , df_count_vect.values
#       , relation='problemlemmas'
#       , names=df_count_vect.columns)






# import gensim.downloader as api
# from gensim.models import KeyedVectors


# from sklearn.decomposition import PCA

# dtm = ''



problem_class_labels = ["problem"]*15 + ["non_problem"]*15
pragraph_class_labels = ["problematic_paragraph"]*15+  ["nonproblematic_paragraph "]*15



print ("bag of words for leemas in corpora")
unigramControl = robjects.r('list(removePunctuation=TRUE,stopwords=TRUE,removeNumbers=TRUE,stripWhiteSpace=TRUE)')
dtm = base.as_matrix(tm.DocumentTermMatrix(tm.Corpus(tm.VectorSource(actualwords)),control=unigramControl))



# # dtm = base.cbind(dtm,class_label=problem_class_labels)
# # dtm_syntax = base.cbind(dtm,class_label=pragraph_class_labels)

# # waikatoWriteArff(base.data_frame(dtm_syntax),file="baseline_method2.arff",class_col="class_label")



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



# print ("WordPositioning_method1")


# def indexing_fun(words,largest_length,sentence):
#     print ("WordPositioning")
    
#     indexies = []
#     sentence_group =' '
#     sentence_indexies = []
    
    
    
#     for i,s in enumerate(sentence): 
#             sentence_group = sentence_group+s
    
    
#             for w in words[i]:
        
#                 try:
#                     sentence_indexies.append(sentence_group.index(''.join(w)))
             
#                 except ValueError:
#                     continue
    
#             while (largest_length > len(sentence_indexies)):
#                 sentence_indexies.append(-1)
                
  
#             indexies.append(sentence_indexies)


#             sentence_indexies = []   
     
          
    
#     return indexies
 



# for parg problematic

total_length = 0

problem_words2 = []
largest_length = 0
for i,s in enumerate(actualwords[0:10]):
    problem_words2.append(word_tokenize(s))
    
    
    if largest_length < len(problem_words2[i]):
        largest_length = len(problem_words2[i])
      
total_length = largest_length


nonproblem_words2 = []
largest_length = 0
for i,s in enumerate(actualwords[10:20]):
    nonproblem_words2.append(word_tokenize(s))
    
    if largest_length < len(nonproblem_words2[i]):
        largest_length = len(nonproblem_words2[i])


total_length = total_length + largest_length

# # # for parg non problematic
# # problem_words = []
# # largest_length = 0
# # for i,s in enumerate(problem_string_paragraph_method2[0:15]):
# #     problem_words.append(word_tokenize(s))
    
# #     if largest_length < len(problem_words[i]):
# #         largest_length = len(problem_words[i])


# # total_length = total_length + largest_length


# # nonproblem_words = []
# # largest_length = 0
# # for i,s in enumerate(non_problem_string_paragraph_method2[0:15]):
# #     nonproblem_words.append(word_tokenize(s))
    
# #     if largest_length < len(nonproblem_words[i]):
# #         largest_length = len(nonproblem_words[i])


# # total_length = total_length + largest_length


# # problem_indexies2 = (index_method2(problem_words2,total_length,sentences_acl_problematic[0:10])) 
# # nonproblem_indexies2 = (index_method2(nonproblem_words2,total_length,sentences_acl_nonproblematic[0:10])) 
# # problem_indexies = (index_method2(problem_words,total_length,sentences_acl_nonproblematic[0:10])) 
# # nonproblem_indexies = (index_method2(nonproblem_words,total_length,sentences_acl_nonproblematic[0:10])) 


problem_indexies = (index_method2(problem_words2,total_length,sentences_acl_problematic[0:10])) 
# # nonproblem_indexies = (indexing_fun(actualwords,total_length,sentences_acl_problematic[0:10])) 
# # problem_indexies2 = (indexing_fun(problem_words2,total_length,sentences_acl_nonproblematic[0:10])) 
nonproblem_indexies2 = (index_method2(nonproblem_words2,total_length,sentences_acl_nonproblematic[110:120])) 


# # problem_indexies2 = np.array(problem_indexies2).T.tolist() 
# nonproblem_indexies2 = np.array(nonproblem_indexies2).T.tolist() 
# problem_indexies = np.array(problem_indexies).T.tolist() 
# # nonproblem_indexies = np.array(nonproblem_indexies).T.tolist() 



for i , s in enumerate(problem_indexies):
    dtm = base.cbind(dtm,ind= problem_indexies [i] + nonproblem_indexies2 [i])







# # print ("word2vec")
# # word2vec_model = gensim.models.word2vec.Word2Vec.load("fuse_word2vec.model")
# # word2vec_vector = []
# # for word in problem_words+nonproblem_words:
# #     try:
# #         word2vec_vector.append(word2vec_model[word])
# #     except:
# #         word2vec_vector.append(np.array([0]*100,dtype=np.float32))
# # for i in range(0,len(word2vec_vector[0])):
# #     dtm = base.cbind(dtm,word2vec=list(float(wordVec[i]) for wordVec in word2vec_vector))
# # dtm_word2vec = base.cbind(dtm,class_label=problem_class_labels)
# # waikatoWriteArff(base.data_frame(dtm_word2vec),file="problem_word2vec.arff",class_col="class_label")


# # dtm=''

def doc2vec(dtm):
    print ("doc2vec")
    doc2vecVectors=[]
    doc2vecModel = Doc2Vec.load("acl_sent_doc2vec.model")
    for s in actualwords:
        doc2vecVectors.append(doc2vecModel.infer_vector(s.split()))
    for i in range(0,len(doc2vecVectors[0])):
        dtm = base.cbind(dtm,doc2vec=list(float(docVec[i]) for docVec in doc2vecVectors))
    
    return dtm


# # # doc2vecVectors = doc2vec(dtm)
# # # avg = []
# # # for vec in doc2vecVectors:
# # #     avg.append(sum(vec) / len(vec))
    

# # # plt.plot (avg)

dtm = doc2vec(dtm)



# # new_tree = [str.join(" ",l.splitlines())for l in tree]

# # with open ("data_upload/non_problem_prediction-tree -MD.txt",'w') as file:
# #     file.write('\n'.join(new_tree))
    
print ("syntax")
vb_count=0
total_count=0
vb_count_total=0
total_count_total=0

sent_posInfo = []

posInfo = []
conc = []

total_posInfo = []

tree = [s.strip() for s in codecs.open("data_upload/problem_prediction-tree -MD.txt","r","ISO-8859-1").readlines()]

tree2 =[s.strip() for s in codecs.open("data_upload/non_problem_prediction-tree -MD.txt","r","ISO-8859-1").readlines()]


tree = tree+tree2
for i,line in enumerate(tree):
    t = Tree.fromstring(line.strip())

    sent_posInfo.append([pos for word,pos in t.pos() if not pos in string.punctuation])
    
    
    
sentence_group =[]
i=0
split = 2
dsplit = split
j=0
s=0
while s < (30): 
   
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
        # total_posInfo.append(posInfo)
        # posInfo = []
        split = 2
        dsplit = split
        j = 0
        i = 0
    
    s+=1          
    
# posInfo.append(sent_posInfo[0])

# i = 1
# for s in sent_posInfo:
        

                
#         conc = conc + posInfo[len(posInfo)-1]
    
#         posInfo.append(conc+sent_posInfo[i])
        
#         if i == 9:
#                 posInfo.sort()
#                 conc = []
               
#         if i == 19:    
#                 posInfo.sort()
#                 conc = []
            
            
#                 break
        
#         i+=1
    
for word,pos in t.pos():
        if pos[0:3]=="VB" and i<500: vb_count+=1
        elif i < 500: vb_count_total+=1
        elif pos[0:3]=="VB" and i>=500: total_count+=1
        elif i >= 500: total_count_total+=1

total_pos_tags = list(set([pos for sent in sent_posInfo for pos in sent]))
print ([pos+"."+str(i) for i,pos in enumerate(total_pos_tags)])

# tempvect = total_posInfo[0]+total_posInfo[1]
pos_tag_vector = []
# for posInfo in total_posInfo:
    
for pos in total_pos_tags:
        pos_tag_vector.append([1 if p.count(pos)>0 else 0 for p in posInfo])



for i,pos in enumerate(total_pos_tags):
   
    dtm = base.cbind(dtm,pos=pos_tag_vector[i])
    print (vb_count,vb_count_total, total_count, total_count_total)

dtm = base.cbind(dtm,class_label=problem_class_labels)
dtm_syntax = base.cbind(dtm,class_label=pragraph_class_labels)

waikatoWriteArff(base.data_frame(dtm_syntax),file="universal-method2-md.arff",class_col="class_label")


# 	 arff = open("problem_baseline.arff", "w")
# 	 RELATION_NAME = "bag_of_words"									 
# 	 arff.write("@RELATION " + RELATION_NAME + "\n")
#      arff.write("@ATTRIBUTE " + RELATION_NAME + "\n")

#      for feature in feature_funcitons:
# 			arff.write("@ATTRIBUTE " +\
# 						str(feature.__name__) + " REAL\n")






# Load dataset
# url = "https://github.com/sudar/pig-samples/blob/master/data/tweets.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = read_csv(url,names=names)
# dataset = pd.read_csv("tweets.csv",delimiter=';', error_bad_lines=False)

# AllText = dataset.iloc[:,7]
# datalist = [x for x in AllText]


# datalenght = len(datalist)
# preprocessedData = []
# edf = dataset.text[0:5]

# finald = dataset.text[0:5].apply(gensim.utils.simple_preprocess)
     
        

# for i,sent in enumerate(datalist):

#     new = gensim.utils.simple_preprocess(sent)
    
#     preprocessedData.append([new])

#     time.sleep(.1)


# preprocessedData = gensim.utils.simple_preprocess(s) for s in datalist

# problem_strings=[l.strip() for l in codecs.open("data_upload\problem_strings.txt","r","utf-8").readlines()]


# ## Extract the third row
# print(dataset.iloc[5])
# ### or ###
# df.iloc[2,]
# ### or ###
# df.iloc[2,:]
# ## Extract the first three rows
# df.iloc[:3]
# ### or ###
# df.iloc[0:3]
# ### or ###
# df.iloc[0:3,:]

# ## Extract the 5th column and a sentence

# AllText = dataset.iloc[:,7]
# sentencex = AllText[4]


# ## Extract the first 5 columns
# df.iloc[:,:5]
# ### or ###
# df.iloc[:,0:5]

# print(dataset.shape)
# print(dataset.head(20))
# print(dataset)
# ...
# # head
#
# print(dataset.plot())
# print(dataset.describe())
# print(dataset.groupby('class').size())

# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# histograms
# dataset.hist()
# plt.hist(line_num_words) #for comprehension list
# pyplot.show()

# scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

# Tokenize each line: tokenized_lines

# lines = [re.sub(pattern, '', l) for l in lines]
# tokenized_lines = [regexp_tokenize(s,"\w+") for s in lines]
# words = word_tokenize(sentencex)
# print(words)

# get list not set of lenghts of words list .
# word_length= [len(w) for w in words]
