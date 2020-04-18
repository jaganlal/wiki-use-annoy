# Sample that shows how to use Universal Sentence Encoder

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

#download the model to local so it can be used again and again
# !mkdir ./use-large-3
# Download the module, and uncompress it to the destination folder. 

# DO NOT DOWNLOAD IT EACH TIME, IF YOU HAD DOWNLOADED IT ONCE, ITS ENOUGH

# !curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC ./use-large-3

#Function so that one session can be called multiple times. 
#Useful while multiple calls need to be done for embedding.

# Reduce logging output.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

embed_fn = embed_useT('./use-large-3')

messages = [
    "How old are you",
    "How are you",
    "What is your age"
]

encoding_matrix = embed_fn(messages)
np.inner(encoding_matrix, encoding_matrix)

from math import*
from decimal import Decimal

class Similarity():
    def euclidean_distance(self,x,y):

        """ return euclidean distance between two lists """

        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    def manhattan_distance(self,x,y):

        """ return manhattan distance between two lists """

        return sum(abs(a-b) for a,b in zip(x,y))

    def minkowski_distance(self,x,y,p_value):

        """ return minkowski distance between two lists """

        return self.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),
           p_value)

    def nth_root(self,value, n_root):

        """ returns the n_root of an value """

        root_value = 1/float(n_root)
        return round (Decimal(value) ** Decimal(root_value),3)

    def cosine_similarity(self,x,y):

        """ return cosine similarity between two lists """

        numerator = sum(a*b for a,b in zip(x,y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return round(numerator/float(denominator),3)

    def square_rooted(self,x):

        """ return 3 rounded square rooted value """

        return round(sqrt(sum([a*a for a in x])),3)

def recommendTopSentences(sentenceIndex):
    similarities = []
    measures = Similarity()
    for index, sentence in enumerate(messages):
        if(index != sentenceIndex):
            similarities.append({'score': measures.cosine_similarity(encoding_matrix[sentenceIndex], encoding_matrix[index]), 'title': sentence})
    return similarities

measures = Similarity()
measures.cosine_similarity(encoding_matrix[0],encoding_matrix[1])

sentenceIndex = 1
sentence = messages[sentenceIndex]
print('Similar sentence to: ', sentence)
similarities = recommendTopSentences(sentenceIndex)
sentencesRecommended = sorted(similarities, key = lambda i: i['score'], reverse=True)
print(sentencesRecommended[:10])
