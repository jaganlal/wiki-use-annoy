from flask import Flask, request
from flask_cors import CORS

import json
from json import JSONEncoder

import os
import time
import sys

import tensorflow as tf
import tensorflow_hub as hub

from annoy import AnnoyIndex

import numpy as np
import pandas as pd

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.config['DEBUG'] = True

# globals
VECTOR_SIZE = 512
default_use_model = 'https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed'
default_csv_file_path = './short-wiki.csv'
model_indexes_path = './model-indexes/'
model_index_reference_file = 'index.txt'
default_index_file = 'wiki.annoy.use.large3.index'
default_index_filepath = model_indexes_path + default_index_file
# default_ann_index_file = './wiki.annoy.index'
default_k = 10
default_batch_size = 32
default_num_trees = 10

class SimilarityResult:
  def __init__(self, sourceGuid, sourceSentence, similarDocs):
    self.sourceGuid = sourceGuid
    self.sourceSentence = sourceSentence
    self.similarDocs = similarDocs


class DocType:
  def __init__(self, guid, content):
    self.guid = guid
    self.content = content

class SimilarityResultEncoder(JSONEncoder):
  def default(self, o):
    return o.__dict__

@app.route('/', methods=['GET'])
def home():
  return '<h1>Sentense Analysis</h1><p>Simple sentense analysis</p>'

@app.route('/train', methods=['GET', 'POST'])
def train_model():
  params = request.get_json()
  result = train(params)
  return json.dumps(result)

@app.route('/similarity', methods=['POST'])
def predict_sentence():
  params = request.get_json()
  result = predict(params)
  return json.dumps(result, cls=SimilarityResultEncoder)

@app.route('/get-model-indexes', methods=['GET'])
def get_model_indexes():
  result = get_index_files()
  return json.dumps(result)

# methods called from the APIs
def get_index_files():
  result = None
  try:
    df = pd.read_csv(model_indexes_path + model_index_reference_file, usecols=['FILENAME', 'MODEL-URL', 'MODEL-SHORT-NAME'])
    result = df.values.tolist()

    # for root, dirs, files in os.walk(model_indexes_path):
    #   for file in files:
    #     print(os.path.join(root, file))
    #     files.append(file)

  except Exception as e:
    print('Exception in read_data: {0}'.format(e))
    result = {
        'error': 'Failure'
    }

  return result

def train(params):
  result = {}

  print('Training', params)

  annoy_vector_dimension = VECTOR_SIZE
  index_file = default_index_file

  data_file = default_csv_file_path
  use_model = default_use_model
  num_trees = default_num_trees

  try:
    if params:
      if params.get('vector_size'):
        annoy_vector_dimension = params.get('vector_size')
      if params.get('index_file'):
        index_file = params.get('index_file')
      if params.get('data_file'):
        data_file = params.get('data_file')
      if params.get('use_model'):
        use_model = params.get('use_model')

    start_time = time.time()
    embed_func = hub.Module(use_model)
    end_time = time.time()
    print_with_time('Load the module: {}'.format(end_time-start_time))

    start_time = time.time()
    sentences = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
    embedding = embed_func(sentences)
    end_time = time.time()
    print_with_time('Init sentences embedding: {}'.format(end_time-start_time))

    start_time = time.time()
    data_frame = read_data(data_file)
    content_array = data_frame.to_numpy()
    end_time = time.time()
    print('Read Data Time: {}'.format(end_time - start_time))

    start_time = time.time()
    ann = build_index(annoy_vector_dimension, embedding, default_batch_size, sentences, content_array)
    end_time = time.time()
    print('Build Index Time: {}'.format(end_time - start_time))

    ann.build(num_trees)
    ann.save(model_indexes_path + index_file)

    result = {
      'message': 'Training successful'
    }

  except Exception as e:
    print('Exception in read_data: {0}'.format(e))
    result = {
        'error': 'Failure'
    }

  return result

def predict(params):
  result = {}

  print('Predict', params)

  annoy_vector_dimension = VECTOR_SIZE
  index_file = default_index_file

  data_file = default_csv_file_path
  use_model = default_use_model
  k = default_k

  input_sentence_id = None

  try:
    if params:
      if params.get('guid'):
        input_sentence_id = params.get('guid')
      if params.get('vector_size'):
        annoy_vector_dimension = params.get('vector_size')
      if params.get('index_file'):
        index_file = params.get('index_file')
      if params.get('data_file'):
        data_file = params.get('data_file')
      if params.get('use_model'):
        use_model = params.get('use_model')
      if params.get('k'):
        k = params.get('k')

    if len(input_sentence_id) <= 0:
      print_with_time('Input Sentence Id: {}'.format(input_sentence_id))
      result = {
        'error': 'Invalid Input id'
      }
      return result

    start_time = time.time()
    annoy_index = AnnoyIndex(annoy_vector_dimension, metric='angular')
    annoy_index.load(model_indexes_path + index_file)
    end_time = time.time()
    print_with_time('Annoy Index load time: {}'.format(end_time-start_time))

    start_time = time.time()
    data_frame = read_data(data_file)
    content_array = data_frame.to_numpy()
    end_time = time.time()
    print_with_time('Time to read data file: {}'.format(end_time-start_time))

    start_time = time.time()
    embed_func = hub.Module(use_model)
    end_time = time.time()
    print_with_time('Load the module: {}'.format(end_time-start_time))

    start_time = time.time()
    sentences = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
    embedding = embed_func(sentences)
    end_time = time.time()
    print_with_time('Init sentences embedding: {}'.format(end_time-start_time))

    start_time = time.time()
    sess = tf.compat.v1.Session()
    sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    end_time = time.time()
    print_with_time('Time to create session: {}'.format(end_time-start_time))

    print_with_time('Input Sentence id: {}'.format(input_sentence_id))
    params_filter = 'GUID == "' + input_sentence_id + '"'
    input_data_object = data_frame.query(params_filter)
    input_sentence = input_data_object['CONTENT']

    start_time = time.time()
    sentence_vector = sess.run(embedding, feed_dict={sentences:input_sentence})
    nns = annoy_index.get_nns_by_vector(sentence_vector[0], k)
    end_time = time.time()
    print_with_time('nns done: Time: {}'.format(end_time-start_time))

    similar_sentences = []
    similarities = [content_array[nn] for nn in nns]
    for sentence in similarities[1:]:
      similar_sentences.append({
        'guid': sentence[0],
        'content': sentence[1]
      })
      print(sentence[0])

    result = SimilarityResult(input_sentence_id, input_sentence.values[0], similar_sentences)
  
  except Exception as e:
    print('Exception in predict: {0}'.format(e))
    result = {
        'error': 'Failure'
    }

  return result    


# private methods
def print_with_time(msg):
  print('{}: {}'.format(time.ctime(), msg))
  sys.stdout.flush()

def read_data(path):
  df_docs = None

  try:
    df_docs = pd.read_csv(path, usecols=['GUID', 'CONTENT', 'ENTITY'])
  except Exception as e:
      print('Exception in read_data: {0}'.format(e))
      raise

  return df_docs

def build_index(annoy_vector_dimension, embedding_fun, batch_size, sentences, content_array):
  ann = AnnoyIndex(annoy_vector_dimension, metric='angular')
  batch_sentences = []
  batch_indexes = []
  last_indexed = 0
  num_batches = 0
  with tf.compat.v1.Session() as sess:
    sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    for sindex, sentence in enumerate(content_array):
      batch_sentences.append(sentence[1])
      batch_indexes.append(sindex)

      if len(batch_sentences) == batch_size:
        context_embed = sess.run(embedding_fun, feed_dict={sentences: batch_sentences})
        for index in batch_indexes:
          ann.add_item(index, context_embed[index - last_indexed])
          batch_sentences = []
          batch_indexes = []
        last_indexed += batch_size
        if num_batches % 10000 == 0:
          print_with_time('sindex: {} annoy_size: {}'.format(sindex, ann.get_n_items()))
        num_batches += 1
    if batch_sentences:
      context_embed = sess.run(embedding_fun, feed_dict={sentences: batch_sentences})
      for index in batch_indexes:
        ann.add_item(index, context_embed[index - last_indexed])

  return ann


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=1975)