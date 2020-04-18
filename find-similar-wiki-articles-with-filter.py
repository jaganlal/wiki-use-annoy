import argparse
import time
import sys

import tensorflow as tf
import tensorflow_hub as hub

from annoy import AnnoyIndex

import numpy as np
import pandas as pd
import csv

# Globals
D = 512

def print_with_time(msg):
    print('{}: {}'.format(time.ctime(), msg))
    sys.stdout.flush()


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sentences')
    parser.add_argument('-use_model', default='./use-large-3', type=str)
    parser.add_argument('-csv_file_path', default='./short-wiki.csv', type=str)
    parser.add_argument('-ann', default='wiki.annoy.index', type=str)
    parser.add_argument('-filter_data', default=1, type=int, help='1 to on, 0 to off')
    parser.add_argument('-k', default=10, type=int, help='# of neighbors')
    return parser.parse_args()


def load_sentences(file):
    with open(file) as fr:
        return [line.strip() for line in fr]

def read_data(path, filter_data):
  df_docs = pd.read_csv(path, usecols=['GUID', 'CONTENT', 'ENTITY'])

  if filter_data:
    data_filter_1 = df_docs['ENTITY'] == 'country-related'
    data_filter_2 = df_docs['ENTITY'] == 'person-related'
    df_docs = df_docs[data_filter_1 & data_filter_2]

    # data_filter = df_docs['ENTITY'] == 'human-related'
    # df_docs = df_docs[data_filter]
  
  return df_docs.to_numpy()

def read_data_as_dict(path, key_column, value_column, filter_data):
  print(filter_data)
  data_dict = {}
  with open(path, "r") as infile:
      reader = csv.DictReader(infile)

      for row in reader:
          if filter_data:
            if row['ENTITY'] in ['country-related', 'person-related']:
            # if row['ENTITY'] in ['humen-related']:
              data_dict[row[key_column]] = row[value_column]
          else:
            data_dict[row[key_column]] = row[value_column]
      return data_dict

def main():
    args = setup_args()
    print_with_time(args)

    print(args.k)
    print(args.filter_data)
    print(args.ann)
    print(args.csv_file_path)

    start_time = time.time()
    ann = AnnoyIndex(D)
    ann.load(args.ann)
    end_time = time.time()
    print('Load Time: {}'.format(end_time - start_time))

    print_with_time('Annoy Index: {}'.format(ann.get_n_items()))

    start_time = time.time()
    content_array = read_data(args.csv_file_path, args.filter_data)
    end_time = time.time()
    print_with_time('Sentences: {} Time: {}'.format(len(content_array), end_time - start_time))

    start_time = time.time()
    content_dict = read_data_as_dict(args.csv_file_path, 'GUID', 'CONTENT', args.filter_data)
    end_time = time.time()
    print_with_time('Dictionary: Time: {}'.format(end_time - start_time))

    start_time = time.time()
    embed = hub.Module(args.use_model)
    sentences_ph = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
    embedding_fun = embed(sentences_ph)

    sess = tf.compat.v1.Session()
    sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    end_time = time.time()
    

    print_with_time('Ready! TF setup time: {}'.format(end_time - start_time))
    while True:
        input_sentence_id = input('Enter sentence id: ').strip()

        if input_sentence_id == 'q':
            return
        print_with_time('Input Sentence: {}'.format(input_sentence_id))
        input_sentence = content_dict[input_sentence_id]

        start_time = time.time()
        sentence_vector = sess.run(embedding_fun, feed_dict={sentences_ph:[input_sentence]})
        print_with_time('vec done')
        nns = ann.get_nns_by_vector(sentence_vector[0], args.k)
        end_time = time.time()
        print_with_time('nns done: Time: {}'.format(end_time-start_time))
        similar_sentences = [content_array[nn] for nn in nns]
        for sentence in similar_sentences[1:]:
            print(sentence[0])

if __name__ == '__main__':
    main()