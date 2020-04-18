import argparse
import time
import sys

import tensorflow as tf
import tensorflow_hub as hub

from annoy import AnnoyIndex

import numpy as np
import pandas as pd

print('TF version: {}'.format(tf.__version__))
print('TF-Hub version: {}'.format(hub.__version__))

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
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-num_trees', default=10, type=int)
    return parser.parse_args()


def read_data(path):
  df_docs = pd.read_csv(path, usecols=['GUID', 'CONTENT'])
  df_docs.head()
  return df_docs.to_numpy()

def build_index(embedding_fun, batch_size, sentences, content_array):
    ann = AnnoyIndex(D)
    batch_sentences = []
    batch_indexes = []
    last_indexed = 0
    num_batches = 0
    with tf.compat.v1.Session() as sess:
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        for sindex, sentence in enumerate(content_array):
            batch_sentences.append(sentence[1]) #'CONTENT'
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

def main():
    args = setup_args()
    print_with_time(args)

    embed = hub.Module(args.use_model)
    sentences = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
    embedding_fun = embed(sentences)

    start_time = time.time()
    content_array = read_data(args.csv_file_path)
    end_time = time.time()
    print('Read Data Time: {}'.format(end_time - start_time))

    start_time = time.time()
    ann = build_index(embedding_fun, args.batch_size, sentences, content_array)
    end_time = time.time()
    print('Build Index Time: {}'.format(end_time - start_time))

    ann.build(args.num_trees)
    ann.save(args.ann)

if __name__ == '__main__':
    main()