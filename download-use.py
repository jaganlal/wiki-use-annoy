import os 

#download the model to local so it can be used again and again
os.system('mkdir use-large-3')

os.system('curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC ./use-large-3')