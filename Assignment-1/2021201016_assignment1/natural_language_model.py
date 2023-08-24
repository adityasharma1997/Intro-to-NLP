# -*- coding: utf-8 -*-
"""Question3_NLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KtGE5c9ZvZK6fX4XbgICRaJuqAGQJTFh
"""

from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM, Dense
from nltk.tokenize import word_tokenize
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import nltk
import os
import re
# import os
import string
# import re
import random
import math

from nltk.corpus.reader.plaintext import PlaintextCorpusReader

# from google.colab import drive
# # drive.mount('/content/drive')

path ="/content/drive/MyDrive/NLP/Pride and Prejudice - Jane Austen.txt"
# path="/content/drive/MyDrive/NLP/Ulysses - James Joyce.txt"

def preprocessing(raw_input):
  #mention
  raw_input= re.sub('@\w+', '<MENTION>', raw_input)
  #url
  raw_input =re.sub(r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}[^\W]|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}[^\W]|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}[^\W]|www\.[a-zA-Z0-9]+\.[^\s]{2,}[^\W]',r'<URL>',raw_input)
  #hashtag
  raw_input= re.sub('\#[a-zA-Z0-9]\w+', '<HASHTAG>', raw_input)
  #number
  raw_input =re.sub(r'(\d+(?:\.\d+)?)','<NUM>',raw_input)
  #remove punctuations
  raw_input =re.sub(r"[^\s\w\d.?!<>]", '',raw_input,0,re.MULTILINE)
  # raw_input= re.sub("[^\w\d\s\<\>\.\?\!]+",'',raw_input)
  return raw_input

def extractSentences(temp_content):
    output_sentence = []
    temp_out = ""
    temp_list = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",temp_content)
    temp = ""
    keep_punctuation=r'[.?!]'
    for i in temp_list:
        temp = re.sub(keep_punctuation, '', i).strip()
        if(len(temp.split())>=3):
            output_sentence.extend([temp])
            temp = ""
        else:
            if(len(output_sentence)>=1):
                temp = output_sentence[-1] +  " " + temp
                output_sentence = output_sentence[:-1]
                output_sentence.extend([temp])
                temp = ""
            else:
                temp = temp+ " "
    return output_sentence

strip_content=""
with open(path, "r") as fp:
  for line in fp:
    temp_content=line.strip()
    strip_content+=" "+temp_content
  parsed_string=preprocessing(strip_content.lower())
  final_sentence=extractSentences(parsed_string)
  print(final_sentence)

word_to_index={}
index_to_word={}
count=1
for data in final_sentence:
  datas=data.split()
  for word in datas:
    if word not in word_to_index:
      word_to_index[word]=count
      index_to_word[count]=word
      count=count+1


total_words=len(word_to_index)+1

sentence_with_threshold = list()
for sentence in final_sentence:
  temp_tokens = sentence.split()
  if len(temp_tokens) > 1000:
    while len(sentence) > 1000:
      sentence_split = sentence.split()
      temp=sentence_split[:1000]
      sentence_with_threshold.append(" ".join(temp))
      sentence = " ".join(sentence_split[1000:])
  else:
    sentence_with_threshold.append(sentence)

def train_test_split(final_sentence):
  np.random.seed(37)
  train_sentence=[]
  test_sentence=[]
  tot=len(final_sentence)
  training_set = np.random.choice(tot, 1000, replace=False)
  for i in range(len(final_sentence)):
    if i in training_set:
      test_sentence.append(final_sentence[i])
    else:
      train_sentence.append(final_sentence[i])
  return train_sentence,test_sentence

train_sentences, test_sentences = train_test_split(sentence_with_threshold)

print(len(train_sentences))
print(len(test_sentences))

def sequences_to_texts(sentence,word_to_index):
  tokens=list()
  for word in sentence:
    if word in word_to_index:
      tokens.append(word_to_index[word])
  return tokens

train_input_sentence=[]
for sentence in train_sentences:
  sentence=sentence.split()
  temp_tokens=sequences_to_texts(sentence,word_to_index)
  temp=len(temp_tokens)
  for i in range(1,temp):
    train_input_sentence.append(temp_tokens[:i+1])
# print(train_input_sentence[0])

max_length_of_sequence=0
for i in train_input_sentence:
  if len(i)>max_length_of_sequence:
    max_length_of_sequence=len(i)
max_length_of_sequence

def pad_sequence(seq, max_sequence_len):
  if len(seq) < max_sequence_len:
      for i in range(0, max_sequence_len - len(seq)):
        seq.insert(0, 0)
  return seq

def pad_sequences(input_sequence, max_sequence_len):
  padded_input_sequence = list()
  for seq in input_sequence:
    padded_input_sequence.extend([pad_sequence(seq, max_sequence_len)])
  return input_sequence

train_input_sequences = np.array(pad_sequences(train_input_sentence, max_length_of_sequence))

predictors, label = train_input_sequences[:,:-1],train_input_sequences[:,-1]
label = ku.to_categorical(label, num_classes=total_words)

checkpoint_dir = os.path.dirname(os.getcwd())
checkpoint_path = checkpoint_dir + "lstm_model_5.ckpt"
batch_size = 32
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max', 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

activation='softmax'
input_len = max_length_of_sequence - 1 
model = Sequential()
model.add(Embedding(total_words, input_len, input_length=input_len))
model.add(LSTM(150))
model.add(Dense(total_words, activation=activation))

print(model.summary)

model.save(checkpoint_path)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_model = model.fit(predictors, label, epochs=20, batch_size=32, callbacks=[cp_callback], verbose=1, validation_split=0.15)

print(len(test_sentences))

# def generate_report(perplexity_avg, report_filename, perplexity_list,test_sentence):
#     line_no = 0
#     with open(report_filename, 'w') as f:
#         f.write("avg_perplexity is:\t" + str(perplexity_avg) + '\n')
#         for perplex in perplexity_list:
#             f.write(test_sentence[line_no] + '\t' +"pp=" +str(perplex) + '\n')
#             line_no =line_no+ 1

testing_perplexity_list = list()
testing_sentences_count = 0
testing_perplexity_sum = 0

sentence_count = 1
test_sentences=input()
for sentence in test_sentences:
  # print(sentence)
  tokens = sentence.split()
  encoded_text = sequences_to_texts(tokens, word_to_index)
  # print(tokens)
  final_prob = 1
  tot=len(tokens)
  # print(tot)
  for i in range(1, tot):
    input_sentence = temp_tokens[:i+1]
    # print(input_sentence)
    # encoded_text = sequences_to_texts(input_sentence, word_to_index)
    # print(len(encoded_text))
    exp_output = encoded_text[-1]
    encoded_text = encoded_text[:-1]
    pad_encoded = pad_sequence(encoded_text, max_length_of_sequence-1)
    predicted = model.predict([pad_encoded])
    final_prob *= predicted[0][exp_output]
  # print(final_prob)

  perplexity = (1 / final_prob) ** (1 / max_length_of_sequence)
  testing_perplexity_list.append(perplexity)
  sentence_count += 1

testing_perplexity_sum = 0
# max_perplexity_score = max([x for x in testing_perplexity_list if x != float("inf")])
# print(max_perplexity_score)
for perplexity in testing_perplexity_list:
    testing_perplexity_sum += perplexity

  #   perplexity = max_perplexity_score



testing_perplexity_avg = testing_perplexity_sum / len(test_sentences)
print(testing_perplexity_avg)


# generate_report(testing_perplexity_avg, "2021201016_LM5_test-perplexity.txt", testing_perplexity_list,test_sentences)
