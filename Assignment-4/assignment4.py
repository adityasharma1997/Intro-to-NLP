# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NwjzaI6DDorAELyiofQAet_wpZymi407
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install datasets

!pip install nltk

!pip install -U torchtext

import nltk
nltk.download('punkt')
nltk.download('stopwords')

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from torchtext.vocab import GloVe
import pickle
import re
from tensorflow.keras.utils import pad_sequences
from torch.utils.data import Dataset, DataLoader
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import precision_score,confusion_matrix, accuracy_score, roc_auc_score, roc_curve

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

stop_words = stopwords.words('english')
porter = PorterStemmer()
# glove = GloVe(name='6B', dim=100)

# fp = open('/content/drive/MyDrive/NLP_Assg4/glove.txt', 'wb')
# pickle.dump(glove,fp)
# fp.close()

fp = open('/content/drive/MyDrive/NLP_Assg4/glove.txt', 'rb')
glove= pickle.load(fp)
fp.close()

max_len = 250

from datasets import load_dataset
data = load_dataset('sst')
# data=load_dataset('multi_nli')

train_data = data['train']
valid_data = data["validation"]
test_data = data["test"]
# print(train_data[0])

def remove_punctuation(temp):
  temp=temp.translate(str.maketrans('', '', string.punctuation))
  return temp

def remove_spaces(temp):
  temp=re.sub(' +',' ',temp)
  return temp

def preprocessing(data):
    global stop_words
    global porter

    data_x,data_y = ([] for i in range(2))
    for i in range(len(data)):
        # lowercase
        temp_sen = data[i]['sentence'].lower()

        # remove punctuations
        temp_sen = remove_punctuation(temp_sen)

        #remove extra spaces
        temp_sen = remove_spaces(temp_sen)

        #create tokens
        words = word_tokenize(temp_sen)

        # remove stopwords
        filtered_words=list()
        for word in words:
          if word not in stop_words:
            filtered_words.append(word)
        
        # print(filtered_words)
        
        
        # stemming
        stemmed=list()
        for word in filtered_words:
          stemmed.append(porter.stem(word))

        # print(stemmed)

        data_x.append(stemmed)
        data_y.append(data[i]['label'])
    return data_x, data_y

def create_vocab(data):
    freq,word_to_idx,idx_to_word = [dict() for _ in range(3)]
    threshold = 1
    for sen in data:
        for word in sen:
            if word not in freq:
                freq[word] = 1
            else:
                freq[word] = freq[word] + 1
    
    idx_to_word[0] = '<PAD>'
    idx_to_word[1] = '<UNK>'
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1

    i = 2
    for key,val in freq.items():
        if val > threshold:
            word_to_idx[key] = i
            idx_to_word[i] = key
            i =i + 1
    
    return word_to_idx, idx_to_word

def convert_labels(data):
    labels = list()
    for val in data:
        if val>=0.5:
            labels.extend([1])
        else:
            labels.extend([0])
    labels = np.array(labels)
    return labels

def word_to_tokens(data,vocab):
    tokens_list = []
    temp=vocab['<UNK>']
    for sent in data:
        sen_tokens = []
        for word in sent:
            if word in vocab:
                token=vocab[word]
                sen_tokens.extend([token])
            else:
                sen_tokens.extend([temp])
                
        tokens_list.append(sen_tokens)
    return tokens_list

def pad_sentences(data,max_len):
    data = pad_sequences(data, maxlen=max_len,padding='post')
    return data

def get_embeddings(vocab):
    embedding_dim=100
    temp_embedding_matrix = torch.zeros((len(vocab), embedding_dim))
    for word in vocab.keys():
        if word in glove.stoi:
            temp_embedding_matrix[vocab[word]] = glove[word]
        
    embedding_matrix = temp_embedding_matrix.detach().clone()

    return embedding_matrix

class ELMoDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.data = data
        self.data = pad_sentences(data,max_len)
        self.vocab = vocab
        self.forward_data = list()
        for sent in self.data:
            temp = sent[1:]
            self.forward_data.append(temp)

        self.back_data = []

        for sent in self.data:
            temp = sent[:-1]
            self.back_data.append(temp)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent = torch.tensor(self.data[index])
        forward_data = torch.tensor(self.forward_data[index])
        back_data = torch.tensor(self.back_data[index])
        return forward_data, back_data

train_x, train_y = preprocessing(train_data)
val_x, val_y = preprocessing(valid_data)
test_x, test_y = preprocessing(test_data)

# print(train_x)

train_y = convert_labels(train_y)
val_y = convert_labels(val_y)
test_y = convert_labels(test_y)

word_to_idx, idx_to_word = create_vocab(train_x)
vocab_len = len(word_to_idx)

train_data = word_to_tokens(train_x,word_to_idx)
valid_data = word_to_tokens(val_x, word_to_idx)
test_data = word_to_tokens(test_x, word_to_idx)

elmo_train_data =  ELMoDataset(train_data, word_to_idx, 200)
elmo_valid_data =  ELMoDataset(valid_data, word_to_idx, 200)
elmo_test_data =  ELMoDataset(test_data, word_to_idx, 200)

print(elmo_train_data[0])

embedding_matrix = get_embeddings(word_to_idx)
embedding_matrix[2]

class ELMo(nn.Module):
    def __init__(self, vocab_size, weights):
         super(ELMo, self).__init__()
         self.vocab_size = vocab_size
         self.embedding_dim = 100
         self.hidden_dim = 128
         self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
         self.embedding.weight.data.copy_(weights)
         self.embedding.requires_grad = True
         self.bi_lstm1 = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
         self.bi_lstm2 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, batch_first=True, bidirectional=True)
         self.temp_linear = nn.Linear(self.embedding_dim, self.hidden_dim)
         self.linear = nn.Linear(self.hidden_dim*2, vocab_size)

    def forward(self, data):
        embed = self.embedding(data)   # (batch_size, max_len, embedding_dim)
        h1, _ = self.bi_lstm1(embed)   # (batch_size, max_len, hidden_dim*2)
        h2, _ = self.bi_lstm2(h1)      # (batch_size, max_len, hidden_dim*2)
        linear_layer = self.linear(h2)
        return linear_layer

model_elmo = ELMo(vocab_len, embedding_matrix)
model_elmo.to(device)

batchsize=32

train_dataloader = DataLoader(elmo_train_data, batch_size=batchsize, shuffle=True)
valid_dataloader = DataLoader(elmo_valid_data, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(elmo_test_data, batch_size=batchsize, shuffle=True)

forward_train, back_train = next(iter(train_dataloader))
forward_valid, back_valid = next(iter(valid_dataloader))
forward_test, back_test = next(iter(test_dataloader))

# Optimizers specified in the torch.optim package
learning_rate=0.001
optimizer = torch.optim.Adam(model_elmo.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

total_index=len(word_to_idx)

def one_epoch_training(epoch_number, dataloader,epochs):
    train_loss = last_loss= 0

    for i, data in enumerate(dataloader):
        forward_data, back_data = data
        forward_data = forward_data.type(torch.LongTensor) 
        forward_data, back_data = forward_data.to(device), back_data.to(device)
        target = forward_data.view(-1)
        
        optimizer.zero_grad()
        outputs = model_elmo(back_data)
        outputs = outputs.view(-1, total_index)
        
        loss_val = loss_fn(outputs,target)
        loss_val.backward()
        optimizer.step()
        train_loss = train_loss + loss_val.item()

        if i % 200 == 0:
            last_loss = train_loss / 200
            print('Epoch: {}/{}'.format(epoch_number+1, 10), '  Step {} loss: {}'.format(i, last_loss))
            train_loss = 0
    return last_loss

best_vloss,epochs = 100,10
model_path = '/content/drive/MyDrive/NLP_Assg4/elmo_sst.pt'

for epoch in range(0,epochs):
    model_elmo.train(True)
    avg_loss = one_epoch_training(epoch, train_dataloader,epochs)

    model_elmo.train(False)
    valid_loss = 0
    for i, data in enumerate(valid_dataloader):
        forward_data, back_data = data
        forward_data = forward_data.type(torch.LongTensor) 
        forward_data, back_data = forward_data.to(device), back_data.to(device)
        target = forward_data.view(-1)

        optimizer.zero_grad()
        outputs = model_elmo(back_data)
        outputs = outputs.view(-1, total_index)
        
        loss_val = loss_fn(outputs,target)
        valid_loss = valid_loss + loss_val.item()

    den=i+1
    avg_vloss = valid_loss / den
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    if avg_vloss < best_vloss:
        best_vloss = avg_loss
        torch.save(model_elmo.state_dict(), model_path)
        print('Model is updated and saved')

model_elmo.load_state_dict(torch.load(model_path))

elmo_bi_lstm1 = model_elmo.bi_lstm1
elmo_bi_lstm2 = model_elmo.bi_lstm2

parameters = list(model_elmo.parameters())
model_elmo_embeddings = parameters[0].cpu().detach().numpy()

embedding_path = '/content/drive/MyDrive/NLP_Assg4/embeddings_sst.pt'
torch.save(model_elmo_embeddings,embedding_path)

print(model_elmo_embeddings)

parameters = list(model_elmo.parameters())
model_elmo_embeddings = parameters[0].to(device)
print(model_elmo_embeddings)

class Classifier(nn.Module):
    def __init__(self, model_elmo_embeddings, elmo_bi_lstm1, elmo_bi_lstm2, num_classes):
        super(Classifier, self).__init__()
        self.embedding_dim = 100
        self.hidden_dim = 128
        self.embeddings = nn.Embedding.from_pretrained(model_elmo_embeddings)
        self.l1 = nn.Linear(self.embedding_dim, self.hidden_dim*2)
        self.bi_lstm1 = elmo_bi_lstm1
        self.bi_lstm2 = elmo_bi_lstm2
        self.l2 = nn.Linear(self.hidden_dim*2, num_classes)

        self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]), requires_grad=True)


    def forward(self, data):
        embed = self.embeddings(data)
        embeds = self.l1(embed)
        h1,_ = self.bi_lstm1(embed)
        h2,_ = self.bi_lstm2(h1)

        elmo_embed = self.weights[0]*embeds + self.weights[1]*h1 + self.weights[2]*h2
        elmo_max = torch.max(elmo_embed, dim=1)
        output = self.l2(elmo_max[0])
        return output

classify_model = Classifier(model_elmo_embeddings,elmo_bi_lstm1, elmo_bi_lstm2,2)
print(classify_model)

classify_model.to(device)
learning_rate=0.05
optimizer = torch.optim.Adam(classify_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
epochs = 25

pad_train_x = pad_sentences(train_data, max_len) 
pad_valid_x = pad_sentences(valid_data, max_len)
pad_test_x = pad_sentences(test_data, max_len)

x_train =  torch.tensor(pad_train_x)
y_train = torch.tensor(train_y)

x_val =  torch.tensor(pad_valid_x)
y_val = torch.tensor(val_y)

x_test =  torch.tensor(pad_test_x)
y_test = torch.tensor(test_y)

import torch
from torch.utils.data import DataLoader, TensorDataset

batch_size = 32
shuffle = True

train_set = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

val_set = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)

test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

model_path = '/content/drive/MyDrive/NLP_Assg4/classifier_sst.pt'

best_vloss = 100
for epoch in range(0,epochs):
    train_loss = 0.0
    val_loss = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    classify_model.train(True)
    t_loss = 0
    for i, data in enumerate(train_dataloader):
        input_features, label = data
        input_features, label = input_features.to(device), label.to(device)

        optimizer.zero_grad()

        output = classify_model(input_features)
        # output = output.to(device)
        loss = loss_fn(output,label)
        loss.backward()
        train_loss =train_loss + loss.item()
        optimizer.step()

        if i%200==0:
            avg_loss = train_loss/(i+1)
            print('Epoch: {}/{}'.format(epoch+1, epochs), '  Step {} loss: {}'.format(i, avg_loss))
        
    classify_model.train(False)
    for i, data in enumerate(val_dataloader):
        val_inp_feature, val_label = data
        val_inp_feature, val_label = val_inp_feature.to(device), val_label.to(device)
        val_output = classify_model(val_inp_feature)
        loss = loss_fn(val_output, val_label).item()
        val_loss =val_loss + loss
        
    den=i+1
    avg_vloss = val_loss / den
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(classify_model.state_dict(), model_path)
        print('Model updated and saved')

classify_model.load_state_dict(torch.load(model_path))
classify_model.eval()

def print_metrics(true_y, pred_y):
    # Calculate the evaluation metrics
    f1 = f1_score(true_y, pred_y)
    print('F1 score is :',end="")
    print(f1)
    recall = recall_score(true_y, pred_y)
    print('Recall Score is :',end="")
    print(recall)
    accuracy = accuracy_score(true_y, pred_y)
    print('Accuracy is:',end="")
    print(accuracy)
    precision = precision_score(true_y, pred_y)
    print('Precision is :',end="")
    print(precision)
    cf_mat = confusion_matrix(true_y, pred_y)
    print('Evaluated Confusion matrix is :')
    print(cf_mat)

with torch.no_grad():
    preds = []
    for data in test_dataloader:
        input, label = data
        input, label = input.to(device), label.to(device)

        output = classify_model(input)
        pred = torch.argmax(output, dim=1)
        preds.append(pred.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    print_metrics(test_y, preds)

from sklearn import metrics

import matplotlib.pyplot as plt
def plot_roc_curve(test_y, preds):
    fpr, tpr, thresholds = metrics.roc_curve(test_y, preds)
    auc = metrics.roc_auc_score(test_y, preds)
    plt.plot(fpr, tpr, label="AUC={:.3f}".format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.legend(loc=4)
    plt.show()

plot_roc_curve(test_y, preds)

"""Natural Languagle Inference: Dataset

"""

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = re.sub(' +', ' ', sentence)
    words = word_tokenize(sentence)
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [porter.stem(word) for word in filtered_words]
    return stemmed_words

def nli_preprocessing(data, max_samples=45000):
    data_x = []
    data_y = []
    for i, example in enumerate(data):
        premise = preprocess_sentence(example['premise'])
        hypothesis = preprocess_sentence(example['hypothesis'])
        data_x.extend([premise, hypothesis])
        data_y.extend([example['label'], example['label']])
        if i == max_samples - 1:
            break
    return data_x, data_y

data_nli = load_dataset('multi_nli')

nli_train_data = data_nli['train']
nli_valid_data = data_nli["validation_matched"]
nli_test_data = data_nli["validation_mismatched"]

# print(nli_train_data[0])
train_x, train_y = nli_preprocessing(nli_train_data)
val_x, val_y = nli_preprocessing(nli_valid_data)
test_x, test_y = nli_preprocessing(nli_test_data)

word_to_idx, idx_to_word = create_vocab(train_x)
vocab_len = len(word_to_idx)

train_data = word_to_tokens(train_x,word_to_idx)
valid_data = word_to_tokens(val_x, word_to_idx)
test_data = word_to_tokens(test_x, word_to_idx)

elmo_train_data =  ELMoDataset(train_data,word_to_idx, max_len)
elmo_valid_data =  ELMoDataset(valid_data,word_to_idx, max_len)
elmo_test_data =  ELMoDataset(test_data,word_to_idx, max_len)

print(elmo_train_data[1])