import os
import string
import re
import random
import math



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


# def train_test_split(final_sentence):
#   np.random.seed(37)
#   train_sentence=[]
#   test_sentence=[]
#   tot=len(final_sentence)
#   training_set = np.random.choice(tot, 1000, replace=False)
#   for i in range(len(final_sentence)):
#     if i in training_set:
#       test_sentence.append(final_sentence[i])
#     else:
#       train_sentence.append(final_sentence[i])
#   return train_sentence,test_sentence


def findngram(data,n):
  total_tokens=len(data)
  ngrams=[]
  out=[]
  key=""
  for i in range(0,total_tokens):
    if(len(ngrams))==n:
      toke=[str(elem) for elem in ngrams]
      # print(toke)
      key=" ".join(toke)
      out.append(key)
      ngrams=ngrams[1:]
    ngrams.append(data[i])
  if(len(ngrams))==n:
    toke=[str(elem) for elem in ngrams]
    key=" ".join(toke)
    out.append(key)
  return out



def ngrams(input,n):
  stored_ngrams=list()
  for data in input:
    datas=data.split()
    output=findngram(datas,n)
    for i in output:
      stored_ngrams.append(i)
  return stored_ngrams
  
  
  



def findingwords(ngrams):
  no_of_words={}
  for i in range(0,len(ngrams)):
    if ngrams[i] in no_of_words:
      no_of_words[ngrams[i]]+=1
    else:
      no_of_words[ngrams[i]]=1
  return no_of_words


def formula_values(ngram_vocab):
  distinct_words={}
  total_words={}
  for key in ngram_vocab:
    t_word=key.split()
    t_word=t_word[:-1]
    token=[str(elem) for elem in t_word]
    temp_word = ' '.join(token)
    if temp_word in distinct_words:
      distinct_words[temp_word]+=1
      total_words[temp_word]+=ngram_vocab[key]
    else:
      distinct_words[temp_word]=1
      total_words[temp_word]=ngram_vocab[key]
  return distinct_words,total_words



def reverse_count(ngram_vocab):
  reverse_count={}
  for key in ngram_vocab:
    temp=key.split(" ")
    word=temp[-1]
    if word in reverse_count:
      reverse_count[word]+=1
    else:
      reverse_count[word]=1
  return reverse_count
  


def kneser_ney_helper(sequence, vocabs, prev_word_count,prev_world_totcount, tot_reverse_count, total_count, first_step,d, n):
    '''
    sequence: setence ngram
    vocabs: contain vocabulary for unigram, bigram, trigram and quadgram
    prev_word_count: count for distinct history sentences with which n-gram starts with
    cont_count: count for curr_word coming as end word in n-gram
    total_count: unigram, bigram, trigram, quadgram toatl count
    d: value of delta
    n: n in n-gram

    '''
    # taking out last word from given n_gram
    curr_word = sequence[-1]

    if n==1:
        if curr_word in vocabs[n-1]:
          first_term=max(vocabs[n-1][curr_word]-d, 0)/total_count[n-1] 
          final_prob = first_term + d/len(vocabs[n-1])
          return final_prob
        else:
          return d/len(vocabs[n-1])

    history = sequence[:-1]
    history_sen = ' '.join([str(elem) for elem in history])
    sentence = history_sen + ' ' + curr_word

    # total_hist_count: total ngrams starting with history
    # distinct_hist_count: distinct ngrams starting with history
    first_term = 0.0
    lambda_term = 0.0
    total_hist_count = 0.0
    distinct_hist_count = 0.0
    


    if history_sen not in prev_world_totcount[n-1]:
      total_hist_count=0.0
    else:
      total_hist_count = prev_world_totcount[n-1][history_sen]

    
    #calculation for first term
    if first_step:
      if total_hist_count==0 or sentence not in vocabs[n-1]:
        first_term=0.0
      else:
        first_term = max(vocabs[n-1][sentence]-d,0)/total_hist_count
    else:
      if curr_word not in tot_reverse_count[n-1]:
        first_term=0.0
      else:
        count=0
        count=tot_reverse_count[n-1][curr_word]
        first_term=max(count-d,0)/len(vocabs[n-1])

    # calculation for lambda
    
    if history_sen not in prev_word_count[n-1]:
      distinct_hist_count = 0.0
    else:
      distinct_hist_count = prev_word_count[n-1][history_sen]


    if total_hist_count==0:
      lambda_term = 1.0
    else:
      lambda_term = (d*distinct_hist_count)/total_hist_count
      if(lambda_term==0):
        lambda_term = d/len(vocabs[n-1])

 

    # call for recursion
    new_seq = sequence[1:]
    p_cont = kneser_ney_helper(new_seq, vocabs, prev_word_count,prev_world_totcount,tot_reverse_count, total_count, False, 0.75, n-1)

    final_prob = first_term + lambda_term*p_cont    
    return final_prob
  
  

def kneser_ney(sentence, vocab, prev_word_count, prev_world_totcount, tot_reverse_count, total_count):
    '''
    vocabs: contain vocabulary for unigram, bigram, trigram and quadgram
    prev_word_count: count for distinct history sentences with which n-gram starts with
    tot_reverse_count: count for curr_word coming as end word in n-gram
    total_count: unigram, bigram, trigram, quadgram toatl count
    '''
    
    # sentence_words = sentence.split()
    sentence_words = sentence.split()
    # num_words = len(sentence_words)
    n = min(len(sentence_words),4)
    sequence = list()
    first_step = True
    final_prob = 1.0
    perplex = 0.0
    temp_prob = 1.0
    temp_final_prob = 0.0
    for i in range(len(sentence_words)):
        if len(sequence) == n:
            temp_prob = kneser_ney_helper(sequence, vocab, prev_word_count,prev_world_totcount, tot_reverse_count, total_count, first_step, 0.75,n)
            perplex += math.log(temp_prob)
            final_prob *= temp_prob
            sequence = sequence[1:]
        sequence.append(sentence_words[i])
    
    temp_prob = kneser_ney_helper(sequence, vocab, prev_word_count,prev_world_totcount, tot_reverse_count, total_count, first_step, 0.75, n)
    final_prob *= temp_prob
    perplex += math.log(temp_prob)
    perplex = perplex/len(sentence_words)
    perplex = math.exp(-1*perplex)

    return final_prob, perplex



def witten_bell_helper(sequence, vocab, prev_word_count,prev_world_totcount,total_count,n):

  last_word=sequence[-1]
  if n==1:
    if last_word in vocab[0]:
      lamda=1/len(vocab[0])
      pml=vocab[0][last_word]/total_count[0]
      return lamda+(1-lamda)*pml
    else:
      return 1/len(vocab[0])
  

  temp_prev_word=sequence[:-1]
  prev_word=" ".join([str(elem) for elem in temp_prev_word])
  sentence=prev_word+" "+last_word

  total_count_of_word=0
  if prev_word in prev_world_totcount[n-1]:
    total_count_of_word=prev_world_totcount[n-1][prev_word]
  

  distinct_count=0
  if prev_word in prev_word_count[n-1]:
    distinct_count=prev_word_count[n-1][prev_word]
  
  numerator=distinct_count
  denominator=(distinct_count+total_count_of_word)
  lamda=1
  if denominator!=0:
    lamda=numerator/denominator
  

  pml=0

  if sentence in vocab[n-1]:
    pml=vocab[n-1][sentence]/total_count_of_word
  
  updated_sequence=sequence[1:]
  wb_prob=(1-lamda)*pml+lamda*witten_bell_helper(updated_sequence, vocab, prev_word_count,prev_world_totcount,total_count,n-1)
  return wb_prob



def written_bell(sentence, vocab, prev_word_count, prev_world_totcount, total_count):
    
    
    sentence=sentence.split()
    n = min(len(sentence),4)
    sequence = []
    perplex = 0.0
    temp_prob = 0.0
    final_prob = 1.0

    for i in range(len(sentence)):
        if len(sequence) == n:
            temp_prob = witten_bell_helper(sequence, vocab, prev_word_count, prev_world_totcount,total_count,n)
            final_prob *= temp_prob
            perplex += math.log(temp_prob)
            sequence = sequence[1:]
        sequence.append(sentence[i])
    
    temp_prob = witten_bell_helper(sequence, vocab, prev_word_count, prev_world_totcount,total_count,n)
    final_prob *= temp_prob
    perplex += math.log(temp_prob)
    perplex = perplex/len(sentence)
    perplex = math.exp(-1*perplex)
    return final_prob, perplex









