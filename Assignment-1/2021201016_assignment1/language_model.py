import sys
import math
from utils import *


smoothing_technique = sys.argv[1]
data_path = sys.argv[2]

strip_content=""
with open(data_path, "r") as file:
    for readline in file:
        line_strip = readline.strip()
        strip_content += " "  + line_strip
        

parsed_string=preprocessing(strip_content.lower())
train_sentence=extractSentences(parsed_string)

# train_sentence,test_split=train_test_split(final_sentence) 
#where final_sentence is statement coming after extractSentences
# print(test_sentence)

unigram=ngrams(train_sentence,1)
bigram=ngrams(train_sentence,2)
trigram=ngrams(train_sentence,3)
quadgram=ngrams(train_sentence,4)

unigram_words=findingwords(unigram)
bigram_words=findingwords(bigram)
trigram_words=findingwords(trigram)
quadgram_words=findingwords(quadgram)


quad_dis_words,quad_tot_words=formula_values(quadgram_words)
tri_dis_words,tri_tot_words=formula_values(trigram_words)
bi_dis_words,bi_tot_words=formula_values(bigram_words)
uni_dis_words=dict()
uni_tot_words=dict()
			
			
quad_reverse_count=reverse_count(quadgram_words)
tri_reverse_count=reverse_count(trigram_words)
bi_reverse_count=reverse_count(bigram_words)
uni_reverse_count=reverse_count(unigram_words)

			
tot_reverse_count=[]			
vocab=[unigram_words,bigram_words,trigram_words,quadgram_words]
prev_word_count=[uni_dis_words,bi_dis_words,tri_dis_words,quad_dis_words]
prev_world_totcount=[uni_tot_words,bi_tot_words,tri_tot_words,quad_tot_words]
tot_reverse_count=[uni_reverse_count,bi_reverse_count,tri_reverse_count,quad_reverse_count]
total_count = [len(unigram), len(bigram), len(trigram), len(quadgram)]
#print(tot_reverse_count)		
			
test_sentence=input()		

if smoothing_technique == 'k':
	kn_train_sentence_prob = list()
	kn_train_perplex = list()
	# test_report_filename="2021201016_LM3_train-perplexity.txt"
	training_perplexity_avg = 0.0
	#print(tot_reverse_count[0])
	#print("hi")
	for i in train_sentence:
	  if len(i)>=4:
	    prob, sen_perplex= kneser_ney(i, vocab, prev_word_count, prev_world_totcount, tot_reverse_count, total_count)
	    kn_train_perplex.append(sen_perplex)
	    training_perplexity_avg += sen_perplex
	    kn_train_sentence_prob.append([i, prob, sen_perplex])
	training_perplexity_avg /= len(train_sentence)
	# generate_report(training_perplexity_avg, test_report_filename, kn_train_perplex,train_sentence)
	#print(training_perplexity_avg)
			
	kn_test_sentence_prob = list()
	kn_test_perplex = list()
	kn_test_avg_perplex = 0.0
	# test_report_filename="2021201016_LM3_test-perplexity.txt"
	if len(test_sentence) >=4:
	  prob, sen_perplex= kneser_ney(test_sentence, vocab, prev_word_count, prev_world_totcount, tot_reverse_count, total_count)
	  kn_test_perplex.append(sen_perplex)
	  kn_test_avg_perplex += sen_perplex
	  kn_test_sentence_prob.append([test_sentence, prob, sen_perplex])
	  print("prob of given statement is:", prob)
	kn_test_avg_perplex /= len(test_sentence)
	# print(kn_test_avg_perplex)
	
elif smoothing_technique=='w':
	wb_train_sentence_prob = list()
	wb_train_perplex = list()
	wb_train_avg_perplex = 0.0
	# train_report_filename="2021201016_LM4_train-perplexity.txt"
	for i in train_sentence:
	    prob, sen_perplex= written_bell(i, vocab, prev_word_count, prev_world_totcount,total_count)
	    wb_train_perplex.append(sen_perplex)
	    wb_train_avg_perplex += sen_perplex
	    wb_train_sentence_prob.append([i, prob, sen_perplex])
	wb_train_avg_perplex /= len(train_sentence)
	# generate_report(wb_train_avg_perplex, train_report_filename, wb_train_perplex,train_sentence)
	# print(wb_train_avg_perplex)
	
	wb_test_sentence_prob = list()
	wb_test_perplex = list()
	wb_test_avg_perplex = 0.0
	# test_report_filename="2021201016_LM4_test-perplexity.txt"
	#for i in test_sentence:
	prob, sen_perplex= written_bell(test_sentence, vocab, prev_word_count, prev_world_totcount,total_count)
	wb_test_perplex.append(sen_perplex)
	wb_test_avg_perplex += sen_perplex
	wb_test_sentence_prob.append([test_sentence, prob, sen_perplex])
	print("prob of given statement is:", prob)
	wb_test_avg_perplex /= len(test_sentence)
else:
	print("wrong smoothing technique")			
