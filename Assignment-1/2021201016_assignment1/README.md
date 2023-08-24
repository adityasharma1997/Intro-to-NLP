## Introduction to NLP - Assignment 1 
### Language-Modelling

------


##### Aditya Sharma (2021201016)
###### 18th February, 2023



-----
##Steps to execute the file

python3 language_model.py <smoothing_type> <path_to_corpus> test_sentence

###### smoothing_type = k for Kneser-Ney Smoothing and

###### smoothing_type = w for Witten-Bell Smoothing



Perplexity is calculated in the following:
1. The given corpus is first divided into train set and test set using random.shuffle.
2. Then the language model is created on train test.
3. Then each sentence in the test set is evaluated.
4. Probability of each sentence is calculated by the two smoothing methods(given above).
5. Then each probability is written in the file along with the sentence.
6. At last the average perplexity score is put in the file.



### neurak_Language-Modelling

neural language model is saved at drive for which link is 
	https://drive.google.com/file/d/1Tw2M3GQasy5MHYP5jpvVTnNA4yMEYeqG/view?usp=sharing
