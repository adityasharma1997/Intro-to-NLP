import re
import json
from keras.models import model_from_json


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
loaded_model.load_weights('my_model.h5')
loaded_model.compile(loss='categorical_crossentropy',optimizer=Adam(0.005),
                metrics=['accuracy'])

fileObj = open('data.obj', 'rb')
exampleObj = pickle.load(fileObj)

word2index=exampleObj["vocab_word"]
reverse_tag2idx=exampleObj["reverse"]
MAX_LENGTH=exampleObj["len"]

def preprocessing(input):
  # print(input)
  modified_data=list()
  input=input.split(' ')
  for sentence in input:
    # print(sentence)
    sentence = re.sub(r'[^\w\s]','',sentence) # remove punctuation
    sentence=sentence.lower()
    # sentence=sentence.split(' ')
    # temp=list()
    # for word in sentence:
    #   temp.append(word)
    modified_data.append(sentence)
  # print(modified_data)
  return modified_data
  
  
def find_encoding(input,vocab):
  final_store=list()
  for word in input:
    if word in vocab:
        final_store.append(vocab[word])
    else:
        final_store.append(vocab['-OOV-'])
  return final_store


def encoding(input):
  data_encoded=find_encoding(input,word2index)
  return data_encoded


from tensorflow.keras.preprocessing.sequence import pad_sequences
def padding_data(data):
  return pad_sequences(data,maxlen=MAX_LENGTH,padding='post')
  
  
sentence=input("Enter the sentence:")
# print(len(sentence))
temp=preprocessing(sentence)
true_temp = temp
temp=encoding(temp)
temp_list = []
temp_list.append(temp)
#print(temp)
final_padded=padding_data(temp_list)

prediction=loaded_model.predict(final_padded)

for i in range(prediction.shape[0]):
  for j in range(len(true_temp)):
    idx = np.argmax(prediction[i][j])
    index = reverse_tag2idx[idx]
    to_print = true_temp[j] + "\t" + str(index)
    print(to_print)
    

#print(final_padded)
