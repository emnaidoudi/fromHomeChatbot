# things we need for NLP
import nltk
from spellchecker import SpellChecker
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import requests
# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import spacy
from fromDatabase import get_manager
import datetime
import requests

def get_wheather(loc):
     if(loc ==None):
       loc="tunis"
    # print("inin %s" %(loc))
     r = requests.get('http://api.openweathermap.org/data/2.5/weather?q=%s&units=metric&APPID=acf0a678438a992a21999196194f42c0'%(loc))
     j=r.json()
     x="Temperature : %s°C, %s "  % (j['main']["temp"],j["weather"][0]["description"])
     return  x
def spacy_entity(sentence):
    all_extractions = list()
    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load('en_core_web_sm')
    def entity_extraction():
        #sentence=sentence.upper()
        print(sentence)
        doc = nlp(sentence)
        for entity in doc.ents:
            all_extractions.append((entity.text.lower(), entity.label_))
    entity_extraction()        
    return all_extractions


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words):
    # clean up the sentence
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                break
    return(np.array(bag))

# restore all of our data structures
import pickle
data = pickle.load( open("training_data_entities", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('entities.json') as json_data:
    entities = json.load(json_data)


# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net) #tensorboard_dir='tflearn_entities_logs'    

# load our saved model
model.load('./model_entities.tflearn')


def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) ]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def entity_exact(sentence):
    sentence=sentence.lower()
    entities_stemmed=""
    entity=classify(sentence)[0][0] # location
    for i in entities["entities"]:
        if(i["tag"]==entity):
            examples=i["examples"]
            entities_stemmed=[stemmer.stem(word.lower()) for word in examples ]
            break
    for i in entities_stemmed:
        if(i in sentence):
            x=entities_stemmed.index(i)
            return examples[x]  


tf.reset_default_graph()


# restore all of our data structures
import pickle
data_intent = pickle.load( open( "training_data", "rb" ) )
words_intent = data_intent['words']
classes_intent = data_intent['classes']
train_x_intent = data_intent['train_x']
train_y_intent = data_intent['train_y']

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x_intent[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y_intent[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model_intent = tflearn.DNN(net, tensorboard_dir='tflearn_logs')    

# load our saved model
model_intent.load('./model.tflearn')



# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify_intent(sentence):
    # generate probabilities from the model
    results = model_intent.predict([bow(sentence, words_intent)])[0]
    # filter out predictions below a threshold_intent
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes_intent[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

from watson_developer_cloud import LanguageTranslatorV3
import json
language_translator = LanguageTranslatorV3(
    version='2018-05-01',
    iam_apikey='6N2fgkLzRaDtj7vM90uDNYiBZY8rxYRdfzXKvQ0vqio9',
    url='https://gateway-lon.watsonplatform.net/language-translator/api'
)

def ibm_watson_translation(sentence):
    translation = language_translator.translate(
    text=sentence,
    model_id='en-fr').get_result()
    return translation["translations"][0]["translation"]

def response(sentence, userID='123', show_details=False):       
    for i in intents['intents']:
        if 'context_filter' in i and len(context)>0 and i["context_filter"]==context[userID]:
            context.clear()
            # Check if it is a TRANSALATION
            if i["context_filter"]=="ibm_translation":
                return ibm_watson_translation(sentence)
            return random.choice(i['responses'])
        
    results = classify_intent(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if results[0][0]=="weather":
                    loc=entity_exact(sentence)
                    print(loc)
                    return get_wheather(loc)
                if results[0][0]=="askManager":
                    try:
                        now = datetime.datetime.now()
                        year=now.year
                        group=entity_exact(sentence)
                        sp=spacy_entity(sentence)
                        for i in sp:
                            if('DATE' in i ):
                                year=i[0]
                                print(type(year))
                                break
                        return get_manager(group,str(year)) 
                    except:
                        return "which group or manager?"                 
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return random.choice(i['responses'])

            results.pop(0)     