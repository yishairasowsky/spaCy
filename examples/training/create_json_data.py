import spacy
from spacy import load 
from spacy.gold import docs_to_json
nlp = load("en_core_web_sm")
f = open('train.txt')
docs = [] # initialize a list to be populated wih strings
lines = f.readlines()  # returns a list of srings from the txt file
for line in lines:
    # print(line[:]) # display the sentence from that line
    doc = nlp(line) # convert string into a spacy doc object using nlp
    docs.append(doc) # add new doc to the list of docs
json_data = docs_to_json(docs) # convert doc into a json file

import json
with open('train.json', 'w+') as outfile:
    json.dump(json_data, outfile)

# import srsly
# srsly.write_json('real_data.json', [spacy.gold.docs_to_json(docs)])
