import spacy
from spacy import load 
from spacy.gold import docs_to_json
nlp = load("en_core_web_sm")

# naming of files
fname = 'validation'
txt_file = fname + '.txt'
json_file = fname + '.json'

f = open(txt_file) # will scrape strings from here
lines = f.readlines()  # list of srings from the txt file

docs = [] # initialize a list to be populated wih nlp doc objects
for line in lines:
    # print(line[:]) # display the sentence from that line
    doc = nlp(line) # convert string into a spacy doc object using nlp
    docs.append(doc) # add new doc to the list of docs

json_data = docs_to_json(docs) # convert doc into a json file

# import json
# with open('json/' + json_file, 'w+') as outfile:
#     json.dump(json_data, outfile)

import srsly
srsly.write_json('json/' + json_file, [spacy.gold.docs_to_json(docs)])
