import os
pdf_files = os.listdir()

from tika import parser
for file_name in pdf_files:
# Parse data from file
    file_data = parser.from_file(file_name)
    # Get files text content
    text = file_data['content']
    print(text[999:1100])

import spacy # NLP library, with many features, including named entity recognition (NER)
nlp = spacy.load("en_core_web_sm") # function that loads the model

data_fname = 'real_data.txt'
data_file = open(data_fname,"r") 
lines = data_file.readlines() 
data_file.close() 

print()
print("TRAIN_DATA = [")
for line in lines:
    doc = nlp(line)
    ent_list = []
    for ent in doc.ents:
        ent_list.append( (ent.start_char,ent.end_char,ent.label_) )
    info = "('" + line[:-1] + "', \n{'entities': "+str(ent_list)+"}),"
    print(info)
print("]")
print()