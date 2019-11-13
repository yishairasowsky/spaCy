import spacy # NLP library, with many features, including named entity recognition (NER)
nlp = spacy.load("en_core_web_sm") # function that loads the model

# I want to train two types of examples: "New" and "Old".
# 
# "New" sentences contain start and end dates.
# I must provide the start/end positions of each date in the string.
# 
# "Old" sentences contain other entity types which the basic model already recognizes.
# The model can recignize the start/end position of these entities.

sentences = [
    
]



print()
print("TRAIN_DATA = [")
for sentence in old_sentences:
    doc = nlp(sentence[:])
    ent_list = []
    for ent in doc.ents:
        ent_list.append((ent.start_char,ent.end_char,ent.label_))
    test = "('" + sentence + "', \n{'entities': "+str(ent_list)+"}),"
    print(test)
print("]")
print()