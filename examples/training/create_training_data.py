import spacy # NLP library, with many features, including named entity recognition (NER)
nlp = spacy.load("en_core_web_sm") # function that loads the model

# I want to train two types of examples: "New" and "Old".
# 
# "New" sentences contain start and end dates.
# I must provide the start/end positions of each date in the string.
# 
# "Old" sentences contain other entity types which the basic model already recognizes.
# The model can recignize the start/end position of these entities.

New_sentences = [

]

Old_sentences = [
                "Google bought SMRTflow for $9 billion",
                "United States is against the PLO according to Fox News",
                "Amazon has a new GUI for Tony Blair in Harvard University Press",
                'what is the Prime Minister doing on a Wednesday in Africa?',
                'Netanyahu declares war on terror in the Arutz Sheva broadcast',
                "you live on Main Street",
                "we are located on River Road",
                "you can find us at the office on Joshua Court",
                "this Landlord is helping the Tenant all the time",
                "the house is on Oak Tree Drive",
                "the State University of New York called Bank of America for a CPA because of the IRS audit"
]
print("TRAIN_DATA = [")
for sentence in FP_sentences:
    doc = nlp(sentence[:])
    ent_list = []
    for ent in doc.ents:
        ent_list.append((ent.start_char,ent.end_char,ent.label_))
    test = "('" + sentence + "', \n{'entities': "+str(ent_list)+"}),"
    print(test)
print("]")