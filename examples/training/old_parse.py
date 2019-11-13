import spacy # NLP library, with many features, including named entity recognition (NER)
nlp = spacy.load("en_core_web_sm") # function that loads the model

from spacy import displacy # visualizer -- i think it only works in notebooks, not py files.

text = """

The commencement date of the rental shall be February 1, 2012 and shall expire on January 31, 2022 unless 
The said Lease is amended, effective September 25, 2012,  
agree that the said Lease is amended, effective January 16, 2013 as follows
the initial term of this lease shall be for sixty months beginning on October 1, 2015 (the "Initial Term") together with 
this lease is entered into on this 25th day of July, 2017 between
This office lease agreement ("Lease") is entered into and made this 21st day of November, 2018, by and between 
The commencement date of the rental shall be October 21, 2002 and shall expire on December 31, 2022 unless 
The said Lease is amended, effective July 25, 2022, 
agree that the said Lease is amended, effective March 16, 2023 as follows
the initial term of this lease shall be for eighty months beginning on June 23, 2025 (the "Initial Term") together with 
this lease is entered into on this 2nd day of August, 2027 between
This office lease agreement ("Lease") is entered into and made this 1st day of October, 2011, by and between 
"""

doc = nlp(text) # ocs for spacy

displacy.serve(doc, style="ent") # highlight and label the 'ent' objects, i.e. entities