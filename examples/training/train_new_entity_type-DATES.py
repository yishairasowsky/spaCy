#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# new entity label
START_LABEL = "START_DATE"
END_LABEL = "END_DATE"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting



TRAIN_DATA = [

]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="animal", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else: # no model was selected, so by default, use "en"
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(START_LABEL)  # add new entity label to entity recognizer
    ner.add_label(END_LABEL)  # add new entity label to entity recognizer
    ner.add_label("ORG") # Adding extraneous labels shouldn't mess anything up
    ner.add_label("MONEY") # Adding extraneous labels shouldn't mess anything up
    ner.add_label("GPE") # Adding extraneous labels shouldn't mess anything up
    ner.add_label("PERSON") # Adding extraneous labels shouldn't mess anything up
    ner.add_label("DATE") # Adding extraneous labels shouldn't mess anything up
    ner.add_label("LOC") # Adding extraneous labels shouldn't mess anything up
    ner.add_label("CARDINAL") # Adding extraneous labels shouldn't mess anything up
    ner.add_label("FAC") # Adding extraneous labels shouldn't mess anything up
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # ********** test the trained model *************
    
    # test_text = "this lease will end on Jan 1, 2010 and begin on Dec 31, 2003"
    
    test_text = """
                THIS COMMERCIAL LEASE AGREEMENT is made and entered into effective starting
                February 1, 2012 until January 31, 2013, by and between FRONTAGE ROAD COMMERCIAL PROPERTIES, LLC, 
                with mailing address of 607 Triple Tree Road, Bozeman, Montana, 59715, hereinafter 
                referred to as “Landlord,” and MSU Extension, Housing & Environmental Health 
                Program, a division of Montana State University, a state institution of higher education, 
                hereinafter referred to as “Tenant.” 
                """
    doc = nlp(test_text)
    print()
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        if True: # ent.label_ in ["START_DATE","END_DATE"]:
            print()
            # print(ent.label_, ent.text)
    print()

    
    from spacy import displacy # visualizer -- i think it only works in notebooks, not py files.

    text = """
    THIS COMMERCIAL LEASE AGREEMENT is made and entered into effective 
    February 1, 2012, by and between FRONTAGE ROAD COMMERCIAL PROPERTIES, LLC, 
    with mailing address of 607 Triple Tree Road, Bozeman, Montana, 59715, hereinafter 
    referred to as “Landlord,” and MSU Extension, Housing & Environmental Health 
    Program, a division of Montana State University, a state institution of higher education, 
    hereinafter referred to as “Tenant.”     
    """

    doc = nlp(text) # ocs for spacy

    displacy.serve(doc, style="ent") # highlight and label the 'ent' objects, i.e. entities

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
