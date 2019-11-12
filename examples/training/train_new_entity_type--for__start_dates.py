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

# directive to the compiler that a particular module should be compiled 
# using syntax or semantics that will be available 
# in a specified future release of Python
from __future__ import unicode_literals, print_function 

import plac # command line arguments parser
import random # random number generation

# Path is used to examine, locate, and manipulate files
from pathlib import Path 

import spacy # process and “understand” large volumes of text

from spacy.util import minibatch, compounding 
# 'minibatch' iterates over batches of items
# in 'compounding' a new value is produced by multiplying the
# previous value by the compound rate

# new entity label
LABEL = "END_DATE"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    (
        "parties terminate the of January 14, 2011",
        {"entities": [(25, 41, LABEL)]},
    ),
    ('the beginning will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(69, 82, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(70, 83, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(70, 83, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(70, 83, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(71, 84, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(69, 82, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(62, 75, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(60, 73, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(59, 72, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(59, 72, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(61, 74, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(61, 74, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(60, 73, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(60, 73, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(62, 75, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(62, 75, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(62, 75, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),

    (
        "consideration of the agreements hereinafter set forth", 
        {"entities": []}
    ),
    (
        "and that it is terminated on January 31, 2012",
        {"entities": [(29, 45, LABEL)]},
    ),
    (
        "and it will be expiring on January 31, 2017", 
        {"entities": [(27, 43, LABEL)]}
    ),
    (
        "and end on January 31, 2018, with the successive optional",
        {"entities": [(11, 27, LABEL)]}
    ),
    ('the beginning will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the beginning will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(69, 82, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(70, 83, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(70, 83, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(70, 83, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the commencement will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(71, 84, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the inception will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(68, 81, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the initiation will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(69, 82, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(62, 75, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(60, 73, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(59, 72, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(59, 72, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(61, 74, LABEL)]}),
    ('the onset will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(61, 74, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(60, 73, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(60, 73, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(62, 75, LABEL)]}),
    ('the outset will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the cessation will be on June 14, 2034', 
    {'entities': [(65, 78, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the closure will be on June 14, 2034', 
    {'entities': [(63, 76, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the conclusion will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the completion will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the ending will be on June 14, 2034', 
    {'entities': [(62, 75, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the expiration will be on June 14, 2034', 
    {'entities': [(66, 79, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the finish will be on June 14, 2034', 
    {'entities': [(62, 75, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the stopping will be on June 14, 2034', 
    {'entities': [(64, 77, LABEL)]}),
    ('the starting will be on Jan 4, 2019 and the termination will be on June 14, 2034', 
    {'entities': [(67, 80, LABEL)]}),

    (
        "and ending January 31, 2019, and", 
        {"entities": [(11, 27, LABEL)]}
    ),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="end_date", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, i.e. "ner" is in fact in the pipeline, 
    # then get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names) # not sure what this does...
    # get names of other pipes to disable them during training
    # i am not sure what the "pipes" are...
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter): # for each iteration
            random.shuffle(TRAIN_DATA) 
            # process your training examples in batches
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "the contract begins on June 6, 2019"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

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
