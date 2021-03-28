# Ad Detection from Podcast transcripts

Ad Detection model that takes builds a context of previous **K** sentences, where each sentence is represented by the mean of each of its word embeddings.
The problem is treated as a binary classification problem, IOB-encoding is changed to just (0/1)).

Due to the small amounts of

Necessary packages: `numpy`, `sklearn`

## Metrics

Ad detection's purpose should be to keep the amount False Positives down, as a high number of the latter would mean filtering out actual podcast content.
A case could be made for precision being the most important metric, as a low precision score means a high number of False Positives and that actual podcast content gets filtered out.

### Results

## Possible Improvements

- Take into account the relative location of sentences in the podcast as they're most likely to occur in the beginning, on 1/3, 1/2 2/3 and in the end. Could also use the training data to build a probability distribution to estimate the occurences of ads in positions.
- Build a probability distribution for the length of ads and incorporate it into the model, so that if the beginning of an ad is detected, the following sentences are more likely to be classified as ads as well.
- Neural model is a possibility as ads can be extracted from the training data and added to any other transcript in random positions. Thus creation of synthetic training data is possible.
