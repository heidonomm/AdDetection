# Ad Detection from Podcast transcripts

Ad Detection model that takes builds a context of previous **K** sentences, where each sentence is represented with the mean of each of its word embeddings. The context and the sentence being classified are concatenated to form a 50-dimensional sample.
The problem is treated as a binary classification problem, IOB-encoding is changed to just (0/1)).

### Requirements to run

Necessary packages: `numpy`, `sklearn`  
Glove 25-dimensinonal embeddings. Run in project root - `wget http://nlp.stanford.edu/data/glove.twitter.27B.zip && unzip glove.twitter.27B.zip && mv ./glove.twitter.27B/glove.twitter.27B.25d.txt .`

## Metrics

Ad detection's purpose should be to keep the amount False Positives down, as a high number of the latter would mean filtering out actual podcast content. Thus precision is a key metric to follow. A Linear Regression model achieves a precision score of ~ 63%. Recall on the otherhand is ~17%.

In addition the the confidence with which the model labels the data should be taken into account. An additional metric to implement would be the models ability to correctly label full chunks of ads.

### Results

- Average precision of ~ 63%
- Average recall of 17%
- Average confidence for TP ~ 70%
- Average confidence for FN ~ 18%
- Average confidence for FP ~ 73%

## Possible Improvements

- Take into account the relative location of sentences in the podcast as they're most likely to occur in the beginning, on 1/3, 1/2 2/3 and in the end. Could also use the training data to build a probability distribution to estimate the occurences of ads in positions.
- Build a probability distribution for the length of ads and incorporate it into the model, so that if the beginning of an ad is detected, the following sentences are more likely to be classified as ads as well.
- Neural model is a possibility as ads can be extracted from the training data and added to any other transcript in random positions. Thus creation of synthetic training data is possible.
