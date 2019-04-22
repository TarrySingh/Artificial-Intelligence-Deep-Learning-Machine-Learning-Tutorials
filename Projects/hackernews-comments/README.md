# hncynic
The best Hacker News comments are written with a complete disregard for the linked article.
`hncynic` is an attempt at capturing this phenomenon by training a model to predict
Hacker News comments just from the submission title. More specifically, I trained a
[Transformer](http://jalammar.github.io/illustrated-transformer/) encoder-decoder model on
[Hacker News data](https://archive.org/details/14566367HackerNewsCommentsAndStoriesArchivedByGreyPanthersHacker).
In my second attempt, I also included data from Wikipedia.

The generated comments are fun to read, but often turn out meaningless or contradictory
-- see [here](examples/2019-03-09_wiki-hn.md) for some examples generated from recent HN titles.

There is a demo live at [https://hncynic.leod.org/](https://hncynic.leod.org/).

## Steps
### Hacker News
Train a model on Hacker News data only:
1. [data](data/): Prepare the data and extract title-comment pairs from the HN data dump.
2. [train](train/): Train a Transformer translation model on the title-comment pairs using
   [TensorFlow](https://www.tensorflow.org/) and [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf).

### Transfer Learning
Train a model on Wikipedia data, then switch to Hacker News data:
1. [data-wiki](data-wiki/): Prepare data from Wikipedia articles.
2. [train-wiki](train-wiki/): Train a model to predict Wikipedia section texts from titles.
3. [train-wiki-hn](train-wiki-hn/): Continue training on HN data.

### Hosting
1. [serve](serve/): Serve the model with TensorFlow serving.
2. [ui](ui/): Host a web interface for querying the model.

## Future Work
- Acquire GCP credits, train for more steps.
- It's probably nonideal to use encoder-decoder models. In retrospect, I should have trained
  a language model instead, on data like `title <SEP> comment` (see also: [GPT-2](https://github.com/openai/gpt-2)).
- I've completely excluded HN comments that are replies from the training data. It might be
  interesting to train on these as well.
