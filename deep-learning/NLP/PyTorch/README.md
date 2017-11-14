# Table of Contents:
1. Introduction to Torch's Tensor Library
2. Computation Graphs and Automatic Differentiation
3. Deep Learning Building Blocks: Affine maps, non-linearities, and objectives
4. Optimization and Training
5. Creating Network Components in Pytorch
  * Example: Logistic Regression Bag-of-Words text classifier
6. Word Embeddings: Encoding Lexical Semantics
  * Example: N-Gram Language Modeling
  * Exercise: Continuous Bag-of-Words for learning word embeddings
7. Sequence modeling and Long-Short Term Memory Networks
  * Example: An LSTM for Part-of-Speech Tagging
  * Exercise: Augmenting the LSTM tagger with character-level features
8. Advanced: Dynamic Toolkits, Dynamic Programming, and the BiLSTM-CRF
  * Example: Bi-LSTM Conditional Random Field for named-entity recognition
  * Exercise: A new loss function for discriminative tagging

# What is this tutorial?
I am writing this tutorial because, although there are plenty of other tutorials out there, they all seem to have one of three problems:
* They have a lot of content on computer vision and conv nets, which is irrelevant for most NLP (although conv nets have been applied in cool ways to NLP problems).
* Pytorch is brand new, and so many deep learning for NLP tutorials are in older frameworks, and usually not in dynamic frameworks like Pytorch, which have a totally different flavor.
* The examples don't move beyond RNN language models and show the awesome stuff you can do when trying to do lingusitic structure prediction.  I think this is a problem, because Pytorch's dynamic graphs make structure prediction one of its biggest strengths.

Specifically, I am writing this tutorial for a Natural Language Processing class at Georgia Tech, to ease into a problem set I wrote for the class on deep transition parsing.
The problem set uses some advanced techniques.  The intention of this tutorial is to cover the basics, so that students can focus on the more challenging aspects of the problem set.
The aim is to start with the basics and move up to linguistic structure prediction, which I feel is almost completely absent in other Pytorch tutorials.
The general deep learning basics have short expositions.  Topics more NLP-specific received more in-depth discussions, although I have referred to other sources when I felt a full description would be reinventing the wheel and take up too much space.

### Dependency Parsing Problem Set

As mentioned above, [here](https://github.com/jacobeisenstein/gt-nlp-class/tree/master/psets/ps4) is the problem set that goes through implementing
a high-performing dependency parser in Pytorch.  I wanted to add a link here since it might be useful, provided you ignore the things that were specific to the class.
A few notes:

* There is a lot of code, so the beginning of the problem set was mainly to get people familiar with the way my code represented the relevant data, and the interfaces you need to use.  The rest of the problem set is actually implementing components for the parser.  Since we hadn't done deep learning in the class before, I tried to provide an enormous amount of comments and hints when writing it.
* There is a unit test for every deliverable, which you can run with nosetests.
* Since we use this problem set in the class, please don't publically post solutions.
* The same repo has some notes that include a section on shift-reduce dependency parsing, if you are looking for a written source to complement the problem set.
* The link above might not work if it is taken down at the start of a new semester.

# References:
* I learned a lot about deep structure prediction at EMNLP 2016 from [this](https://github.com/clab/dynet_tutorial_examples) tutorial on [Dynet](http://dynet.readthedocs.io/en/latest/), given by Chris Dyer and Graham Neubig of CMU and Yoav Goldberg of Bar Ilan University.  Dynet is a great package, especially if you want to use C++ and avoid dynamic typing.  The final BiLSTM CRF exercise and the character-level features exercise are things I learned from this tutorial.
* A great book on structure prediction is [Linguistic Structure Prediction](https://www.amazon.com/Linguistic-Structure-Prediction-Synthesis-Technologies/dp/1608454053/ref=sr_1_1?ie=UTF8&qid=1489510387&sr=8-1&keywords=Linguistic+Structure+Prediction) by Noah Smith.  It doesn't use deep learning, but that is ok.
* The best deep learning book I am aware of is [Deep Learning](http://deeplearningbook.org), which is by some major contributors to the field and very comprehensive, although there is not an NLP focus.  It is free online, but worth having on your shelf.

# Exercises:
There are a few exercises in the tutorial, which are either to implement a popular model (CBOW) or augment one of my models.
The character-level features exercise especially is very non-trivial, but very useful (I can't quote the exact numbers, but I have run the experiment before and usually the character-level features increase accuracy 2-3%).
Since they aren't simple exercises, I will soon implement them myself and add them to the repo.

# Suggestions:
Please open a GitHub issue if you find any mistakes or think there is a particular model that would be useful to add.
