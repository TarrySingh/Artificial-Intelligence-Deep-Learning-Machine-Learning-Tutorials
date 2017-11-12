Awesome Deep Learning for Natural Language Processing (NLP) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
====

Contents
----

- __[Courses ](#courses)__  
- __[Books](#books)__  
- __[Tutorials / Demos](#tutorials)__  
- __[Talks / Lectures](#talks)__  
- __[Frameworks](#frameworks)__  
- __[Papers](#papers)__  
- __[Blog Posts](#blog-posts)__
- __[Researchers](#researchers)__  
- __[Datasets](#datasets)__
- __[NLP Specific Areas](#specific-areas)__
- __[Miscellaneous](#miscellaneous)__  
- __[Contributing](#contributing)__  

Courses 
----
1. CS224d: Deep Learning for Natural Language Processing from Stanford
	- [Course homepage](http://web.stanford.edu/class/cs224n/) A complete survey of the field with videos, lecture slides, and sample student projects.
	- [Course Lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) Video playlist.
	- [Course notes](https://github.com/stanfordnlp/cs224n-winter17-notes) Probably the best "book" on DL for NLP.
1. Neural Networks for NLP from Carnegie Mellon University
	- [Coures homepage](http://phontron.com/class/nn4nlp2017/)
	- [Course Lectures](https://www.youtube.com/user/neubig/videos)
	- [Course code](https://github.com/neubig/nn4nlp2017-code/)
1. Deep Learning for Natural Language Processing from University of Oxford and DeepMind
	- [Coures homepage](https://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/)
	- [Coures Slides](https://github.com/oxford-cs-deepnlp-2017/lectures)
	- [Course Lectures](https://www.youtube.com/playlist?list=PL613dYIGMXoZBtZhbyiBqb0QtgK6oJbpm)

Books
-----
1. [Neural Network Methods in Natural Language Processing](https://www.amazon.com/gp/product/1627052984) by Yoav Goldberg and Graeme Hirst
2. [Deep Learning in Natural Language Processing](http://www.springer.com/us/book/9789811052088) by Li Deng and Yang Liu
3. [Natural Language Processing in Action](https://www.manning.com/books/natural-language-processing-in-action) by Hobson Lane, Cole Howard, and Hannes Hapke

Tutorials
----
1. [Deep Learning for Natural Language Processing (without Magic)](http://www.socher.org/index.php/DeepLearningTutorial/DeepLearningTutorial)
1. [A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726) 
1. [Deep Learning for Natural Language Processing: Theory and Practice (Tutorial)](https://www.microsoft.com/en-us/research/publication/deep-learning-for-natural-language-processing-theory-and-practice-tutorial/) 
1. [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/mandelbrot)
1. [Practical Neural Networks for NLP](https://github.com/clab/dynet_tutorial_examples) from EMNLP 2016 using DyNet framework
1. [Recurrent Neural Networks with Word Embeddings](http://deeplearning.net/tutorial/rnnslu.html)
1. [LSTM Networks for Sentiment Analysis](http://deeplearning.net/tutorial/lstm.html)
1. [TensorFlow demo using the Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
1. [LSTMVis: Visual Analysis for Recurrent Neural Networks](http://lstm.seas.harvard.edu/client/index.html)  

Talks
----
1. Ali Ghodsi's lecture on word2vec [part 1](https://www.youtube.com/watch?v=TsEGsdVJjuA) and [part 2](https://www.youtube.com/watch?v=nuirUEmbaJU)
2. [Richard Socher's talk on sentiment analysis, question answering, and sentence-image embeddings](https://www.youtube.com/watch?v=tdLmf8t4oqM)
3. [Deep Learning, an interactive introduction for NLP-ers](http://www.slideshare.net/roelofp/220115dlmeetup)
4. [Deep Natural Language Understanding](http://videolectures.net/deeplearning2016_cho_language_understanding/) 
5. [Deep Learning Summer School, Montreal 2016](http://videolectures.net/deeplearning2016_montreal/) Includes state-of-art language modeling.

Frameworks
----
1. [Keras](https://keras.io/) - _The Python Deep Learning library_ Emphasis on user friendliness, modularity, easy extensibility, and Pythonic.
1. [TensorFlow](https://www.tensorflow.org/) - A cross-platform, general purpose Machine Intelligence library with Python and C++ API.
1. [Genism: Topic modeling for humans](https://pypi.python.org/pypi/gensim) - A Python package that includes word2vec and doc2vec implementations.
1. [DyNet](https://github.com/clab/dynet) - _The Dynamic Neural Network Toolkit_ "work well with networks that have dynamic structures that change for every training instance".
1. [Google’s original word2vec implementation](https://code.google.com/archive/p/word2vec/)
1. [Deeplearning4j’s NLP framework](http://deeplearning4j.org/nlp) - Java implementation.
1. [deepnl](https://github.com/attardi/deepnl) - A Python library for NLP based on Deep Learning neural network architecture.
1. [PyTorch](http://pytorch.org/) - PyTorch is a deep learning framework that puts Python first. "Tensors and Dynamic neural networks in Python with strong GPU acceleration."

Papers
----
1. [Deep or shallow, NLP is breaking out](http://dl.acm.org/citation.cfm?id=2874915) - General overview of how Deep Learning is impacting NLP.
2. [Natural Language Processing from Research at Google](http://research.google.com/pubs/NaturalLanguageProcessing.html) - Not all Deep Learning (but mostly).
2. [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - The original word2vec paper.
3. [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
4. [Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)
5. [Context Dependent Recurrent Neural Network Language Model](http://www.msr-waypoint.com/pubs/176926/rnn_ctxt.pdf)
6. [Translation Modeling with Bidirectional Recurrent Neural Networks](https://www-i6.informatik.rwth-aachen.de/publications/download/936/SundermeyerMartinAlkhouliTamerWuebkerJoernNeyHermann--TranslationModelingwithBidirectionalRecurrentNeuralNetworks--2014.pdf)
7. [Contextual LSTM (CLSTM) models for Large scale NLP tasks](https://arxiv.org/abs/1602.06291)
8. [LSTM Neural Networks for Language Modeling](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.248.4448&rep=rep1&type=pdf)
9. [Exploring the Limits of Language Modeling](http://arxiv.org/pdf/1602.02410.pdf)
10. [Conversational Contextual Cues](https://arxiv.org/abs/1606.00372) - Models context and participants in conversations.
11. [Sequence to sequence learning with neural networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
12. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)
13. [Learning Character-level Representations for Part-of-Speech Tagging](http://jmlr.org/proceedings/papers/v32/santos14.pdf)
14. [Representation Learning for Text-level Discourse Parsing](http://www.cc.gatech.edu/~jeisenst/papers/ji-acl-2014.pdf)
15. [Fast and Robust Neural Network Joint Models for Statistical Machine Translation](http://acl2014.org/acl2014/P14-1/pdf/P14-1129.pdf)
16. [Parsing With Compositional Vector Grammars](http://www.socher.org/index.php/Main/ParsingWithCompositionalVectorGrammars)
17. [Smart Reply: Automated Response Suggestion for Email](https://arxiv.org/abs/1606.04870)
18. [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) - State-of-the-art performance in NER with bidirectional LSTM with a sequential conditional random layer and transition-based parsing with stack LSTMs.
19. [GloVe: Global Vectors for Word Representation](http://www-nlp.stanford.edu/pubs/glove.pdf) - A "count-based"/co-occurrence model to learn word embeddings.
20. [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449) - State-of-the-art syntactic constituency parsing using generic sequence-to-sequence approach.
22. Skip-Thought Vectors - "unsupervised learning of a generic, distributed sentence encoder"
    - [Paper](http://arxiv.org/abs/1506.06726)
    - [Code](https://github.com/ryankiros/skip-thoughts)


Blog Posts
----

1. [the morning paper: The amazing power of word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/) - Overview of word vectors.
1. [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017/)
2. [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
3. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
1. [Neural Language Modeling From Scratch](http://ofir.io/Neural-Language-Modeling-From-Scratch/?a=1)
4. [Machine Learning for Emoji Trends](http://instagram-engineering.tumblr.com/post/117889701472/emojineering-part-1-machine-learning-for-emoji)
5. [Teaching Robots to Feel: Emoji & Deep Learning](http://getdango.com/emoji-and-deep-learning.html)
6. [Computational Linguistics and Deep Learning](http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00239) - Opinion piece on how Deep Learning fits into the broader picture of text processing.
7. [Deep Learning NLP Best Practices](http://ruder.io/deep-learning-nlp-best-practices/index.html)


Researchers
----
1. [Christopher Manning](http://nlp.stanford.edu/manning/)
2. [Ali Ghodsi](https://uwaterloo.ca/data-science/)
3. [Richard Socher](http://www.socher.org/)
4. [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html)


Datasets
----
1. [Dataset from "One Billion Word Language Modeling Benchmark"](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz) - Almost 1B words, already pre-processed text.
1. [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html) - Fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences.
1. [Quora Question Pairs Dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) - Identify question pairs that have the same intent.

Specific Areas
----

## Deep Learning for NLP 
[Stanford Natural Language Processing](https://www.coursera.org/learn/nlp)  
Intro NLP course with videos. This has **no deep learning**. But it is a good primer for traditional nlp. Covers topics such as [sentence segmentation](https://github.com/diasks2/pragmatic_segmenter), word tokenizing, word normalization, n-grams, named entity recognition, part of speech tagging.  **Currently not available**  

[Stanford CS 224D: Deep Learning for NLP class](http://cs224d.stanford.edu/syllabus.html)  
[Richard Socher](https://scholar.google.com/citations?user=FaOcyfMAAAAJ&hl=en). (2016)  Class with syllabus, and slides.  
Videos: [2015 lectures](https://www.youtube.com/channel/UCsGC3XXF1ThHwtDo18d7WVw/videos) / [2016 lectures](https://www.youtube.com/playlist?list=PLcGUo322oqu9n4i0X3cRJgKyVy7OkDdoi)   

[A Primer on Neural Network Models for Natural Language Processing](https://www.jair.org/media/4992/live-4992-9623-jair.pdf)  
Yoav Goldberg. Submitted 9/2015, published 11/16. 75 page summary of state of the art.  

[Oxford Deep Learning for NLP class](http://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/)  
[Phil Blunsom](https://scholar.google.co.uk/citations?user=eJwbbXEAAAAJ&hl=en). (2017) Class by Deep Mind NLP Group.   
Lecture slides, videos, and practicals: [Github Repository](https://github.com/oxford-cs-deepnlp-2017)  

## Word Vectors
Resources about word vectors, aka word embeddings, and distributed representations for words.  
Word vectors are numeric representations of words where similar words have similar vectors. Word vectors are often used as input to deep learning systems. This process is sometimes called pretraining. 

[A neural probabilistic language model.](http://papers.nips.cc/paper/1839-a-neural-probabilistic-language-model.pdf)  
Bengio 2003. Seminal paper on word vectors.  

___
[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)  
Mikolov et al. 2013. Word2Vec generates word vectors in an unsupervised way by attempting to predict words from a corpus. Describes Continuous Bag-of-Words (CBOW) and Continuous Skip-gram models for learning word vectors.  
Skip-gram takes center word and predict outside words. Skip-gram is better for large datasets.  
CBOW - takes outside words and predict the center word. CBOW is better for smaller datasets.  

[Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
Mikolov et al. 2013. Learns vectors for phrases such as "New York Times." Includes optimizations for skip-gram: heirachical softmax, and negative sampling. Subsampling frequent words. (i.e. frequent words like "the" are skipped periodically to speed things up and improve vector for less frequently used words)  

[Linguistic Regularities in Continuous Space Word Representations](http://www.aclweb.org/anthology/N13-1090)  
[Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. 2013. Performs well on word similarity and analogy task.  Expands on famous example: King – Man + Woman = Queen  
[Word2Vec source code](https://code.google.com/p/word2vec/)  
[Word2Vec tutorial](http://tensorflow.org/tutorials/word2vec/index.html) in [TensorFlow](http://tensorflow.org/)  

[word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)  
Rong 2014  

Articles explaining word2vec: [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) and 
[The amazing power of word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)

___
[GloVe: Global vectors for word representation](http://nlp.stanford.edu/projects/glove/glove.pdf)  
Pennington, Socher, Manning. 2014. Creates word vectors and relates word2vec to matrix factorizations.  [Evalutaion section led to controversy](http://rare-technologies.com/making-sense-of-word2vec/) by [Yoav Goldberg](https://plus.google.com/114479713299850783539/posts/BYvhAbgG8T2)  
[Glove source code and training data](http://nlp.stanford.edu/projects/glove/) 

___
[Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606v1.pdf)  
Bojanowski, Grave, Joulin, Mikolov 2016  
[FastText Code](https://github.com/facebookresearch/fastText)  

## Sentiment Analysis
Thought vectors are numeric representations for sentences, paragraphs, and documents.  This concept is used for many text classification tasks such as sentiment analysis.      

[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)  
Socher et al. 2013.  Introduces Recursive Neural Tensor Network and dataset: "sentiment treebank."  Includes [demo site](http://nlp.stanford.edu/sentiment/
). Uses a parse tree.

[Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)  
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ), Mikolov. 2014.  Introduces Paragraph Vector. Concatenates and averages pretrained, fixed word vectors to create vectors for sentences, paragraphs and documents. Also known as paragraph2vec.  Doesn't use a parse tree.  
Implemented in [gensim](https://github.com/piskvorky/gensim/).  See [doc2vec tutorial](http://rare-technologies.com/doc2vec-tutorial/)

[Deep Recursive Neural Networks for Compositionality in Language](http://www.cs.cornell.edu/~oirsoy/files/nips14drsv.pdf)  
Irsoy & Cardie. 2014.  Uses Deep Recursive Neural Networks. Uses a parse tree.

[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://aclweb.org/anthology/P/P15/P15-1150.pdf)  
Tai et al. 2015  Introduces Tree LSTM. Uses a parse tree.

[Semi-supervised Sequence Learning](http://arxiv.org/pdf/1511.01432.pdf)  
Dai, Le 2015  
Approach: "We present two approaches that use unlabeled data to improve sequence learning with recurrent networks. The first approach is to predict what comes next in a sequence, which is a conventional language model in natural language processing.
The second approach is to use a sequence autoencoder..."  
Result: "With pretraining, we are able to train long short term memory recurrent networks up to a few hundred
timesteps, thereby achieving strong performance in many text classification tasks, such as IMDB, DBpedia and 20 Newsgroups."

[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)  
Joulin, Grave, Bojanowski, Mikolov 2016 Facebook AI Research.  
"Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation."  
[FastText blog](https://research.facebook.com/blog/fasttext/)  
[FastText Code](https://github.com/facebookresearch/fastText)  

## Neural Machine Translation
In 2014, neural machine translation (NMT) performance became comprable to state of the art statistical machine translation(SMT).  

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078v3.pdf) ([abstract](https://arxiv.org/abs/1406.1078))    
Cho et al. 2014 Breakthrough deep learning paper on machine translation. Introduces basic sequence to sequence model which includes two rnns, an encoder for input and a decoder for output.  

[Neural Machine Translation by jointly learning to align and translate](http://arxiv.org/pdf/1409.0473v6.pdf) ([abstract](https://arxiv.org/abs/1409.0473))     
Bahdanau, Cho, Bengio 2014.  
Implements attention mechanism. "Each time the proposed model generates a word in a translation, it
(soft-)searches for a set of positions in a source sentence where the most relevant information is
concentrated"  
Result: "comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation."  
[English to French Demo](http://104.131.78.120/)  

[On Using Very Large Target Vocabulary for Neural Machine Translation](https://arxiv.org/pdf/1412.2007v2.pdf)  
Jean, Cho, Memisevic, Bengio 2014.    
"we try replacing each [UNK] token with the aligned source word or its most likely translation determined by another word alignment model."  
Result: English -> German bleu score = 21.59 (target vocabulary of 50,000)    

[Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215v3.pdf)  
Sutskever, Vinyals, Le 2014.  ([nips presentation](http://research.microsoft.com/apps/video/?id=239083)). Uses seq2seq to generate translations.  
Result: English -> French bleu score = 34.8 (WMT’14 dataset)  
A key contribution is improvements from reversing the source sentences.  
[seq2seq tutorial](http://tensorflow.org/tutorials/seq2seq/index.html) in [TensorFlow](http://tensorflow.org/).   

[Addressing the Rare Word Problem in Neural Machine Translation](https://arxiv.org/pdf/1410.8206v4.pdf) ([abstract](https://arxiv.org/abs/1410.8206))  
Luong, Sutskever, Le, Vinyals, Zaremba 2014    
Replace UNK words with dictionary lookup.  
Result: English -> French BLEU score = 37.5.  

[Effective Approaches to Attention-based Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf)  
Luong, Pham, Manning. 2015  
2 models of attention: global and local.  
Result: English -> German 25.9 BLEU points  

[Context-Dependent Word Representation for Neural
Machine Translation](http://arxiv.org/pdf/1607.00578v1.pdf)  
Choi, Cho, Bengio 2016  
"we propose to contextualize the word embedding vectors using a nonlinear bag-of-words representation of the source sentence."  
"we propose to represent special tokens (such as numbers, proper nouns and acronyms) with typed symbols to facilitate translating those words that are not well-suited to be translated via continuous vectors."   

[Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](http://arxiv.org/abs/1609.08144)  
Wu et al. 2016  
[blog post](https://research.googleblog.com/2016/09/a-neural-network-for-machine.html)  
"WMT’14 English-to-French, our single model scores 38.95 BLEU"  
"WMT’14 English-to-German, our single model scores 24.17 BLEU"  

[Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/abs/1611.04558)  
Johnson et al. 2016  
[blog post](https://research.googleblog.com/2016/11/zero-shot-translation-with-googles.html)  
Translations between untrained language pairs.  

Google has started [rolling out NMT](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/) to it's production system, and it's a [significant improvement](http://www.nytimes.com/2016/12/14/magazine/the-great-ai-awakening.html?_r=0).  

[Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)  
Gehring et al. 2017 Facebook AI research 
[blog post](https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation/)  
Architecture: Convolutional sequence to sequence. ConvS2s.  
Results: "We outperform the accuracy of the deep LSTM setup of Wu et al. (2016) on both WMT'14 English-German and WMT'14 English-French translation at an order of magnitude faster speed, both on GPU and CPU."
  
[Facebook is transitioning entirely to neural machine translation](https://code.facebook.com/posts/289921871474277/transitioning-entirely-to-neural-machine-translation/)
  
[Transformer: A Novel Neural Network Architecture for Language Understanding](https://research.googleblog.com/2017/08/transformer-novel-neural-network.html)  
Arcitecture: Transformer, a T2T model introduced by Google in [Attention is all you need](https://arxiv.org/abs/1706.03762)  
Results: "we show that the Transformer outperforms both recurrent and convolutional models on academic English to German and English to French translation benchmarks."  
[T2T Source code](https://github.com/tensorflow/tensor2tensor)  
[T2T blog post](https://research.googleblog.com/2017/06/accelerating-deep-learning-research.html)   

[DeepL Translator](https://www.deepl.com/translator) claims to [outperform competitors](https://www.deepl.com/press.html) but doesn't disclose their architecture.
"Specific details of our network architecture will not be published at this time. DeepL Translator is based on a single, non-ensemble model."  
  
## Image Captioning
[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/pdf/1502.03044v3.pdf)  
Xu et al. 2015 Creates captions by feeding image into a CNN which feeds into hidden state of an RNN that generates the caption. At each time step the RNN outputs next word and the next location to pay attention to via a probability over grid locations. Uses 2 types of attention soft and hard. Soft attention uses gradient descent and backprop and is deterministic. Hard attention selects the element with highest probability. Hard attention uses reinforcement learning, rather than backprop and is stochastic.  

[Open source implementation in TensorFlow](https://research.googleblog.com/2016/09/show-and-tell-image-captioning-open.html)  

## Conversation modeling / Dialog
[Neural Responding Machine for Short-Text Conversation](http://arxiv.org/pdf/1503.02364v2.pdf)  
Shang et al. 2015  Uses Neural Responding Machine.  Trained on Weibo dataset.  Achieves one round conversations with 75% appropriate responses.  

[A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](http://arxiv.org/pdf/1506.06714v1.pdf)  
Sordoni et al. 2015.  Generates responses to tweets.   
Uses [Recurrent Neural Network Language Model (RLM) architecture
of (Mikolov et al., 2010).](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)  source code: [RNNLM Toolkit](http://www.rnnlm.org/)

[Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](http://arxiv.org/pdf/1507.04808v3.pdf)  
Serban, Sordoni, Bengio et al. 2015. Extends [hierarchical recurrent encoder-decoder](https://arxiv.org/abs/1507.02221) neural network (HRED).

[Attention with Intention for a Neural Network Conversation Model](http://arxiv.org/pdf/1510.08565v3.pdf)  
Yao et al. 2015 Architecture is three recurrent networks: an encoder, an intention network and a decoder.  

[A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](http://arxiv.org/pdf/1605.06069v3.pdf)  
Serban, Sordoni, Lowe, Charlin, Pineau, Courville, Bengio 2016  
Proposes novel architecture: VHRED.  Latent Variable Hierarchical Recurrent Encoder-Decoder  
Compares favorably against LSTM and HRED.  
___
[A Neural Conversation Model](http://arxiv.org/pdf/1506.05869v3.pdf)  
Vinyals, [Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ) 2015.  Uses LSTM RNNs to generate conversational responses. Uses [seq2seq framework](http://tensorflow.org/tutorials/seq2seq/index.html).  Seq2Seq was originally designed for machine translation and it "translates" a single sentence, up to around 79 words, to a single sentence response, and has no memory of previous dialog exchanges.  Used in Google [Smart Reply feature for Inbox](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html)  

[Incorporating Copying Mechanism in Sequence-to-Sequence Learning](http://arxiv.org/pdf/1603.06393v3.pdf)  
Gu et al. 2016 Proposes CopyNet, builds on seq2seq.  

[A Persona-Based Neural Conversation Model](http://arxiv.org/pdf/1603.06155v2.pdf)  
Li et al. 2016  Proposes persona-based models for handling the issue of speaker consistency in neural response generation. Builds on seq2seq.  

[Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/pdf/1606.01541v3.pdf)  
Li et al. 2016. Uses reinforcement learing to generate diverse responses. Trains 2 agents to chat with each other. Builds on seq2seq.   

[Adversarial Learning for Neural Dialogue Generation](https://arxiv.org/pdf/1701.06547.pdf)  
Li et al. 2017.  
[Source code for Li papers](https://github.com/jiweil/Neural-Dialogue-Generation)  
___
[Deep learning for chatbots](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/)  
Article summary of state of the art, and challenges for chatbots.  
[Deep learning for chatbots. part 2](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/)  
Implements a retrieval based dialog agent using dual encoder lstm with TensorFlow, based on the Ubuntu dataset [[paper](http://arxiv.org/pdf/1506.08909v3.pdf)] includes [source code](https://github.com/dennybritz/chatbot-retrieval/)  

[ParlAI](https://github.com/facebookresearch/ParlAI) A framework for training and evaluating AI models on a variety of openly available dialog datasets. Released by FaceBook.  

## Memory and Attention Models
Attention mechanisms allows the network to refer back to the input sequence, instead of forcing it to encode all information into one fixed-length vector.  - [Attention and Memory in Deep Learning and NLP](http://www.opendatascience.com/blog/attention-and-memory-in-deep-learning-and-nlp/)  

[Memory Networks](http://arxiv.org/pdf/1410.3916v10.pdf) Weston et. al 2014, and 
[End-To-End Memory Networks](http://arxiv.org/pdf/1503.08895v4.pdf) Sukhbaatar et. al 2015.  
Memory networks are implemented in [MemNN](https://github.com/facebook/MemNN).  Attempts to solve task of reason attention and memory.  
[Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/pdf/1502.05698v7.pdf)  
Weston 2015. Classifies QA tasks like single factoid, yes/no etc. Extends memory networks.  
[Evaluating prerequisite qualities for learning end to end dialog systems](http://arxiv.org/pdf/1511.06931.pdf)  
Dodge et. al 2015. Tests Memory Networks on 4 tasks including reddit dialog task.  
See [Jason Weston lecture on MemNN](https://www.youtube.com/watch?v=Xumy3Yjq4zk)  
  
[Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf)  
Graves, Wayne, Danihelka 2014.  
We extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes. The combined system is analogous to a Turing Machine or Von Neumann architecture but is differentiable end-toend, allowing it to be efficiently trained with gradient descent. Preliminary results demonstrate
that Neural Turing Machines can infer simple algorithms such as copying, sorting, and associative recall from input and output examples.
[Olah and Carter blog on NTM](http://distill.pub/2016/augmented-rnns/#neural-turing-machines)  

[Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/pdf/1503.01007v4.pdf)  
Joulin, Mikolov 2015. [Stack RNN source code](https://github.com/facebook/Stack-RNN) and [blog post](https://research.facebook.com/blog/1642778845966521/inferring-algorithmic-patterns-with-stack/)  


[Reasoning, Attention and Memory RAM workshop at NIPS 2015. slides included](http://www.thespermwhale.com/jaseweston/ram/)  

# Deep Learning for NLP resources

State of the art resources for NLP sequence modeling tasks such as machine translation, image captioning, and dialog.

[My notes on neural networks, rnn, lstm](https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md)  

Miscellaneous
----
1. [word2vec analogy demo](http://deeplearner.fz-qqq.net/)

-----
Contributing
----
Have anything in mind that you think is awesome and would fit in this list? Feel free to send a pull request!

-----
License
----

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Tarry Singh](http://www.linkedin.com/in/tarrysingh/) has waived all copyright and related or neighboring rights to this work.


