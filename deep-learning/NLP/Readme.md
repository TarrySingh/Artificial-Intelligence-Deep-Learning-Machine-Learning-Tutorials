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


# NLP-DATASETS -- all you can eat and burn on your GPUs 😅
-----

Alphabetical list of free/public domain datasets with text data for use in Natural Language Processing (NLP). Most stuff here is just raw unstructured text data, if you are looking for annotated corpora or Treebanks refer to the sources at the bottom.

## Datasets
*   [Apache Software Foundation Public Mail Archives](http://aws.amazon.com/de/datasets/apache-software-foundation-public-mail-archives/): all publicly available Apache Software Foundation mail archives as of July 11, 2011 (200 GB)

*   [Blog Authorship Corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm): consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. 681,288 posts and over 140 million words. (298 MB)

*   [Amazon Fine Food Reviews [Kaggle]](https://www.kaggle.com/snap/amazon-fine-food-reviews): consists of 568,454 food reviews Amazon users left up to October 2012. [Paper](http://i.stanford.edu/~julian/pdfs/www13.pdf). (240 MB)

*   [Amazon Reviews](https://snap.stanford.edu/data/web-Amazon.html): Stanford collection of 35 million amazon reviews. (11 GB)

*   [ArXiv](http://arxiv.org/help/bulk_data_s3): All the Papers on archive as fulltext (270 GB) + sourcefiles (190 GB).

*   [ASAP Automated Essay Scoring [Kaggle]](https://www.kaggle.com/c/asap-aes/data): For this competition, there are eight essay sets. Each of the sets of essays was generated from a single prompt. Selected essays range from an average length of 150 to 550 words per response. Some of the essays are dependent upon source information and others are not. All responses were written by students ranging in grade levels from Grade 7 to Grade 10. All essays were hand graded and were double-scored. (100 MB)

*   [ASAP Short Answer Scoring [Kaggle]](https://www.kaggle.com/c/asap-sas/data): Each of the data sets was generated from a single prompt. Selected responses have an average length of 50 words per response. Some of the essays are dependent upon source information and others are not. All responses were written by students primarily in Grade 10. All responses were hand graded and were double-scored. (35 MB)

*   [Classification of political social media](https://www.crowdflower.com/data-for-everyone/): Social media messages from politicians classified by content. (4 MB)

*   [CLiPS Stylometry Investigation (CSI) Corpus](http://www.clips.uantwerpen.be/datasets/csi-corpus): a yearly expanded corpus of student texts in two genres: essays and reviews. The purpose of this corpus lies primarily in stylometric research, but other applications are possible. (on request)

*   [ClueWeb09 FACC](http://lemurproject.org/clueweb09/FACC1/): [ClueWeb09](http://lemurproject.org/clueweb09/) with Freebase annotations (72 GB)

*   [ClueWeb11 FACC](http://lemurproject.org/clueweb12/FACC1/): [ClueWeb11](http://lemurproject.org/clueweb12/) with Freebase annotations (92 GB)

*   [Common Crawl Corpus](http://aws.amazon.com/de/datasets/common-crawl-corpus/): web crawl data composed of over 5 billion web pages (541 TB)

*   [Cornell Movie Dialog Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html): contains a large metadata-rich collection of fictional conversations extracted from raw movie scripts: 220,579 conversational exchanges between 10,292 pairs of movie characters, 617 movies (9.5 MB)

*   [Corporate messaging](http://aws.amazon.com/de/datasets/common-crawl-corpus/): A data categorization job concerning what corporations actually talk about on social media. Contributors were asked to classify statements as information (objective statements about the company or it’s activities), dialog (replies to users, etc.), or action (messages that ask for votes or ask users to click on links, etc.). (600 KB)

*   [Crosswikis](http://nlp.stanford.edu/data/crosswikis-data.tar.bz2/): English-phrase-to-associated-Wikipedia-article database. Paper. (11 GB) 

*   [DBpedia](http://aws.amazon.com/de/datasets/dbpedia-3-5-1/?tag=datasets%23keywords%23encyclopedic): a community effort to extract structured information from Wikipedia and to make this information available on the Web (17 GB)

*   [Death Row](http://www.tdcj.state.tx.us/death_row/dr_executed_offenders.html): last words of every inmate executed since 1984 online (HTML table)

*   [Del.icio.us](http://arvindn.livejournal.com/116137.html): 1.25 million bookmarks on delicious.com

*   [Disasters on social media](https://www.crowdflower.com/data-for-everyone/): 10,000 tweets with annotations whether the tweet referred to a disaster event (2 MB).

*   [Economic News Article Tone and Relevance](https://www.crowdflower.com/data-for-everyone/): News articles judged if relevant to the US economy and, if so, what the tone of the article was. Dates range from 1951 to 2014. (12 MB)

*   [Enron Email Data](http://aws.amazon.com/de/datasets/enron-email-data/): consists of 1,227,255 emails with 493,384 attachments covering 151 custodians (210 GB)

*   [Event Registry](http://eventregistry.org/): Free tool that gives real time access to news articles by 100.000 news publishers worldwide. [Has API](https://github.com/gregorleban/EventRegistry/). (query tool)

*   [Examiner.com - Spam Clickbait News Headlines [Kaggle]](https://www.kaggle.com/therohk/examine-the-examiner): 3 Million crowdsourced News headlines published by now defunct clickbait website The Examiner from 2010 to 2015. (200 MB)

*   [Federal Contracts from the Federal Procurement Data Center (USASpending.gov)](http://aws.amazon.com/de/datasets/federal-contracts-from-the-federal-procurement-data-center-usaspending-gov/): data dump of all federal contracts from the Federal Procurement Data Center found at USASpending.gov (180 GB)

*   [Flickr Personal Taxonomies](http://www.isi.edu/~lerman/downloads/flickr/flickr_taxonomies.html): Tree dataset of personal tags (40 MB)

*   [Freebase Data Dump](http://aws.amazon.com/de/datasets/freebase-data-dump/): data dump of all the current facts and assertions in Freebase (26 GB) 

*   [Freebase Simple Topic Dump](http://aws.amazon.com/de/datasets/freebase-simple-topic-dump/): data dump of the basic identifying facts about every topic in Freebase (5 GB)

*   [Freebase Quad Dump](http://aws.amazon.com/de/datasets/freebase-quad-dump/): data dump of all the current facts and assertions in Freebase (35 GB)

*   [GigaOM Wordpress Challenge [Kaggle]](https://www.kaggle.com/c/predict-wordpress-likes/data): blog posts, meta data, user likes (1.5 GB)

*   [Google Books Ngrams](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html): available also in hadoop format on amazon s3 (2.2 TB)

*   [Google Web 5gram](https://catalog.ldc.upenn.edu/LDC2006T13): contains English word n-grams and their observed frequency counts (24 GB)

*   [Gutenberg Ebook List](http://www.gutenberg.org/wiki/Gutenberg:Offline_Catalogs): annotated list of ebooks (2 MB)

*   [Hansards text chunks of Canadian Parliament](http://www.isi.edu/natural-language/download/hansard/): 1.3 million pairs of aligned text chunks (sentences or smaller fragments) from the official records (Hansards) of the 36th Canadian Parliament. (82 MB)

*   [Harvard Library](http://library.harvard.edu/open-metadata#Harvard-Library-Bibliographic-Dataset): over 12 million bibliographic records for materials held by the Harvard Library, including books, journals, electronic resources, manuscripts, archival materials, scores, audio, video and other materials. (4 GB)

*   [Hate speech identification](https://github.com/t-davidson/hate-speech-and-offensive-language): Contributors viewed short text and identified if it a) contained hate speech, b) was offensive but without hate speech, or c) was not offensive at all. Contains nearly 15K rows with three contributor judgments per text string. (3 MB)

*   [Hillary Clinton Emails [Kaggle]](https://www.kaggle.com/kaggle/hillary-clinton-emails): nearly 7,000 pages of Clinton's heavily redacted emails (12 MB)

*   [Home Depot Product Search Relevance [Kaggle]](https://www.kaggle.com/c/home-depot-product-search-relevance/data): contains a number of products and real customer search terms from Home Depot's website. The challenge is to predict a relevance score for the provided combinations of search terms and products. To create the ground truth labels, Home Depot has crowdsourced the search/product pairs to multiple human raters. (65 MB)

*   [Identifying key phrases in text](https://www.crowdflower.com/data-for-everyone/): Question/Answer pairs + context; context was judged if relevant to question/answer. (8 MB)

*   [Jeopardy](http://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/): archive of 216,930 past Jeopardy questions (53 MB)

*   [200k English plaintext jokes](https://github.com/taivop/joke-dataset): archive of 208,000 plaintext jokes from various sources.

*   [Machine Translation of European Languages](http://statmt.org/wmt11/translation-task.html#download): (612 MB)

*   [Material Safety Datasheets](http://aws.amazon.com/de/datasets/material-safety-data-sheets/): 230,000 Material Safety Data Sheets. (3 GB)

*   [Million News Headlines - ABC Australia [Kaggle]](https://www.kaggle.com/therohk/million-headlines): 1.3 Million News headlines published by ABC News Australia from 2003 to 2017. (56 MB)

*   [MCTest](http://research.microsoft.com/en-us/um/redmond/projects/mctest/index.html): a freely available set of 660 stories and associated questions intended for research on the machine comprehension of text; for question answering (1 MB)

*   [NEGRA](http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/negra-corpus.html): A Syntactically Annotated Corpus of German Newspaper Texts. Available for free for all Universities and non-profit organizations. Need to sign and send form to obtain. (on request)

*   [News Headlines of India - Times of India [Kaggle]](https://www.kaggle.com/therohk/india-headlines-news-dataset): 2.7 Million News Headlines with category published by Times of India from 2001 to 2017. (185 MB)

*   [News article / Wikipedia page pairings](https://www.crowdflower.com/data-for-everyone/): Contributors read a short article and were asked which of two Wikipedia articles it matched most closely. (6 MB)

*   [NIPS2015 Papers (version 2) [Kaggle]](https://www.kaggle.com/benhamner/nips-2015-papers/version/2): full text of all NIPS2015 papers (335 MB)

*   [NYTimes Facebook Data](http://minimaxir.com/2015/07/facebook-scraper/): all the NYTimes facebook posts (5 MB)

*   [One Week of Global News Feeds [Kaggle]](https://www.kaggle.com/therohk/global-news-week): News Event Dataset of 1.4 Million Articles published globally in 20 languages over one week of August 2017. (115 MB)

*   [Objective truths of sentences/concept pairs](https://www.crowdflower.com/data-for-everyone/): Contributors read a sentence with two concepts. For example “a dog is a kind of animal” or “captain can have the same meaning as master.” They were then asked if the sentence could be true and ranked it on a 1-5 scale. (700 KB)

*   [Open Library Data Dumps](https://openlibrary.org/developers/dumps): dump of all revisions of all the records in Open Library. (16 GB)

*   [Personae Corpus](http://www.clips.uantwerpen.be/datasets/personae-corpus): collected for experiments in Authorship Attribution and Personality Prediction. It consists of 145 Dutch-language essays by 145 different students. (on request)

*   [Reddit Comments](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/): every publicly available reddit comment as of july 2015. 1.7 billion comments (250 GB)

*   [Reddit Comments (May ‘15) [Kaggle]](https://www.kaggle.com/reddit/reddit-comments-may-2015): subset of above dataset (8 GB)

*   [Reddit Submission Corpus](https://www.reddit.com/r/datasets/comments/3mg812/full_reddit_submission_corpus_now_available_2006/): all publicly available Reddit submissions from January 2006 - August 31, 2015). (42 GB)

*   [Reuters Corpus](http://trec.nist.gov/data/reuters/reuters.html): a large collection of Reuters News stories for use in research and development of natural language processing, information retrieval, and machine learning systems. This corpus, known as "Reuters Corpus, Volume 1" or RCV1, is significantly larger than the older, well-known Reuters-21578 collection heavily used in the text classification community. Need to sign agreement and sent per post to obtain. (2.5 GB)

*   [SaudiNewsNet](https://github.com/ParallelMazen/SaudiNewsNet): 31,030 Arabic newspaper articles alongwith metadata, extracted from various online Saudi newspapers. (2 MB)

*   [SMS Spam Collection](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/): 5,574 English, real and non-enconded SMS messages, tagged according being legitimate (ham) or spam.  (200 KB) 

*   [SouthparkData](https://github.com/BobAdamsEE/SouthParkData): .csv files containing script information including: season, episode, character, & line. (3.6 MB)

*   [Stackoverflow](http://data.stackexchange.com/): 7.3 million stackoverflow questions + other stackexchanges (query tool)

*   [Twitter Cheng-Caverlee-Lee Scrape](https://archive.org/details/twitter_cikm_2010): Tweets from September 2009 - January 2010, geolocated. (400 MB)

*   [Twitter New England Patriots Deflategate sentiment](https://www.crowdflower.com/data-for-everyone/): Before the 2015 Super Bowl, there was a great deal of chatter around deflated footballs and whether the Patriots cheated. This data set looks at Twitter sentiment on important days during the scandal to gauge public sentiment about the whole ordeal. (2 MB)

*   [Twitter Progressive issues sentiment analysis](https://www.crowdflower.com/data-for-everyone/): tweets regarding a variety of left-leaning issues like legalization of abortion, feminism, Hillary Clinton, etc. classified if the tweets in question were for, against, or neutral on the issue (with an option for none of the above). (600 KB)

*   [Twitter Sentiment140](http://help.sentiment140.com/for-students/): Tweets related to brands/keywords. Website includes papers and research ideas. (77 MB)

*   [Twitter sentiment analysis: Self-driving cars](https://www.crowdflower.com/data-for-everyone/): contributors read tweets and classified them as very positive, slightly positive, neutral, slightly negative, or very negative. They were also prompted asked to mark if the tweet was not relevant to self-driving cars. (1 MB)

*   [Twitter Tokyo Geolocated Tweets](http://followthehashtag.com/datasets/200000-tokyo-geolocated-tweets-free-twitter-dataset/): 200K tweets from Tokyo. (47 MB)

*   [Twitter UK Geolocated Tweets](http://followthehashtag.com/datasets/170000-uk-geolocated-tweets-free-twitter-dataset/): 170K tweets from UK. (47 MB)

*   [Twitter USA Geolocated Tweets](http://followthehashtag.com/datasets/free-twitter-dataset-usa-200000-free-usa-tweets/): 200k tweets from the US (45MB)

*   [Twitter US Airline Sentiment [Kaggle]](https://www.kaggle.com/crowdflower/twitter-airline-sentiment): A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service"). (2.5 MB)

*   [U.S. economic performance based on news articles](https://www.crowdflower.com/data-for-everyone/): News articles headlines and excerpts ranked as whether relevant to U.S. economy. (5 MB)

*   [Urban Dictionary Words and Definitions [Kaggle]](https://www.kaggle.com/therohk/urban-dictionary-words-dataset): Cleaned CSV corpus of 2.6 Million of all Urban Dictionary words, definitions, authors, votes as of May 2016. (238 MB)

*   [Wesbury Lab Usenet Corpus](http://aws.amazon.com/de/datasets/the-westburylab-usenet-corpus/): anonymized compilation of postings from 47,860 English-language newsgroups from 2005-2010 (40 GB)

*   [Wesbury Lab Wikipedia Corpus](http://www.psych.ualberta.ca/~westburylab/downloads/westburylab.wikicorp.download.html) Snapshot of all the articles in the English part of the Wikipedia that was taken in April 2010. It was processed, as described in detail below, to remove all links and irrelevant material (navigation text, etc) The corpus is untagged, raw text. Used by [Stanford NLP](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=9060444488071171966&as_sdt=5) (1.8 GB).

*   [Wikipedia Extraction (WEX)](http://aws.amazon.com/de/datasets/wikipedia-extraction-wex/): a processed dump of english language wikipedia (66 GB)

*   [Wikipedia XML Data](http://aws.amazon.com/de/datasets/wikipedia-xml-data/): complete copy of all Wikimedia wikis, in the form of wikitext source and metadata embedded in XML. (500 GB)

*   [Yahoo! Answers Comprehensive Questions and Answers](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): Yahoo! Answers corpus as of 10/25/2007. Contains 4,483,032 questions and their answers. (3.6 GB)

*   [Yahoo! Answers consisting of questions asked in French](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): Subset of the Yahoo! Answers corpus from 2006 to 2015 consisting of 1.7 million questions posed in French, and their corresponding answers. (3.8 GB)

*   [Yahoo! Answers Manner Questions](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): subset of the Yahoo! Answers corpus from a 10/25/2007 dump, selected for their linguistic properties. Contains 142,627 questions and their answers. (104 MB)

*   [Yahoo! HTML Forms Extracted from Publicly Available Webpages](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): contains a small sample of pages that contain complex HTML forms, contains 2.67 million complex forms. (50+ GB)

*   [Yahoo! Metadata Extracted from Publicly Available Web Pages](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): 100 million triples of RDF data (2 GB)

*   [Yahoo N-Gram Representations](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): This dataset contains n-gram representations. The data may serve as a testbed for query rewriting task, a common problem in IR research as well as to word and sentence similarity task, which is common in NLP research. (2.6 GB)

*   [Yahoo! N-Grams, version 2.0](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): n-grams (n = 1 to 5), extracted from a corpus of 14.6 million documents (126 million unique sentences, 3.4 billion running words) crawled from over 12000 news-oriented sites (12 GB)

*   [Yahoo! Search Logs with Relevance Judgments](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): Annonymized Yahoo! Search Logs with Relevance Judgments (1.3 GB)

*   [Yahoo! Semantically Annotated Snapshot of the English Wikipedia](http://webscope.sandbox.yahoo.com/catalog.php?datatype=l): English Wikipedia dated from 2006-11-04 processed with a number of publicly-available NLP tools. 1,490,688 entries. (6 GB)

*   [Yelp](https://www.yelp.com/academic_dataset): including restaurant rankings and 2.2M reviews (on request)

*   [Youtube](https://www.reddit.com/r/datasets/comments/3gegdz/17_millions_youtube_videos_description/): 1.7 million youtube videos descriptions (torrent)

## Sources
*   [Awesome public datasets/NLP](https://github.com/caesar0301/awesome-public-datasets#natural-language) (includes more lists)
*   [AWS Public Datasets](http://aws.amazon.com/de/datasets/)
*   [CrowdFlower: Data for Everyone](https://www.crowdflower.com/data-for-everyone/) (lots of little surveys they conducted and data obtained by crowdsourcing for a specific task)
*   [Kaggle 1](https://www.kaggle.com/datasets), [2](https://www.kaggle.com/competitions) (make sure though that the kaggle competition data can be used outside of the competition!)
*   [Open Library](https://openlibrary.org/developers/dumps)
*   [Quora](https://www.quora.com/Datasets-What-are-the-major-text-corpora-used-by-computational-linguists-and-natural-language-processing-researchers-and-what-are-the-characteristics-biases-of-each-corpus) (mainly annotated corpora)
*   [/r/datasets](https://www.reddit.com/r/datasets) (endless list of datasets, most is scraped by amateurs though and not properly documented or licensed)
*   [rs.io](http://rs.io/100-interesting-data-sets-for-statistics/) (another big list)
*   [Stackexchange: Opendata](http://opendata.stackexchange.com/)
*   [Stanford NLP group](http://www-nlp.stanford.edu/links/statnlp.html) (mainly annotated corpora and TreeBanks or actual NLP tools)
*   [Yahoo! Webscope](http://webscope.sandbox.yahoo.com/) (also includes papers that use the data that is provided)



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


