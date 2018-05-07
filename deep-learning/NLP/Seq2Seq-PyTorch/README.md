# Sequence to Sequence models with PyTorch

This repository contains implementations of Sequence to Sequence (Seq2Seq) models in PyTorch

At present it has implementations for : 

    * Vanilla Sequence to Sequence models

    * Attention based Sequence to Sequence models from https://arxiv.org/abs/1409.0473 and https://arxiv.org/abs/1508.04025

    * Faster attention mechanisms using dot products between the **final** encoder and decoder hidden states

    * Sequence to Sequence autoencoders (experimental)

## Sequence to Sequence models

A vanilla sequence to sequence model presented in https://arxiv.org/abs/1409.3215, https://arxiv.org/abs/1406.1078 consits of using a recurrent neural network such as an LSTM (http://dl.acm.org/citation.cfm?id=1246450) or GRU (https://arxiv.org/abs/1412.3555) to encode a sequence of words or characters in a *source* language into a fixed length vector representation and then deocoding from that representation using another RNN in the *target* language.

![Sequence to Sequence](/images/Seq2Seq.png)

An extension of sequence to sequence models that incorporate an attention mechanism was presented in https://arxiv.org/abs/1409.0473 that uses information from the RNN hidden states in the source language at each time step in the deocder RNN. This attention mechanism significantly improves performance on tasks like machine translation. A few variants of the attention model for the task of machine translation have been presented in https://arxiv.org/abs/1508.04025.

![Sequence to Sequence with attention](/images/Seq2SeqAttention.png)

The repository also contains a simpler and faster variant of the attention mechanism that doesn't attend over the hidden states of the encoder at each time step in the deocder. Instead, it computes the a single batched dot product between all the hidden states of the decoder and encoder once after the decoder has processed all inputs in the target. This however comes at a minor cost in model performance. One advantage of this model is that it is possible to use the cuDNN LSTM in the attention based decoder as well since the attention is computed after running through all the inputs in the decoder.

## Results on English - French WMT14

The following presents the model architecture and results obtained when training on the WMT14 English - French dataset. The training data is the english-french bitext from Europral-v7. The validation dataset is newstest2011

The model was trained with following configuration

    * Source and target word embedding dimensions - 512

    * Source and target LSTM hidden dimensions - 1024

    * Encoder - 2 Layer Bidirectional LSTM

    * Decoder - 1 Layer LSTM

    * Optimization - ADAM with a learning rate of 0.0001 and batch size of 80

    * Decoding - Greedy decoding (argmax)


| Model | BLEU | Train Time Per Epoch |
| ------------- | ------------- | ------------- |
| Seq2Seq | 11.82 | 2h 50min |
| Seq2Seq FastAttention | 18.89 | 3h 45min |
| Seq2Seq Attention | 22.60 | 4h 47min |

Times reported are using a Pre 2016 Nvidia GeForce Titan X

## Running

To run, edit the config file and execute python nmt.py --config <your_config_file>

NOTE: This only runs on a GPU for now.