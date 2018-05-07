#!/u/subramas/miniconda2/bin/python
"""Main script to run things"""
import sys

sys.path.append('/u/subramas/Research/nmt-pytorch/')

from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from evaluate import evaluate_model
import math
import numpy as np
import logging
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)
experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


print 'Reading data ...'

src, trg = read_nmt_data(
    src=config['data']['src'],
    config=config,
    trg=config['data']['trg']
)

src_test, trg_test = read_nmt_data(
    src=config['data']['test_src'],
    config=config,
    trg=config['data']['test_trg']
)

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
src_vocab_size = len(src['word2id'])
trg_vocab_size = len(trg['word2id'])

logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Source Language : %s ' % (config['model']['src_lang']))
logging.info('Target Language : %s ' % (config['model']['trg_lang']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_trg']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (1))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['model']['n_layers_trg']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words in src ' % (src_vocab_size))
logging.info('Found %d words in trg ' % (trg_vocab_size))

weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[trg['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()

if config['model']['seq2seq'] == 'vanilla':

    model = Seq2Seq(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

elif config['model']['seq2seq'] == 'attention':

    model = Seq2SeqAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        ctx_hidden_dim=config['model']['dim'],
        attention_mode='dot',
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

elif config['model']['seq2seq'] == 'fastattention':

    model = Seq2SeqFastAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()

if load_dir:
    model.load_state_dict(torch.load(
        open(load_dir)
    ))

# __TODO__ Make this more flexible for other learning methods.
if config['training']['optimizer'] == 'adam':
    lr = config['training']['lrate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
    optimizer = optim.Adadelta(model.parameters())
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['lrate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

for i in xrange(1000):
    losses = []
    for j in xrange(0, len(src['data']), batch_size):

        input_lines_src, _, lens_src, mask_src = get_minibatch(
            src['data'], src['word2id'], j,
            batch_size, max_length, add_start=True, add_end=True
        )
        input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
            trg['data'], trg['word2id'], j,
            batch_size, max_length, add_start=True, add_end=True
        )

        decoder_logit = model(input_lines_src, input_lines_trg)
        optimizer.zero_grad()

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, trg_vocab_size),
            output_lines_trg.view(-1)
        )
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if j % config['management']['monitor_loss'] == 0:
            logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (
                i, j, np.mean(losses))
            )
            losses = []

        if (
            config['management']['print_samples'] and
            j % config['management']['print_samples'] == 0
        ):
            word_probs = model.decode(
                decoder_logit
            ).data.cpu().numpy().argmax(axis=-1)

            output_lines_trg = output_lines_trg.data.cpu().numpy()
            for sentence_pred, sentence_real in zip(
                word_probs[:5], output_lines_trg[:5]
            ):
                sentence_pred = [trg['id2word'][x] for x in sentence_pred]
                sentence_real = [trg['id2word'][x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted : %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real : %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')

        if j % config['management']['checkpoint_freq'] == 0:

            logging.info('Evaluating model ...')
            bleu = evaluate_model(
                model, src, src_test, trg,
                trg_test, config, verbose=False,
                metric='bleu',
            )

            logging.info('Epoch : %d Minibatch : %d : BLEU : %.5f ' % (i, j, bleu))

            logging.info('Saving model ...')

            torch.save(
                model.state_dict(),
                open(os.path.join(
                    save_dir,
                    experiment_name + '__epoch_%d__minibatch_%d' % (i, j) + '.model'), 'wb'
                )
            )

    bleu = evaluate_model(
        model, src, src_test, trg,
        trg_test, config, verbose=False,
        metric='bleu',
    )

    logging.info('Epoch : %d : BLEU : %.5f ' % (i, bleu))

    torch.save(
        model.state_dict(),
        open(os.path.join(
            save_dir,
            experiment_name + '__epoch_%d' % (i) + '.model'), 'wb'
        )
    )
