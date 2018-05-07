#!/u/subramas/miniconda2/bin/python
"""Main script to run things"""
import sys

sys.path.append('/u/subramas/Research/nmt-pytorch/')

from data_utils import read_nmt_data, get_autoencode_minibatch, read_config, hyperparam_string
from model import Seq2Seq, Seq2SeqAutoencoder
from evaluate import evaluate_model, evaluate_autoencode_model
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

src, _ = read_nmt_data(
    src=config['data']['src'],
    trg=None
)

src_test, _ = read_nmt_data(
    src=config['data']['test_src'],
    trg=None
)

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
src_vocab_size = len(src['word2id'])

logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Source Language : %s ' % (config['model']['src_lang']))
logging.info('Target Language : %s ' % (config['model']['src_lang']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_trg']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (config['model']['n_layers_trg']))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['data']['batch_size']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words in src ' % (src_vocab_size))

weight_mask = torch.ones(src_vocab_size).cuda()
weight_mask[src['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()

model = Seq2SeqAutoencoder(
    src_emb_dim=config['model']['dim_word_src'],
    trg_emb_dim=config['model']['dim_word_trg'],
    src_vocab_size=src_vocab_size,
    src_hidden_dim=config['model']['dim'],
    trg_hidden_dim=config['model']['dim'],
    batch_size=batch_size,
    bidirectional=config['model']['bidirectional'],
    pad_token_src=src['word2id']['<pad>'],
    nlayers=config['model']['n_layers_src'],
    nlayers_trg=config['model']['n_layers_trg'],
    dropout=0.,
).cuda()

if load_dir:
    model.load_state_dict(torch.load(
        open(load_dir)
    ))


def clip_gradient(model, clip):
    """Compute a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

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

torch.save(
    model.state_dict(),
    open(os.path.join(
        save_dir,
        experiment_name + 'epoch_0.model'), 'wb'
    )
)

bleu = evaluate_autoencode_model(
    model, src, src_test, config, verbose=False,
    metric='bleu',
)
logging.info('Epoch : %d : BLEU : %.5f ' % (0, bleu))


for i in xrange(1000):
    losses = []
    for j in xrange(0, len(src['data']), batch_size):

        input_lines_src, output_lines_src, lens_src, mask_src = get_autoencode_minibatch(
            src['data'], src['word2id'], j,
            batch_size, max_length, add_start=True, add_end=True
        )

        decoder_logit = model(input_lines_src)
        optimizer.zero_grad()

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, src_vocab_size),
            output_lines_src.view(-1)
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
            output_lines_trg = input_lines_src.data.cpu().numpy()
            for sentence_pred, sentence_real in zip(
                word_probs[:5], output_lines_trg[:5]
            ):
                sentence_pred = [src['id2word'][x] for x in sentence_pred]
                sentence_real = [src['id2word'][x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info(' '.join(sentence_pred))
                logging.info('-----------------------------------------------')
                logging.info(' '.join(sentence_real))
                logging.info('===============================================')

    bleu = evaluate_autoencode_model(
        model, src, src_test, config, verbose=False,
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
