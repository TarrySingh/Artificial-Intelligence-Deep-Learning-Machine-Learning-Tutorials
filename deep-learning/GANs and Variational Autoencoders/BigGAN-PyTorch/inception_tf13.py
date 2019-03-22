''' Tensorflow inception score code
Derived from https://github.com/openai/improved-gan
Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
THIS CODE REQUIRES TENSORFLOW 1.3 or EARLIER to run in PARALLEL BATCH MODE 

To use this code, run sample.py on your model with --sample_npz, and then 
pass the experiment name in the --experiment_name.
This code also saves pool3 stats to an npz file for FID calculation
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile
import math
from tqdm import tqdm, trange
from argparse import ArgumentParser

import numpy as np
from six.moves import urllib
import tensorflow as tf

MODEL_DIR = ''
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

def prepare_parser():
  usage = 'Parser for TF1.3- Inception Score scripts.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--experiment_name', type=str, default='',
    help='Which experiment''s samples.npz file to pull and evaluate')
  parser.add_argument(
    '--experiment_root', type=str, default='samples',
    help='Default location where samples are stored (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=500,
    help='Default overall batchsize (default: %(default)s)')
  return parser


def run(config):
  # Inception with TF1.3 or earlier.
  # Call this function with list of images. Each of elements should be a 
  # numpy array with values ranging from 0 to 255.
  def get_inception_score(images, splits=10):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
    bs = config['batch_size']
    with tf.Session() as sess:
      preds, pools = [], []
      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
      for i in trange(n_batches):
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred, pool = sess.run([softmax, pool3], {'ExpandDims:0': inp})
        preds.append(pred)
        pools.append(pool)
      preds = np.concatenate(preds, 0)
      scores = []
      for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
      return np.mean(scores), np.std(scores), np.squeeze(np.concatenate(pools, 0))
  # Init inception
  def _init_inception():
    global softmax, pool3
    if not os.path.exists(MODEL_DIR):
      os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
        MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session() as sess:
      pool3 = sess.graph.get_tensor_by_name('pool_3:0')
      ops = pool3.graph.get_operations()
      for op_idx, op in enumerate(ops):
        for o in op.outputs:
          shape = o.get_shape()
          shape = [s.value for s in shape]
          new_shape = []
          for j, s in enumerate(shape):
            if s == 1 and j == 0:
              new_shape.append(None)
            else:
              new_shape.append(s)
          o._shape = tf.TensorShape(new_shape)
      w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
      logits = tf.matmul(tf.squeeze(pool3), w)
      softmax = tf.nn.softmax(logits)

  # if softmax is None: # No need to functionalize like this.
  _init_inception()

  fname = '%s/%s/samples.npz' % (config['experiment_root'], config['experiment_name'])
  print('loading %s ...'%fname)
  ims = np.load(fname)['x']
  import time
  t0 = time.time()
  inc_mean, inc_std, pool_activations = get_inception_score(list(ims.swapaxes(1,2).swapaxes(2,3)), splits=10)
  t1 = time.time()
  print('Saving pool to numpy file for FID calculations...')
  np.savez('%s/%s/TF_pool.npz' % (config['experiment_root'], config['experiment_name']), **{'pool_mean': np.mean(pool_activations,axis=0), 'pool_var': np.cov(pool_activations, rowvar=False)})
  print('Inception took %3f seconds, score of %3f +/- %3f.'%(t1-t0, inc_mean, inc_std))
def main():
  # parse command line and run
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()