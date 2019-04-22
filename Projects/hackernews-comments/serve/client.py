#!/usr/bin/env python3
""" Client for TensorFlow serving.

Reads titles from STDIN, and writes comment samples to STDOUT.
"""

# adapted from https://github.com/OpenNMT/OpenNMT-tf/blob/master/examples/serving/ende_client.py

import sys

import argparse

import tensorflow as tf
import subprocess

import grpc

import subword_nmt.apply_bpe as apply_bpe

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

class Generator:
  def __init__(self,
               host,
               port,
               model_name,
               preprocessor,
               postprocessor,
               bpe_codes):
    channel = grpc.insecure_channel("%s:%d" % (host, port))
    self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    self.model_name = model_name

    self.preprocessor = preprocessor
    self.postprocessor = postprocessor
    with open(bpe_codes) as f:
      self.bpe = apply_bpe.BPE(f)

  def __call__(self, title, n=8, timeout=50.0):
    if self.preprocessor:
      # FIXME: Tried to reuse the process, but something seems to be buffering
      preprocessor = subprocess.Popen([self.preprocessor],
                                      stdout=subprocess.PIPE,
                                      stdin=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL)

      title_pp = preprocessor.communicate((title.strip() + '\n').encode())[0].decode('utf-8')
    else:
      title_pp = title

    title_bpe = self.bpe.segment_tokens(title_pp.strip().lower().split(' '))
    #print(title_bpe)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = self.model_name
    request.inputs['tokens'].CopyFrom(
      tf.make_tensor_proto([title_bpe] * n, shape=(n, len(title_bpe))))
    request.inputs['length'].CopyFrom(
      tf.make_tensor_proto([len(title_bpe)] * n, shape=(n,)))
    future = self.stub.Predict.future(request, timeout)

    result = future.result()

    batch_predictions = tf.make_ndarray(result.outputs["tokens"])
    batch_lengths = tf.make_ndarray(result.outputs["length"])
    batch_scores = tf.make_ndarray(result.outputs["log_probs"])

    hyps = []
    for (predictions, lengths, scores) in zip(batch_predictions, batch_lengths, batch_scores):
      # ignore </s>
      prediction = predictions[0][:lengths[0]-1]

      comment = ' '.join([token.decode('utf-8') for token in prediction])
      comment = comment.replace('@@ ', '')
      #comment = comment.replace('<NL>', '\n')

      # FIXME: Tried to reuse the process, but something seems to be buffering
      if self.postprocessor:
        postprocessor = subprocess.Popen([self.postprocessor],
                                         stdout=subprocess.PIPE,
                                         stdin=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL)
        prediction_ready = postprocessor.communicate((comment + '\n').encode())[0].decode('utf-8')
      else:
        prediction_ready = comment

      hyps.append((prediction_ready.strip(), float(scores[0])))

    #hyps.sort(key=lambda hyp: hyp[1] / len(hyp), reverse=True)

    return hyps

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--host', default='localhost', help='model server host')
  parser.add_argument('--port', type=int, default=9000, help='model server port')
  parser.add_argument('--model_name', required=True, help='model name')
  parser.add_argument('--preprocessor', help='tokenization script')
  parser.add_argument('--postprocessor', help='postprocessing script')
  parser.add_argument('--bpe_codes', required=True, help='BPE codes')
  parser.add_argument('-n', type=int, default=5, help='Number of comments to sample per title')

  args = parser.parse_args()
 
  generator = Generator(host=args.host,
                        port=args.port,
                        model_name=args.model_name,
                        preprocessor=args.preprocessor,
                        postprocessor=args.postprocessor,
                        bpe_codes=args.bpe_codes)

  for title in sys.stdin:
    hyps = generator(title, args.n)

    for prediction, score in hyps:
      sys.stdout.write('{}\t{}\n'.format(title.strip(), prediction))
  
if __name__ == "__main__":
  main()
