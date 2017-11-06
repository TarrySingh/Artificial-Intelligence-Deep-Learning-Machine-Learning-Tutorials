import numpy as np
import tensorflow as tf

from config import cfg
from utils import load_mnist
from utils import save_images
from capsNet import CapsNet


if __name__ == '__main__':
    capsNet = CapsNet(is_training=cfg.is_training)
    tf.logging.info('Graph loaded')

    teX, teY = load_mnist(cfg.dataset, cfg.is_training)
    with capsNet.graph.as_default():
        sv = tf.train.Supervisor(logdir=cfg.logdir)
        # with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
            tf.logging.info('Restored')

            reconstruction_err = []
            for i in range(10000 // cfg.batch_size):
                start = i * cfg.batch_size
                end = start + cfg.batch_size
                recon_imgs = sess.run(capsNet.decoded, {capsNet.X: teX[start:end]})
                orgin_imgs = np.reshape(teX[start:end], (cfg.batch_size, -1))
                squared = np.square(recon_imgs - orgin_imgs)
                reconstruction_err.append(np.mean(squared))

                if i % 5 == 0:
                    imgs = np.reshape(recon_imgs, (cfg.batch_size, 28, 28, 1))
                    size = 6
                    save_images(imgs[0:size * size, :], [size, size], 'results/test_%03d.png' % i)
            print('test acc:')
            print((1. - np.mean(reconstruction_err)) * 100)