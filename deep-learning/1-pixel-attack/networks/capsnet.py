from keras.utils import to_categorical
import numpy as np

from networks.capsulenet.capsule_net import CapsNetv1, train as train_net

# Capsule Network taken from https://github.com/theblackcat102/dynamic-routing-capsule-cifar
class CapsNet:
    def __init__(self, epochs=200, batch_size=128, load_weights=True):
        self.name               = 'capsnet'
        self.model_filename     = 'networks/models/capsnet.h5'
        self.num_classes        = 10
        self.input_shape        = 32, 32, 3
        self.num_routes         = 3
        self.batch_size         = batch_size
        self.epochs             = epochs

        self._model = CapsNetv1(input_shape=self.input_shape,
                        n_class=self.num_classes,
                        n_route=self.num_routes)

        if load_weights:
            try:
                self._model.load_weights(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)

    def count_params(self):
        return self._model.count_params()

    def train(self):
        self._model = train_net()

    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        label = to_categorical(np.repeat([0], len(img)), self.num_classes) # Don't care what label it is, just needs to be fed into the network
        processed = self.color_process(img)
        input_ = [processed, label]
        pred, _ = self._model.predict(input_, batch_size=self.batch_size)
        return pred
    
    def predict_one(self, img):
        return self.predict(img)[0]
