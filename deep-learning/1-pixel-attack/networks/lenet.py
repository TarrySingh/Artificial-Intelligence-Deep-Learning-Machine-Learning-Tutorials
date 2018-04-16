import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from networks.train_plot import PlotLearning

# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class LeNet:
    def __init__(self, epochs=200, batch_size=128, load_weights=True):
        self.name               = 'lenet'
        self.model_filename     = 'networks/models/lenet.h5'
        self.num_classes        = 10
        self.input_shape        = 32, 32, 3
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = 391
        self.weight_decay       = 0.0001
        self.log_filepath       = r'networks/models/lenet/'

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)
    
    def count_params(self):
        return self._model.count_params()

    def color_preprocessing(self, x_train, x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        return x_train, x_test

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay), input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay) ))
        model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay) ))
        model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay) ))
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def scheduler(self, epoch):
        if epoch <= 60:
            return 0.05
        if epoch <= 120:
            return 0.01
        if epoch <= 160:    
            return 0.002
        return 0.0004

    def train(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        # build network
        model = self.build_model()
        model.summary()

        # Save the best model during each training checkpoint
        checkpoint = ModelCheckpoint(self.model_filename,
                                    monitor='val_loss', 
                                    verbose=0,
                                    save_best_only= True,
                                    mode='auto')
        plot_callback = PlotLearning()
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)

        cbks = [checkpoint, plot_callback, tb_cb]

        # using real-time data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

        datagen.fit(x_train)

        # start traing 
        model.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
                            steps_per_epoch=self.iterations,
                            epochs=self.epochs,
                            callbacks=cbks,
                            validation_data=(x_test, y_test))
        # save model
        model.save(self.model_filename)

        self._model = model

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
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self._model.evaluate(x_test, y_test, verbose=0)[1]



