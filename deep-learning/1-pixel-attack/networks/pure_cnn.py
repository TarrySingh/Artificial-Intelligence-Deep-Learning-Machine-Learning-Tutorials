import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.constraints import maxnorm
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

from networks.train_plot import PlotLearning

# A pure CNN model from https://arxiv.org/pdf/1412.6806.pdf
# Code taken from https://github.com/09rohanchopra/cifar10
class PureCnn:
    def __init__(self, epochs=350, batch_size=128, load_weights=True):
        self.name               = 'pure_cnn'
        self.model_filename     = 'networks/models/pure_cnn.h5'
        self.num_classes        = 10
        self.input_shape        = 32, 32, 3
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.learn_rate         = 1.0e-4
        self.log_filepath       = r'networks/models/pure_cnn/'

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

    def pure_cnn_network(self, input_shape):
        model = Sequential()
        
        model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', input_shape=input_shape))    
        model.add(Dropout(0.2))
        
        model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))  
        model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2))    
        model.add(Dropout(0.5))
        
        model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))    
        model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
        model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2))    
        model.add(Dropout(0.5))    
        
        model.add(Conv2D(192, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Conv2D(192, (1, 1),padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(10, (1, 1), padding='valid'))

        model.add(GlobalAveragePooling2D())
        
        model.add(Activation('softmax'))

        return model
    
    def train(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        model = self.pure_cnn_network(self.input_shape)
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

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

        datagen.fit(x_train)

        model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
                    optimizer=Adam(lr=self.learn_rate), # Adam optimizer with 1.0e-4 learning rate
                    metrics = ['accuracy']) # Metrics to be evaluated by the model

        model.fit_generator(datagen.flow(x_train, y_train, batch_size = self.batch_size),
                            epochs = self.epochs,
                            validation_data= (x_test, y_test),
                            callbacks=cbks,
                            verbose=1)

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