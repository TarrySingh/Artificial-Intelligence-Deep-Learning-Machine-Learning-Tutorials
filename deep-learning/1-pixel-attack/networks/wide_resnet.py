import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras import optimizers
from keras import regularizers

from networks.train_plot import PlotLearning

# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class WideResNet:
    def __init__(self, epochs=200, batch_size=128, load_weights=True):
        self.name               = 'wide_resnet'
        self.model_filename     = 'networks/models/wide_resnet.h5'
        
        self.depth              = 16
        self.wide               = 8
        self.num_classes        = 10
        self.img_rows, self.img_cols = 32, 32
        self.img_channels       = 3
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = 391
        self.weight_decay       = 0.0005
        self.log_filepath       = r'networks/models/wide_resnet/'

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)
    
    def count_params(self):
        return self._model.count_params()

    def scheduler(self, epoch):
        if epoch <= 60:
            return 0.1
        if epoch <= 120:
            return 0.02
        if epoch <= 160:
            return 0.004
        return 0.0008

    def color_preprocessing(self, x_train,x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        return x_train, x_test

    def wide_residual_network(self, img_input,classes_num,depth,k):

        print('Wide-Resnet %dx%d' %(depth, k))
        n_filters  = [16, 16*k, 32*k, 64*k]
        n_stack    = (depth - 4) / 6
        in_filters = 16

        def conv3x3(x,filters):
            return Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',
            kernel_initializer=he_normal(),
            kernel_regularizer=regularizers.l2(self.weight_decay))(x)

        def residual_block(x,out_filters,increase_filter=False):
            if increase_filter:
                first_stride = (2,2)
            else:
                first_stride = (1,1)
            pre_bn   = BatchNormalization()(x)
            pre_relu = Activation('relu')(pre_bn)
            conv_1 = Conv2D(out_filters,kernel_size=(3,3),strides=first_stride,padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay))(pre_relu)
            bn_1   = BatchNormalization()(conv_1)
            relu1  = Activation('relu')(bn_1)
            conv_2 = Conv2D(out_filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay))(relu1)
            if increase_filter or in_filters != out_filters:
                projection = Conv2D(out_filters,kernel_size=(1,1),strides=first_stride,padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay))(x)
                block = add([conv_2, projection])
            else:
                block = add([conv_2,x])
            return block

        def wide_residual_layer(x,out_filters,increase_filter=False):
            x = residual_block(x,out_filters,increase_filter)
            in_filters = out_filters
            for _ in range(1,int(n_stack)):
                x = residual_block(x,out_filters)
            return x

        x = conv3x3(img_input,n_filters[0])
        x = wide_residual_layer(x,n_filters[1])
        x = wide_residual_layer(x,n_filters[2],increase_filter=True)
        x = wide_residual_layer(x,n_filters[3],increase_filter=True)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        return x

    def train(self):
        # load data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        # build network
        img_input = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
        output = self.wide_residual_network(img_input,self.num_classes,self.depth,self.wide)
        resnet = Model(img_input, output)
        resnet.summary()
        
        # set optimizer
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # set callback
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(self.scheduler)
        checkpoint = ModelCheckpoint(self.model_filename, 
                monitor='val_loss', verbose=0, save_best_only= True, mode='auto')
        plot_callback = PlotLearning()
        cbks = [change_lr,tb_cb,checkpoint,plot_callback]

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

        datagen.fit(x_train)

        # start training
        resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=self.batch_size),
                            steps_per_epoch=self.iterations,
                            epochs=self.epochs,
                            callbacks=cbks,
                            validation_data=(x_test, y_test))
        resnet.save(self.model_filename)

        self.param_count = self._model.count_params()
        self._model = resnet

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