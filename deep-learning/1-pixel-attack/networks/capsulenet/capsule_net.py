from keras.layers import (
    Input,
    Conv2D,
    Activation,
    Dense,
    Flatten,
    Reshape,
    Dropout
)
from keras.layers.merge import add
from keras.regularizers import l2
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import optimizers
import numpy as np

from networks.capsulenet.capsule_layers import CapsuleLayer, PrimaryCapsule, Length,Mask
from networks.capsulenet.capsulenet import CapsNet as CapsNetv1
from networks.capsulenet.helper_function import load_cifar_10,load_cifar_100


def convolution_block(input,kernel_size=8,filters=16,kernel_regularizer=l2(1.e-4)):
    conv2 = Conv2D(filters=filters,kernel_size=kernel_size,kernel_regularizer=kernel_regularizer,
                    kernel_initializer="he_normal",padding="same")(input)
    norm = BatchNormalization(axis=3)(conv2)
    activation = Activation("relu")(norm)
    return activation    

def CapsNet(input_shape,n_class,n_route,n_prime_caps=32,dense_size = (512,1024)):
    conv_filter = 256
    n_kernel = 24
    primary_channel =64
    primary_vector = 9
    vector_dim = 9

    target_shape = input_shape

    input = Input(shape=input_shape)

    # TODO: try leaky relu next time
    conv1 = Conv2D(filters=conv_filter,kernel_size=n_kernel, strides=1, padding='valid', activation='relu',name='conv1',kernel_initializer="he_normal")(input)

    primary_cap = PrimaryCapsule(conv1,dim_vector=8, n_channels=64,kernel_size=9,strides=2,padding='valid')

    routing_layer = CapsuleLayer(num_capsule=n_class, dim_vector=vector_dim, num_routing=n_route,name='routing_layer')(primary_cap)

    output = Length(name='output')(routing_layer)

    y = Input(shape=(n_class,))
    masked = Mask()([routing_layer,y])
    
    x_recon = Dense(dense_size[0],activation='relu')(masked)

    for i in range(1,len(dense_size)):
        x_recon = Dense(dense_size[i],activation='relu')(x_recon)
    # Is there any other way to do  
    x_recon = Dense(target_shape[0]*target_shape[1]*target_shape[2],activation='relu')(x_recon)
    x_recon = Reshape(target_shape=target_shape,name='output_recon')(x_recon)

    return Model([input,y],[output,x_recon])

# why using 512, 1024 Maybe to mimic original 10M params?
def CapsNetv2(input_shape,n_class,n_route,n_prime_caps=32,dense_size = (512,1024)):
    conv_filter = 64
    n_kernel = 16
    primary_channel =64
    primary_vector = 12
    capsule_dim_size = 8

    target_shape = input_shape

    input = Input(shape=input_shape)

    # TODO: try leaky relu next time
    conv_block_1 = convolution_block(input,kernel_size=16,filters=64)
    primary_cap = PrimaryCapsule(conv_block_1,dim_vector=capsule_dim_size,n_channels=primary_channel,kernel_size=9,strides=2,padding='valid')    
    # Suppose this act like a max pooling 
    routing_layer = CapsuleLayer(num_capsule=n_class,dim_vector=capsule_dim_size*2,num_routing=n_route,name='routing_layer_1')(primary_cap)
    output = Length(name='output')(routing_layer)

    y = Input(shape=(n_class,))
    masked = Mask()([routing_layer,y])
    
    x_recon = Dense(dense_size[0],activation='relu')(masked)

    for i in range(1,len(dense_size)):
        x_recon = Dense(dense_size[i],activation='relu')(x_recon)
    # Is there any other way to do  
    x_recon = Dense(np.prod(target_shape),activation='relu')(x_recon)
    x_recon = Reshape(target_shape=target_shape,name='output_recon')(x_recon)

    # conv_block_2 = convolution_block(routing_layer)
    # b12_sum = add([conv_block_2,conv_block_1])

    return Model([input,y],[output,x_recon])

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(epochs=50,batch_size=64,mode=1):
    import numpy as np
    import os
    from keras import callbacks
    from keras.utils.vis_utils import plot_model
    if mode==1:
        num_classes = 10
        (x_train,y_train),(x_test,y_test) = load_cifar_10()
    else:
        num_classes = 100
        (x_train,y_train),(x_test,y_test) = load_cifar_100()
    model = CapsNetv1(input_shape=[32, 32, 3],
                        n_class=num_classes,
                        n_route=3)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model.summary()
    log = callbacks.CSVLogger('networks/models/results/capsule-cifar-'+str(num_classes)+'-log.csv')
    tb = callbacks.TensorBoard(log_dir='networks/models/results/tensorboard-capsule-cifar-'+str(num_classes)+'-logs',
                               batch_size=batch_size, histogram_freq=True)
    checkpoint = callbacks.ModelCheckpoint('networks/models/capsnet.h5',
                                           save_best_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # plot_model(model, to_file='models/capsule-cifar-'+str(num_classes)+'.png', show_shapes=True)

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.1],
                  metrics={'output_recon':'accuracy','output':'accuracy'})
    from networks.capsulenet.helper_function import data_generator

    generator = data_generator(x_train,y_train,batch_size)
    # Image generator significantly increase the accuracy and reduce validation loss
    model.fit_generator(generator,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        validation_data=([x_test, y_test], [y_test, x_test]),
                        epochs=epochs, verbose=1, max_q_size=100,
                        callbacks=[log,tb,checkpoint,lr_decay])
    
    return model

def test(epoch, mode=1):
    import matplotlib.pyplot as plt
    from PIL import Image
    from networks.capsulenet.helper_function import combine_images

    if mode == 1:
        num_classes =10
        _,(x_test,y_test) = load_cifar_10()
    else:
        num_classes = 100
        _,(x_test,y_test) = load_cifar_100()
    
    model = CapsNetv2(input_shape=[32, 32, 3],
                        n_class=num_classes,
                        n_route=3)
    model.load_weights('weights/capsule_weights/capsule-cifar-'+str(num_classes)+'weights-{:02d}.h5'.format(epoch)) 
    print("Weights loaded, start validation")   
    # model.load_weights('weights/capsule-weights-{:02d}.h5'.format(epoch))    
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-'*50)
    # Test acc: 0.7307
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save("results/real_and_recon.png")
    print('Reconstructed images are saved to ./results/real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("results/real_and_recon.png", ))
    plt.show()
