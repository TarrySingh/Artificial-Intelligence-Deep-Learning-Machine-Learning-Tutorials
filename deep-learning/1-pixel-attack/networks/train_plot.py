from matplotlib import pyplot as plt
from IPython.display import clear_output
import keras

# Live loss plot as the network trains
# https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
class PlotLearning(keras.callbacks.Callback):
    def __init__(self, clear_on_begin=False):
        self.clear_on_begin = clear_on_begin
        self.reset()
    
    def on_train_begin(self, logs={}):
        if (self.clear_on_begin):
            self.reset()
    
    def reset(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        if (self.i < 3):
            return
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        
        clear_output(wait=True)
        
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(['train', 'val'], loc='best')
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend(['train', 'val'], loc='best')
        
        plt.show()