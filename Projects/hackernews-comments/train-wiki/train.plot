load '../train/train.plot' 

plot 'model.step-trainloss' t 'train loss' w lines ls 1, \
     'model.step-trainlossavg10' t 'train loss, moving average' w lines ls 3, \
     'model.step-devloss' t 'dev loss' w lines ls 2, \
