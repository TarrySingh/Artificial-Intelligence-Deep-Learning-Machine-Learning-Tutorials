load '../train/train.plot' 

plot '../train-wiki/model.step-trainlossavg10' t 'wikipedia train loss' w lines ls 1 dt '.', \
     'model.step-trainlossavg10' t 'hn train loss' w lines ls 2 dt '.', \
     'model-3.step-trainlossavg10' u ($1+197000):2 t 'hn train loss 3' w lines ls 4 dt '.', \
     '../train-wiki/model.step-devloss' t 'wikipedia dev loss' w lines ls 1, \
     'model.step-devloss' t 'hn dev loss' w lines ls 2, \
     'model-2.step-devloss' t 'hn dev loss 2' w lines ls 3, \
     'model-3.step-devloss' u ($1+197000):2 t 'hn dev loss 3' w lines ls 4
