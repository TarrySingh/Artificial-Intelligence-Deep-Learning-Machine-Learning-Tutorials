load '../train/train.plot' 

set yrange [:60]
set ylabel 'perplexity'
plot 'model.step-devperplexity' t 'dev perplexity' w lines ls 2
