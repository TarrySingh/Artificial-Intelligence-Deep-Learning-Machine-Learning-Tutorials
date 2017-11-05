#! /bin/env Rscript

library(ggplot2)

data<-read.csv(file='results/accuracy.csv',header=T)
png('results/accuracy.png')
p<-ggplot(data,aes(x=step,y=test_acc))
p + geom_line(color='red') + ylim(c(0.97,1))
dev.off()