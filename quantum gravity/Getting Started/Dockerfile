FROM andrewosh/binder-base
MAINTAINER Jonah Kanner <jonah.kanner@ligo.org>


USER root
RUN apt-get update
RUN apt-get install -y libhdf5-dev
RUN apt-get install -y build-essential
RUN pip install cython
RUN pip install h5py
#RUN mkdir /home/main/notebook; 
#WORKDIR /home/main/notebook
#RUN wget https://losc.ligo.org/s/events/GW150914/H-H1_LOSC_4_V1-1126259446-32.hdf5 && \
#	wget https://losc.ligo.org/s/events/GW150914/L-L1_LOSC_4_V1-1126259446-32.hdf5 && \
#	wget https://losc.ligo.org/s/events/GW150914/H-H1_LOSC_16_V1-1126259446-32.hdf5 && \
#	wget https://losc.ligo.org/s/events/GW150914/L-L1_LOSC_16_V1-1126259446-32.hdf5 && \
#	wget https://losc.ligo.org/s/events/GW150914/GW150914_4_NR_waveform.txt

# Adopted from Kyle Cranmer <kyle.cranmer@nyu.edu> 