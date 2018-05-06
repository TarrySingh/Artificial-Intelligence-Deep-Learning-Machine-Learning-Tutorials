# ConvWave: Searching for Gravitational Waves with Fully Convolutional Neural Nets

**TL;DR:** 

We propose a method based on fully convolutional neurals nets to detect and localize simulated GW signals in arbitrarily long stretches of real detector noise, using multiple channels in parallel.



**Abstract:**

The first detection of gravitational waves (GWs) from a binary black hole merger in 2015 was a milestone in modern physics, and just recently awarded with the Nobel Prize. However, despite the unparalleled sensitivity of the LIGO detectors, there still exist challenges in the analysis of the recorded data. 

We apply ConvWave, a dilated, fully convolutional neural net directly on the time series strain data to localize simulated GW signals from black hole mergers in real, non-Gaussian background measurements from the LIGO detectors. ConvWave performs well on simulated signals with masses and distances chosen from ranges that contain the estimated parameters of all previously detected real events. It efficiently runs on strain data of arbitrary length from any number of detectors in real time. Through our proposed evaluation approach, it has the potential to develop into a complementary trigger generator in the existing LIGO search pipeline.