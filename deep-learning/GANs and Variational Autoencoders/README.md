# Delving deep into Generative Adversarial Networks (GANs) 	


## A curated, quasi-exhaustive list of state-of-the-art publications and resources about Generative Adversarial Networks (GANs) and their applications.

### Background
Generative models are models that can learn to create data that is similar to data that we give them. One of the most promising approaches of those models are Generative Adversarial Networks (GANs), a branch of unsupervised machine learning implemented by a system of two neural networks competing against each other in a zero-sum game framework. They were first introduced by Ian Goodfellow et al. in 2014. This repository aims at presenting an elaborate list of the state-of-the-art works on the field of Generative Adversarial Networks since their introduction in 2014.

<p align="center">
  <img src="https://raw.githubusercontent.com/GKalliatakis/Delving-deep-into-GANs/master/GAN2.gif?raw=true"/><br>
  Image taken from http://multithreaded.stitchfix.com/blog/2016/02/02/a-fontastic-voyage/<br>
</p>

**_This is going to be an evolving repo and I will keep updating it (at least twice monthly) so make sure you have starred and forked this repository before moving on !_**

---

### :link: Contents
* [Opening Publication](#pushpin-opening-publication)
* [State-of-the-art papers](#clipboard-state-of-the-art-papers-descending-order-based-on-google-scholar-citations)
* [Theory](#notebook_with_decorative_cover-theory)
* [Presentations](#nut_and_bolt-presentations)
* [Courses](#books-courses--tutorials--blogs-webpages-unless-other-is-stated)
* [Code / Resources / Models](#package-resources--models-descending-order-based-on-github-stars)
* [Frameworks & Libraries](#electric_plug-frameworks--libraries-descending-order-based-on-github-stars)
---

### :busts_in_silhouette: Contributing
Contributions are welcome !! If you have any suggestions (missing or new papers, missing repos or typos) you can pull a request or start a discussion.

---


### :pushpin: Opening Publication 	
Generative Adversarial Nets (GANs) (2014) [[pdf]](https://arxiv.org/pdf/1406.2661v1.pdf)  [[presentation]](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf) [[code]](https://github.com/goodfeli/adversarial) [[video]](https://www.youtube.com/watch?v=HN9NRhm9waY)

---

### :clipboard: State-of-the-art papers (Descending order based on Google Scholar Citations)

S/N|Paper|Year|Citations
:---:|:---:|:---:|:---:
|1|Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGANs)  [[pdf]](https://arxiv.org/pdf/1511.06434v2.pdf)|2015|423
|2|Explaining and Harnessing Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)|2014|285
|3|Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (LAPGAN)  [[pdf]](https://arxiv.org/pdf/1506.05751.pdf)|2015|252
|4|Semi-Supervised Learning with Deep Generative Models  [[pdf]](https://arxiv.org/pdf/1406.5298v2.pdf )|2014|246
|5|Conditional Generative Adversarial Nets (CGAN)  [[pdf]](https://arxiv.org/pdf/1411.1784v1.pdf)|2014|125
|6|Improved Techniques for Training GANs  [[pdf]](https://arxiv.org/pdf/1606.03498v1.pdf )|2016|99
|7|Context Encoders: Feature Learning by Inpainting  [[pdf]](https://arxiv.org/pdf/1604.07379)|2016|96
|8|Deep multi-scale video prediction beyond mean square error  [[pdf]](https://arxiv.org/pdf/1511.05440.pdf)|2015|94
|9|Generative Adversarial Text to Image Synthesis  [[pdf]](https://arxiv.org/pdf/1605.05396)|2016|94
|10|Adversarial Autoencoders  [[pdf]](https://arxiv.org/pdf/1511.05644.pdf)|2015|91
|11|InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1606.03657)|2016|86
|12|Autoencoding beyond pixels using a learned similarity metric (VAE-GAN)  [[pdf]](https://arxiv.org/pdf/1512.09300.pdf)|2015|80
|13|Wasserstein GAN (WGAN)  [[pdf]](https://arxiv.org/pdf/1701.07875.pdf)|2017|80
|14|Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)  [[pdf]](https://arxiv.org/pdf/1611.07004.pdf)|2016|79
|15|Generative Moment Matching Networks  [[pdf]](https://arxiv.org/pdf/1502.02761.pdf)|2015|78
|16|Energy-based Generative Adversarial Network (EBGAN)  [[pdf]](https://arxiv.org/pdf/1609.03126.pdf)|2016|76
|17|Conditional Image Generation with PixelCNN Decoders  [[pdf]](https://arxiv.org/pdf/1606.05328.pdf)|2015|69
|18|Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)  [[pdf]](https://arxiv.org/pdf/1609.04802.pdf)|2016|68
|19|Adversarial Feature Learning (BiGAN)  [[pdf]](https://arxiv.org/pdf/1605.09782v6.pdf)|2016|59
|20|f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization  [[pdf]](https://arxiv.org/pdf/1606.00709.pdf)|2016|55
|21|Generating Images with Perceptual Similarity Metrics based on Deep Networks   [[pdf]](https://arxiv.org/pdf/1602.02644)|2016|54
|22|Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1602.02697.pdf)|2016|50
|23|Adversarially Learned Inference (ALI)  [[pdf]](https://arxiv.org/pdf/1606.00704.pdf)|2016|49
|24|Generative Visual Manipulation on the Natural Image Manifold (iGAN)  [[pdf]](https://arxiv.org/pdf/1609.03552.pdf)|2016|47
|25|Improving Variational Inference with Inverse Autoregressive Flow  [[pdf]](https://arxiv.org/pdf/1606.04934)|2016|47
|26|Generative Image Modeling using Style and Structure Adversarial Networks (S^2GAN)  [[pdf]](https://arxiv.org/pdf/1603.05631.pdf)|2016|41
|27|Unsupervised Learning for Physical Interaction through Video Prediction   [[pdf]](https://arxiv.org/pdf/1605.07157)|2016|41
|28|Generating Videos with Scene Dynamics (VGAN)  [[pdf]](http://web.mit.edu/vondrick/tinyvideo/paper.pdf)|2016|41
|29|Training generative neural networks via Maximum Mean Discrepancy optimization  [[pdf]](https://arxiv.org/pdf/1505.03906.pdf)|2015|39
|30|Conditional generative adversarial nets for convolutional face generation [[pdf]](https://pdfs.semanticscholar.org/42f6/f5454dda99d8989f9814989efd50fe807ee8.pdf)|2014|37
|31|Towards Principled Methods for Training Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1701.04862.pdf)|2017|37
|32|Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks (CatGAN)  [[pdf]](https://arxiv.org/pdf/1511.06390.pdf)|2015|35
|33|Generating images with recurrent adversarial networks  [[pdf]](https://arxiv.org/pdf/1602.05110.pdf)|2016|33
|34|Attend, infer, repeat: Fast scene understanding with generative models  [[pdf]](https://arxiv.org/pdf/1603.08575.pdf)|2016|31
|35|Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (3D-GAN)  [[pdf]](https://arxiv.org/pdf/1610.07584)|2016|31
|36|Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks (MGAN)  [[pdf]](https://arxiv.org/pdf/1604.04382.pdf)|2016|30
|37|Coupled Generative Adversarial Networks (CoGAN)  [[pdf]](https://arxiv.org/pdf/1606.07536)|2016|30
|38|Conditional Image Synthesis with Auxiliary Classifier GANs (AC-GAN)  [[pdf]](https://arxiv.org/pdf/1610.09585.pdf)|2016|30
|39|Unrolled Generative Adversarial Networks (Unrolled GAN)  [[pdf]](https://arxiv.org/pdf/1611.02163.pdf)|2016|29
|40|Synthesizing the preferred inputs for neurons in neural networks via deep generator networks   [[pdf]](https://arxiv.org/pdf/1605.09304)|2016|28
|41|StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1612.03242.pdf)|2016|28
|42|SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient    [[pdf]](https://arxiv.org/pdf/1609.05473.pdf)|2016|28
|43|Amortised MAP Inference for Image Super-resolution (AffGAN)  [[pdf]](https://arxiv.org/pdf/1610.04490.pdf)|2016|27
|44|Learning in Implicit Generative Models  [[pdf]](https://arxiv.org/pdf/1610.03483.pdf)|2016|25
|45|Unsupervised Cross-Domain Image Generation (DTN)  [[pdf]](https://arxiv.org/pdf/1611.02200.pdf)|2016|24
|46|Semantic Image Inpainting with Perceptual and Contextual Losses   [[pdf]](https://arxiv.org/pdf/1607.07539.pdf)|2016|23
|47|Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space (PPGN)  [[pdf]](https://arxiv.org/pdf/1612.00005.pdf)|2016|22
|48|VIME: Variational Information Maximizing Exploration  [[pdf]](https://arxiv.org/pdf/1605.09674)|2016|21
|49|Generative Adversarial Imitation Learning  [[pdf]](https://arxiv.org/pdf/1606.03476)|2016|20
|50|Learning What and Where to Draw (GAWWN)  [[pdf]](https://arxiv.org/pdf/1610.02454v1.pdf)|2016|20
|51|Pixel-Level Domain Transfer   [[pdf]](https://arxiv.org/pdf/1603.07442)|2016|16
|52|Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)  [[pdf]](https://arxiv.org/pdf/1703.10593.pdf)|2017|16
|53|Neural Photo Editing with Introspective Adversarial Networks (IAN)  [[pdf]](https://arxiv.org/pdf/1609.07093.pdf)|2016|15
|54|Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN) by Apple  [[pdf]](https://arxiv.org/pdf/1612.07828v1.pdf)|2016|15
|55|BEGAN: Boundary Equilibrium Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1703.10717.pdf)|2017|15
|56|On the Quantitative Analysis of Decoder-Based Generative Models  [[pdf]](https://arxiv.org/pdf/1611.04273.pdf)|2016|14
|57|Full Resolution Image Compression with Recurrent Neural Networks [[pdf]](https://arxiv.org/pdf/1608.05148)|2016|13
|58|Semantic Segmentation using Adversarial Networks   [[pdf]](https://arxiv.org/pdf/1611.08408.pdf)|2016|12
|59|Learning to Protect Communications with Adversarial Neural Cryptography [[pdf]](https://arxiv.org/pdf/1610.06918)|2016|11
|60|Adversarial Discriminative Domain Adaptation  [[pdf]](https://arxiv.org/pdf/1702.05464)|2017|10
|61|Generalization and Equilibrium in Generative Adversarial Nets (GANs)  [[pdf]](https://arxiv.org/pdf/1703.00573)|2017|9
|62|Generalization and Equilibrium in Generative Adversarial Nets (MIX+GAN)  [[pdf]](https://arxiv.org/pdf/1703.00573.pdf)|2016|9
|63|Mode Regularized Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1612.02136)|2016|8
|64|Learning a Driving Simulator [[pdf]](https://arxiv.org/pdf/1608.01230)|2016|8
|65|Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro [[pdf]](https://arxiv.org/pdf/1701.07717)|2017|8
|66|Connecting Generative Adversarial Networks and Actor-Critic Methods  [[pdf]](https://arxiv.org/pdf/1610.01945.pdf)|2016|8
|67|Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (DiscoGAN) [[pdf]](https://arxiv.org/pdf/1703.05192.pdf)|2017|8
|68|Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities (LS-GAN)  [[pdf]](https://arxiv.org/pdf/1701.06264.pdf)|2017|8
|69|Cooperative Training of Descriptor and Generator Network  [[pdf]](https://arxiv.org/pdf/1609.09408)|2016|6
|70|Invertible Conditional GANs for image editing (IcGAN)  [[pdf]](https://arxiv.org/pdf/1611.06355.pdf)|2016|6
|71|Generative Multi-Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1611.01673.pdf)|2016|6
|72|Boundary-Seeking Generative Adversarial Networks (BS-GAN)  [[pdf]](https://arxiv.org/pdf/1702.08431.pdf)|2017|6
|73|AdaGAN: Boosting Generative Models  [[pdf]](https://arxiv.org/pdf/1701.02386.pdf)|2017|6
|74|Stacked Generative Adversarial Networks (SGAN)  [[pdf]](https://arxiv.org/pdf/1612.04357.pdf)|2016|5
|75|Adversarial Attacks on Neural Network Policies [[pdf]](https://arxiv.org/pdf/1702.02284.pdf)|2017|5
|76|LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation   [[pdf]](https://arxiv.org/pdf/1703.01560.pdf)|2017|5
|77|Contextual RNN-GANs for Abstract Reasoning Diagram Generation (Context-RNN-GAN)  [[pdf]](https://arxiv.org/pdf/1609.09444.pdf)|2016|5
|78|Disentangled Representation Learning GAN for Pose-Invariant Face Recognition   [[pdf]](http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf)|2017|5
|79|Temporal Generative Adversarial Nets (TGAN)  [[pdf]](https://arxiv.org/pdf/1611.06624.pdf)|2016|4
|80|C-RNN-GAN: Continuous recurrent neural networks with adversarial training    [[pdf]](https://arxiv.org/pdf/1611.09904.pdf)|2016|4
|81|McGan: Mean and Covariance Feature Matching GAN    [[pdf]](https://arxiv.org/pdf/1702.08398.pdf)|2017|4
|82|Imitating Driver Behavior with Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1701.06699)|2017|3
|83|DualGAN: Unsupervised Dual Learning for Image-to-Image Translation    [[pdf]](https://arxiv.org/pdf/1704.02510.pdf)|2017|3
|84|Unsupervised Image-to-Image Translation with Generative Adversarial Networks   [[pdf]](https://arxiv.org/pdf/1701.02676.pdf)|2017|3
|85|Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN (MalGAN)  [[pdf]](https://arxiv.org/pdf/1702.05983.pdf)|2016|3
|86|Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks (SSL-GAN)  [[pdf]](https://arxiv.org/pdf/1611.06430.pdf)|2016|3
|87|Improved generator objectives for GANs  [[pdf]](https://arxiv.org/pdf/1612.02780.pdf)|2017|3
|88|3D Shape Induction from 2D Views of Multiple Objects (PrGAN)  [[pdf]](https://arxiv.org/pdf/1612.05872.pdf)|2016|3
|89| :new: Adversarial Deep Structural Networks for Mammographic Mass Segmentation  [[pdf]](https://arxiv.org/abs/1612.05970)|2017|3
|90|Learning to Generate Images of Outdoor Scenes from Attributes and Semantic Layouts (AL-CGAN)  [[pdf]](https://arxiv.org/pdf/1612.00215.pdf)|2016|2
|91|SEGAN: Speech Enhancement Generative Adversarial Network    [[pdf]](https://arxiv.org/pdf/1703.09452.pdf)|2017|2
|92|SeGAN: Segmenting and Generating the Invisible   [[pdf]](https://arxiv.org/pdf/1703.10239.pdf)|2017|2
|93|Robust LSTM-Autoencoders for Face De-Occlusion in the Wild   [[pdf]](https://arxiv.org/pdf/1612.08534.pdf)|2016|2
|94|Inverting The Generator Of A Generative Adversarial Network  [[pdf]](https://arxiv.org/pdf/1611.05644)|2016|2
|95|Ensembles of Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1612.00991.pdf)|2016|2
|96|Precise Recovery of Latent Vectors from Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.04782)|2016|2
|97|RenderGAN: Generating Realistic Labeled Data    [[pdf]](https://arxiv.org/pdf/1611.01331.pdf)|2016|2
|98|Texture Synthesis with Spatial Generative Adversarial Networks (SGAN)  [[pdf]](https://arxiv.org/pdf/1611.08207.pdf)|2016|2
|99|Least Squares Generative Adversarial Networks (LSGAN)  [[pdf]](https://pdfs.semanticscholar.org/0bbc/35bdbd643fb520ce349bdd486ef2c490f1fc.pdf)|2017|1
|100|Adversarial Training For Sketch Retrieval (SketchGAN)  [[pdf]](https://arxiv.org/pdf/1607.02748.pdf)|2016|1
|101|SAD-GAN: Synthetic Autonomous Driving using Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1611.08788.pdf)|2017|1
|102|Message Passing Multi-Agent GANs (MPM-GAN)  [[pdf]](https://arxiv.org/pdf/1612.01294.pdf)|2017|1
|103|Improved Training of Wasserstein GANs (WGAN-GP)  [[pdf]](https://arxiv.org/pdf/1704.00028.pdf)|2017|1
|104|Deep and Hierarchical Implicit Models (Bayesian GAN)  [[pdf]](https://arxiv.org/pdf/1702.08896.pdf)|2017|1
|105|A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection   [[pdf]](https://arxiv.org/pdf/1704.03414.pdf)|2017|1
|106|Maximum-Likelihood Augmented Discrete Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.07983)|2017|1
|107|Simple Black-Box Adversarial Perturbations for Deep Networks  [[pdf]](https://arxiv.org/pdf/1612.06299)|2016|1
|108| :new: Unsupervised Diverse Colorization via Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.06674.pdf)|2017|1
|109|ArtGAN: Artwork Synthesis with Conditional Categorial GANs   [[pdf]](https://arxiv.org/pdf/1702.03410.pdf)|2017|0
|110|GP-GAN: Towards Realistic High-Resolution Image Blending   [[pdf]](https://arxiv.org/pdf/1703.07195.pdf)|2017|0
|111|Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (AnoGAN)  [[pdf]](https://arxiv.org/pdf/1703.05921.pdf)|2017|0
|112|Generative Adversarial Nets with Labeled Data by Activation Maximization (AMGAN)  [[pdf]](https://arxiv.org/pdf/1703.02000.pdf)|2017|0
|113|MAGAN: Margin Adaptation for Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1704.03817.pdf)|2017|0
|114|CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training   [[pdf]](https://arxiv.org/pdf/1703.10155.pdf)|2017|0
|115|Multi-Agent Diverse Generative Adversarial Networks (MAD-GAN)  [[pdf]](https://arxiv.org/pdf/1704.02906.pdf)|2017|0
|116|Image De-raining Using a Conditional Generative Adversarial Network (ID-CGAN)  [[pdf]](https://arxiv.org/pdf/1701.05957)|2017|0
|117|Generative Mixture of Networks  [[pdf]](https://arxiv.org/pdf/1702.03307.pdf)|2017|0
|118|Generative Temporal Models with Memory  [[pdf]](https://arxiv.org/pdf/1702.04649.pdf)|2017|0
|119|Stopping GAN Violence: Generative Unadversarial Networks  [[pdf]](https://arxiv.org/pdf/1703.02528.pdf)|2016|0
|120|Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking (GoGAN)  [[pdf]](https://arxiv.org/pdf/1704.04865.pdf)|2017|0
|121|Deep Unsupervised Representation Learning for Remote Sensing Images (MARTA-GAN)    [[pdf]](https://arxiv.org/pdf/1612.08879.pdf)|2017|0
|122|Generating Multi-label Discrete Electronic Health Records using Generative Adversarial Networks (MedGAN)  [[pdf]](https://arxiv.org/pdf/1703.06490.pdf)|2017|0
|123|Semi-Latent GAN: Learning to generate and modify facial images from attributes (SL-GAN)   [[pdf]](https://arxiv.org/pdf/1704.02166.pdf)|2017|0
|124|TAC-GAN - Text Conditioned Auxiliary Classifier Generative Adversarial Network    [[pdf]](https://arxiv.org/pdf/1703.06412.pdf)|2017|0
|125|Triple Generative Adversarial Nets (Triple-GAN)  [[pdf]](https://arxiv.org/pdf/1703.02291.pdf)|2017|0
|126|Image Generation and Editing with Variational Info Generative Adversarial Networks (ViGAN)  [[pdf]](https://arxiv.org/pdf/1701.04568.pdf)|2016|0
|127|Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis (TP-GAN)  [[pdf]](https://arxiv.org/pdf/1704.04086.pdf)|2017|0
|128|Generative Adversarial Networks as Variational Training of Energy Based Models (VGAN)  [[pdf]](https://arxiv.org/pdf/1611.01799.pdf)|2017|0
|129|SalGAN: Visual Saliency Prediction with Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1701.01081.pdf)|2016|0
|130|WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images    [[pdf]](https://arxiv.org/pdf/1702.07392.pdf)|2017|0
|131|Multi-view Generative Adversarial Networks (MV-BiGAN)  [[pdf]](https://arxiv.org/pdf/1611.02019.pdf)|2017|0
|132|Recurrent Topic-Transition GAN for Visual Paragraph Generation (RTT-GAN)  [[pdf]](https://arxiv.org/pdf/1703.07022.pdf)|2017|0
|133|Generative face completion   [[pdf]](https://arxiv.org/pdf/1704.05838)|2016|0
|134|MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation using 1D and 2D Conditions   [[pdf]](https://arxiv.org/pdf/1703.10847.pdf)|2016|0
|135|Multi-View Image Generation from a Single-View   [[pdf]](https://arxiv.org/pdf/1704.04886)|2016|0
|136|Towards Large-Pose Face Frontalization in the Wild   [[pdf]](https://arxiv.org/pdf/1704.06244)|2016|0
|137|Adversarial Training Methods for Semi-Supervised Text Classification  [[pdf]](https://arxiv.org/abs/1605.07725)|2016|0
|138|An Adversarial Regularisation for Semi-Supervised Training of Structured Output Neural Networks  [[pdf]](https://arxiv.org/pdf/1702.02382)|2017|0
|139|Associative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1611.06953)|2017|0
|140|Generative Adversarial Parallelization  [[pdf]](https://arxiv.org/pdf/1612.04021)|2015|0
|141|Generative Adversarial Residual Pairwise Networks for One Shot Learning  [[pdf]](https://arxiv.org/pdf/1703.08033)|2017|0
|142|Generative Adversarial Structured Networks  [[pdf]](https://sites.google.com/site/nips2016adversarial/WAT16_paper_14.pdf)|2017|0
|143|On the effect of Batch Normalization and Weight Normalization in Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.03971)|2017|0
|144|Softmax GAN [[pdf]](https://arxiv.org/pdf/1704.06191)|2017|0
|145|Adversarial Networks for the Detection of Aggressive Prostate Cancer [[pdf]](https://arxiv.org/pdf/1702.08014)|2017|0
|146|Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation [[pdf]](https://arxiv.org/pdf/1705.00389.pdf)|2017|0
|147|Age Progression / Regression by Conditional Adversarial Autoencoder [[pdf]](https://arxiv.org/pdf/1702.08423)|2017|0
|148|Auto-painter: Cartoon Image Generation from Sketch by Using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1705.01908.pdf)|2017|0
|149|Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.01705)|2017|0
|150|Outline Colorization through Tandem Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.08834.pdf)|2017|0
|151|Supervised Adversarial Networks for Image Saliency Detection [[pdf]](https://arxiv.org/pdf/1704.07242)|2017|0
|152|Towards Diverse and Natural Image Descriptions via a Conditional GAN [[pdf]](https://arxiv.org/pdf/1703.06029.pdf?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)|2017|0
|153|Reconstruction of three-dimensional porous media using generative adversarial neural networks [[pdf]](https://arxiv.org/pdf/1704.03225)|2017|0
|154|Steganographic Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1703.05502)|2017|0
|155|Generative Cooperative Net for Image Generation and Data Augmentation  [[pdf]](https://arxiv.org/pdf/1705.02887.pdf)|2017|0
|156|The Space of Transferable Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1704.03453)|2017|0
|157|Deep Generative Adversarial Compression Artifact Removal  [[pdf]](https://arxiv.org/pdf/1704.02518)|2017|0
|158|Adversarial Generator-Encoder Networks  [[pdf]](https://arxiv.org/pdf/1704.02304)|2017|0
|159|Training Triplet Networks with GAN  [[pdf]](https://arxiv.org/pdf/1704.02227)|2017|0
|160|Universal Adversarial Perturbations Against Semantic Image Segmentation  [[pdf]](https://arxiv.org/pdf/1704.05712)|2017|0
|161|Learning Representations of Emotional Speech with Deep Convolutional Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1705.02394)|2017|0
|162|CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1705.02355)|2017|0
|163|Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN  [[pdf]](https://arxiv.org/pdf/1705.03387)|2017|0
|164|Geometric GAN  [[pdf]](https://arxiv.org/pdf/1705.02894.pdf)|2017|0
|165|Face Super-Resolution Through Wasserstein GANs  [[pdf]](https://arxiv.org/pdf/1705.02438)|2017|0
|166|Training Triplet Networks with GAN  [[pdf]](https://arxiv.org/pdf/1704.02227)|2017|0
|167|Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks  [[pdf]](https://arxiv.org/pdf/1704.01155)|2017|0
|168|Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1704.00849.pdf)|2017|0
|169|Adversarial Image Perturbation for Privacy Protection--A Game Theory Perspective  [[pdf]](https://arxiv.org/pdf/1703.09471)|2017|0
|170|Adversarial Transformation Networks: Learning to Generate Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1703.09387)|2017|0
|171|SCAN: Structure Correcting Adversarial Network for Chest X-rays Organ Segmentation  [[pdf]](https://arxiv.org/pdf/1703.08770)|2017|0
|172|Adversarial Examples for Semantic Segmentation and Object Detection  [[pdf]](https://arxiv.org/pdf/1703.08603.pdf)|2017|0
|173|GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data  [[pdf]](https://arxiv.org/pdf/1705.04932.pdf)|2017|0
|174|Generative Adversarial Networks for Multimodal Representation Learning in Video Hyperlinking [[pdf]](https://arxiv.org/abs/1705.05103)|2017|0
|175| Learning Texture Manifolds with the Periodic Spatial GAN [[pdf]](https://arxiv.org/abs/1705.06566)|2017|0
|176|Continual Learning in Generative Adversarial Nets [[pdf]](https://arxiv.org/abs/1705.08395)|2017|0
|177|Flow-GAN: Bridging implicit and prescribed learning in generative models [[pdf]](https://arxiv.org/abs/1705.08868)|2017|0
|178|How to Train Your DRAGAN [[pdf]](https://arxiv.org/abs/1705.07215)|2017|0
|179|Improved Semi-supervised Learning with GANs using Manifold Invariances [[pdf]](https://arxiv.org/abs/1705.08850)|2017|0
|180|From source to target and back: symmetric bi-directional adaptive GAN [[pdf]](https://arxiv.org/abs/1705.08824)|2017|0
|181|Semantically Decomposing the Latent Spaces of Generative Adversarial Networks (SD-GAN) [[pdf]](https://arxiv.org/abs/1705.07904)|2017|0
|182|Conditional CycleGAN for Attribute Guided Face Image Generation [[pdf]](https://arxiv.org/abs/1705.09966)|2017|0
|183|Good Semi-supervised Learning that Requires a Bad GAN [[pdf]](https://arxiv.org/abs/1705.09783)|2017|0
|184|Stabilizing Training of Generative Adversarial Networks through Regularization [[pdf]](https://arxiv.org/abs/1705.09367)|2017|0
|185|Bayesian GAN [[pdf]](https://arxiv.org/abs/1705.09558)|2017|0
|186|MMD GAN: Towards Deeper Understanding of Moment Matching Network [[pdf]](https://arxiv.org/pdf/1705.08584.pdf)|2017|0
|187|Relaxed Wasserstein with Applications to GANs (RWGAN ) [[pdf]](https://arxiv.org/abs/1705.07164)|2017|0
|188|VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning () [[pdf]](https://arxiv.org/abs/1705.07761)|2017|0
|189|Weakly Supervised Generative Adversarial Networks for 3D Reconstruction  [[pdf]](https://arxiv.org/abs/1705.10904)|2017|0
|190|Adversarial Generation of Natural Language [[pdf]](https://arxiv.org/abs/1705.10929)|2017|0
|191|Objective-Reinforced Generative Adversarial Networks (ORGAN) [[pdf]](https://arxiv.org/pdf/1705.10843.pdf)|2017|0
|192|SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation  [[pdf]](https://arxiv.org/abs/1706.01805)|2017|0
|193|Language Generation with Recurrent Generative Adversarial Networks without Pre-training  [[pdf]](https://arxiv.org/abs/1706.01399)|2017|0
|194|DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data  [[pdf]](https://arxiv.org/abs/1706.02071)|2017|0
|195|Depth Structure Preserving Scene Image Generation  [[pdf]](https://arxiv.org/abs/1706.00212)|2017|0
|196| From source to target and back: symmetric bi-directional adaptive GAN [[pdf]](https://arxiv.org/abs/1705.08824)|2017|0
|197|Synthesizing Filamentary Structured Images with GANs  [[pdf]](https://arxiv.org/abs/1706.02185)|2017|0
|198|TextureGAN: Controlling Deep Image Synthesis with Texture Patches  [[pdf]](https://arxiv.org/abs/1706.02823)|2017|0
|199|Generate Identity-Preserving Faces by Generative Adversarial Networks  [[pdf]](https://arxiv.org/abs/1706.03227)|2017|0
|200|Style Transfer for Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN  [[pdf]](https://arxiv.org/abs/1706.03319)|2017|0
|201|Gradient descent GAN optimization is locally stable  [[pdf]](https://arxiv.org/abs/1706.04156)|2017|0
|202|Neural Face Editing with Intrinsic Image Disentangling  [[pdf]](https://arxiv.org/abs/1704.04131)|2017|0
|203|Interactive 3D Modeling with a Generative Adversarial Network [[pdf]](https://arxiv.org/abs/1706.05170)|2017|0
|204| Perceptual Generative Adversarial Networks for Small Object Detection [[pdf]](https://arxiv.org/abs/1706.05274)|2017|0
|205|CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms [[pdf]](https://arxiv.org/pdf/1706.07068.pdf)|2017|0
|206| :new: Perceptual Adversarial Networks for Image-to-Image Transformation  [[pdf]](https://arxiv.org/pdf/1706.09138)|2017|0
|207| :new: Retinal Vessel Segmentation in Fundoscopic Images with Generative Adversarial Networks  [[pdf]](https://arxiv.org/abs/1706.09318)|2017|0
|208| :new: Auto-Encoder Guided GAN for Chinese Calligraphy Synthesis  [[pdf]](https://arxiv.org/abs/1706.08789)|2017|0





----------

### :notebook_with_decorative_cover: Theory
- Improved Techniques for Training GANs [[pdf]](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans.pdf) 	
- Energy-Based GANs & other Adversarial things by Yann Le Cun [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Mode RegularizedGenerative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1612.02136.pdf)

----------

### :nut_and_bolt: Presentations
- Generative Adversarial Networks (GANs) by Ian Goodfellow [[pdf]](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf) 	
- Learning Deep Generative Models by Russ Salakhutdinov [[pdf]](http://www.cs.toronto.edu/~rsalakhu/talk_Montreal_2016_Salakhutdinov.pdf) 	

----------

### :books: Courses / Tutorials / Blogs (Webpages unless other is stated)
- NIPS 2016 Tutorial: Generative Adversarial Networks (2016) [[pdf]](https://arxiv.org/pdf/1701.00160.pdf)
- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
- [Generative Models by OpenAI](https://openai.com/blog/generative-models/)
- [MNIST Generative Adversarial Model in Keras](https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/)
- [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/)
- [Attacking machine learning with adversarial examples by OpenAI](https://openai.com/blog/adversarial-example-research/)
- [On the intuition behind deep learning & GANs—towards a fundamental understanding](https://blog.waya.ai/introduction-to-gans-a-boxing-match-b-w-neural-nets-b4e5319cc935)
- [SimGANs - a game changer in unsupervised learning, self driving cars, and more](https://blog.waya.ai/simgans-applied-to-autonomous-driving-5a8c6676e36b)
----------


### :package: Resources / Models (Descending order based on GitHub stars)
S/N|Name|Repo|Stars
:---:|:---:|:---:|:---:
|1|Image super-resolution through deep learning|https://github.com/david-gpu/srez|4325
|2|Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)|https://github.com/junyanz/CycleGAN|3955
|3|Image-to-image translation with conditional adversarial nets (pix2pix)|https://github.com/phillipi/pix2pix|3068
|4|Deep Convolutional Generative Adversarial Networks (DCGAN)|https://github.com/Newmu/dcgan_code|2170
|5|Τensorflow implementation of Deep Convolutional Generative Adversarial Networks (DCGAN)|https://github.com/carpedm20/DCGAN-tensorflow|2110
|6|Generative Visual Manipulation on the Natural Image Manifold (iGAN)|https://github.com/junyanz/iGAN|2021
|7|Neural Photo Editing with Introspective Adversarial Networks|https://github.com/ajbrock/Neural-Photo-Editor|1487
|8|Generative Adversarial Text to Image Synthesis |https://github.com/paarthneekhara/text-to-image|1291
|9|Wasserstein GAN|https://github.com/martinarjovsky/WassersteinGAN|1127
|10|Improved Techniques for Training GANs|https://github.com/openai/improved-gan|837
|11|cleverhans: A library for benchmarking vulnerability to adversarial examples|https://github.com/openai/cleverhans|771
|12|StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks|https://github.com/hanzhanggit/StackGAN|682
|13|Semantic Image Inpainting with Perceptual and Contextual Losses (2016) |https://github.com/bamos/dcgan-completion.tensorflow|660
|14|Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (The Eyescream Project)|https://github.com/facebook/eyescream|498
|15|Improved Training of Wasserstein GANs|https://github.com/igul222/improved_wgan_training|481
|16|Unsupervised Cross-Domain Image Generation|https://github.com/yunjey/domain-transfer-network|466
|17|HyperGAN|https://github.com/255bits/HyperGAN|442
|18|Learning to Discover Cross-Domain Relations with Generative Adversarial Networks|https://github.com/carpedm20/DiscoGAN-pytorch|431
|19|Generating Videos with Scene Dynamics|https://github.com/cvondrick/videogan|419
|20|Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (KERAS-DCGAN)|https://github.com/jacobgil/keras-dcgan|382
|21|Synthesizing the preferred inputs for neurons in neural networks via deep generator networks|https://github.com/Evolving-AI-Lab/synthesizing|361
|22|Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space|https://github.com/Evolving-AI-Lab/ppgn|352
|23|Image-to-image translation using conditional adversarial nets|https://github.com/yenchenlin/pix2pix-tensorflow|344
|24|Deep multi-scale video prediction beyond mean square error|https://github.com/dyelax/Adversarial_Video_Generation|291
|25|Learning from Simulated and Unsupervised Images through Adversarial Training|https://github.com/carpedm20/simulated-unsupervised-tensorflow|285
|26|Learning What and Where to Draw|https://github.com/reedscot/nips2016|253
|27|Conditional Image Synthesis With Auxiliary Classifier GANs|https://github.com/buriburisuri/ac-gan|213
|28|Precomputed real-time texture synthesis with markovian generative adversarial networks|https://github.com/chuanli11/MGANs|193
|29|A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection|https://github.com/xiaolonw/adversarial-frcnn|190
|30|Unrolled Generative Adversarial Networks|https://github.com/poolio/unrolled_gan|190
|31|Adversarially Learned Inference (2016) (ALI)|https://github.com/IshmaelBelghazi/ALI|190
|32|Generating images with recurrent adversarial networks (sequence_gan)|https://github.com/ofirnachum/sequence_gan|185
|33|Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling|https://github.com/zck119/3dgan-release|179
|34|Energy-based generative adversarial network|https://github.com/buriburisuri/ebgan|178
|35|Autoencoding beyond pixels using a learned similarity metric|https://github.com/andersbll/autoencoding_beyond_pixels|145
|36|Pixel-Level Domain Transfer|https://github.com/fxia22/PixelDTGAN|143
|37|Sampling Generative Networks|https://github.com/dribnet/plat|139
|38|Invertible Conditional GANs for image editing|https://github.com/Guim3/IcGAN|120
|39|Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network|https://github.com/leehomyc/Photo-Realistic-Super-Resoluton|118
|40|Generative Image Modeling using Style and Structure Adversarial Networks (ss-gan)|https://github.com/xiaolonw/ss-gan|88
|41|Adversarial Autoencoders |https://github.com/musyoku/adversarial-autoencoder|81
|42|SalGAN: Visual Saliency Prediction with Generative Adversarial Networks|https://github.com/imatge-upc/saliency-salgan-2017|81
|43|Coupled Generative Adversarial Networks|https://github.com/mingyuliutw/CoGAN|65
|44|InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets|https://github.com/buriburisuri/supervised_infogan|59
|45|C-RNN-GAN: Continuous recurrent neural networks with adversarial training|https://github.com/olofmogren/c-rnn-gan|58
|46|Generative face completion (2017)|https://github.com/Yijunmaverick/GenerativeFaceCompletion|57
|47|Context Encoders: Feature Learning by Inpainting (2016)|https://github.com/jazzsaxmafia/Inpainting|55
|48|Conditional Generative Adversarial Nets|https://github.com/zhangqianhui/Conditional-Gans|28
|49|Least Squares Generative Adversarial Networks|https://github.com/pfnet-research/chainer-LSGAN|8
|50|Improving Generative Adversarial Networks with Denoising Feature Matching|https://github.com/hvy/chainer-gan-denoising-feature-matching|5



----------

### :electric_plug: Frameworks & Libraries (Descending order based on GitHub stars)
- Tensorflow by Google  [C++ and CUDA]: [[homepage]](https://www.tensorflow.org/) [[github]](https://github.com/tensorflow/tensorflow)
- Caffe by Berkeley Vision and Learning Center (BVLC)  [C++]: [[homepage]](http://caffe.berkeleyvision.org/) [[github]](https://github.com/BVLC/caffe) [[Installation Instructions]](Caffe_Installation/README.md)
- Keras by François Chollet  [Python]: [[homepage]](https://keras.io/) [[github]](https://github.com/fchollet/keras)
- Microsoft Cognitive Toolkit - CNTK  [C++]: [[homepage]](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/) [[github]](https://github.com/Microsoft/CNTK)
- MXNet adapted by Amazon  [C++]: [[homepage]](http://mxnet.io/) [[github]](https://github.com/dmlc/mxnet)
- Torch by Collobert, Kavukcuoglu & Clement Farabet, widely used by Facebook  [Lua]: [[homepage]](http://torch.ch/) [[github]](https://github.com/torch) 	
- Convnetjs by Andrej Karpathy [JavaScript]: [[homepage]](http://cs.stanford.edu/people/karpathy/convnetjs/) [[github]](https://github.com/karpathy/convnetjs)
- Theano by Université de Montréal  [Python]: [[homepage]](http://deeplearning.net/software/theano/) [[github]](https://github.com/Theano/Theano) 	
- Deeplearning4j by startup Skymind  [Java]: [[homepage]](https://deeplearning4j.org/) [[github]](https://github.com/deeplearning4j/deeplearning4j) 	
- :arrow_up_small: Caffe2 by Facebook Open Source [C++ & Python]: [[github]](https://github.com/caffe2/caffe2) [[web]](https://caffe2.ai/)
- Paddle by Baidu  [C++]: [[homepage]](http://www.paddlepaddle.org/) [[github]](https://github.com/PaddlePaddle/Paddle)
- Deep Scalable Sparse Tensor Network Engine (DSSTNE) by Amazon  [C++]: [[github]](https://github.com/amznlabs/amazon-dsstne)
- Neon by Nervana Systems  [Python & Sass]: [[homepage]](http://neon.nervanasys.com/docs/latest/) [[github]](https://github.com/NervanaSystems/neon) 	
- Chainer  [Python]: [[homepage]](http://chainer.org/) [[github]](https://github.com/pfnet/chainer) 	
- h2o  [Java]: [[homepage]](http://www.h2o.ai/) [[github]](https://github.com/h2oai/h2o-3) 	
- Brainstorm by Istituto Dalle Molle di Studi sull’Intelligenza Artificiale (IDSIA)  [Python]: [[github]](https://github.com/IDSIA/brainstorm)
- Matconvnet by Andrea Vedaldi  [Matlab]: [[homepage]](http://www.vlfeat.org/matconvnet/) [[github]](https://github.com/vlfeat/matconvnet) 	


----

License

MIT


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [@thomasfuchs]: <http://twitter.com/thomasfuchs>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [keymaster.js]: <https://github.com/madrobby/keymaster>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]:  <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
