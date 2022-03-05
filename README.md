# [Mei, N. Santana, R., & Soto, D. (2022). Informative neural representations of unseen objects during higher-order processing in human brains and deep artificial networks, Nature Human Behavior. https://doi.org/10.1038/s41562-021-01274-7](https://www.nature.com/articles/s41562-021-01274-7) [biorxiv](https://www.biorxiv.org/content/10.1101/2021.01.12.426428v4.abstract)

---
# System Information
## Hardware
- [Platform](https://dipc.ehu.es):      Linux-3.10.0-514.el7.x86_64-x86_64-with-centos-7.3.1611-Core
- CPU:           x86_64: 16 cores
## Python environment
- Python:        3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42)  [GCC 7.2.0]
- Numpy:         1.19.1 {blas=mkl_rt, lapack=mkl_rt}
- Scipy:         1.3.1
- Matplotlib:    3.1.3 {backend=agg}
- Scikit-learn:  0.24.2
- Seaborn:       0.11.1
- Pandas:        1.0.1
- Tensorflow:    2.0.0
- Pytorch:       1.7.1
- Nilearn:       0.7.1
- Nipype:        1.4.2
- [LegrandNico/metadPy](https://github.com/LegrandNico/metadPy)
## R environment - R base
- R:             4.0.3 # for 3-way repeated measure ANOVAs
## Brain image processing backends
- mricrogl
- mricron:       10.2014
- FSL:           6.0.0
- Freesurfer:    6.0.0

# Main results

## MVPA as a function of awareness states
![decode](https://github.com/nmningmei/unconfeats/blob/main/figures/figure3.png)

## Simulation - full
![cnn_full](https://github.com/nmningmei/unconfeats/blob/main/figures/figure4.png)
The black dots illustrate the classification performance of the FCNN models, as a function of noise level, type of pre-trained model configuration (column) and activation functions (row). The blue dots illustrate the classification performance of the linear SVMs applied to the hidden layer of the FCNN model when the FCNN classification performance was lower than 0.55.
![cnn_chance](https://github.com/nmningmei/unconfeats/blob/main/figures/figure5.png)
Image classification performance of the linear SVMs applied to the FCNN hidden layers when a given FCNN failed to discriminate the living v.s. nonliving categories, as a function of the noise level. The superimposed subplot depicts the proportion of times in which the linear SVM was able to decode the FCNN hidden layers as a function of low and high noise levels. The blue bar represents the proportion of linear SVMs being able to decode the FCNN hidden layers, while the orange bar represents the proportion of linear SVMs decoding the FCNN hidden layers at chance level.

MobileNet produced more informative hidden representations that could be decoded by the SVMs compared to other candidate model configurations. We also observed that the classification performance of ResNet models trained with different configurations (e.g., varying number of hidden units, dropout rates) did not fall to chance level until the noise level was relatively high (closer to the dashed line) and the proportion of SVMs being able to decode FCNN hidden layers was higher compared to other model configurations (50.62\% v.s. 46.92\% for MobileNet, 35.51\% for AlexNet, 31.35\% for DenseNet, and 30.22\% for VGGNet). Additionally, we observed that even when the noise level was high, the MobileNet models provided a higher proportion of hidden representations that were decodable by the linear SVMs (34.77\% v.s. 29.53\% for ResNet, 27.84\% for DenseNet, 26.35\% for AlexNet, and 21.62\% for VGGNet, see Figure \ref{CNN_chance}). In summary, Mobilenet and Resnet generated the most informative hidden representations. These networks have an intermediate level of depth. By comparison, the deepest network Densenet did not produce better hidden representations. Hence the depth of the network does not appear to determine the quality or informativeness of the hidden representations significantly.
