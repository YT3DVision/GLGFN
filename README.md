# GLGFN: Global-Local Grafting Fusion Network for High-Resolution Image Deraining
**Tao Yan, Xiangjie Zhu, Xianglong Chen, Weijiang He, Chenglong Wang, Yang Yang, Yinghui Wang and Xiaojun Chang, "GLGFN: Global-Local Grafting Fusion Network for High-Resolution Image Deraining," IEEE Transactions on Circuits and Systems for Video Technology, June 2024, <a href="https://ieeexplore.ieee.org/document/10552302/">doi: 10.1109/TCSVT.2024.3411655</a>**

> Abstract—Image deraining is a hot research topic, which aims to remove various rain streaks (raindrops) from rainy images and restore the backgrounds. Though image deraining has been extensively studied in recent years, few methods are able to effectively and efficiently derain real-world high-resolution rainy images. In general, existing image deraining methods are restricted by two main factors while processing high-resolution images. First, the computational complexity and memory usage of existing deep learning-based methods are high when it comes to derain high-resolution images. Second, as the image resolution increases, it is difficult to simultaneously extract and aggregate both global and local features for clean rain removal. In this paper, we propose a novel network, called Global-Local Grafting Fusion Network (GLGFN), for deraining real-world high-resolution images. Our GLGFN uses a staggered connection structure to achieve deeper sampling depth while maintaining low computational cost. It adopts the Transformer and CNN based encoders (backbones) to extract global and local features, respectively, and then grafts global features into local features to guide the extraction of rain streaks. In addition, for well fusing global and local features, we also propose a Grafting Fusion Module (GFM), which adopts Cross Sparse Attention (CSA) and Selective Kernel Fusion (SK Fusion) to efficiently aggregate global and local features. Extensive experiments conducted on several high-resolution real rainy datasets have demonstrated the effectiveness and efficiency of our proposed GLGFN. We will release our code and dataset.


## Network Architecture
![NetworkArchitecture](https://github.com/YT3DVision/GLGFN/blob/main/images/NetworkArchitecture.png)
## Quantitative Comparisons
![QuantitativeComparisons](https://github.com/YT3DVision/GLGFN/blob/main/images/QuantitativeComparisons.png)
## Memory Comparisons
![MemoryComparisons](https://github.com/YT3DVision/GLGFN/blob/main/images/MemoryComparisons.png)
## Runtime Comparisons
![RuntimeComparisons](https://github.com/YT3DVision/GLGFN/blob/main/images/RuntimeComparisons.png)
## Inference results and pre-training models
**Inference results and pre-training models are as follows:<br/> https://drive.google.com/drive/folders/1KZ-7bZq4Z_IwwF5BfUM6euWDeKVS0Vk_?usp=drive_link;<br/> https://drive.google.com/drive/folders/1RyCR3BKSyzn9wNio8gfYVcm_0ZyM3lnk.**
