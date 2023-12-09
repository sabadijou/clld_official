# Contrastive Learning for Lane Detection via Cross-Similarity
>[**Contrastive Learning for Lane Detection via Cross-Similarity**](https://arxiv.org/abs/2308.08242)<br>
>[![arXiv](https://img.shields.io/badge/arXiv-2312.02151-b31b1b)](https://arxiv.org/abs/2308.08242)

## Overview of CLLD

Contrastive Learning for Lane Detection via cross-similarity (CLLD), is a self-supervised learning method that tackles this challenge by enhancing lane detection models’ resilience to real-world conditions that cause lane low visibility. CLLD is a novel multitask contrastive learning that trains lane detection approaches to detect lane markings even in low visible situations by integrating local feature contrastive learning (CL) with our new proposed operation cross-similarity. To ease of understanding some details are listed in the following:

- CLLD employs similarity learning to improve the performance of deep neural networks in lane detection, particularly in challenging scenarios. 
- The approach aims to enhance the knowledge base of neural networks used in lane detection.
- Our experiments were carried out using `ImageNet` as a pretraining dataset. We employed pioneering lane detection models like  `RESA`, `CLRNet`, and `UNet`, to evaluate the impact of our approach on model performances.

<p align="center">
  <img src="utils/images/architecture.jpg" alt="CLLD architecture" style="width:768px;"><br>
  <i>CLLD architecture</i>
</p>

