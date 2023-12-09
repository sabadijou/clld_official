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

## Get started
1. Clone the repository
    ```
    git clone https://github.com/sabadijou/clld_official.git
    ```
    We call this directory as `$RESA_ROOT`

2. Create an environment and activate it (We've used conda. but it is optional)

    ```Shell
    conda create -n clld python=3.9 -y
    conda activate clld
    ```

3. Install dependencies

    ```Shell
    # Install pytorch firstly, the cudatoolkit version should be same in your system. (you can also use pip to install pytorch and torchvision)
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
      
    # Install kornia and einops
    pip install kornia
    pip install einops

    # Install other dependencies
    pip install -r requirements.txt
    ```
## Run CLLD
We conducted pretraining using the training data from `ImageNet`. However, you are free to utilize other datasets and configurations as needed. The configuration file for our approach can be found in the `configs` folder.

Once the dataset and new configurations are in place, you can execute the approach using the following command:

```Shell
python main.py --dataset_path /Imagenet/train --encoder resnet50 --alpha 1 --batch_size 1024 --world_size 1 --gpus_id 0 1 
```
The following is a quick guide on arguments:
- `dataset_path`: Path to training data directory 
- `encoder`: Select an encoder for training. `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`,`resnext101_32x8d`, `wide_resnet50_2`, `wide_resnet101_2`.
- `alpha`: Cross similarity window size
- `batch_size`: Select a batch size that suits the GPU infrastructure you are using.
- `world_size`: For example, if you are training a model on a single machine with 4 GPUs, the world size is 4. If you have 2 machines, each with 4 GPUs, and you use all of them for training, the world size would be 8.
- `gpus_id`: Please specify all the GPU IDs that you used for training the approach.

  
