# Binarized ResNet: Enabling Robust Automatic ModulationClassification at the resource-constrained Edge
This repository is a research oriented-repository dedicated to the Binarized ResNet: Enabling Automatic ModulationClassification at the resource-constrained Edge. 

The main idea is to release the code of our papers in a very friendly manner such that it cultivates and accelerates further research on this topic. Our codes are based on the following github repos https://github.com/itayhubara/BinaryNet.pytorch and https://github.com/lmbxmu/RBNN.
If you have any comments, please contact 

Nitin P Shankar  (ee20d425@smail.iitm.ac.in )

Deepsayan Sadukhan (ee20s001@smail.iitm.ac.in)

Nancy Nayak (ee17d408@smail.iitm.ac.in)

I hope you enjoy it and good luck with your research.

## Citing this Repo
If you use any part of this repo, please consider citing our work:
Shankar, N.P., Sadhukhan, D., Nayak, N. and Kalyani, S., 2021. Binarized ResNet: Enabling Automatic Modulation Classification at the resource-constrained Edge. arXiv preprint arXiv:2110.14357.

# 1. Our Works and Our Codes
## Binarized ResNet
- The paper is available at __[[Paper]](https://arxiv.org/abs/2110.14357)__

- **About the work:** This work is based on Deep Learning methods used for Automatic Modulation Classification at the Edge with constrained resources. Firstly we propose a new architecture for AMC and then we binarize and rotate the weights and activations to bring down the memory and comutational complexity to the minimal. Next we improve the accuracy by ensembling those architectures. Our model also has higher adversarial robustness against gradient based attacks.

- **System Requirements**
-- CUDA compatible graphics card.

- **Dependencies**
- Any package manager of your choice. If you use conda, use the following code to create a new envirinment and activate it.
```
conda create --name myenv
conda activate myenv
```

- Install pytorch
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

- Install scipy
```
conda install -c anaconda scipy
```

- Install dali and torch summary using pip
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```

```
pip install torchsummary
```

-Install the remaining packages

```
conda install -c anaconda pandas seaborn scikit-learn
```

- **Executing the code**
- Download the RML2018.01a dataset and use Dataset2018.py file to create the torch datasets for training and testing.
- Once the datasets are created, change the paths in the accordingly to execute the main file of both BNN and RBNN APIs. Some of our pretrained models are stored for reference.
- Navigate to [./RBNN/cifar](https://github.com/deepsy1998/RBLResNet/tree/main/RBNN/cifar) or [./BNN/BinaryNet.pytorch](https://github.com/deepsy1998/RBLResNet/tree/main/BNN/BinaryNet.pytorch) where you will find the main files and respective .sh files to execute.
