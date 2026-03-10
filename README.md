# DMS-Mamba
Official PyTorch implementation of the paper"Dual-Domain Multi-Scale State Space Model for Multi-Contrast MRI Reconstruction"


# Environment

CUDA Version: 11.7

python=3.8.18 

pytorch=1.13.1



## Links for downloading the public datasets:

We evaluate **DMS-Mamba** on two widely used public multi-contrast MRI datasets. Please follow the official links to request access and download the raw data:

1) BraTS2020 Dataset - <a href="https://www.kaggle.com/datasets/awsaf49/brats2020-training-data"> Link </a> 
2) fastMRI Dataset - <a href="https://fastmri.med.nyu.edu/"> Link </a>

## Train DMS-Mamba
```bash 
bash train.sh
```

## Ackonwledgements

We give acknowledgements to [fastMRI](https://github.com/facebookresearch/fastMRI),  [Pan-Mamba](https://github.com/alexhe101/Pan-Mamba), [PW-FNet ](https://github.com/deng-ai-lab/PW-FNet)and [MMR-Mamba](https://github.com/zoujing925/MMR-Mamba).

