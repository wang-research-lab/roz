# Benchmarking Zero-Shot Robustness of Multimodal Foundation Models: A Pilot Study

Reference: "Benchmarking Zero-Shot Robustness of Multimodal Foundation Models: A Pilot Study". 

## Installation 

Dependencies: python 3.7, requirements.txt 

```
conda create --name roz python=3.7
conda activate roz
pip install -r requirements.txt
```

## Hardware Requirements

We run our experiments on 4 GeForce GTX 1080 GPUs. Inernet connection is required for downloading ImageNet and CIFAR-10 datasets.

##  Data Preparation

### Download ImageNet data

https://image-net.org/download.php

### Download CIFAR-10 data

```
python download_cifar.py
```

After downloading ImageNet data, you may need to modify the dataset path in the code in scripts/common_adversarial_attack/load_utils.py. We use ImageNetPATH to indicate the dataset location.

## Scripts for Reproducing Results

### Natural Distribution Shifts

```
bash eval_natural_distribution_shifts.sh
```

### Synthetic Distribution Shifts

```
bash eval_synthetic_distribution_shifts.sh
```

### Common Adversarial Attacks

```
bash eval_common_adversarial_attacks.sh
```

### Typographic Attacks

Both CIFAR-10-T and ImageNet-T datasets are under the "dataset" folder.

```
bash eval_typographic_attacks.sh
```

## Scripts for Recreating ImageNet-T and CIFAR-10-T

```
bash create_benchmark.sh 
```

Please refer to the paper and appendix for more results.

## Citation
```bibtex
@article{wang2024roz,
  title={Benchmarking Zero-Shot Robustness of Multimodal Foundation Models: A Pilot Study},
  author={Wang, Chenguang and Jia, Ruoxi and Liu, Xin and Song, Dawn},
  journal={arXiv preprint},
  year={2024}
}
