# CLIPText Reproducibility Project

This repository contains replication materials for the paper  
**"CLIPText: A New Paradigm for Zero-shot Text Classification" (Qin et al., ACL 2023)[[PDF](https://aclanthology.org/2023.findings-acl.69)].**

## Contents
- `scripts/` – custom Python scripts for label–image mapping, dataset preprocessing and dataset creation.
- `data/new_data/` – constructed datasets (Wikipedia topics, IMDB reviews).
- `results/` – output logs and evaluation metrics.
- `report/` – final reproducibility report (PDF).

## Original Source
The official CLIPText implementation and pretrained model are available at  
[https://github.com/LightChen233/CLIPText](https://github.com/LightChen233/CLIPText)

## How to Reproduce
1. Clone the original CLIPText repo and follow its setup instructions.
2. Replace or point `data_dir` to `data/new_data/`.
3. Run:
   ```bash
   python main.py --test --dataset topic --data_dir data/new_data
   python main.py --test --text_prompt --dataset topic --data_dir data/new_data
