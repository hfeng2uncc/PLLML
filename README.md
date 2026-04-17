# PLLML

<div align='center'>
 
<!-- [![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.12508&color=B31B1B)](https://www.google.com/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**Title** - Persistent local Laplacian prediction of protein-ligand binding affinities.

**Authors** - Jian Liu and Hongsong Feng

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Prerequisites](#prerequisites)
- [Visualization tools](#Visualization-tools)
- [Datasets](#datasets)
- [Modeling with mGLI-based features](#Modeling-with-mGLI-based-features)
    - [mGLI-based B-factor prediction](#i-mgli-based-b-factor-prediction)
    - [Generation of mGLI-based features for protein-ligand complex](#II-Generation-of-mGLI-based-features-for-protein-ligand-complex)
    - [Generation of mGLI-based features for small molecule](#III-Generation-of-mGLI-based-features-for-small-molecule)
    - [Generation of sequence-based features for protein or small molecules](#IV-Generation-of-sequence-based-features-for-protein-or-small-molecules)

- [Results](#results)
    - [I. Modeling the B-factor datasets]()
    - [II. Modeling the PDBbind datasets]()
- [License](#license)
- [Citation](#citation)

---

## Introduction

Accurate prediction of protein–ligand binding affinity remains a central challenge in structure-based drug discovery. The effectiveness of machine learning models critically depends on the quality of molecular representations, for which advanced mathematical frameworks provide powerful tools. In this work, we employ a novel mathematical theory, termed the persistent local Laplacian (PLL), to construct molecular descriptors that capture localized geometric and topological features of biomolecular structures. The PLL framework addresses key limitations of traditional topological data analysis methods, such as persistent homology and the persistent Laplacian, which are often insensitive to local structural variations, while maintaining high computational efficiency. The resulting molecular descriptors are integrated with advanced machine learning algorithms to develop accurate predictive models for protein–ligand binding affinity. The proposed models are systematically evaluated on three well-established benchmark datasets including PDBbind-v2007, PDBbind-v2013, and PDBbind-v2016, demonstrating consistently strong and competitive predictive performance. Computational results show that the PLL-based models outperform existing approaches, highlighting their potential as a powerful tool for drug discovery, protein engineering, and broader applications in science and engineering.

> **Keywords**: Persistent Local Laplacian, Machine Learning, Protein–Ligand Binding Affinity

---

## Model Architecture

Schematic illustration of the overall persistent Local Laplacian platform for protein-ligand binding affinity prediction. The model architecture is shown in below.

![Model Architecture](concept.png)

Further explain the details in the [paper](https://arxiv.org/abs/2603.21503), providing context and additional information about the architecture and its components.

---

## Prerequisites

- numpy                     1.21.0
- scipy                     1.7.3
- scikit-learn              1.0.2
- python                    3.10.12
- biopandas                 0.4.1
- Biopython                 1.75

---

## Datasets

A brief introduction about the benchmarks.

| Datasets                |Total    | Training Set                 | Test Set                                             |
|-|-----------------------------|------------------------------|------------------------------                        |-                                                            |
| PDBbind-v2007       |1300 |1105  [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                        | 195 [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                         |
| PDBbind-v2013       |2959|2764  [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                        | 195 [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                         |
| PDBbind-v2016       |4057|3767  [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                        | 290 [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                         |



- PDBbind RawData: the protein-ligand complex structures. Download from [PDBbind database](http://www.pdbbind.org.cn/)
- Label: the .csv file, which contains the protein ID and corresponding binding affinity for PDBbind data.
---

## Modeling with mGLI-based features

### I. Generation of mGLI-based features for protein-ligand complex
Example with PDB 1c87, generating PLL features. Output: 2eg8-complex-median-bin.npy
```shell
python codes/mGLI-protein-ligand.py --pdbid 2eg8 --bin_or_all bin --integral_type median
```

### II. Generation of mGLI-based features for small molecule
Example with the ligand in protein complex PDB 2eg8, generating mGLI features with "bin" manner and "median" statistics for atom-by-atom Gauss linking integral. Statistics of "all" can also be used.  output: 2eg8-ligand-median-bin.npy
```shell
python codes/mGLI-ligand.py --mol2_path datasets/PDBbind/2eg8/2eg8_ligand.mol2 --mol2_id 2eg8 --bin_or_all bin --integral_type median
```

---

## Results

### II. Modeling the PDBbind datasets

#### 1. Modeling with \#{mGLI-all & mGLI-lig-all,TF} features
|Datasets                                        | Training Set                  | Test Set| PCC | RMSE (kcal/mol) |
|-------------------------------------------------|-------------                  |---------|-    |-                |
| PDBbind-v2007 [result](./Results)      |1300 |1105  | **0.835** |1.888|
| PDBbind-v2013 [result](./Results)      |2959|2764  | **0.819** |1.930|
| PDBbind-v2016 [result](./Results)      |4057|3767  | 0.857 |1.673|

#### 2. Modeling with \#{mGLI-bin & mGLI-lig-all,TF} features
|Datasets                                        | Training Set                  | Test Set| PCC | RMSE (kcal/mol) |
|-------------------------------------------------|-------------                  |---------|-    |-                |
| PDBbind-v2007 [result](./Results)      |1105| 195  | 0.831 |1.932|
| PDBbind-v2013 [result](./Results)      |2764| 195  | **0.819** | 1.948|
| PDBbind-v2016 [result](./Results)      |3767| 290  | **0.862** |1.671|


Note, twenty gradient boosting regressor tree (GBRT) models were built for each dataset with distinct random seeds such that initialization-related errors can be addressed. The mGLI-based features and transformer-based features were paired with GBRT, respectively. The consensus predictions (\#{mGLI-all & mGLI-lig-all,TF} or \#{mGLI-bin & mGLI-lig-all,TF}) were obtained using predictions from the two types of models. The predictions can be found in the [results](./Results) folder. 

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

- Jian Liu and Hongsong Feng, "Persistent local Laplacian prediction of protein-ligand binding affinities"

---
