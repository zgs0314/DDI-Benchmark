## Benchmarking Graph Learning for Drug-Drug Interaction Prediction

### Environment Setup

Make sure you have Anaconda or Miniconda installed on your system before you start. This guide is designed for systems with a CUDA-enabled GPU.

```
# Create and activate a new environment
conda create -n DDIBench python=3.8

# install dependencies
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### Dataset Preparation

This work uses 2 different datasets, which can be downloaded from this [link](https://drive.google.com/file/d/1Scpur6Lj7QwoHemSh6K941T1-XSCfSbp/view?usp=sharing). Please unzip the downloaded files into a folder naned `data` within the directory. 

### Benchmarking

Running the model training step.

```
python main.py --model MLP --dataset drugbank
```

+ Model choice: CompGCN, SkipGNN, ComplEx, MSTE, MLP, KGDDI, CSMDDI, HINDDI, Decagon, SumGNN, KnowDDI, EmerGNN
+ Dataset choice: drugbank, twosides
+ Other hypermeters can also be adapted. 

### Dataset Information

|Dataset|#Nodes|#Relations|#Triplets|
|---|---|---|---|
|DrugBank|1710|86|188509|
|TWOSIDES|645|209|116650|
