# SGCN-DTRS
This method can predict CMap scores using the structure of pairwise compounds.
The paper is "Predicting drug transcriptional response similarity using Signed Graph Convolutional Network".
The authors of the paper are Ziyan Wang, Chengzhi Hong, Xuan Liu, Zhankun Xiong, Feng Liu, Wen Zhang.

## environment
conda create --name SGCN-DTRS python=3.6


torch 1.9.0+cu102
numpy
pandas
tqdm
torch_geometric
torch_scatter
scipy
texttable
sklearn
rdkit
networkx
deepchem

## Training and evaluating the method in transductive task
You can train and evaluat the method by going to src/ and run python main.py

## Training and evaluating the method in inductive task
You can go to kk_ku/ and run python main.py
