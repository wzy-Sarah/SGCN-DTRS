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

## Cite
@INPROCEEDINGS{9994907,
  author={Wang, Ziyan and Hong, Chengzhi and Liu, Xuan and Xiong, Zhankun and Liu, Feng and Zhang, Wen},
  booktitle={2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={Predicting drug transcriptional response similarity using Signed Graph Convolutional Network}, 
  year={2022},
  volume={},
  number={},
  pages={340-345},
  keywords={Drugs;Training;Databases;Training data;Predictive models;Biology;Graph neural networks;drug transcriptional response;similarity;CMap score;signed graph neural network},
  doi={10.1109/BIBM55620.2022.9994907}}
