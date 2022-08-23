"""Data reading utils."""
import json
import os.path
import pickle
from scipy import sparse
from texttable import Texttable
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, roc_auc_score
from rdkit import Chem
import numpy as np
import random
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import torch

def get_dataset(path):
    return pickle.load(open(path, 'rb'))

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
 
    # "../data/labels.txt" contains the drug association triplets[drug1(number),drug2(number),score]
    dataset = np.loadtxt(args.edge_path)
    edges = {}
    edges["positive_edges"] = [edge[0:3] for edge in dataset if edge[2] > 0]
    print("positive_edges without zero",len(edges["positive_edges"]))
    edges["negative_edges"] = [edge[0:3] for edge in dataset if edge[2] < 0]
    edges["ecount"] = len(dataset)
    edges["ncount"] = len(set([edge[0] for edge in dataset] + [edge[1] for edge in dataset])) 
    return edges


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows=[]
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def calculate_auc(targets, predictions, edges):
    """
    Calculate performance measures on test dataset.
    :param targets: Target vector to predict.
    :param predictions: Predictions vector.
    :param edges: Edges dictionary with number of edges etc.
    :return auc: AUC value.
    :return f1: F1-score.
    """
    neg_ratio = len(edges["negative_edges"]) / edges["ecount"]
    targets = [0 if target == 1 else 1 for target in targets]
    auc = roc_auc_score(targets, predictions)
    f1 = f1_score(targets, [1 if p > neg_ratio else 0 for p in predictions])
    return auc, f1


def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable()
    t.add_rows([per for i, per in enumerate(logs["performance"])])
    print(t.draw())


def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.mkdir(os.path.dirname(args.log_path))
    with open(args.log_path, "w") as f:
        json.dump(logs, f)


def setup_features(args):

    print("general_features",args.general_features)
    print("maccskeys",args.maccskeys)

    if args.general_features:
        X = np.array(get_dataset(args.features_path))
    elif args.maccskeys:
        fp = open("..\data\Drug-smiles-mol_MACCS.pkl","rb")
        X = pickle.load(fp)
        X = np.array(X)
        print(X.shape)
    elif args.mol2vec:
        fp = open("..\data\Drug-smiles-mol2vec.pkl", "rb")
        X = pickle.load(fp)
        X = np.array(X)
        print("X", X.shape)

    return X




def evaluation(y_pred, y_true, ncount=None):
    # save_pred_matrix(y_pred, ncount)
    
    y_true_nonzero = (np.nonzero(y_true))

    y_true = y_true[y_true_nonzero]
    y_pred = y_pred[y_true_nonzero]

    #np.savetxt(".\data\true_matrix.txt", np.array(y_true), fmt='%.05f')
    #np.savetxt(".\data\pred_matrix.txt", np.array(y_pred), fmt='%.05f')

    corr = pearsonr(y_pred, y_true)[0] 

    msetotal = mse_at_k(y_pred, y_true, 1.0)
    mse1 = mse_at_k(y_pred, y_true, 0.01)
    mse2 = mse_at_k(y_pred, y_true, 0.02)
    mse5 = mse_at_k(y_pred, y_true, 0.05)

    auroc = float('nan')
    if len([x for x in y_true if x > 0.9]) > 0:
        auroc = roc_auc_score([1 if x > 0.9 else 0 for x in y_true], y_pred)
    precision1 = precision_at_k(y_pred, y_true, 0.01)
    precision2 = precision_at_k(y_pred, y_true, 0.02)
    precision5 = precision_at_k(y_pred, y_true, 0.05)
    precision = precision_at_k(y_pred, y_true, 1.0)

    return np.float(corr), np.float(msetotal), np.float(mse1), np.float(mse2), np.float(mse5), np.float(
        auroc), np.float(precision1), np.float(precision2), np.float(precision5), np.float(precision)

def evaluation_new(y_pred, y_true):
       
    corr = pearsonr(y_pred, y_true)[0]

    msetotal = mse_at_k(y_pred, y_true, 1.0)
    mse1 = mse_at_k(y_pred, y_true, 0.01)
    mse2 = mse_at_k(y_pred, y_true, 0.02)
    mse5 = mse_at_k(y_pred, y_true, 0.05)

    auroc = float('nan')
    if len([x for x in y_true if x > 0.9]) > 0:
        auroc = roc_auc_score([1 if x > 0.9 else 0 for x in y_true], y_pred)
    precision1 = precision_at_k(y_pred, y_true, 0.01)
    precision2 = precision_at_k(y_pred, y_true, 0.02)
    precision5 = precision_at_k(y_pred, y_true, 0.05)
    precision = precision_at_k(y_pred, y_true, 1.0)

    return np.float(corr), np.float(msetotal), np.float(mse1), np.float(mse2), np.float(mse5), np.float(
        auroc), np.float(precision1), np.float(precision2), np.float(precision5), np.float(precision)

def precision_at_k(y_pred, y_true, k):
    list_of_tuple = [(x, y) for x, y in zip(y_pred, y_true)]
    sorted_list_of_tuple = sorted(list_of_tuple, key=lambda tup: tup[0], reverse=True)
    topk = sorted_list_of_tuple[:int(len(sorted_list_of_tuple) * k)]
    topk_true = [x[1] for x in topk]
    topk_pred = [x[0] for x in topk]

    precisionk = precision_score([1 if x > 0.9 else 0 for x in topk_true],
                                 [1 if x > -1 else 0 for x in topk_pred], labels=[0, 1], pos_label=1)

    return precisionk


def mse_at_k(y_pred, y_true, k):
    list_of_tuple = [(x, y) for x, y in zip(y_pred, y_true)]
    sorted_list_of_tuple = sorted(list_of_tuple, key=lambda tup: tup[0], reverse=True)
    topk = sorted_list_of_tuple[:int(len(sorted_list_of_tuple) * k)]
    topk_true = [x[1] for x in topk]
    topk_pred = [x[0] for x in topk]

    msek = np.square(np.subtract(topk_pred, topk_true)).mean()
    return msek


def get_label_list(labels, node_count):
    index_1 = [edge[0] for edge in labels]
    index_2 = [edge[1] for edge in labels]
    values = [edge[2] for edge in labels]
    values=np.array(values)    
    matrix = sparse.csr_matrix((values, (index_1, index_2)), shape=(node_count, node_count))
    label_list = np.array(matrix.todense())
    label_mask = np.array(label_list, dtype=bool)

    return label_list.flatten(), label_mask


def div_list(ls, n):
    ls_len = len(ls)
    j = ls_len // n
    ls_return = []
    for i in range(0, (n - 1) * j, j):
        ls_return.append(ls[i:i + j])
    ls_return.append(ls[(n - 1) * j:])
    return ls_return


def get_cross_validation_dataset(edges, args,seed):
    print("seed is: ",42)
    random.seed(42)
    np.random.seed(42)
    pos_reorder = np.arange(len(edges["positive_edges"]))
    neg_reorder = np.arange(len(edges["negative_edges"]))
    np.random.shuffle(pos_reorder)
    np.random.shuffle(neg_reorder)

    pos_order = div_list(pos_reorder.tolist(), args.fold)
    neg_order = div_list(neg_reorder.tolist(), args.fold)
    return pos_reorder, neg_reorder, pos_order, neg_order


# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index



def save_pred_matrix(pred, ncount):
    np.set_printoptions(suppress=True)
    pred = np.array(pred)
    pred = pred.reshape((ncount, ncount), order='C')
    np.savetxt("../data/pred_matrix.txt", pred, fmt='%.05f')

def generate_drug_mol_feature_():
    id_to_smiles={}
    with open("../data/dds_drug_smiles.csv", 'r') as fp:
            fp.readline()
            for line in fp:
                sptlist = line.strip().split(',')
                if sptlist[1].strip()=='drug_id':
                    continue
                id = sptlist[0].strip()
                smiles = sptlist[2].strip()
                id_to_smiles[id]=smiles
    fea = []
    edge = []
    for key,smile in id_to_smiles.items():
        c_size, feature, edge_index = smile_to_graph(smile)
        fea.append(np.array(feature))
        edge.append(np.array(edge_index))

    fea = np.array(fea)
    edge = np.array(edge)

    data = np.array([fea, edge])
    return np.array(data)

    
def FeatureExtract(drug_feature):
    drug_data={}
    for key, value in drug_feature.items():
        feat_mat, adj_list = value[0], value[1]
        drug_data[key] = CalculateGraphFeat(feat_mat, adj_list)
    return drug_data
    
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]
    
def drug_mol_feature_MACCS():
    id_to_smiles={}
    with open("../data/dds_drug_smiles.csv", 'r') as fp:
            fp.readline()
            for line in fp:
                sptlist = line.strip().split(',')
                if sptlist[1].strip()=='drug_id':
                    continue
                id = sptlist[0].strip()
                smiles = sptlist[2].strip()
                id_to_smiles[id]=smiles
    drug_feature = [] 
    
    for key, smiles in id_to_smiles.items():
        
        mol = Chem.MolFromSmiles(smiles)
        fps = MACCSkeys.GenMACCSKeys(mol)
        fp_arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fps,fp_arr)
        fp_arr=np.array(fp_arr)
        print("fp_arr",fp_arr.shape)
        drug_feature.append(fp_arr)
    return drug_feature



