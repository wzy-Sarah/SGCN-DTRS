"""SGCN runner."""
import time

import torch
from tqdm import trange
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv,GCNConv,SAGPooling, global_max_pool,GINConv,LayerNorm, global_add_pool, BatchNorm
from signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep,ListModule
from torch_geometric import data as DATA
import torch.utils.data as Data
from utils import *
from MolDataset import MolDataset, MolDataset_new


class SSI_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(data.x, data.edge_index,
                                                                                             batch=data.batch)
        global_graph_emb = global_add_pool(att_x, att_batch)

        # data = max_pool_neighbor_x(data)
        return data, global_graph_emb

class SignedGraphConvolutionalNetwork2(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    """

    def __init__(self, device, args, X, mode ):
        super(SignedGraphConvolutionalNetwork2, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.mode = mode
        self.X = X
        self.setup_layers()

    def setup_layers(self):

        #self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        if self.mode == "GAT":
            in_features = 75  # TOTAL_ATOM_FEATS 55
            self.hidd_dim = 128
            self.kge_dim = 64
            # self.heads_out_feat_params=[32, 32, 32, 32]
            # self.blocks_params=[2, 2, 2, 2]
            self.heads_out_feat_params = [64, 64, 64, 64]
            self.blocks_params = [2, 2, 2, 2]
            self.n_blocks = len(self.blocks_params)
            self.initial_norm = LayerNorm(in_features)
            self.blocks = []
            self.net_norms = nn.ModuleList()
            for i, (head_out_feats, n_heads) in enumerate(zip(self.heads_out_feat_params, self.blocks_params)):
                block = SSI_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
                self.add_module(f"block{i}", block)
                self.blocks.append(block)
                self.net_norms.append(LayerNorm(head_out_feats * n_heads))
                in_features = head_out_feats * n_heads
            # self.linear = nn.Linear(128, 128)
            # self.relu = nn.ReLU()
            self.positive_base_aggregator = SignedSAGEConvolutionBase(256, self.neurons[0]).to(self.device)
            self.negative_base_aggregator = SignedSAGEConvolutionBase(256, self.neurons[0]).to(self.device)
        elif self.mode == "GCN":
            self.mol_gcn = GCNConv(75, 256)
            self.mol_gcn1 = GCNConv(256, 256)
            self.mol_gcn2 = GCNConv(256, 256)
            self.mol_gcn3 = GCNConv(256, 256)
            self.batch_norm1 = BatchNorm(256)
            self.batch_norm2 = BatchNorm(256)
            self.batch_norm3 = BatchNorm(256)
            self.batch_norm4 = BatchNorm(256)
            self.linear = nn.Linear(256, 256)
            self.relu = nn.ReLU()
            self.positive_base_aggregator = SignedSAGEConvolutionBase(512, self.neurons[0]).to(self.device)
            self.negative_base_aggregator = SignedSAGEConvolutionBase(512, self.neurons[0]).to(self.device)
        elif self.args.mol2vec:
            self.linear = nn.Linear(300, 300)
            self.relu = nn.ReLU()
            self.positive_base_aggregator = SignedSAGEConvolutionBase(600, self.neurons[0]).to(self.device)
            self.negative_base_aggregator = SignedSAGEConvolutionBase(600, self.neurons[0]).to(self.device)
        else:
            self.linear = nn.Linear(2048, 2048)
            self.relu = nn.ReLU()
            self.positive_base_aggregator = SignedSAGEConvolutionBase(4096, self.neurons[0]).to(self.device)
            self.negative_base_aggregator = SignedSAGEConvolutionBase(4096, self.neurons[0]).to(self.device)

        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1, self.layers):
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1],
                                                                       self.neurons[i]).to(self.device))
            self.negative_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1],
                                                                       self.neurons[i]).to(self.device))

        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)

    def calculate_mse_function(self, pred, labels):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred, labels)
        return loss

    def forward(self, positive_edges, negative_edges, labels, label_mask, loader, mode_trainortest):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        if self.mode == "GAT":
            outlist = []
            for i, block in enumerate(self.blocks):
                out = block(loader)[1]
                outlist.append(out)
                loader.x = F.elu(self.net_norms[i](loader.x, loader.batch))
            self.X = outlist[-1]
        elif self.mode == "GCN":
            mol_x, mol_edge_index, mol_batch = loader.x, loader.edge_index, loader.batch
            mol_x = self.mol_gcn(mol_x, mol_edge_index)
            mol_x = F.relu(mol_x)
            mol_x = self.batch_norm1(mol_x)
            mol_x = self.mol_gcn1(mol_x, mol_edge_index)
            mol_x = F.relu(mol_x)
            mol_x = self.batch_norm2(mol_x)
            mol_x = self.mol_gcn2(mol_x, mol_edge_index)
            mol_x = F.relu(mol_x)
            mol_x = self.batch_norm3(mol_x)
            mol_x = self.mol_gcn3(mol_x, mol_edge_index)
            mol_x = F.relu(mol_x)
            mol_x = self.batch_norm4(mol_x)
            self.X = global_max_pool(mol_x, mol_batch)


        # X = self.relu(X)
        self.h_pos, self.h_neg = [], []
        self.h_pos.append(torch.tanh(
            self.positive_base_aggregator(self.X, positive_edges)))  # [2428,64]
        self.h_neg.append(torch.tanh(
            self.negative_base_aggregator(self.X, negative_edges)))

        for i in range(1, self.layers):
            self.h_pos.append(torch.tanh(
                self.positive_aggregators[i - 1](self.h_pos[i - 1], self.h_neg[i - 1], positive_edges, negative_edges)))
            self.h_neg.append(torch.tanh(
                self.negative_aggregators[i - 1](self.h_neg[i - 1], self.h_pos[i - 1], positive_edges, negative_edges)))

        self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1)  # [2428,128]



        self.X_mol = F.normalize(self.z)

        pred = torch.flatten(torch.mm(self.X_mol, self.X_mol.t()) * label_mask)

        loss = self.calculate_mse_function(pred, labels)
        if mode_trainortest == "get_embed":
            return loss, self.X , pred, self.X_mol
        elif mode_trainortest == "train":
            return loss, self.X_mol, pred
        #return loss, self.X_mol, pred


class SignedGraphConvolutionalNetwork(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    """

    def __init__(self, device, args, X, agg_mode, dropout=0.9):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.agg_mode = agg_mode

        #self.tanh = torch.nn.Tanh()
        #self.dropout = torch.nn.Dropout(dropout)
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers.
        Assing Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        input_dim = 2048
        if self.agg_mode == "SGCN":
            self.positive_base_aggregator = SignedSAGEConvolutionBase(input_dim*2,self.neurons[0]).to(self.device)
            self.negative_base_aggregator = SignedSAGEConvolutionBase(input_dim*2,self.neurons[0]).to(self.device)

            self.positive_aggregators = []
            self.negative_aggregators = []
            for i in range(1, self.layers):
                self.positive_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1],
                                                                           self.neurons[i]).to(self.device))

                self.negative_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1],
                                                                           self.neurons[i]).to(self.device))

            self.positive_aggregators = ListModule(*self.positive_aggregators)
            self.negative_aggregators = ListModule(*self.negative_aggregators)

        elif self.agg_mode=="GAT":
            self.gat1 = GATConv(input_dim, 128)
            self.gat2 = GATConv(128, 128)
        elif self.agg_mode=="GCN":
            self.gcn1 = GCNConv(input_dim, 128)
            self.gcn2 = GCNConv(128, 128)
        #self.linear= nn.Linear(2048,2048)
        #self.relu = nn.ReLU()

    def calculate_mse_function(self, pred, labels):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred, labels)
        return loss


    def forward(self, X, positive_edges, negative_edges, labels, label_mask, mode):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        
        #X = self.linear(X)
        #X = self.relu(X)
        self.out = X        

        if self.agg_mode=="SGCN":
            self.h_pos=[]
            self.h_neg=[]
            self.h_pos.append(torch.tanh(
                self.positive_base_aggregator(X, positive_edges)))
            self.h_neg.append(torch.tanh(
                self.negative_base_aggregator(X, negative_edges)))
            for i in range(1, self.layers):
                self.h_pos.append(torch.tanh(
                    self.positive_aggregators[i - 1](self.h_pos[i - 1], self.h_neg[i - 1], positive_edges, negative_edges)))
                self.h_neg.append(torch.tanh(
                    self.negative_aggregators[i - 1](self.h_neg[i - 1], self.h_pos[i - 1], positive_edges, negative_edges)))
            self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1) #[2428,128]
        elif self.agg_mode =="GAT":
            edge = torch.cat((positive_edges,negative_edges),1)
            X = self.gat1(X, edge)
            X = F.normalize(X, p=2, dim=-1)
            self.z = self.gat2(X, edge)
        elif self.agg_mode =="GCN":
            edge = torch.cat((positive_edges, negative_edges), 1)
            X= self.gcn1(X,edge)
            X = F.normalize(X, p=2, dim=-1)
            self.z = self.gcn2(X,edge)


        self.X_mol = F.normalize(self.z)
        pred = torch.flatten(torch.mm(self.X_mol, self.X_mol.t()) * label_mask)
        loss = self.calculate_mse_function(pred, labels)
        if mode == "get_embed":
            return loss, self.out , pred, self.X_mol
        elif mode == "train":
            return loss, self.X_mol, pred


class SignedGCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """

    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure with positive and negative edges separated.
        """
        self.args = args
        self.edges = edges
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print("device",self.device)
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance_ku"] = [
            ["Epoch", "corr", "msetotal", "mse1", "mse2", "mse5", "auroc", "precision1", "precision2",
             "precision5"]]
        self.logs["performance_uu"] = [
            ["Epoch", "corr", "msetotal", "mse1", "mse2", "mse5", "auroc", "precision1", "precision2",
             "precision5"]]
        self.logs["training_loss"] = [["Epoch", "loss"]]

    def setup_dataset(self, positive_edges, test_positive_edges, negative_edges, test_negative_edges, kk_edges,ku_edges,uu_edges):
        """
        Creating train and test split.
        """
        # get the trainset of association graph and testset of asociation graph
        self.positive_edges = positive_edges  # positive edges of kk edges
        self.test_positive_edges = test_positive_edges # none
        self.negative_edges = negative_edges # negative edges of kk edges
        self.test_negative_edges = test_negative_edges #none
        # the whole edges number
        self.ecount = len(self.positive_edges + self.negative_edges)
        
        # the length of self.train_edges is not equal with self.kk_edge,
        # because self.kk_edges has zero cmap scores
        self.train_edges = self.positive_edges + self.negative_edges
        # the whole testset of asociation graph
        self.test_edges = self.test_positive_edges + self.test_negative_edges

        self.kk_edges = kk_edges
        self.ku_edges = ku_edges
        self.uu_edges = uu_edges

        self.train_drugs = []
        for pair in kk_edges:
            if pair[0] not in self.train_drugs:
                self.train_drugs.append(int(pair[0]))
            if pair[1] not in self.train_drugs:
                self.train_drugs.append(int(pair[1]))
        self.test_drugs = []
        for i in range(2428):
            if i not in self.train_drugs:
                self.test_drugs.append(i)
        assert len(self.train_drugs)+len(self.test_drugs)==2428


        # rebulit trainset, only get [drug1,drug2]
        self.positive_edges = np.array(np.array(self.positive_edges)[:, [0, 1]])
        self.negative_edges = np.array(np.array(self.negative_edges)[:, [0, 1]])
        
        # testset do not change
        self.test_positive_edges = np.array(self.test_positive_edges)
        self.test_negative_edges = np.array(self.test_negative_edges)

        # get trainset feature in association graph self.X:[2428, 300], the drug's features
        if self.args.init_embedding == 'GNN':
            with open("..\\data\\Drug-smiles-mol_new.pkl", 'rb') as f:
                drug_feature = pickle.load(f)
            drug_data = FeatureExtract(drug_feature)
            self.data_list_mol = []
            for index in range(len(drug_data)):
                drug_feature = drug_data[f'{index}'][0]
                edge = drug_data[f'{index}'][1]
                GCNData_mol = DATA.Data(x=torch.Tensor(drug_feature), edge_index=torch.LongTensor(edge))
                self.data_list_mol.append(GCNData_mol)
            mol_dataset = MolDataset(self.data_list_mol)
            self.loader = Data.DataLoader(
                dataset=mol_dataset,
                batch_size=len(self.data_list_mol),
                shuffle=False,
                num_workers=0,
                collate_fn=collate
            )

        else:
            X_train = setup_features(self.args,
                                    self.train_drugs
                                    )
            X_test = setup_features(self.args)
        


            self.X_train = torch.from_numpy(X_train).float().to(self.device)
            self.X_test = torch.from_numpy(X_test).float().to(self.device)
        
        # self.positive_edges shape is [2,208084]
        self.positive_edges = torch.from_numpy(np.array(self.positive_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)
        # self.negative_edges shape is [2,7551]
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)

        self.train_labels, self.train_mask = get_label_list(self.train_edges, self.edges["ncount"])
        self.train_labels = torch.from_numpy(self.train_labels).float().to(self.device)
        self.train_mask = torch.from_numpy(self.train_mask).float().to(self.device)

        self.test_labels, self.test_mask = get_label_list(self.test_edges, self.edges["ncount"])
        self.test_labels = torch.from_numpy(self.test_labels).float().to(self.device)
        self.test_mask = torch.from_numpy(self.test_mask).float().to(self.device)

    def create_and_train_model(self, weight_decay, fold_idx=None):
        """
        Model training and scoring.
        """
        print("\n SGCN Training started.\n")
        if self.args.init_embedding=="GNN":
            X=None
            self.model = SignedGraphConvolutionalNetwork2(self.device, self.args, X, self.args.init_embedding_GNN).to(self.device)
        else:
            self.model = SignedGraphConvolutionalNetwork(self.device, self.args, self.X_train, "SGCN").to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=weight_decay)
        self.model.train()
        print("self.model",self.model)
        self.epochs = trange(self.args.epochs, desc="Loss")
        Mol_emb=0
        for epoch in self.epochs:
            self.optimizer.zero_grad()
            if self.args.init_embedding == "GNN":

                t1 = time.time()
                for batch_x in self.loader:
                    #print("timesssssssssssssssssssssss", time.time() - t1)
                    loss, mol_x, pred = self.model(self.positive_edges, self.negative_edges, self.train_labels,
                                       self.train_mask, batch_x.to(self.device), "train")

                    loss.backward()
                    self.epochs.set_description(
                        "SGCN (Loss=%g)" % round(loss.item(), 4))
                    self.optimizer.step()
                    if (epoch + 1) % 100 == 0:
                        self.logs["training_loss"].append([epoch + 1, loss.item()])

            else:
                loss, mol_x, pred = self.model(self.X_train, self.positive_edges, self.negative_edges, self.train_labels,
                                       self.train_mask, "train")
                loss.backward()
                self.epochs.set_description(
                    "SGCN (Loss=%g)" % round(loss.item(), 4))
                self.optimizer.step()
                if (epoch + 1) % 100 == 0:
                    self.logs["training_loss"].append([epoch + 1, loss.item()])

        if self.args.get_embed:
            self.get_embed_model(self.args.epochs-1)


    def get_embed_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """
        self.model.eval()
        with torch.no_grad():
            if self.args.init_embedding == "GNN":
                for batch_x in enumerate(self.loader):
                    self.train_mask, batch_x[1].to(self.device), "train"
                    loss, linear_embedding, pred, sgcn_embedding = self.model(self.positive_edges, self.negative_edges, self.train_labels,
                                       self.train_mask, batch_x[1].to(self.device), "get_embed")
            else:
                loss, linear_embedding, pred, sgcn_embedding = self.model(self.X_test, self.positive_edges,
                                                                          self.negative_edges, self.test_labels,
                                                                          self.test_mask,
                                                                          "get_embed")
        sgcn_embedding = sgcn_embedding.cpu().detach().numpy()

        jasscard_sim = self.get_tanimoto_coefficient(linear_embedding).cpu().detach().numpy()
        unseen_sim = np.zeros([jasscard_sim.shape[0], jasscard_sim.shape[1]])  # [2428,2428]
        unseen_sim[:, self.train_drugs] = jasscard_sim[:, self.train_drugs]  

        sort_unseen_sim_index = (-unseen_sim).argsort()[:, 0:5]  # get unseen drugs top 10 sims [2428,10]
        sort_unseen_sim_score = np.sort(-unseen_sim)[:, 0:5]  # get unseen drugs top 10 sims [2428,10]
        print("sort_unseen_sim_score",sort_unseen_sim_score[0])
        unseen_drugs_embed_from_sgcn = sgcn_embedding[sort_unseen_sim_index]  # [2428,10,128]
        unseen_drugs_embed_from_sgcn = np.mean(unseen_drugs_embed_from_sgcn, axis=1)  # [2428, 128]

        va_pred = []
        va_labels = []
        for pair in self.ku_edges:
            pair1 = int(pair[0])
            pair2 = int(pair[1])
            if pair1 in self.train_drugs:
                embedding1 = sgcn_embedding[pair1]
            else:
                embedding1 = unseen_drugs_embed_from_sgcn[pair1]

            if pair2 in self.train_drugs:
                embedding2 = sgcn_embedding[pair2]
            else:
                embedding2 = unseen_drugs_embed_from_sgcn[pair2]
            pred = np.matmul(embedding1, embedding2.T)
            va_pred.append(pred)
            va_labels.append(pair[2])

        te_pred = []
        te_labels = []
        for pair in self.uu_edges:
            pair1 = int(pair[0])
            pair2 = int(pair[1])
            embedding1 = unseen_drugs_embed_from_sgcn[pair1]
            embedding2 = unseen_drugs_embed_from_sgcn[pair2]
            pred = np.matmul(embedding1, embedding2.T)
            te_pred.append(pred)
            te_labels.append(pair[2])

        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5, precision = evaluation_new(
            va_pred, va_labels)
        self.logs["performance_ku"].append(
            [epoch + 1, corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5])
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5, precision = evaluation_new(
            te_pred, te_labels)
        self.logs["performance_uu"].append(
            [epoch + 1, corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5])





    def get_Jaccard_Similarity(self, interaction_matrix):
        X = interaction_matrix
        E = torch.ones_like(X.T)
        denominator = torch.mm(X, E) + torch.mm(E.T, X.T) - torch.mm(X, X.T)
        denominator_zero_index = torch.where(denominator == 0)
        denominator[denominator_zero_index] = 1
        result = torch.div(torch.mm(X, X.T), denominator)
        result[denominator_zero_index] = 0
        result = result - torch.diag(torch.diag(result))
        return result

    def get_tanimoto_coefficient(self, interaction_matrix):
        X = interaction_matrix
        fenzi = torch.mm(X, X.T)
        a = torch.unsqueeze(torch.sum(torch.pow(X, 2), dim=1), 1)
        b = a + a.T
        fenmu = b - fenzi
        result = torch.div(fenzi, fenmu)
        result = result - torch.diag(torch.diag(result))
        return result

    def get_Cosin_Similarity(self, interaction_matrix):
        X = interaction_matrix
        alpha = torch.sum(torch.mul(X, X), dim=1, keepdim=True)
        # print(np.where(alpha==0))
        norm = torch.mm(alpha, alpha.T)
        index = torch.where(norm == 0)
        norm[index] = 1
        similarity_matrix = torch.div(torch.mm(X, X.T), (torch.sqrt(norm)))
        similarity_matrix[index] = 0
        # similarity_matrix[np.isnan(similarity_matrix)] = 0
        result = similarity_matrix
        result = result - torch.diag(torch.diag(result))
        return result

    def get_CommonNeighbours_Similarity(self, interaction_matrix):
        X = interaction_matrix
        similarity_matrix = torch.mm(X, X.T)
        similarity_matrix[torch.isnan(similarity_matrix)] = 0
        return similarity_matrix

    def get_CommonNeighbours_Similarity(self, interaction_matrix):
        X = interaction_matrix
        similarity_matrix = torch.mm(X, X.T)
        similarity_matrix[torch.isnan(similarity_matrix)] = 0
        return similarity_matrix

    def get_Pearson_Similarity(self, interaction_matrix):
        X = interaction_matrix
        X = X - (X.sum(axis=1) / X.shape[1])
        similarity_matrix = self.get_Cosin_Similarity(X)
        similarity_matrix[np.isnan(similarity_matrix)] = 0

        return similarity_matrix



def collate(data_list):
    batch = Batch.from_data_list([data for data in data_list])
    return batch
