import torch
import dgl
import random
import torch.nn as nn
from task_sampling import sample_datasets, sample_test_datasets
from GNN_model import GINPredictor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from functools import partial
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from dgl.data.utils import load_graphs
from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer, CanonicalAtomFeaturizer
from dataset import MetaMoleDataset
from dgl.data import DGLDataset
from dgllife.data import SIDER, Tox21

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, feature, label).
    '''
    :param samples: 5 samples
    :return: batched graphs, tensor of labels
    '''
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def load_dataset(dataset):
    # load dataset and motif
    if dataset == "sider":
        dataset_dgl = SIDER(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=PretrainAtomFeaturizer(), edge_featurizer=PretrainBondFeaturizer())
    elif dataset == "tox21":
        dataset_dgl = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=PretrainAtomFeaturizer(), edge_featurizer=PretrainBondFeaturizer())
    dataset_motif = load_graphs("/home/ubuntu/Few-Shot-Learning-on-Graphs/motif_preparation/sider_motif.bin")
    return dataset_dgl, dataset_motif


def get_graph_emb(node_emb, batch_nodes):
    '''
    :param node_emb: node embeddding, shape [num_graphs * num_nodes, emb_dim]
    :param batch_nodes: number of nodes for each graph, shape [1, num_graphs]
    :return: graph embedding, shape [num_graphs, emb_dim]
    '''
    graph_emb = []
    t = 0
    for k in batch_nodes:
        graph_emb.append(node_emb[t:t + k].mean(0))
    return torch.stack(graph_emb)


def euclidean_dist(x, y):
    '''
    :param x: embedding_x, tensor vector
    :param y: embedding_y, tensor vector
    :return: euclidean distance, scalar
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def proto_loss_spt(logits, y_t, n_support, add_motif, motif_vectors):
    '''
    :param logits: graph embedding
    :param y_t: graph labels
    :param n_support: size of support
    :param add_motif: If adding motifs
    :param motif_vectors: [num_motifs, emb_dim]
    :return: prototypical loss
    '''
    # put label on cpu
    target_cpu = y_t.squeeze().to('cpu')
    # put logits on cpu
    input_cpu = logits.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # classes = torch.unique(target_cpu)
    # n_classes = len(classes)
    # number of classes always 2 (binary classification)
    classes = [0, 1]
    n_classes = 2
    n_query = n_support

    # len(support_idxs)
    support_idxs = list(map(supp_idxs, classes))

    # select motif
    if add_motif:
        # score matrix [num_graphs, num_motifs]
        score_matrix = torch.matmul(logits, torch.t(motif_vectors))
        # selected motif [num_graphs, top_k]
        selected_motif = torch.topk(score_matrix, 3)[1]

    if len(torch.unique(target_cpu)) < n_classes:
        prototypes_list = []
        for idx_list in support_idxs:
            if len(idx_list) == 0:
                prototypes_list.append(torch.zeros_like(input_cpu[0]), )
            else:
                prototypes_list.append(input_cpu[idx_list].mean(0))
    else:
        prototypes_list = []
        for idx_list in support_idxs:
            original_graph = input_cpu[idx_list].mean(0)
            add_motifs = selected_motif[idx_list]
            add_motifs = torch.flatten(add_motifs)
            new_motif = motif_vectors[add_motifs].mean(0)
            new_prototype = torch.stack([original_graph, new_motif], dim=0)
            new_prototype = new_prototype.mean(0)
            prototypes_list.append(new_prototype)
    # prototypes shape [2, emb_dim]
    prototypes = torch.stack(prototypes_list)
    # query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support], classes))).view(-1)
    # query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(input_cpu, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    loss_val = -log_p_y.squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val, prototypes


def proto_loss_qry(logits, y_t, prototypes, add_motif, motif_vectors):
    '''
    :param logits: graph embedding, shape [num_graphs, emb_dim]
    :param y_t: label tensor, shape [1, num_graphs]
    :param prototypes: number of prototypes = number of classes
    :param add_motif: if adding motifs
    :param motif_vectors: [num_motifs, emb_dim]
    :return: Loss, scalar
    '''
    # put label to cpu
    target_cpu = y_t.squeeze().to('cpu')
    # put graph embedding to cpu
    input_cpu = logits.to('cpu')

    # number of classes = 2
    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # number of queries
    n_query = int(logits.shape[0])

    # query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    # query_samples = input_cpu[query_idxs]

    # compute euclidean distance between graph embedding and prototypes
    dists = euclidean_dist(input_cpu, prototypes)

    # cross entropy loss
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    #
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    loss_val = -log_p_y.squeeze().view(-1).mean()

    #
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val


class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


class Interact_attention(nn.Module):
    def __init__(self, dim, num_tasks):
        super(Interact_attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_tasks * dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Meta_model(nn.Module):
    def __init__(self, args):
        super(Meta_model, self).__init__()

        self.dataset = args.dataset
        self.num_tasks = args.num_tasks
        self.num_train_tasks = args.num_train_tasks
        self.num_test_tasks = args.num_test_tasks
        self.n_way = args.n_way
        self.m_support = args.m_support
        self.k_query = args.k_query
        self.gnn_type = args.gnn_type
        self.emb_dim = args.emb_dim
        self.device = args.device
        self.add_similarity = args.add_similarity
        self.add_selfsupervise = args.add_selfsupervise
        self.add_masking = args.add_masking
        self.add_weight = args.add_weight
        # add prototypes
        self.add_prototype = args.add_prototype
        self.interact = args.interact
        self.batch_size = args.batch_size
        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.criterion = nn.BCEWithLogitsLoss()
        self.graph_model = GINPredictor(num_node_emb_list=[num_atom_type, num_chirality_tag],
                                        num_edge_emb_list=[num_bond_type, num_bond_direction])
        # consider pretrained model ?
        if args.pretrained_gnn:
            self.graph_model.from_pretrained()
        self.motif_model = self.graph_model

        # consider self-supervised loss ?
        if self.add_selfsupervise:
            self.self_criterion = nn.BCEWithLogitsLoss()

        # consider node masking prediction loss?
        if self.add_masking:
            self.masking_criterion = nn.CrossEntropyLoss()
            self.masking_linear = nn.Linear(self.emb_dim, 119)

        # consider task similarity?
        if self.add_similarity:
            self.Attention = attention(self.emb_dim)

        # consider interact?
        if self.interact:
            self.softmax = nn.Softmax(dim=0)
            self.Interact_attention = Interact_attention(self.emb_dim, self.num_train_tasks)

        # Consider motif
        self.add_motif = args.add_motif

        model_param_group = []
        model_param_group.append({"params": self.graph_model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": self.graph_model.pool.parameters(), "lr": args.lr * args.lr_scale})
        model_param_group.append(
            {"params": self.graph_model.predict.parameters(), "lr": args.lr * args.lr_scale})

        if self.add_masking:
            model_param_group.append({"params": self.masking_linear.parameters()})

        if self.add_similarity:
            model_param_group.append({"params": self.Attention.parameters()})

        if self.interact:
            model_param_group.append({"params": self.Interact_attention.parameters()})

        self.optimizer = optim.Adam(model_param_group, lr=args.meta_lr, weight_decay=args.decay)

    def update_params(self, loss, update_lr):
        '''
        :param loss: loss
        :param update_lr: learning rate
        :return: gradient vector,  new model parameter vector with one step update
        '''
        # get gradients
        grads = torch.autograd.grad(loss, self.graph_model.parameters())
        return parameters_to_vector(grads), parameters_to_vector(self.graph_model.parameters()) - parameters_to_vector(
            grads) * update_lr

    def build_negative_edges(self, batched_graphs):
        font_list = batched_graphs.edges()[0][::2].long().tolist()
        back_list = batched_graphs.edges()[1][::2].long().tolist()
        all_edge = {}
        for count, front_e in enumerate(font_list):
            if front_e not in all_edge:
                all_edge[front_e] = [back_list[count]]
            else:
                all_edge[front_e].append(back_list[count])

        negative_edges = []
        for num in range(batched_graphs.nodes().shape[0]):
            if num in all_edge:
                for num_back in range(num, batched_graphs.nodes().shape[0]):
                    if num_back not in all_edge[num] and num != num_back:
                        negative_edges.append((num, num_back))
            else:
                for num_back in range(num, batched_graphs.nodes().shape[0]):
                    if num != num_back:
                        negative_edges.append((num, num_back))

        negative_edge_index = torch.tensor(np.array(random.sample(negative_edges, len(font_list))).T, dtype=torch.long)

        return negative_edge_index

    def forward(self, epoch):
        # set of support dataset
        dataset_dgl, dataset_motif = load_dataset(self.dataset)
        motif_list = dataset_motif[0]
        print('Filtering vocabulary motifs and Getting motif vector list')
        motif_vector_list = []
        for i in range(len(motif_list)):
            # delete one-atom useless motif and delete inorganic motif
            if motif_list[i].number_of_nodes() != 1 and 5 in motif_list[i].ndata['atomic_number']:
                categorical_motif_node_feats = [motif_list[i].ndata['atomic_number'],
                                                motif_list[i].ndata['chirality_type']]
                categorical_motif_edge_feats = [motif_list[i].edata['bond_type'],
                                                motif_list[i].edata['bond_direction_type']]
                _, motif_node_vector = self.motif_model(motif_list[i],
                                           categorical_motif_node_feats,
                                           categorical_motif_edge_feats)
                motif_vector_list.append(motif_node_vector.mean(0))

        self.motif_vectors = torch.stack(motif_vector_list)
        # 631 left for sider
        support_loaders = []
        query_loaders = []
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        # graph_model turn to train state
        self.graph_model.train()
        # tasks_list = random.sample(range(0,self.num_train_tasks), self.batch_size)
        print(self.num_train_tasks)
        for task in range(self.num_train_tasks):
            # for task in tasks_list:
            # load dataset for specific task
            dataset = MetaMoleDataset(dataset_dgl, task)
            # get support set and query dataset for training
            support_dataset, query_dataset = sample_datasets(dataset, self.dataset, task, self.m_support,
                                                             self.k_query)
            # load support set
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size,
                                        shuffle=False, num_workers=1, collate_fn=collate)
            # load query set
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=1, collate_fn=collate)
            # append single task support
            support_loaders.append(support_loader)
            # append single task query
            query_loaders.append(query_loader)

        # loop update step
        for k in range(0, self.update_step):
            print("update step: " + str(k))
            # print(self.fi)
            # store old model parameters at the beginning of update
            old_params = parameters_to_vector(self.graph_model.parameters())
            # start total query loss for all training tasks
            losses_q = torch.tensor([0.0]).to(device)

            # loop train tasks
            for task in range(self.num_train_tasks):
                # start support loss
                losses_s = torch.tensor([0.0]).to(device)
                if self.add_similarity or self.interact:
                    # start task embedding
                    one_task_emb = torch.zeros(300).to(device)
                # loop step and batch
                for step, (batched_graphs, batched_labels) in enumerate(tqdm(support_loaders[task], desc="Iteration")):
                    batched_graphs = batched_graphs.to(device)
                    batched_labels = batched_labels.to(device)
                    # generate prediction and node embedding on batch of graphs
                    categorical_node_feats = [batched_graphs.ndata['atomic_number'],
                                              batched_graphs.ndata['chirality_type']]
                    categorical_edge_feats = [batched_graphs.edata['bond_type'],
                                              batched_graphs.edata['bond_direction_type']]
                    pred, node_emb = self.graph_model(batched_graphs, categorical_node_feats, categorical_edge_feats)
                    # get batch of labels
                    y = batched_labels.view(pred.shape).to(torch.float64)
                    # compute loss
                    loss = torch.sum(self.criterion(pred.double(), y)) / pred.size()[0]
                    # if add self-supervise loss (link prediction)
                    if self.add_selfsupervise:
                        # compute positive pair similarity
                        positive_score = torch.sum(node_emb[batched_graphs.edges()[0][::2].long()] *
                                                   node_emb[batched_graphs.edges()[1][::2].long()], dim=1)
                        # get negative edges
                        negative_edge_index = self.build_negative_edges(batched_graphs)
                        # compute negative pair similarity
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]],
                                                   dim=1)
                        # self_loss = positive score + negative + score
                        self_loss = torch.sum(
                            self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(
                                negative_score, torch.zeros_like(negative_score))) / negative_edge_index[0].size()[0]
                        # self_loss has weight 0.1 in total loss
                        loss += (self.add_weight * self_loss)

                    # if add masking (node prediction)
                    batched_node_feats = torch.cat([batched_graphs.ndata['atomic_number'].reshape(-1, 1),
                                                    batched_graphs.ndata['chirality_type'].reshape(-1, 1)], axis=1)
                    if self.add_masking:
                        # sample masking node index
                        mask_num = random.sample(range(0, node_emb.size()[0]), self.batch_size)
                        # predict embedding on masked node
                        # predict embedding on masked node
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        # add node prediction loss to total loss
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(),
                                                                          batched_node_feats[mask_num, 0]))

                    # if add_similarity (?)
                    if self.add_similarity or self.interact:
                        one_task_emb = torch.div((one_task_emb + torch.mean(node_emb, 0)), 2.0)


                    # if add_prototype
                    if self.add_prototype:
                        # compute prototypes
                        # graph_emb is in shape (num_graphs, emb_dim)
                        graph_emb = get_graph_emb(node_emb, batched_graphs.batch_num_nodes())
                        # prototypical loss and prototypes
                        proto_loss, _, prototypes = proto_loss_spt(graph_emb, y, pred.size()[0], self.add_motif, self.motif_vectors)
                        # add proto_loss
                        loss += proto_loss
                    # batch support loss add
                    losses_s += loss

                if self.add_similarity or self.interact:
                    if task == 0:
                        tasks_emb = []
                    tasks_emb.append(one_task_emb)

                # get new gradient vector and new updated model parameter
                new_grad, new_params = self.update_params(losses_s, update_lr=self.update_lr)

                # assign new updated paramters to model parameters
                vector_to_parameters(new_params, self.graph_model.parameters())

                # start query loss
                this_loss_q = torch.tensor([0.0]).to(device)
                # loop query set
                for step, (batched_graphs, batched_labels) in enumerate(tqdm(query_loaders[task], desc="Iteration")):
                    batched_graphs = batched_graphs.to(device)
                    batched_labels = batched_labels.to(device)
                    # generate prediction and node embedding on batch of graphs
                    categorical_node_feats = [batched_graphs.ndata['atomic_number'],
                                              batched_graphs.ndata['chirality_type']]
                    categorical_edge_feats = [batched_graphs.edata['bond_type'],
                                              batched_graphs.edata['bond_direction_type']]
                    pred, node_emb = self.graph_model(batched_graphs, categorical_node_feats, categorical_edge_feats)
                    # get batch of labels
                    y = batched_labels.view(pred.shape).to(torch.float64)
                    # compute loss
                    loss_q = torch.sum(self.criterion(pred.double(), y)) / pred.size()[0]

                    # if self-supervised
                    if self.add_selfsupervise:
                        positive_score = torch.sum(node_emb[batched_graphs.edges()[0][::2].long()] *
                                                   node_emb[batched_graphs.edges()[1][::2].long()], dim=1)
                        # get negative edges
                        negative_edge_index = self.build_negative_edges(batched_graphs)
                        # compute negative pair similarity
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]],
                                                   dim=1)
                        # self_loss = positive score + negative + score
                        self_loss = torch.sum(
                            self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(
                                negative_score, torch.zeros_like(negative_score))) / negative_edge_index[0].size()[0]
                        # self_loss has weight 0.1 in total loss
                        loss_q += (self.add_weight * self_loss)

                    # for node prediction
                    batched_node_feats = torch.cat([batched_graphs.ndata['atomic_number'].reshape(-1, 1),
                                                    batched_graphs.ndata['chirality_type'].reshape(-1, 1)], axis=1)
                    if self.add_masking:
                        # mask some nodes
                        mask_num = random.sample(range(0, node_emb.size()[0]), self.batch_size)
                        # predict masked nodes
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        # compute loss
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(),
                                                                          batched_node_feats[mask_num, 0]))

                    if self.add_prototype:
                        graph_emb = get_graph_emb(node_emb, batched_graphs.batch_num_nodes())
                        proto_loss_q, acc_q = proto_loss_qry(graph_emb, y, prototypes, self.add_motif, self.motif_vectors)
                        # loss_q += proto_loss_q

                    # sum query loss
                    this_loss_q += loss_q

                # first task
                if task == 0:
                    # total query loss is one batch query loss
                    losses_q = this_loss_q
                else:
                    # total query loss is concatenation of all batch query loss
                    losses_q = torch.cat((losses_q, this_loss_q), 0)
                #
                vector_to_parameters(old_params, self.graph_model.parameters())

            # if add similarity
            if self.add_similarity:
                for t_index, one_task_e in enumerate(tasks_emb):
                    if t_index == 0:
                        tasks_emb_new = one_task_e
                    else:
                        tasks_emb_new = torch.cat((tasks_emb_new, one_task_e), 0)

                tasks_emb_new = torch.reshape(tasks_emb_new, (self.num_train_tasks, self.emb_dim))
                tasks_emb_new = tasks_emb_new.detach()

                tasks_weight = self.Attention(tasks_emb_new)
                losses_q = torch.sum(tasks_weight * losses_q)

            elif self.interact:
                for t_index, one_task_e in enumerate(tasks_emb):
                    if t_index == 0:
                        tasks_emb_new = one_task_e
                    else:
                        tasks_emb_new = torch.cat((tasks_emb_new, one_task_e), 0)

                tasks_emb_new = tasks_emb_new.detach()
                represent_emb = self.Interact_attention(tasks_emb_new)
                represent_emb = F.normalize(represent_emb, p=2, dim=0)

                tasks_emb_new = torch.reshape(tasks_emb_new, (self.num_train_tasks, self.emb_dim))
                tasks_emb_new = F.normalize(tasks_emb_new, p=2, dim=1)

                tasks_weight = torch.mm(tasks_emb_new, torch.reshape(represent_emb, (self.emb_dim, 1)))
                print(tasks_weight)
                print(self.softmax(tasks_weight))
                print(losses_q)

                # tasks_emb_new = tasks_emb_new * torch.reshape(represent_emb_m, (self.batch_size, self.emb_dim))
                losses_q = torch.sum(losses_q * torch.transpose(self.softmax(tasks_weight), 1, 0))
                print(losses_q)

            else:
                # add all query loss across task
                losses_q = torch.sum(losses_q)
            # average all query losses over all tasks
            loss_q = losses_q / self.num_train_tasks
            # update on query set
            self.optimizer.zero_grad()
            loss_q.backward()
            self.optimizer.step()
        return []

    def test(self, support_grads):
        # record accuracy
        accs = []
        # start old_params
        old_params = parameters_to_vector(self.graph_model.parameters())
        # load dataset
        dataset_dgl, dataset_motif = load_dataset(self.dataset)
        # loop all testing tasks
        for task in range(self.num_test_tasks):
            print(self.num_tasks - task)
            # load test dataset
            dataset = MetaMoleDataset(dataset_dgl, self.num_tasks-task-1)
            # support dataset, query_dataset
            support_dataset, query_dataset = sample_test_datasets(dataset, self.dataset, self.num_tasks-task-1,
                                                                  self.n_way, self.m_support, self.k_query)
            # support loader and query loader
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                        collate_fn=collate)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                      collate_fn=collate)

            device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")

            # model turn to evaluation mode
            self.graph_model.eval()

            # loop over all update steps
            for k in range(0, self.update_step_test):
                # start loss
                loss = torch.tensor([0.0]).to(device)
                # loop support set
                for step, (batched_graphs, batched_labels) in enumerate(tqdm(support_loader, desc="Iteration")):
                    batched_graphs = batched_graphs.to(device)
                    batched_labels = batched_labels.to(device)
                    # generate prediction and node embedding on batch of graphs
                    categorical_node_feats = [batched_graphs.ndata['atomic_number'],
                                              batched_graphs.ndata['chirality_type']]
                    categorical_edge_feats = [batched_graphs.edata['bond_type'],
                                              batched_graphs.edata['bond_direction_type']]
                    pred, node_emb = self.graph_model(batched_graphs, categorical_node_feats, categorical_edge_feats)
                    # get batch of labels
                    y = batched_labels.view(pred.shape).to(torch.float64)
                    # sum loss
                    loss += torch.sum(self.criterion(pred.double(), y)) / pred.size()[0]

                    if self.add_selfsupervise:
                        positive_score = torch.sum(node_emb[batched_graphs.edges()[0][::2].long()] *
                                                   node_emb[batched_graphs.edges()[1][::2].long()], dim=1)
                        # get negative edges
                        negative_edge_index = self.build_negative_edges(batched_graphs)
                        # compute negative pair similarity
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]],
                                                   dim=1)
                        # self_loss = positive score + negative + score
                        self_loss = torch.sum(
                            self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(
                                negative_score, torch.zeros_like(negative_score))) / negative_edge_index[0].size()[0]
                        # self_loss has weight 0.1 in total loss
                        loss += (self.add_weight * self_loss)

                    # for node prediction
                    batched_node_feats = torch.cat([batched_graphs.ndata['atomic_number'].reshape(-1, 1),
                                                    batched_graphs.ndata['chirality_type'].reshape(-1, 1)], axis=1)
                    if self.add_masking:
                        mask_num = random.sample(range(0, node_emb.size()[0]), self.batch_size)
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(),
                                                                          batched_node_feats[mask_num, 0]))

                    if self.add_prototype:
                        graph_emb = get_graph_emb(node_emb, batched_graphs.batch_num_nodes())
                        proto_loss, _, prototypes = proto_loss_spt(graph_emb, y, pred.size()[0], self.add_motif, self.motif_vectors)
                        loss += proto_loss

                    print(loss)

                # get new gradients and new model parameters
                new_grad, new_params = self.update_params(loss, update_lr=self.update_lr)

                # if self.add_similarity:
                #     new_params = self.update_similarity_params(new_grad, support_grads)
                vector_to_parameters(new_params, self.graph_model.parameters())

            y_true = []
            y_scores = []
            for step, (batched_graphs, batched_labels) in enumerate(tqdm(query_loader, desc="Iteration")):
                batched_graphs = batched_graphs.to(device)
                batched_labels = batched_labels.to(device)
                # generate prediction and node embedding on batch of graphs
                categorical_node_feats = [batched_graphs.ndata['atomic_number'],
                                          batched_graphs.ndata['chirality_type']]
                categorical_edge_feats = [batched_graphs.edata['bond_type'],
                                          batched_graphs.edata['bond_direction_type']]

                pred, node_emb = self.graph_model(batched_graphs, categorical_node_feats, categorical_edge_feats)

                if self.add_prototype:
                    graph_emb = get_graph_emb(node_emb, batched_graphs.batch_num_nodes())
                    proto_loss_q, acc_q = proto_loss_qry(graph_emb, y, prototypes, self.add_motif, self.motif_vectors)

                # print(pred)
                pred = F.sigmoid(pred)
                pred = torch.where(pred > 0.5, torch.ones_like(pred), pred)
                pred = torch.where(pred <= 0.5, torch.zeros_like(pred), pred)
                y_scores.append(pred)
                y_true.append(batched_labels.view(pred.shape))

            y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
            y_scores = torch.cat(y_scores, dim=0).cpu().detach().numpy()

            roc_list = []
            roc_list.append(roc_auc_score(y_true, y_scores))
            acc = sum(roc_list) / len(roc_list)
            accs.append(acc)

            vector_to_parameters(old_params, self.graph_model.parameters())

        return accs
