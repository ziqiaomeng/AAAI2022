import dgl
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs


class MetaMoleDataset(DGLDataset):
    '''
    Dataset for Meta-Learning on Molecular Property Prediction Task，27 tasks
    '''

    def __init__(self, dataset_dgl, task_id):
        self.dataset_dgl = dataset_dgl
        self.task_id = task_id
        # load sider dgl dataset
        print('splitting dataset for multi-task meta learning')
        self.process(self.dataset_dgl)

    def process(self, single_dataset):
        self.task_graphs = []
        self.task_labels = []
        for i in range(len(single_dataset)):
            # check whether the property is tested:
            if int(single_dataset[i][3][self.task_id]) == 1:
                # append mol dgl graph
                self.task_graphs.append(single_dataset[i][1])
                # append mol label for certain task id
                self.task_labels.append(single_dataset[i][2][self.task_id])

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.task_graphs[idx], self.task_labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.task_graphs)
