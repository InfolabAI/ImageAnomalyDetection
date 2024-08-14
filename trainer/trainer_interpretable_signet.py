# NOTE 이 코드는 SIGNET 을 backbone feature 와 함께 그대로 이용해보기 위한 코드로 현재 사용되지 않음.

from loguru import logger
from torch_geometric.data import Data, InMemoryDataset
import torch.nn as nn
import math
import numpy as np
import wandb
import torch
import tqdm
from utils import plot_segmentation_images
import torch
import pickle
from trainer.trainer import Trainer
from simplenet import Discriminator, Projection
from models.signet_models import GIN, HyperGNN, Explainer_MLP, Explainer_GIN
from datasets.my_inmemory_dataset import MyInMemoryDataset
from argparse import Namespace


class Trainer_Interpretable_SIGNET(Trainer):
    # For graph data construction
    def initialize_model(self, dsc_layers, dsc_hidden, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, dsc_lr, lr, **kwargs):
        self.signet_args = Namespace(dataset='mnist0', batch_size=self.args.batch_size, batch_size_test=64, log_interval=1, num_trials=5, device=0, lr=0.1, epochs=50, encoder_layers=2,
                                     hidden_dim=16, pooling='add', readout='concat', explainer_model='mlp', explainer_layers=2, explainer_hidden_dim=16, explainer_readout='add')
        self.discriminator = SIGNET(
            input_dim=self.target_embed_dimension, input_dim_edge=0, args=self.signet_args, device=self.device).to(self.device)
        self.dsc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.signet_args.lr)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin = dsc_margin
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(
                self.pre_projection.parameters(), lr*.1)

    def _preprocessing_image(self, dict_):
        img = dict_["image"]
        img = img.to(torch.float).to(self.device)
        true_feats, patch_shapes = self._embed(img, evaluation=False)
        # if self.pre_proj > 0:
        #    true_feats = self.pre_projection(
        #        true_feats)  # feature adapter

        true_feats = true_feats.reshape(-1,
                                        np.prod(patch_shapes[0]), true_feats.shape[-1])  # NOTE [batch*#patches, #features] --> [batch, #patches, #features]
        return true_feats

    def _graph_construction_per_graph(self, true_feats, y):
        """
        Parameters
        ----------
        true_feats : torch.Tensor
            [batch, #patches, #features]

        Returns
        -------
        data : dict
            DataBatch 형태의 데이터

        Examples
        --------
        >>>  p data
            DataBatch(x=[8962, 5], edge_index=[2, 74240], edge_attr=[74240, 1], y=[128], node_label=[8962], edge_label=[
                      74240], sp_order=[8962], superpixels=[3584, 28], name=[128], idx=[128], batch=[8962], ptr=[129])
        >>>  p data.x
        tensor([[0.0558, 0.0558, 0.0558, 0.1389, 0.4683],
                ...,
                [0.7900, 0.7900, 0.7900, 0.3791, 0.7527]], device='cuda:0')
        >>>  p data.edge_index
            tensor([[   0,    0,    0,  ..., 8961, 8961, 8961],
                    [   1,    7,   19,  ..., 8938, 8955, 8958]], device='cuda:0')
        >>>  p data.edge_attr
            tensor([[0.2604],
                    [0.2689],
                    [0.1095],
                    ...,
                    [0.2120],
                    [0.1420],
                    [0.1061]], device='cuda:0')
        >>>  p data.y
            tensor([0, 0, ... , 0, 0, 0], device='cuda:0') NOTE 모두 제로
        >>>  p data.node_label
            tensor([1., 1., 1.,  ..., 0., 1., 1.], device='cuda:0')
        >>>  p data.edge_label
            tensor([1., 0., 0.,  ..., 1., 0., 1.], device='cuda:0')
        >>>  p data.sp_order
            tensor([14, 18, 55,  ..., 64, 41, 28],
                   device='cuda:0', dtype=torch.int32)
        >>>  p data.superpixels
            tensor([[ 1,  1,  1,  ...,  9,  9,  9],
                    [ 1,  1,  1,  ...,  9,  9,  9],
                    ...,
                    [60, 60, 60,  ..., 64, 64, 64],
                    [60, 60, 60,  ..., 64, 64, 64]], device='cuda:0')
        >>>  p data.name
            ['MNISTSP-train-58973', 'MNISTSP-train-54764', 'MNISTSP-train-16043', 'MNISTSP-train-50101', ...,  ,'MNISTSP-train-33665', 'MNISTSP-train-46090', 'MNISTSP-train-42579', 'MNISTSP-train-7164', 'MNISTSP-train-50862']
        >>>  p data.idx
            tensor([58973, 54764, 16043, ... , 42579,
                   7164, 50862], device='cuda:0')
        >>>  p data.batch
            tensor([  0,   0,   0,  ..., 127, 127, 127], device='cuda:0')
        >>>  p data.ptr
            tensor([   0,   72,  146,  215,  285,  ... , 8689,
                   8762, 8828, 8898, 8962], device='cuda:0')
        """
        true_feats = true_feats.unsqueeze(3).permute(
            0, 2, 1, 3)  # [batch, #features, #patches, 1]
        # edge_index.shape == [2 (NN, self), batch, #patches, #edges_per_patch]
        edge_index, dist = self.dense_knn_matrix(true_feats)
        _, _, _, edges_per_patch = edge_index.shape
        """
       Example of edge_index
       >>>  p edge_index
           tensor([[[   0,    1,   36,  ...,    4,  144,   73],
                   [   1,   36,    0,  ...,   73,  108,    5],
                   [   2,    1,    3,  ...,    6,   31,   30],
                   ...,
                   [1293, 1292, 1294,  ..., 1273, 1287, 1275],
                   [1294, 1259, 1293,  ..., 1260, 1262, 1222],
                   [1295, 1294, 1259,  ..., 1225, 1290, 1187]]], device='cuda:0')
       """
        # NOTE Transpose edge_index and true_feats into data

        # Flatten the tensors to the required shapes
        batch_size, num_features, num_patches, _ = true_feats.shape
        # assert batch_size == 1, "batch_size should be 1"

        # [2, batch, #patches * #edges_per_patch] --> [batch, 2, #patches * #edges_per_patch]
        edge_index = edge_index.view(
            edge_index.shape[0], edge_index.shape[1], -1).permute(1, 0, 2)

        # [batch, #patches, #features]
        x = true_feats.permute(0, 2, 1, 3).view(
            batch_size, num_patches, num_features)

        # 0-1 normalization
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        # 가까울수록 edge 가중치가 높아야 하므로, 1-dist, [batch, #patches, #patches]
        dist = 1-dist
        # Create dummy data for the example
        batch_indices = torch.arange(
            batch_size, device=edge_index.device).view(-1, 1).repeat(1, edge_index.size(2))

        data_list = []
        for i in range(batch_size):
            data = {}
            data['x'] = x[i]
            data['edge_attr'] = dist[i, edge_index[i]
                                     [0], edge_index[i][1]].unsqueeze(1)
            data['edge_index'] = edge_index[i]
            data['y'] = y[i:i+1]

            # Transform data into torch geometric data
            data = Data(**data)
            data_list.append(data)

        return data_list

    def train(self, training_data, val_data, test_data, dataset_name):
        _ = self.forward_modules.eval()
        data_list = []
        for data_item in training_data:
            out = self._preprocessing_image(data_item)
            data_list += self._graph_construction_per_graph(
                out, data_item['is_anomaly'])

        torch.save(data_list, f'./{dataset_name}_train.pt')
        # train_dataset = self._data_to_inmemorydata(data_list)

        data_list = []
        for data_item in val_data:
            out = self._preprocessing_image(data_item)
            data_list += self._graph_construction_per_graph(
                out, data_item['is_anomaly'])

        torch.save(data_list, f'./{dataset_name}_val.pt')
        # val_dataset = self._data_to_inmemorydata(data_list)

        data_list = []
        for data_item in test_data:
            out = self._preprocessing_image(data_item)
            data_list += self._graph_construction_per_graph(
                out, data_item['is_anomaly'])

        torch.save(data_list, f'./{dataset_name}_test.pt')
        # test_dataset = self._data_to_inmemorydata(data_list)

        # save InMemoryDataset
        # torch.save(train_dataset, f'./{dataset_name}_train.pt')
        # torch.save(val_dataset, f'./{dataset_name}_val.pt')
        # torch.save(test_dataset, f'./{dataset_name}_test.pt')

        exit(0)

    def _data_to_inmemorydata(self, data_list):
        # 커스텀 데이터셋 생성
        dataset = MyInMemoryDataset(data_list)
        return dataset

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        logger.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in tqdm.tqdm(input_data, desc="Training discriminator", leave=False):
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    out = self._preprocessing_image(data_item)
                    data = self._graph_construction(out)
                    all_loss.append(self._train_signet(data))

                    if self.pre_proj > 0:
                        self.proj_opt.step()

                if self.cos_lr:
                    self.dsc_schl.step()

                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{self._loss_str( all_loss)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

        return self._loss_str(all_loss, ret=True)

    def _train_signet(self, data):
        self.dsc_opt.zero_grad()
        data = data.to(self.device)
        y, y_hyper, node_imp, edge_imp = self.discriminator(data)
        # NOTE batch 가 1 일 경우, loss 가 -24 정도가 되는 것은 원래 코드도 동일함. batch 가 커질수록 loss 가 커짐
        # NOTE batch 가 1 이 아닐 경우, 원래 코드는 y, y_hyper 둘 다 batch 가 1이 아님. 그런데, 여기 코드에서는 y_hyper 의 batch 가 1임. 여기서 문제가 발생하는 듯.
        loss = self.discriminator.loss_nce(y, y_hyper).mean()
        loss.backward()
        self.dsc_opt.step()
        return loss.cpu().item()

    def dense_knn_matrix(self, x, k=5, relative_pos=None):
        """Get KNN based on the pairwise distance.
        Args:
            x: (batch_size, num_dims, num_points, 1)
            k: int
        Returns:
            nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
        """
        def pairwise_distance(x):
            """
            Compute pairwise distance of a point cloud.
            Args:
                x: tensor (batch_size, num_points, num_dims)
            Returns:
                pairwise distance: (batch_size, num_points, num_points)
            """
            with torch.no_grad():
                x_inner = -2*torch.matmul(x, x.transpose(2, 1))
                x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
                return x_square + x_inner + x_square.transpose(2, 1)

        def part_pairwise_distance(x, start_idx=0, end_idx=1):
            """
            Compute pairwise distance of a point cloud.
            Args:
                x: tensor (batch_size, num_points, num_dims)
            Returns:
                pairwise distance: (batch_size, num_points, num_points)
            """
            with torch.no_grad():
                x_part = x[:, start_idx:end_idx]
                x_square_part = torch.sum(
                    torch.mul(x_part, x_part), dim=-1, keepdim=True)
                x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
                x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
                return x_square_part + x_inner + x_square.transpose(2, 1)
        with torch.no_grad():
            x = x.transpose(2, 1).squeeze(-1)
            batch_size, n_points, n_dims = x.shape
            ### memory efficient implementation ###
            n_part = 10000
            if n_points > n_part:
                nn_idx_list = []
                groups = math.ceil(n_points / n_part)
                for i in range(groups):
                    start_idx = n_part * i
                    end_idx = min(n_points, n_part * (i + 1))
                    dist = part_pairwise_distance(
                        x.detach(), start_idx, end_idx)
                    if relative_pos is not None:
                        dist += relative_pos[:, start_idx:end_idx]
                    _, nn_idx_part = torch.topk(-dist, k=k)
                    nn_idx_list += [nn_idx_part]
                nn_idx = torch.cat(nn_idx_list, dim=1)
            else:
                # NOTE n_points 가 196 이라 이쪽으로
                dist = pairwise_distance(x.detach())
                if relative_pos is not None:
                    dist += relative_pos
                _, nn_idx = torch.topk(-dist, k=k)  # b, n, k

            ######
            center_idx = torch.arange(0, n_points, device=x.device).repeat(
                batch_size, k, 1).transpose(2, 1)  # 자기 자신
        return torch.stack((nn_idx, center_idx), dim=0), dist


class SIGNET(nn.Module):
    def __init__(self, input_dim, input_dim_edge, args, device):
        super(SIGNET, self).__init__()

        self.device = device

        self.embedding_dim = args.hidden_dim
        if args.readout == 'concat':
            self.embedding_dim *= args.encoder_layers

        if args.explainer_model == 'mlp':
            self.explainer = Explainer_MLP(
                input_dim, args.explainer_hidden_dim, args.explainer_layers)
        else:
            self.explainer = Explainer_GIN(input_dim, args.explainer_hidden_dim,
                                           args.explainer_layers, args.explainer_readout)

        self.encoder = GIN(input_dim, args.hidden_dim,
                           args.encoder_layers, args.pooling, args.readout)
        self.encoder_hyper = HyperGNN(
            input_dim, input_dim_edge, args.hidden_dim, args.encoder_layers, args.pooling, args.readout)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_hyper = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                             nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        node_imp = self.explainer(data.x, data.edge_index, data.batch)
        edge_imp = self.lift_node_score_to_edge_score(
            node_imp, data.edge_index)

        y, _ = self.encoder(data.x, data.edge_index, data.batch, node_imp)
        y_hyper, _ = self.encoder_hyper(
            data.x, data.edge_index, data.edge_attr, data.batch, edge_imp)

        y = self.proj_head(y)
        y_hyper = self.proj_head_hyper(y_hyper)

        return y, y_hyper, node_imp, edge_imp

    @staticmethod
    def loss_nce(x1, x2, temperature=0.2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
            torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)

        loss_0 = - torch.log(loss_0 + 1e-10)
        loss_1 = - torch.log(loss_1 + 1e-10)
        loss = (loss_0 + loss_1) / 2.0
        return loss

    def lift_node_score_to_edge_score(self, node_score, edge_index):
        src_lifted_att = node_score[edge_index[0]]
        dst_lifted_att = node_score[edge_index[1]]
        edge_score = src_lifted_att * dst_lifted_att
        return edge_score
