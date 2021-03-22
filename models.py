import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
class StochasticNetwork(nn.Module):
    def __init__(self):
        super(StochasticNetwork, self).__init__()
        if config.NETWORK == 'SAGE':
            self.layers = [
                dglnn.SAGEConv(config.IN_FEATURES , config.HIDDEN_FEATURES, aggregator_type='mean',feat_drop=config.DROPOUT),
                dglnn.SAGEConv(config.HIDDEN_FEATURES , config.HIDDEN_FEATURES, aggregator_type='mean', feat_drop=config.DROPOUT)
            ]
        elif config.NETWORK == 'GAT':
            self.layers = [
                dglnn.GATConv(config.IN_FEATURES, config.HIDDEN_FEATURES, feat_drop=config.DROPOUT,
                              attn_drop=config.ATTN_DROPOUT, num_heads=config.ATTN_HEADS),
                dglnn.GATConv(config.ATTN_HEADS * config.HIDDEN_FEATURES, config.HIDDEN_FEATURES,
                              feat_drop=config.DROPOUT, num_heads=1)
            ]
        elif config.NETWORK == 'GIN':
            self.mlp1 = MLP(1 , config.IN_FEATURES , config.HIDDEN_FEATURES , config.HIDDEN_FEATURES)
            self.mlp2 = MLP(1 , config.HIDDEN_FEATURES , config.HIDDEN_FEATURES , config.HIDDEN_FEATURES)
            self.layers = [
                dglnn.GINConv(apply_func= self.mlp1 , aggregator_type='mean'),
                dglnn.GINConv(apply_func= self.mlp2 , aggregator_type='mean') ,
            ]

        self.layers = torch.nn.ModuleList(self.layers)
        self.final = nn.Linear(config.HIDDEN_FEATURES, 2)


    def forward(self, blocks, x):
        assert len(blocks) == len(self.layers)
        for conv , block in zip(self.layers , blocks):
            x = F.relu(conv(block , x))
            if config.NETWORK == 'GAT':
                x = x.flatten(1)
        x = self.final(x)
        return x


class MLP(nn.Module):
    # MLP with linear output
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)
