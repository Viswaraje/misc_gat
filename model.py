import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool
from torch.nn import Parameter
from my_utiils import *

from base_model.SGATConv import SGATConv  # Replace SGConv with SGATConv
from base_model.GATDConv import GATDConv  # Replace SGConv with SGATConv

EPS = 1e-15

class NodeRepresentation(nn.Module):
    def __init__(self, gcn_layer, dim_gexp, dim_methy, output, units_list=[256, 256, 256], use_relu=True, use_bn=True,
                 use_GMP=True, use_mutation=True, use_gexpr=True, use_methylation=True):
        super(NodeRepresentation, self).__init__()
        torch.manual_seed(0)
        
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP
        self.use_mutation = use_mutation
        self.use_gexpr = use_gexpr
        self.use_methylation = use_methylation
        
        # Drug representation layers using SGATConv
        self.dropout = nn.Dropout(0.3)  # Add dropout
        self.conv1 = SGATConv(gcn_layer, units_list[0])
        self.batch_conv1 = nn.LayerNorm(units_list[0])  # Change to LayerNorm
        
        self.graph_conv = nn.ModuleList()
        self.graph_bn = nn.ModuleList()
        
        for i in range(len(units_list) - 1):
            self.graph_conv.append(SGATConv(units_list[i], units_list[i + 1]))
            self.graph_bn.append(nn.LayerNorm(units_list[i + 1]))
        
        self.conv_end = SGATConv(units_list[-1], output)
        self.batch_end = nn.LayerNorm(output)
        
        # Cell line representation layers
        self.fc_gexp1 = nn.Linear(dim_gexp, 256)
        self.batch_gexp1 = nn.BatchNorm1d(256)
        self.fc_gexp2 = nn.Linear(256, output)
        
        self.fc_methy1 = nn.Linear(dim_methy, 256)
        self.batch_methy1 = nn.BatchNorm1d(256)
        self.fc_methy2 = nn.Linear(256, output)
        
        self.cov1 = nn.Conv2d(1, 50, (1, 700), stride=(1, 5))
        self.cov2 = nn.Conv2d(50, 30, (1, 5), stride=(1, 2))
        self.fla_mut = nn.Flatten()
        self.fc_mut = nn.Linear(2010, output)
        
        self.fcat = nn.Linear(300, output)
        self.batchc = nn.BatchNorm1d(100)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data):
        # Drug representation
        x_drug = self.conv1(drug_feature, drug_adj)
        print(f"After conv1: {x_drug.shape}")
        x_drug = F.relu(x_drug)
        x_drug = self.batch_conv1(x_drug)
        print(f"After batch_conv1: {x_drug.shape}")

        for i in range(len(self.units_list) - 1):
            x_drug = self.graph_conv[i](x_drug, drug_adj)
            print(f"After graph_conv[{i}]: {x_drug.shape}")
            x_drug = F.relu(x_drug)
            x_drug = self.graph_bn[i](x_drug)
            print(f"After graph_bn[{i}]: {x_drug.shape}")
        
        x_drug = self.conv_end(x_drug, drug_adj)
        print(f"After conv_end: {x_drug.shape}")
        x_drug = F.relu(x_drug)
        x_drug = self.batch_end(x_drug)
        print(f"After batch_end: {x_drug.shape}")
        
        x_drug = gmp(x_drug, ibatch) if self.use_GMP else global_mean_pool(x_drug, ibatch)
        print(f"After pooling (GMP or mean_pool): {x_drug.shape}")

        # Cell line representation (Mutation, Gene Expression, Methylation)
        if self.use_mutation:
            x_mutation = torch.tanh(self.cov1(mutation_data))
            x_mutation = F.max_pool2d(x_mutation, (1, 5))
            x_mutation = F.relu(self.cov2(x_mutation))
            x_mutation = F.max_pool2d(x_mutation, (1, 10))
            x_mutation = self.fla_mut(x_mutation)
            x_mutation = F.relu(self.fc_mut(x_mutation))
            print(f"Mutation data shape: {x_mutation.shape}")

        if self.use_gexpr:
            x_gexpr = torch.tanh(self.fc_gexp1(gexpr_data))
            x_gexpr = self.batch_gexp1(x_gexpr)
            x_gexpr = F.relu(self.fc_gexp2(x_gexpr))
            print(f"Gene expression shape: {x_gexpr.shape}")

        if self.use_methylation:
            x_methylation = torch.tanh(self.fc_methy1(methylation_data))
            x_methylation = self.batch_methy1(x_methylation)
            x_methylation = F.relu(self.fc_methy2(x_methylation))
            print(f"Methylation data shape: {x_methylation.shape}")

        # Concatenating omics features
        omics_features = [x for x in [x_mutation, x_gexpr, x_methylation] if x is not None]
        x_cell = torch.cat(omics_features, dim=1)
        print(f"After concatenation (omics): {x_cell.shape}")
        
        x_cell = F.relu(self.fcat(x_cell))
        print(f"After fcat (omics): {x_cell.shape}")
        
        # Final concatenation and batch normalization
        x_all = torch.cat((x_cell, x_drug), 0)
        print(f"After final concatenation: {x_all.shape}")
        x_all = self.batchc(x_all)
        print(f"After batchc (final): {x_all.shape}")
        
        return x_all




class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=2, num_layers=3):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        
        # Creating multiple GAT layers
        self.convs = nn.ModuleList()
        self.prelu = nn.ModuleList()
        
        self.convs.append(GATDConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.prelu.append(nn.PReLU(hidden_channels * heads))
        
        for _ in range(self.num_layers - 1):
            self.convs.append(GATDConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.prelu.append(nn.PReLU(hidden_channels * heads))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            print(f"After conv_{i}: {x.shape}")
            x = self.prelu[i](x)
            print(f"After prelu_{i}: {x.shape}")
        return x




class Summary(nn.Module):
    def __init__(self, ino, inn):
        super(Summary, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=ino + inn, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(ino + inn, 1)  # Used for weight computation

    def forward(self, xo, xn):
        # Concatenate inputs (batch, features)
        x = torch.cat((xo, xn), dim=1).unsqueeze(1)  # (batch, 1, features)
        print(f"After concatenation: {x.shape}")

        # Self-attention processing
        attn_output, _ = self.attn(x, x, x)  # (batch, 1, features)
        print(f"After self-attention: {attn_output.shape}")

        # Compute importance weights
        m = self.fc1(attn_output.squeeze(1))  # (batch, 1)
        m = torch.tanh(m)  # Activation function
        m = torch.exp(m) / torch.exp(m).sum(dim=0, keepdim=True)  # Softmax normalization
        print(f"After softmax normalization: {m.shape}")

        # Weighted sum of xn
        x = torch.matmul(m.T, xn)  # (batch, features)
        print(f"After weighted sum: {x.shape}")
        
        return x



class GraphCDR(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, feat, index):
        super(GraphCDR, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.feat = feat
        self.index = index
        self.dropout = nn.Dropout(0.4)  # Increase dropout

        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(100, 10)
        self.fd = nn.Linear(100, 10)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        glorot(self.weight)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data, edge):
        # ---CDR_graph_edge and corrupted CDR_graph_edge
        if not isinstance(edge, torch.Tensor):
            edge = torch.tensor(edge)  # Convert NumPy array to PyTorch tensor

        pos_edge = edge[edge[:, 2] == 1, :2].T.contiguous().long()
        neg_edge = edge[edge[:, 2] == -1, :2].T.contiguous().long()
        
        # ---cell+drug node attributes
        feature = self.feat(drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data)
        
        # ---cell+drug embedding from the CDR graph and the corrupted CDR graph
        pos_z = self.encoder(feature, pos_edge)
        neg_z = self.encoder(feature, neg_edge)
        
        # ---graph-level embedding (summary)
        summary_pos = self.summary(feature, pos_z)
        summary_neg = self.summary(feature, neg_z)
        
        # ---embedding at layer l
        cellpos = pos_z[:self.index, ]
        drugpos = pos_z[self.index:, ]
        
        # ---embedding at layer 0
        cellfea = self.fc(feature[:self.index, ])
        drugfea = self.fd(feature[self.index:, ])
        cellfea = torch.sigmoid(cellfea)
        drugfea = torch.sigmoid(drugfea)
        
        # ---concatenate embeddings at different layers (0 and l)
        cellpos = torch.cat((cellpos, cellfea), 1)
        drugpos = torch.cat((drugpos, drugfea), 1)
        
        # ---inner product
        pos_adj = torch.matmul(cellpos, drugpos.t())
        pos_adj = self.act(pos_adj)
        
        return pos_z, neg_z, summary_pos, summary_neg, pos_adj.view(-1)

    def discriminate(self, z, summary, sigmoid=True):
        z_dim = z.size(-1)
        summary_dim = summary.size(-1)

        if self.weight.size(0) != z_dim or self.weight.size(1) != summary_dim:
            self.weight = Parameter(torch.Tensor(z_dim, summary_dim)).to(z.device)
            # Initialize the new weight using Xavier (Glorot) initialization
            nn.init.xavier_uniform_(self.weight)

        value = torch.matmul(z, torch.matmul(self.weight, summary.T))  # Transposing summary

        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(torch.clamp(self.discriminate(pos_z, summary, sigmoid=True), min=EPS)).mean()
        neg_loss = -torch.log(torch.clamp(1 - self.discriminate(neg_z, summary, sigmoid=True), min=EPS)).mean()
        return pos_loss + neg_loss

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
