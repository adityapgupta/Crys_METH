import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        assert n_layers >= 2
        super(MLP, self).__init__()
        self.map = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(0, n_layers-2):
            self.map.append(nn.Linear(hidden_dim, hidden_dim))
        self.map.append(nn.Linear(hidden_dim, out_dim))
        self.act = nn.SiLU()
        self.n_layers = n_layers
    
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.act(self.map[i](x))
        x = self.map[-1](x)
        return x

class MHA(nn.Module):
    def __init__(self, attn_head, dim, dropout):
        super(MHA, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dim = dim
        self.attn_head = attn_head
        self.softmax = nn.Softmax(dim=-1)
        self.WO = nn.Linear(dim, dim)
    
    def forward(self, q, k, v, mask):
        b, lq, _ = q.shape
        b, kq, _ = k.shape
        b, vq, _ = v.shape

        dim_head = self.dim // self.attn_head
        q = q.reshape(b, lq, self.attn_head, dim_head).transpose(1, 2)
        k = k.reshape(b, kq, self.attn_head, dim_head).transpose(1, 2)
        v = v.reshape(b, vq, self.attn_head, dim_head).transpose(1, 2)
        
        mask = mask.unsqueeze(1).unsqueeze(-1).float()
        mask = mask @ mask.transpose(-1, -2)

        attn_scores = q @ k.transpose(2, 3) / math.sqrt(dim_head)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-1e3'))
        attn = self.softmax(attn_scores)
        attn = self.dropout(attn)
        
        attn_out = (attn @ v).transpose(1, 2).reshape(b, lq, self.dim)
        attn_out = self.dropout(attn_out)
        attn_out = self.WO(attn_out)
        return attn_out

class Decoder_layer(nn.Module):
    def __init__(self, dim, attn_head, dropout):
        assert dim % attn_head == 0
        super(Decoder_layer, self).__init__()
        self.LN_MHA_SA = nn.LayerNorm(dim)
        self.LN_MLP = nn.LayerNorm(dim)
        
        self.qkv = nn.Linear(dim, 3*dim)
        self.mha_sa = MHA(attn_head, dim, dropout)
        self.MLP = nn.Sequential(nn.Linear(dim, 4*dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
    
    def forward(self, h, mask):
        ## self_attention
        h[~mask] = 0.
        h_ini = h
        h = self.LN_MHA_SA(h)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        sa_attn_out = self.mha_sa(q, k, v, mask)
        sa_attn_out[~mask] = 0.
        h = h_ini + sa_attn_out

        ## add & norm
        h = h + self.MLP(self.LN_MLP(h))
        h[~mask] = 0.
        return h
    
class Transformer(nn.Module):
    def __init__(self, hidden_dim, layers, attn_head, dropout):
        super(Transformer, self).__init__()

        self.pe_emb = nn.Parameter(torch.zeros(500, hidden_dim))
        nn.init.trunc_normal_(self.pe_emb, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.block = nn.ModuleList()
        for _ in range(layers):
            self.block.append(Decoder_layer(hidden_dim, attn_head, dropout))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, mask):
        b, l = mask.shape

        cls_tokens = self.cls_token.expand(b, 1, -1)
        h = torch.cat([cls_tokens, h], 1)
        p_emb = self.pe_emb[:l].unsqueeze(0).repeat(b, 1, 1)
        h = h + p_emb

        for i in range(len(self.block)):
            h[~mask] = 0.
            h = self.block[i](h, mask)
        h = self.norm(h)
        h[~mask] = 0.
        return h


def translate(frac, b, batch):
    device = frac.device
    vec_t = torch.rand(b, 3).to(device)
    frac_trans = (frac + vec_t[batch]) % 1.
    return frac_trans, vec_t

def _random_rotation_matrix_so3(batch=1):
    """
    Uniform random rotation matrix (SO(3)) using random unit quaternions.
    Returns: (3,3) if batch=1 else (B,3,3)
    """
    u1 = torch.rand(batch)
    u2 = torch.rand(batch)
    u3 = torch.rand(batch)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1)     * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1)     * torch.cos(2 * torch.pi * u3)  # w

    # Quaternion to rotation matrix
    x, y, z, w = q1, q2, q3, q4
    R = torch.empty((batch, 3, 3))

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)

    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - x*w)

    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    return R[0] if batch == 1 else R, torch.stack([u1, u2, u3], -1)

def rotate(lattice, b):
    R, vec_r = _random_rotation_matrix_so3(b)
    Rt = R.transpose(-1, -2).to(lattice.device)
    lat_rot = lattice @ Rt
    return lat_rot, vec_r.to(lattice.device)


def batch_type_coords(coords, atomic_numbers):
    atomic_tensor = atomic_numbers.unsqueeze(-1)
    all_tensor = torch.cat([atomic_tensor, coords], -1)
    coe = 1000*all_tensor[:, :, 0] + 100*all_tensor[:, :, 1] + 10*all_tensor[:, :, 2] + 1*all_tensor[:, :, 3]
    batch_index_order = torch.argsort(coe)
    return batch_index_order
    
class JEPA(nn.Module):
    def __init__(self, config, matrix_scaler=None):
        super().__init__()
        self.pre_backbone = MLP(109, config.model.hidden_dim, config.model.hidden_dim)
        self.backbone = Transformer(config.model.hidden_dim, config.model.layers, config.model.attn_head, config.model.dropout)
        self.predictor = MLP(config.model.hidden_dim, config.model.hidden_dim, config.model.hidden_dim)
        self.cond_emb = MLP(6, config.model.hidden_dim, config.model.hidden_dim)
        self.device = config.device
        self.matrix_scaler = matrix_scaler

    def to_device(self, data):
        return [i.to(self.device) for i in data]
    
    def encode(self, x, batch):
        x, mask = to_dense_batch(x, batch)
        mask = torch.cat([torch.BoolTensor([[True]]).to(mask.device).repeat(mask.shape[0], 1), mask], 1) # one more True for cls_token
        x = self.pre_backbone(x)
        x = self.backbone(x, mask)
        cls_out = x[:, 0]
        return cls_out

    def cal_weight(self, ef_per_atom):
        weight = 1. - torch.exp(-torch.abs(ef_per_atom.unsqueeze(0) - ef_per_atom.unsqueeze(1)))
        return weight
    
    def get_loss(self, context, target, ef_per_atom):
        b = context.shape[0]
        
        context_norm = torch.norm(context, dim=-1, keepdim=True)
        target_norm = torch.norm(target, dim=-1, keepdim=True)
        dot_numerator = torch.mm(context, target.t())
        dot_denominator = torch.mm(context_norm, target_norm.t())
        sim = dot_numerator / dot_denominator
        
        weight = self.cal_weight(ef_per_atom.reshape(-1))
        diag = torch.eye(b, dtype=torch.bool).float().to(self.device)
        weight = weight + diag

        sim = torch.exp(sim * weight / 0.1)
        sim = sim / (sim.sum(-1).view(-1, 1) + 1e-8)
        return -torch.log(sim.diag() + 1e-8).mean()
    
    def forward(self, data):
        frac_coords, matrix, atomic_numbers, ori_matrix, num_atoms, ef_per_atom = self.to_device(data)
        frac_aug, matrix_aug, atomic_numbers_aug, aug_params = self.aug_batch(frac_coords, ori_matrix, atomic_numbers, num_atoms)
        
        b = matrix.shape[0]
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        
        target = torch.cat([frac_coords, F.one_hot(atomic_numbers, 100).float(), matrix[batch]], -1)
        target = self.encode(target, batch)

        cond = self.cond_emb(aug_params)
        context = torch.cat([frac_aug.float(), F.one_hot(atomic_numbers_aug, 100).float(), matrix_aug[batch].float()], -1)
        context = self.encode(context, batch)
        context = self.predictor(context + cond)

        loss = self.get_loss(context, target, ef_per_atom)
        return loss

    def aug_batch(self, frac_coords, ori_matrix, atomic_numbers, num_atoms):
        b = ori_matrix.shape[0]
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        
        ## aug trans
        frac_aug, vec_t = translate(frac_coords, b, batch)
        batch_frac_aug, mask = to_dense_batch(frac_aug, batch)
        batch_frac_aug[~mask] = 1000
        batch_atomic_numbers, _ = to_dense_batch(atomic_numbers, batch)
        batch_atomic_numbers[~mask] = 1000
        batch_index_order = batch_type_coords(batch_frac_aug, batch_atomic_numbers)

        batch_frac_aug = batch_frac_aug.gather(dim=1, index=batch_index_order.unsqueeze(-1).expand_as(batch_frac_aug))
        batch_atomic_numbers = batch_atomic_numbers.gather(dim=1, index=batch_index_order)
        frac_aug, atomic_numbers_aug = batch_frac_aug[mask], batch_atomic_numbers[mask]

        ## aug rot
        matrix_aug, vec_r = rotate(ori_matrix, b)

        tri_indices = torch.triu_indices(3, 3).to(self.device)
        matrix_aug = matrix_aug[:, tri_indices[0], tri_indices[1]]
        matrix_aug = matrix_aug / num_atoms.reshape(-1, 1)**(1/3)
        matrix_aug = self.matrix_scaler.transform(matrix_aug)

        ## aug params
        aug_params = torch.cat([vec_t, vec_r], -1)
        return frac_aug, matrix_aug, atomic_numbers_aug, aug_params

    def get_emb(self, data):
        frac_coords, matrix, atomic_numbers, _, num_atoms, ef_per_atom = self.to_device(data)
        b = matrix.shape[0]
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        ori_x = torch.cat([frac_coords, F.one_hot(atomic_numbers, 100).float(), matrix[batch]], -1)
        backbone_out = self.encode(ori_x, batch)

        return backbone_out.data.cpu(), ef_per_atom.data.cpu()
