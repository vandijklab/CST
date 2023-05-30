# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange, repeat, reduce

from continuous_transformer.utils import split_last, merge_last


from distutils.version import LooseVersion

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def input_mapping(x, B): 
    if B is None:
        return x
    else:
        x_proj = (2.*np.pi*x) @ B.T
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(cfg["dim"]))
        self.beta  = nn.Parameter(torch.zeros(cfg["dim"]))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.operation_with_pos_encoding = cfg["operation_with_pos_encoding"]
        
        self.pos_embed = nn.Embedding(cfg["max_len"], cfg["dim"]) 
        if cfg["operation_with_pos_encoding"] == "concatenate":
            self.pos_embed_3DLin_GaussMap = nn.Linear(512, int(cfg["dim"]/2)) # position embedding: x,y,t or 512 after gaussian-fourier mapping
            if self.Spectral_normalization:
                self.pos_embed_3DLin = SpectralNorm(nn.Linear(3, int(cfg["dim"]/2))) # position embedding: x,y,t or 512 after gaussian-fourier mapping
            else:
                self.pos_embed_3DLin = nn.Linear(3, int(cfg["dim"]/2)) # position embedding: x,y,t or 512 after gaussian-fourier mapping
            
        elif cfg["operation_with_pos_encoding"] == "sum":   
            self.pos_embed_3DLin = nn.Linear(3, cfg["dim"]) # position embedding: x,y,t or 512 after gaussian-fourier mapping
        self.max_len = cfg["max_len"]
        self.frame_size = cfg["frame_size"]
        self.patch_size = cfg["patch_size"]
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg["p_drop_hidden"])

    def forward(self, x, T, P_row, P_col, map_fn):
        verbose =False
        
        pos_t = (T/(self.max_len-1)) 
        pos_t = pos_t.unsqueeze(0).T 

        pos_x = (P_row/(self.frame_size["rows"] - self.patch_size[0]-1))
        pos_x = pos_x.unsqueeze(0).T # (S,) -> (B, S) 
        
        pos_y = (P_col/(self.frame_size["cols"] - self.patch_size[1]-1))
        pos_y = pos_y.unsqueeze(0).T # (S,) -> (B, S) 
        
        mapped_input = input_mapping(torch.cat((pos_t, pos_x, pos_y), 1), map_fn)
        
        if map_fn is None: 
            pos_embed_result = self.pos_embed_3DLin(torch.Tensor(mapped_input).type_as(x)).unsqueeze(0)
            
        else: 
            pos_embed_result = self.pos_embed_3DLin_GaussMap(torch.Tensor(mapped_input).type_as(x)).unsqueeze(0)
            
        pos_embed_result = pos_embed_result.expand_as(x) # (S,) -> (B, S)
            
        if self.operation_with_pos_encoding == "concatenate":
            e = torch.cat((x,pos_embed_result),dim=-1)
        elif self.operation_with_pos_encoding == "sum":   
            e = x + pos_embed_result 
        return self.drop(self.norm(e))

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    verbose = False
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# linear attention classes with softmax kernel
# non-causal linear attention
def linear_attention(q, k, v):
    verbose = False
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps = 1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out

# class FastAttention(nn.Module):
#     """ Linear attention from the Performer
#     Original implementation: https://github.com/lucidrains/performer-pytorch/
#     """
#     def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
#         super().__init__()
#         verbose = False
#         nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

#         self.dim_heads = dim_heads
#         self.nb_features = nb_features
#         self.ortho_scaling = ortho_scaling

#         self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
#         projection_matrix = self.create_projection()
#         self.register_buffer('projection_matrix', projection_matrix)

#         self.generalized_attention = generalized_attention
#         self.kernel_fn = kernel_fn

#         # if this is turned on, no projection will be used
#         # queries and keys will be softmax-ed as in the original efficient attention paper
#         self.no_projection = no_projection

#         self.causal = causal
#         if causal:
#             try:
#                 import fast_transformers.causal_product.causal_product_cuda
#                 self.causal_linear_fn = partial(causal_linear_attention)
#             except ImportError:
#                 print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
#                 self.causal_linear_fn = causal_linear_attention_noncuda

#     @torch.no_grad()
#     def redraw_projection_matrix(self, device):
#         projections = self.create_projection(device = device)
#         self.projection_matrix.copy_(projections)
#         del projections

#     def forward(self, q, k, v):
#         '''
#         # queries / keys / values with heads already split and transposed to first dimension
#         # 8 heads, dimension of head is 64, sequence length of 512
#         q = torch.randn(1, 8, 512, 64)
#         k = torch.randn(1, 8, 512, 64)
#         v = torch.randn(1, 8, 512, 64)
#         '''
        
#         device = q.device

#         if self.no_projection:
#             q = q.softmax(dim = -1)
#             k = torch.exp(k) if self.causal else k.softmax(dim = -2)

#         elif self.generalized_attention:
#             create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
#             q, k = map(create_kernel, (q, k))

#         else:
#             create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
#             q = create_kernel(q, is_query = True)
#             k = create_kernel(k, is_query = False)

#         attn_fn = linear_attention if not self.causal else self.causal_linear_fn
#         out = attn_fn(q, k, v)
#         return out
    
def moore_penrose_iter_pinv(x, iters = 6):
    """
    Helper function for Nystformer Attention
    """
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

class Nystroformer_Attention(nn.Module):
    def __init__(self, cfg):
        """
        Attention Module
        """
        super().__init__()
        pinv_iterations = 6
        qkv_bias=False
        proj_drop=0.

        num_landmarks = cfg["num_landmarks"]
        dim = cfg["dim"]
        num_heads = cfg["n_heads"]
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, visualization=False):
        B, N, C = x.shape
        original_N = N
        m, iters = self.num_landmarks, self.pinv_iterations

        remainder = N % m
        if remainder > 0:
            padding = m - (N % m)
            x = torch.nn.functional.pad(x, (0, 0, padding, 0), value = 0)
            B, N, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), (q, k, v))

        q = q * self.scale

        l = math.ceil(N / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)
        q_landmarks /= l
        k_landmarks /= l
        
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = torch.einsum(einops_eq, q, k_landmarks)
        sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = torch.einsum(einops_eq, q_landmarks, k)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        
        attn = attn1 @ attn2_inv @ attn3
        
        x = (attn1 @ attn2_inv) @ (attn3 @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x[:, -original_N:]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        
        self.proj_q = nn.Linear(cfg["dim"], cfg["dim"])
        self.proj_k = nn.Linear(cfg["dim"], cfg["dim"])
        self.proj_v = nn.Linear(cfg["dim"], cfg["dim"])
        
        self.drop = nn.Dropout(cfg["p_drop_attn"])
        self.scores = None # for visualization
        self.n_heads = cfg["n_heads"]
        # self.fast_attention = cfg["fast_attention"]
        # if self.fast_attention:
        #     self.attn_fn = FastAttention(
        #         dim_heads = cfg["dim"]//cfg["n_heads"],
        #         generalized_attention = True,
        #         causal = False
        #     )

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        verbose = False
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        B,S,D = x.shape
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        
        # if self.fast_attention:
        #     h = self.attn_fn(q, k, v) # (1, 8, 512, 64) 
        #     h = h.transpose(1,2)
        #     scores = None # fix it later to get the attention. 
            
        # else: 
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = torch.FloatTensor(mask[:, None, None, :]).type_as(x)
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()

        # -merge-> (B, S, D)
        h = merge_last(h, 2)

        self.scores = scores
        return h, scores

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["dim"], cfg["dim_ff"])
        self.fc2 = nn.Linear(cfg["dim_ff"], cfg["dim"])

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        output = self.fc2(gelu(self.fc1(x)))
        return output


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        if cfg["attention_type"] == 'nystroformer':
            self.attn = Nystroformer_Attention(cfg)
        else: 
            self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg["dim"], cfg["dim"])
            
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg["p_drop_hidden"])


    def forward(self, x, mask):
        h, scores = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        
        output_pwff = self.pwff(h)
        h = self.norm2(h + self.drop(output_pwff))
        
        return h, scores


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.output_layer = 4
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])

    def forward(self, x, T, P_row, P_col, map_fn, mask):
        h = self.embed(x, T, P_row, P_col, map_fn)
        embedded_patches = h.clone().detach()
        scores_per_layer=[]
        output_layer=[]
        for block, i in zip(self.blocks, range(len(self.blocks))):
            h, scores = block(h, mask)
            scores_per_layer.append(scores)
            if i==self.output_layer:
                output_layer = h

        return h, scores_per_layer, embedded_patches


# if __name__ == "__main__":
# #     sample = torch.ones((16, 1, 256,256))
# #     model = ViT(img_size=256, patch_size=16, in_channel=1, n_classes=7, emb_dim=768, depth=12, n_heads=12, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.)

# #     print(model)

# #     print('-------------------------------------------------------------------------------------------------------')
# #     print('Total number of parameters', sum(p.numel() for p in model.parameters()))
# #     print('Total number of learnable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
# #     print('-------------------------------------------------------------------------------------------------------')
#     # model.eval()
#     # with torch.no_grad():
#     #     pred = model(sample)
#     #     print(pred.shape)
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param))) # Select T samples with 
#     P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches)))
#     P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches)))