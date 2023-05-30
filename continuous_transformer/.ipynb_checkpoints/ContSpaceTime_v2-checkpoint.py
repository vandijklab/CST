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
from scipy.special import lambertw

from functools import partial
from einops import rearrange, repeat

from continuous_transformer.utils import split_last, merge_last
from continuous_transformer.spectral_normalization import SpectralNorm
# import torch.nn.utils.spectral_norm as SpectralNorm


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
#         print('x_proj.shape: ',x_proj.shape)
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
        self.Lipschitz_regularization = cfg["Lipschitz_regularization"]
        self.Spectral_normalization = cfg["Spectral_normalization"]
        
        self.pos_embed = nn.Embedding(cfg["max_len"], cfg["dim"]) # nn.Embedding(cfg["max_len"], cfg["dim"]) # position embedding
#         self.pos_embed_Lin = nn.Linear(1, cfg["dim"]) # nn.Embedding(cfg["max_len"], cfg["dim"]) # position embedding
        if cfg["operation_with_pos_encoding"] == "concatenate":
            # self.pos_embed_3DLin_GaussMap = nn.Linear(512, int(cfg["dim"]/2)) # position embedding: x,y,t or 512 after gaussian-fourier mapping
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
        
        pos_t = (T/(self.max_len-1)) # 32 frames/batch. 16 patches [0...1] [0,0,0,0,..., 29,29,29] pos_t: [0.....1]
        if verbose: print('\npos_t.shape, pos_t.unsqueeze(0).shape, x.shape, T.shape: ', pos_t.shape, pos_t.unsqueeze(0).shape, x.shape, T.shape)
            
#         # original
#         pos_t = pos_t.unsqueeze(0).expand_as(x[:,:,0]).T # (S,) -> (B, S) # Since 'x' already has the embedding here, expand_as(x[:,:])
        pos_t = pos_t.unsqueeze(0).T # (S,) -> (B, S) # Since 'x' already has the embedding here, expand_as(x[:,:])
        if verbose: print('\nmap_fn: ',map_fn)
        if verbose: print('\npos_t: ',pos_t.flatten())
        pos_x = (P_row/(self.frame_size["rows"] - self.patch_size-1))
#         # original
#         pos_x = pos_x.unsqueeze(0).expand_as(x[:,:,0]).T # (S,) -> (B, S) 
        pos_x = pos_x.unsqueeze(0).T # (S,) -> (B, S) 
        if verbose: print('\npos_x: ',pos_x.flatten())
        
        pos_y = (P_col/(self.frame_size["cols"] - self.patch_size-1))
#         # original
#         pos_y = pos_y.unsqueeze(0).expand_as(x[:,:,0]).T # (S,) -> (B, S) 
        pos_y = pos_y.unsqueeze(0).T # (S,) -> (B, S) 
        if verbose: print('pos_y.shape: ',pos_y.shape)
        if verbose: print('\npos_y: ',pos_y.flatten())

        if verbose: print('torch.cat((pos_t, pos_x, pos_y), 1).shape: ', torch.cat((pos_t, pos_x, pos_y), 1).shape)
        if verbose: print('torch.cat((pos_t, pos_x, pos_y), 1): ', torch.cat((pos_t, pos_x, pos_y), 1))
        
        mapped_input = input_mapping(torch.cat((pos_t, pos_x, pos_y), 1), map_fn)

        if verbose: print('mapped_input.shape: ',mapped_input.shape)
        if verbose: print('mapped_input: ',mapped_input)
        
        if map_fn is None: 
            pos_embed_result = self.pos_embed_3DLin(torch.Tensor(mapped_input).type_as(x)).unsqueeze(0)
            if self.Lipschitz_regularization:
                param = self.pos_embed_3DLin.weight
                sym = torch.mm(param, torch.t(param))
                sym -= torch.eye(param.shape[0]).to(device)
                ls_ort_pos_embed_3DLin = sym.pow(2.0).sum()  # Loss for orthogonality
            
        else: 
#             if verbose: print('using gauss maps')
            pos_embed_result = self.pos_embed_3DLin_GaussMap(torch.Tensor(mapped_input).type_as(x)).unsqueeze(0)
            
        if verbose: print('pos_embed_result.shape: ',pos_embed_result.shape)
        if verbose: print('pos_embed_result: ',pos_embed_result)
            
        pos_embed_result = pos_embed_result.expand_as(x) # (S,) -> (B, S)
        if verbose: print('pos_embed_result.shape: ',pos_embed_result.shape)
            
        if self.operation_with_pos_encoding == "concatenate":
            e = torch.cat((x,pos_embed_result),dim=-1)
        elif self.operation_with_pos_encoding == "sum":   
            e = x + pos_embed_result #+ self.seg_embed(seg) # Remove seg_emb since there is only one segment
        if verbose: print('\ne.shape: {}, e.min(): {}, e.max(): {}'.format(e.shape, e.min(), e.max()))
        if verbose: print('\ne: ',e)
#         if verbose: print(aaa)
        if self.Lipschitz_regularization:
            return self.drop(self.norm(e)), ls_ort_pos_embed_3DLin
        else: 
            return self.drop(self.norm(e))

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    verbose = False
    b, h, *_ = data.shape

    if verbose: print("data.shape: ",data.shape)
    if verbose: print("projection_matrix.shape: ",projection_matrix.shape)
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    if verbose: print("data_normalizer: ",data_normalizer)

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)
    if verbose: print("projection.shape: ",projection.shape) #projection.shape:  torch.Size([2, 8, 621, 128])

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
    if verbose: print('q,k,v.shapes: ',q.shape,k.shape,v.shape)
    k_cumsum = k.sum(dim = -2)
    if verbose: print('k_cumsum.shape: ',k_cumsum.shape)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    if verbose: print('D_inv.shape: ',D_inv.shape)
    context = torch.einsum('...nd,...ne->...de', k, v)
    if verbose: print('context.shape: ',context.shape)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    if verbose: print('out.shape: ',out.shape)
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

class FastAttention(nn.Module):
    """ Linear attention  """
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        verbose = False
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        if verbose: 
            print('nb_features: ',nb_features)
            print('dim_heads: ',dim_heads)

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        '''
        # queries / keys / values with heads already split and transposed to first dimension
        # 8 heads, dimension of head is 64, sequence length of 512
        q = torch.randn(1, 8, 512, 64)
        k = torch.randn(1, 8, 512, 64)
        v = torch.randn(1, 8, 512, 64)
        '''
        
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.Lipschitz_regularization = cfg["Lipschitz_regularization"]
        self.Spectral_normalization = cfg["Spectral_normalization"]
        
        if cfg["Lipschitz_regularization"]:
            self.proj_q = nn.Linear(cfg["dim"], cfg["dim"])
            self.proj_k = nn.Linear(cfg["dim"], cfg["dim"])
            self.proj_v = nn.Linear(cfg["dim"], cfg["dim"])
            nn.init.orthogonal_(self.proj_q.weight)
            nn.init.orthogonal_(self.proj_k.weight)
            nn.init.orthogonal_(self.proj_v.weight)
            
        elif cfg["Spectral_normalization"]:
            # Orginal 
            self.proj_q = SpectralNorm(nn.Linear(cfg["dim"], cfg["dim"]))
            self.proj_k = SpectralNorm(nn.Linear(cfg["dim"], cfg["dim"]))
            
            # #making the output smaller 
            # self.proj_q = SpectralNorm(nn.Linear(cfg["dim"], 8))
            # self.proj_k = SpectralNorm(nn.Linear(cfg["dim"], 8))
            
            self.proj_v = SpectralNorm(nn.Linear(cfg["dim"], cfg["dim"]))
            # self.proj_q = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], cfg["dim"]))
            # self.proj_k = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], cfg["dim"]))
            # self.proj_v = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], cfg["dim"]))
            
        else:
            self.proj_q = nn.Linear(cfg["dim"], cfg["dim"])
            self.proj_k = nn.Linear(cfg["dim"], cfg["dim"])
            self.proj_v = nn.Linear(cfg["dim"], cfg["dim"])
        
        self.drop = nn.Dropout(cfg["p_drop_attn"])
        self.scores = None # for visualization
        self.n_heads = cfg["n_heads"]
        self.fast_attention = cfg["fast_attention"]
        if self.fast_attention:
            self.attn_fn = FastAttention(
                dim_heads = cfg["dim"]//cfg["n_heads"], #16, # concatenated (x, pos_enc) = 128. Then divided by number of heads. Thus 128/8 = 16
                generalized_attention = True,
                # nb_features = 256,
                causal = False
            )

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        verbose = False
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        B,S,D = x.shape
        # x = x.view(x.size(0), -1)
        # x = (x-x.min(1, keepdim=True)[0])/(x.max(1, keepdim=True)[0]-x.min(1, keepdim=True)[0])
        # # AA = (AA-AA.min(keepdim=True))/(AA.max(keepdim=True)-AA.min(keepdim=True))
        # x = x.view(B,S,D)
        # if verbose:
        #     print('x.shape: ',x.shape)
        #     print('x[1,:].min(): ',x[1,:].min())
        #     print('x[1,:].max(): ',x[1,:].max())
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        # _, SingVal, _ = torch.linalg.svd(self.proj_q.module.weight_bar, full_matrices=False)
        # # self.logger.experiment.add_scalar("Loss/Val_ort_fc2",avg_ls_ort_fc2,self.current_epoch)
        # # U, S, Vh = torch.linalg.svd(self.proj_q.weight, full_matrices=False)
        # print('\nSingVal.max(): {}'.format(SingVal.max()))
        # del SingVal
        # U, S, Vh = torch.linalg.svd(self.proj_k.weight, full_matrices=False)
        # print('max, S_k: ',S.max(),S)
        # U, S, Vh = torch.linalg.svd(self.proj_v.weight, full_matrices=False)
        # print('max, S_v: ',S.max(),S)
        
        # do SVD on the weights of the self.proj_ to verify the largest eigevalue has to be one.
        
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        if verbose: print("\n[after split] q.shape: , k.shape: , v.shape: ",q.shape , k.shape , v.shape) # torch.Size([2, 8, 3588, 16]) torch.Size([2, 8, 3588, 16]) torch.Size([2, 8, 3588, 16])
        
        if self.fast_attention:
            if verbose: print("\n[after transpose for the fast_attention] q.shape: , k.shape: , v.shape: ",q.shape , k.shape , v.shape) # torch.Size([2, 8, 16, 3588]) torch.Size([2, 8, 16, 3588]) torch.Size([2, 8, 16, 3588])
            h = self.attn_fn(q, k, v) # (1, 8, 512, 64) -> 8 heads, dimension of head is 64, sequence length of 512
            h = h.transpose(1,2)
            scores = None # fix it later to get the attention. It should be Q*K, but it is not cleat how to get it from the approximation
            
#         elif self.Spectral_normalization: # Lipschitz from deepmind
#             # q_norm = torch.diag(torch.linalg.norm(q,dim=2)**2)
#             q_norm = torch.diag_embed(torch.linalg.norm(q,dim=3)**2)
#             if verbose: 
#                 print('q_norm.shape: ',q_norm.shape)
#                 print('q_norm: ',q_norm)
#             qk_norm = 2*(q@q.transpose(-2,-1))
#             if verbose: 
#                 print('qk_norm.shape: ',qk_norm.shape)
#                 print('qk_norm: ',qk_norm)
#             k_norm = torch.diag_embed(torch.linalg.norm(q,dim=3)**2)
#             if verbose: 
#                 print('k_norm.shape: ',k_norm.shape)
#                 print('k_norm: ',k_norm)
                
#             scores = -(q_norm - qk_norm + k_norm)/np.sqrt(k.size(-1))
#             if mask is not None:
#                 mask = torch.FloatTensor(mask[:, None, None, :]).type_as(x)
#                 scores -= 10000.0 * (1.0 - mask)
#             # scores = self.drop(F.softmax(-(q_norm - qk_norm + k_norm)/np.sqrt(k.size(-1)),dim=-1))
#             scores = self.drop(F.softmax(scores, dim=-1))
                
#             if verbose: 
#                 print('scores.shape: ',scores.shape)
#                 print('scores: ',scores)
#             h = (scores @ v).transpose(1, 2).contiguous()
            
#             # To ensure 1-Lipschitz 
#             if verbose:
#                 print('( (np.sqrt(S)*np.sqrt(self.n_heads) / (np.sqrt(k.size(-1)))) * np.real(4*lambertw(S/np.e) + 1) ): ',( (np.sqrt(S)*np.sqrt(self.n_heads) / (np.sqrt(k.size(-1)))) * np.real(4*lambertw(S/np.e) + 1) ))
#             # ensure it is the same value for a couple of runs
            
#             # h = h / ( (np.sqrt(S)*np.sqrt(self.n_heads) / (np.sqrt(k.size(-1)))) * np.real(4*lambertw(S/np.e) + 1) )
#             if verbose: 
#                 print('h.shape: ',h.shape)
#                 print('h: ',h)
            
#             # h = h / (self.n_heads * np.sqrt(S/D)*(4*lambertw(S/np.e) + 1))
            
        else: 
            # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
            if verbose: print('q,k,v.shapes: ',q.shape,k.shape,v.shape)
            scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
            if verbose: print('scores.shape: ',scores.shape)
            if verbose: print('mask.shape: ',mask.shape)
            if mask is not None:
                mask = torch.FloatTensor(mask[:, None, None, :]).type_as(x)
                if verbose: print('mask: ', mask)
                scores -= 10000.0 * (1.0 - mask)
            scores = self.drop(F.softmax(scores, dim=-1))
            if verbose: print('scores.shape [second]: ',scores.shape)
            # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
            h = (scores @ v).transpose(1, 2).contiguous()
            
        if self.Lipschitz_regularization: 
            param = self.proj_q.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_proj_q = sym.pow(2.0).sum()  # Loss for orthogonality

            param = self.proj_k.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_proj_k = sym.pow(2.0).sum()  # Loss for orthogonality

            param = self.proj_v.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_proj_v = sym.pow(2.0).sum()  # Loss for orthogonality
        
        if verbose: print("h.shape: ",h.shape) # usual attention: torch.Size([2, 3588, 8, 16]) ; for fast_attention: h.shape:  torch.Size([2, 8, 3588, 16])
        # -merge-> (B, S, D)
        h = merge_last(h, 2)

        self.scores = scores
        if self.Lipschitz_regularization:
            return h, scores, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v
        else: 
            return h, scores

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        
        self.Lipschitz_regularization = cfg["Lipschitz_regularization"]
        self.Spectral_normalization = cfg["Spectral_normalization"]
        
        if self.Lipschitz_regularization:
            self.fc1 = nn.Linear(cfg["dim"], cfg["dim_ff"])
            self.fc2 = nn.Linear(cfg["dim_ff"], cfg["dim"])
            nn.init.orthogonal_(self.fc1.weight)
            nn.init.orthogonal_(self.fc2.weight)
        elif self.Spectral_normalization:
            self.fc1 = SpectralNorm(nn.Linear(cfg["dim"], cfg["dim_ff"]))
            self.fc2 = SpectralNorm(nn.Linear(cfg["dim_ff"], cfg["dim"]))
            # self.fc1 = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], cfg["dim_ff"]))
            # self.fc2 = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim_ff"], cfg["dim"]))
        else:
            self.fc1 = nn.Linear(cfg["dim"], cfg["dim_ff"])
            self.fc2 = nn.Linear(cfg["dim_ff"], cfg["dim"])
            
        #self.activ = lambda x: activ_fn(cfg["activ_fn"], x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        output = self.fc2(gelu(self.fc1(x)))
        
        if self.Lipschitz_regularization:
            param = self.fc2.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_fc2 = sym.pow(2.0).sum()  # Loss for orthogonality

            param = self.fc1.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_fc1 = sym.pow(2.0).sum()  # Loss for orthogonality
        
        if self.Lipschitz_regularization:
            return output, ls_ort_fc1, ls_ort_fc2 #self.fc2(gelu(self.fc1(x)))
        else:
            return output


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.Lipschitz_regularization = cfg["Lipschitz_regularization"]
        self.Spectral_normalization = cfg["Spectral_normalization"]
        self.attn = MultiHeadedSelfAttention(cfg)
        
        if self.Lipschitz_regularization:
            self.proj = nn.Linear(cfg["dim"], cfg["dim"])
            nn.init.orthogonal_(self.proj.weight)
        elif self.Spectral_normalization: 
            self.proj = SpectralNorm(nn.Linear(cfg["dim"], cfg["dim"]))
            # self.proj = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], cfg["dim"]))
        else: 
            self.proj = nn.Linear(cfg["dim"], cfg["dim"])
            
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg["p_drop_hidden"])
        self.Lipschitz_regularization = cfg["Lipschitz_regularization"]

    def forward(self, x, mask):
        if self.Lipschitz_regularization:
            h, scores, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v = self.attn(x, mask)
        else:
            h, scores = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        
        if self.Lipschitz_regularization:
            param = self.proj.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_proj = sym.pow(2.0).sum()  # Loss for orthogonality
        
        # h = self.norm2(h + self.drop(self.pwff(h)))
        if self.Lipschitz_regularization:
            output_pwff, ls_ort_fc1, ls_ort_fc2 = self.pwff(h)
        else:
            output_pwff = self.pwff(h)
        h = self.norm2(h + self.drop(output_pwff))
        
        if self.Lipschitz_regularization:
            return h, scores, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v, ls_ort_proj, ls_ort_fc1, ls_ort_fc2
        else:
            return h, scores


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.output_layer = 4
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.Lipschitz_regularization = cfg["Lipschitz_regularization"]
#         self.scores_per_layer=[]
        #print('\nself.output_layer: ',self.output_layer)

    def forward(self, x, T, P_row, P_col, map_fn, mask):
#         print('x.shape[Transformer-ContSpaceTime]: ', x.shape)
        if self.Lipschitz_regularization:
            h, ls_ort_pos_embed_3DLin = self.embed(x, T, P_row, P_col, map_fn)
        else: 
            h = self.embed(x, T, P_row, P_col, map_fn)
        embedded_patches = h.clone().detach()
        scores_per_layer=[]
        # print('\nembedded_patches.shape [in Transformer-ContSpaceTime]: ',embedded_patches.shape)
#         print('h [in Transformer-ContSpaceTime]: ',h)
        output_layer=[]
        for block, i in zip(self.blocks, range(len(self.blocks))):
            if self.Lipschitz_regularization:
                h, scores, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v, ls_ort_proj, ls_ort_fc1, ls_ort_fc2 = block(h, mask)
            else:
                h, scores = block(h, mask)
            scores_per_layer.append(scores)
#             print('len(scores_per_layer): {}, len(scores): {}'.format(len(scores_per_layer), len(scores)))
            if i==self.output_layer:
                output_layer = h
                # print('output_layer.shape: ',output_layer.shape)
        if self.Lipschitz_regularization:
            return h, scores_per_layer, embedded_patches, ls_ort_pos_embed_3DLin, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v, ls_ort_proj, ls_ort_fc1, ls_ort_fc2
        else:
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