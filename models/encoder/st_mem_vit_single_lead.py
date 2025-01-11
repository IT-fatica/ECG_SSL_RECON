# 단일 리드로 실험할 때
# 해당 lead_embedding 값 가져오도록 !!!!

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from models.encoder.vit import TransformerBlock


__all__ = ['ST_MEM_ViT_SL', 'st_mem_vit_small_sl', 'st_mem_vit_base_sl']


class ST_MEM_ViT_SL(nn.Module):
    def __init__(self,
                 seq_len: int,
                 patch_size: int,
                 num_leads: int,
                 num_classes: Optional[int] = None,
                 single_lead: int = 2, ## 추가 - 어떤 리드?
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'num_classes': num_classes if num_classes is not None else 'None',
                           'single_lead': single_lead, ## 추가 - 어떤 리드?
                           'width': width,
                           'depth': depth,
                           'mlp_dim': mlp_dim,
                           'heads': heads,
                           'dim_head': dim_head,
                           'qkv_bias': qkv_bias,
                           'drop_out_rate': drop_out_rate,
                           'attn_drop_out_rate': attn_drop_out_rate,
                           'drop_path_rate': drop_path_rate}
        self.width = width
        self.depth = depth
        self.single_lead = single_lead
        print(f'selecg lead: {self.single_lead} !!!')

        # embedding layers
        num_patches = seq_len // patch_size
        patch_dim = patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (n p) -> b c n p', p=patch_size),
                                                nn.LayerNorm(patch_dim),
                                                nn.Linear(patch_dim, width),
                                                nn.LayerNorm(width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width))
                                                for _ in range(12)) ## num_leads > 12로 변경 !!

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series):
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')

        x = self.to_patch_embedding(series)
        b, _, n, _ = x.shape
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        # lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings]).unsqueeze(0)
        # lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        lead_embeddings = self.lead_embeddings[self.single_lead - 1].unsqueeze(0).unsqueeze(0).expand(b, -1, n + 2, -1)
        x = x + lead_embeddings
        x = rearrange(x, 'b c n p -> b (c n) p')

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        # remove SEP embeddings
        x = rearrange(x, 'b (c n) p -> b c n p', c=num_leads)
        x = x[:, :, 1:-1, :]

        x = torch.mean(x, dim=(1, 2))
        return self.norm(x)

    def forward(self, series):
        x = self.forward_encoding(series)
        return self.head(x)

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_small_sl(num_leads, num_classes=None, seq_len=2250, patch_size=75, single_lead=1, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      single_lead=single_lead, ## 추가 - 어떤 리드?
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return ST_MEM_ViT_SL(**model_args)


def st_mem_vit_base_sl(num_leads, num_classes=None, seq_len=2250, patch_size=75, single_lead=1, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      single_lead=single_lead, ## 추가 - 어떤 리드?
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return ST_MEM_ViT_SL(**model_args)


'''
patching 후 순서 섞어서 해보기 !!
'''
class ST_MEM_ViT_SL_shffle(nn.Module):
    def __init__(self,
                 seq_len: int,
                 patch_size: int,
                 num_leads: int,
                 num_classes: Optional[int] = None,
                 single_lead: int = 2, ## 추가 - 어떤 리드?
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'num_classes': num_classes if num_classes is not None else 'None',
                           'single_lead': single_lead, ## 추가 - 어떤 리드?
                           'width': width,
                           'depth': depth,
                           'mlp_dim': mlp_dim,
                           'heads': heads,
                           'dim_head': dim_head,
                           'qkv_bias': qkv_bias,
                           'drop_out_rate': drop_out_rate,
                           'attn_drop_out_rate': attn_drop_out_rate,
                           'drop_path_rate': drop_path_rate}
        self.width = width
        self.depth = depth
        self.single_lead = single_lead
        print(f'selecg lead: {self.single_lead} !!!')

        # embedding layers
        num_patches = seq_len // patch_size
        patch_dim = patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (n p) -> b c n p', p=patch_size),
                                                nn.LayerNorm(patch_dim),
                                                nn.Linear(patch_dim, width),
                                                nn.LayerNorm(width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width))
                                                for _ in range(12)) ## num_leads > 12로 변경 !!

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series, mode):
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')

        x = self.to_patch_embedding(series)
        b, _, n, _ = x.shape

        # 패치 순서 랜덤 섞기
        if mode == 'inference':
            print('start inference and shffle patch !!!')
            idx = torch.randperm(n) # 무작위 인덱스 생성
            x = x[:, :, idx, :] # 패치 순서 섞기
        
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        # lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings]).unsqueeze(0)
        # lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        lead_embeddings = self.lead_embeddings[self.single_lead - 1].unsqueeze(0).unsqueeze(0).expand(b, -1, n + 2, -1)
        x = x + lead_embeddings
        x = rearrange(x, 'b c n p -> b (c n) p')

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        # remove SEP embeddings
        x = rearrange(x, 'b (c n) p -> b c n p', c=num_leads)
        x = x[:, :, 1:-1, :]

        x = torch.mean(x, dim=(1, 2))
        return self.norm(x)

    def forward(self, series, mode='train'):
        x = self.forward_encoding(series, mode)
        return self.head(x)

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


'''
patching 후 순서 섞어서 해보기 !! (일부만)
'''
class ST_MEM_ViT_SL_random(nn.Module):
    def __init__(self,
                 seq_len: int,
                 patch_size: int,
                 num_leads: int,
                 num_classes: Optional[int] = None,
                 single_lead: int = 2, ## 추가 - 어떤 리드?
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'num_classes': num_classes if num_classes is not None else 'None',
                           'single_lead': single_lead, ## 추가 - 어떤 리드?
                           'width': width,
                           'depth': depth,
                           'mlp_dim': mlp_dim,
                           'heads': heads,
                           'dim_head': dim_head,
                           'qkv_bias': qkv_bias,
                           'drop_out_rate': drop_out_rate,
                           'attn_drop_out_rate': attn_drop_out_rate,
                           'drop_path_rate': drop_path_rate}
        self.width = width
        self.depth = depth
        self.single_lead = single_lead
        print(f'selecg lead: {self.single_lead} !!!')

        # embedding layers
        num_patches = seq_len // patch_size
        patch_dim = patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (n p) -> b c n p', p=patch_size),
                                                nn.LayerNorm(patch_dim),
                                                nn.Linear(patch_dim, width),
                                                nn.LayerNorm(width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width))
                                                for _ in range(12)) ## num_leads > 12로 변경 !!

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series, mode):
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')

        x = self.to_patch_embedding(series)
        b, _, n, _ = x.shape

        # 일부의 패치만 순서 랜덤 섞기
        if mode == 'inference':
            print('start inference and shffle patch !!!')
             # 3~4개의 패치 인덱스 선택
            selected_idx = torch.randperm(n)[:4]  # 임의로 4개의 패치 인덱스 선택 (3개면 [:3]으로 조정)
            
            # 선택한 인덱스의 순서를 섞음
            shuffled_idx = selected_idx[torch.randperm(selected_idx.size(0))]
            
            # 선택된 패치끼리 순서 섞기
            x[:, :, selected_idx, :] = x[:, :, shuffled_idx, :]
        
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        # lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings]).unsqueeze(0)
        # lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        lead_embeddings = self.lead_embeddings[self.single_lead - 1].unsqueeze(0).unsqueeze(0).expand(b, -1, n + 2, -1)
        x = x + lead_embeddings
        x = rearrange(x, 'b c n p -> b (c n) p')

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        # remove SEP embeddings
        x = rearrange(x, 'b (c n) p -> b c n p', c=num_leads)
        x = x[:, :, 1:-1, :]

        x = torch.mean(x, dim=(1, 2))
        return self.norm(x)

    def forward(self, series, mode='train'):
        x = self.forward_encoding(series, mode)
        return self.head(x)

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str