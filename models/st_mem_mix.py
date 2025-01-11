"""
ST_MEM 에서 순서 섞는 버전으로 변경 !!
"""
from functools import partial
import torch.nn.functional as F

import torch
import torch.nn as nn
from einops import rearrange

from models.encoder.st_mem_vit import ST_MEM_ViT, TransformerBlock


__all__ = ['ST_MEM_MIX', 'st_mem_vit_mix_small_dec256d4b', 'st_mem_vit_mix_base_dec256d4b']


def get_1d_sincos_pos_embed(embed_dim: int,
                            grid_size: int,
                            temperature: float = 10000,
                            sep_embed: bool = False):
    """Positional embedding for 1D patches.
    """
    assert (embed_dim % 2) == 0, \
        'feature dimension must be multiple of 2 for sincos emb.'
    grid = torch.arange(grid_size, dtype=torch.float32)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / (temperature ** omega)

    grid = grid.flatten()[:, None] * omega[None, :]
    pos_embed = torch.cat((grid.sin(), grid.cos()), dim=1)
    if sep_embed:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed, torch.zeros(1, embed_dim)],
                              dim=0)
    return pos_embed


class ST_MEM_MIX(nn.Module):
    def __init__(self,
                 seq_len: int = 2250,
                 patch_size: int = 75,
                 num_leads: int = 12,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm,
                 norm_pix_loss: bool = False):
        super().__init__()
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'embed_dim': embed_dim,
                           'depth': depth,
                           'num_heads': num_heads,
                           'decoder_embed_dim': decoder_embed_dim,
                           'decoder_depth': decoder_depth, # 4
                           'decoder_num_heads': decoder_num_heads,
                           'mlp_ratio': mlp_ratio,
                           'qkv_bias': qkv_bias,
                           'norm_layer': str(norm_layer),
                           'norm_pix_loss': norm_pix_loss}
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.num_leads = num_leads
        # --------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = ST_MEM_ViT(seq_len=seq_len,
                                  patch_size=patch_size,
                                  num_leads=num_leads,
                                  width=embed_dim,
                                  depth=depth,
                                  mlp_dim=mlp_ratio * embed_dim,
                                  heads=num_heads,
                                  qkv_bias=qkv_bias)
        self.to_patch_embedding = self.encoder.to_patch_embedding
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # MAE decoder specifics
        self.to_decoder_embedding = nn.Linear(embed_dim, decoder_embed_dim)

        # self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 2, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                              output_dim=decoder_embed_dim,
                                                              hidden_dim=decoder_embed_dim * mlp_ratio,
                                                              heads=decoder_num_heads,
                                                              dim_head=64,
                                                              qkv_bias=qkv_bias)
                                             for _ in range(decoder_depth)])

        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size)
        
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.encoder.pos_embedding.shape[-1],
                                            self.num_patches,
                                            sep_embed=True)
        self.encoder.pos_embedding.data.copy_(pos_embed.float().unsqueeze(0))
        self.encoder.pos_embedding.requires_grad = False

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.num_patches,
                                                    sep_embed=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.sep_embedding, std=.02)
        # torch.nn.init.normal_(self.mask_embedding, std=.02)
        for i in range(self.num_leads):
            torch.nn.init.normal_(self.encoder.lead_embeddings[i], std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, num_leads, n, patch_size)
        """
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b c n p', p=p)
        return x

    def unpatchify(self, x):
        """
        x: (batch_size, num_leads, n, patch_size)
        series: (batch_size, num_leads, seq_len)
        """
        series = rearrange(x, 'b c n p -> b c (n p)')
        return series
        
    def suffle_patch(self, x, ratio):
        """
        패치에서 ratio에 따라 일부 선택
        선택된 이 패치들 간의 순서 섞음
        """
        b, num_leads, n, d = x.shape ## ## [256, 12, 30, 768]
        
        # ratio 만큼 순서 섞일 패치 수 선정 
        shuffle_nums = int(n * ratio)
        
        # 선택한 인덱스의 순서 섞기
        for c_b in range(b):
            for c_l in range(num_leads):
                # 어떤 패치?
                # selected_idx = torch.randperm(n)[:shuffle_nums].sort()[0]
                selected_idx = torch.randperm(n)[:shuffle_nums]#.sort()[0]
                
                # 선택된 패치의 순서 섞기                
                # shuffled_idx = selected_idx[torch.randperm(selected_idx.size(0))]
                shuffled_idx = selected_idx[torch.randperm(shuffle_nums)]
                
                # x에서 순서 섞기
                x[c_b, c_l, selected_idx, :] = x[c_b, c_l, shuffled_idx, :]
                
            
        return x
        # x -> [256, 12, 30, 768]
        # target, target_idx -> [256, 12, 22]
            
        

    def forward_encoder(self, x, ratio):
        """
        x: (batch_size, num_leads, seq_len)
        """
        # embed patches
        x = self.to_patch_embedding(x) ## [256, 12, 30, 768]
        b, _, n, _ = x.shape
        
        # 순서 섞는 코드 
        x = self.suffle_patch(x, ratio)
            # x: 일부 패치 순서 섞인 것!
            # target: 정답 (원래 순서)
            # target_idx: 해당 patch

        # add positional embeddings
        x = x + self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # apply lead indicating modules
        ## 시작과 끝 알려주는 임베딩
        # 1) SEP embedding
        sep_embedding = self.encoder.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
            ## 구분자 + 원본 시퀀스 + 구분자
        
        # 2) lead embeddings        
        n = x.shape[2] ## 패치에 해당
        lead_embeddings = torch.stack([self.encoder.lead_embeddings[i] for i in range(self.num_leads)]).unsqueeze(0) ## 리드별 임베딩을 한 배열로 모음(stack) -> unsqueeze 통해서 batch 차원 추가
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n, -1) ## unsqueeze(2) 통해서 sequence 차원 추가 -> expand 통해서 배치 크기와 리드 개수, 시퀀스 길이 만큼 리드 임베딩 확장
        x = x + lead_embeddings ## 리드 임베딩 더해줌

        x = rearrange(x, 'b c n p -> b (c n) p')
        for i in range(self.encoder.depth):
            x = getattr(self.encoder, f'block{i}')(x)
        x = self.encoder.norm(x)

        return x
            # x: 순서 섞인 상태에서의 representation 값
            # target: 정답 (해당 위치의 원래 순서)
            # target_idx: 순서가 섞인 패치의 idx

    def forward_decoder(self, x):
        ## x: 일부 패치 순서 섞인 representation 값 [64, 384, 768]
        ## ids_restore: 순서 섞인 패치 idx
        
        # ids_restore = ids_restore.to(torch.int64) ## 차원 변경
        
        x = self.to_decoder_embedding(x) ## Linear 층 태움. # [64, 384, 768]

        # append mask embeddings to sequence
        ## 리드별로 시퀀스 분리 (채널 차원 복구)
        x = rearrange(x, 'b (c n) p -> b c n p', c=self.num_leads) # [64, 12, 32, 256]
        # b, _, n_masked_with_sep, d = x.shape
        b, _, n, d = x.shape ## [64, 12, 32, 256]
        n = n - 2
        # n_shffle = ids_restore.shape[2] # 22
        
        # ## 마스크 임베딩 준비
        # mask_embeddings = self.mask_embedding.unsqueeze(1) ## 마스크 임베딩에 리드 차원 추가
        # mask_embeddings = mask_embeddings.repeat(b, self.num_leads, n + 2 - n_masked_with_sep, 1) ## 배치와 리드에 맞게 확장

        # ## 구분자 제외한 시퀀스에 마스크 임베딩 추가 (Unshuffle)
        # # Unshuffle without SEP embedding
        # x_wo_sep = torch.cat([x[:, :, 1:-1, :], x], dim=2) ## 구분자 제외한 시퀀스에 마스크 임베딩 추가 (패치의 구분자 - 구분자 ECG(패치 수) 구분자)
        # x_wo_sep = torch.gather(x_wo_sep, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, d)) ## 복원한 순서대로 재 배열

        ## 위치 임베딩 및 SEP 임베딩 추가
        # positional embedding and SEP embedding
        x = x[:,:,1:-1,:] + self.decoder_pos_embed[:, 1:n + 1, :].unsqueeze(1) ## 위치 임베딩 추가
        left_sep = x[:, :, :1, :] + self.decoder_pos_embed[:, :1, :].unsqueeze(1) ## 왼쪽 구분자에 위치 임베딩 추가
        right_sep = x[:, :, -1:, :] + self.decoder_pos_embed[:, -1:, :].unsqueeze(1) ## 오른쪽 구분자에 위치 임베딩 추가
        x = torch.cat([left_sep, x, right_sep], dim=2) ## 구분자와 시퀀스 합침

        # lead-wise decoding
        ## 리드별 디코딩
        x_decoded = []
        for i in range(self.num_leads):
            x_lead = x[:, i, :, :] ## 리드별 데이터 추출
            for block in self.decoder_blocks: ## 각 리드에 디코더 블록 적용
                x_lead = block(x_lead)
            # x_lead = self.decoder_norm(x_lead) ## 정규화
            # x_lead -> [64, 32, 256]
            x_lead = self.decoder_norm(x_lead) ## 정규화 [256, 32, 256]
            x_lead = self.decoder_head(x_lead) ## 디코더 헤드 적용 [256, 32, 256]
            x_decoded.append(x_lead[:, 1:-1, :]) ## 구분자 제외하고 저장 [256, 30, 256]
        x = torch.stack(x_decoded, dim=1) ## 리드별 디코딩 결과를 다시 합침
        return x
            
    
    def forward_loss(self, series, pred):
        """
        series: (batch_size, num_leads, seq_len)
        pred: (batch_size, num_leads, n, patch_size)
        """
        ## series 데이터 패치화
        target = self.patchify(series)
        
        ## 픽셀값 정규화 (옵션, default True 같음)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True) ## 패치별 평균
            var = target.var(dim=-1, keepdim=True) ## 패치별 분산
            target = (target - mean) / (var + 1.e-6)**.5 ## 정규화 수행

        ## 예측값과 실제값의 차이 제곱
        loss = (pred - target) ** 2 ## MSE Loss
        
        ## 각 패치에 대한 평균 손실 계산
        loss = loss.mean()#(dim=-1)  # (batch_size, num_leads, n), mean loss per patch ## 패치 단위로 손실 정규화
        
        ## 마스킹된 패치에 대해서만 손실 적용
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches ## 마스킹된 패치만을 대상으로 손실 계산
        return loss
    
    

    def forward(self,
                series,
                mask_ratio=0.75):
        recon_loss = 0
        pred = None
        mask = None

        # latent, mask, ids_restore = self.forward_encoder(series, mask_ratio)
        # pred = self.forward_decoder(latent, ids_restore)
        # recon_loss = self.forward_loss(series, pred, mask)
        
        latent = self.forward_encoder(series, mask_ratio)
        pred = self.forward_decoder(latent)
        # pred -> [64, 12, 30, 30]
        # target -> [64, 12, 30]
        
        recon_loss = self.forward_loss(series, pred)
        # order_loss, accuracy = self.forward_order_loss(pred, target, target_idx)
        # order_loss = self.forward_order_loss(pred, target, target_idx)

        # return {"loss": order_loss, "accuracy": accuracy, "pred": pred, "mask": mask}
        return {"loss": recon_loss, "pred": pred, "mask": mask}

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_mix_small_dec256d4b(**kwargs):
    model = ST_MEM_MIX(embed_dim=384,
                   depth=12,
                   num_heads=6,
                   decoder_embed_dim=256,
                   decoder_depth=4,
                   decoder_num_heads=4,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                   **kwargs)
    return model


def st_mem_vit_mix_base_dec256d4b(**kwargs):
    model = ST_MEM_MIX(embed_dim=768,
                   depth=12,
                   num_heads=12,
                   decoder_embed_dim=256,
                   decoder_depth=4,
                   decoder_num_heads=4,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                   **kwargs)
    return model
