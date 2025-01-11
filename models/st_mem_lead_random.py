"""
2024-10-22
리드 인코더 랜덤으로 주기
"""

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from models.encoder.st_mem_vit import ST_MEM_ViT, TransformerBlock

import random ##


__all__ = ['ST_MEM_LEAD_Random', 'st_mem_vit_lead_random_small_dec256d4b', 'st_mem_vit_lead_random_base_dec256d4b']


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


class ST_MEM_LEAD_Random(nn.Module):
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
                           'decoder_depth': decoder_depth,
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
        
        self.decoder_lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(decoder_embed_dim))
                                                        for _ in range(num_leads)) ## lead_embedding 추가

        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
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
        torch.nn.init.normal_(self.mask_embedding, std=.02)
        
        for i in range(self.num_leads):
            torch.nn.init.normal_(self.encoder.lead_embeddings[i], std=.02)
        
        for i in range(self.num_leads): ## lead embedding
            torch.nn.init.normal_(self.decoder_lead_embeddings[i], std=.02)

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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: (batch_size, num_leads, n, embed_dim)
        """
        b, num_leads, n, d = x.shape
        len_keep = int(n * (1 - mask_ratio))  ## 마스킹되지 않고 남은 패치 개수

        noise = torch.rand(b, num_leads, n, device=x.device)  # noise in [0, 1]  ## 0에서 11사이의 램덤 노이즈 생성 (무직위로 마스킹할 패치 결정에 사용됨)
                ## (batch_size, num_leads, n) => 배치 내 각 리드별로 시퀀스에 랜덤하게 노이즈 부여됨
                
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove ## noise 값이 작은 순서대로 패치의 인덱스 배치 (작은 패치는 유지, 큰 패치 마스크)
        ids_restore = torch.argsort(ids_shuffle, dim=2) ## 원래 순서로 복원하기 위해 ids_shuffle의 역순서 저장. 나중에 디코딩 과정에서 마스크된 부분 원래 위치에 복원하기 위함.

        # keep the first subset ## 마스킹되지 않은 패치 유지
        ids_keep = ids_shuffle[:, :, :len_keep] ## 마스킹되지 않고 남길 패치의 인덱스
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, d)) ## ids_keep에 해당하는 패치만 추출 (추출된 패치는 마스킹X)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, num_leads, n], device=x.device) ## 원래 시퀀스와 동일한 마스크 텐서 생성 (처음 모든 패치 1로 생성 => 마스킹X 부분은 0)
        mask[:, :, :len_keep] = 0 ## 마스킹 X 부분은 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore) ## 마스크를 원래 순서대로 복원 => 마스킹된 패치와 유지된 패치의 순서를 입력 시퀀스와 동일하게 맞춤.

        return x_masked, mask, ids_restore
        ## x_masked: 마스킹되지 않은 패치로 이뤄진 시퀀스
        ## mask: 각 패치가 마스크되었는지 여부 확인하는 바이너리 마스크
        ## ids_restore: 원래 시퀀스로 복원하기 위한 인덱스

    def forward_encoder(self, x):
        """
        x: (batch_size, num_leads, seq_len) [64, 12, 2250]
        """
        # embed patches
        x = self.to_patch_embedding(x) ## [64, 12, 30, 768]
        b, _, n, _ = x.shape
        
        #------- 각 샘플마다 랜덤하게 리드 선택 -------#
        random_leads = torch.randint(0, 12, (b,)) ## 어떤 리드??
        x = x[torch.arange(b), random_leads, :, :] ## [64, 30, 768] ## 해당 리드 선택
        #------- 각 샘플마다 랜덤하게 리드 선택 -------#

        # add positional embeddings
        x = x + self.encoder.pos_embedding[:, 1:n + 1, :] ## 하나의 lead만 존재하니깐 .unsqueeze(1) 불필요
        ## [64, 30, 768]
        
        # x = x + self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)
            ## self.encoder.pos_embedding[:, 1:n + 1, :] > [1, 30, 768]
            ## self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1) > [1, 1, 30, 768]

        # apply lead indicating modules
        ## 시작과 끝 알려주는 임베딩
        # 1) SEP embedding
        sep_embedding = self.encoder.sep_embedding[None, None, :] ## [1, 1, 768]
        left_sep = sep_embedding.expand(b, 1, -1) + self.encoder.pos_embedding[:, :1, :]
        right_sep = sep_embedding.expand(b, 1, -1) + self.encoder.pos_embedding[:, -1:, :]
        x = torch.cat([left_sep, x, right_sep], dim=1) # [64, 32, 768]
            ## 구분자 + 원본 시퀀스 + 구분자
            
        # 2) Lead embedding
        lead_embeddings = torch.stack([self.encoder.lead_embeddings[i] for i in range(12)]) ## 리드 임베딩 값 가져옴. [12, 768]
        selected_lead_embeddings = lead_embeddings[random_leads] ## 각 배치에서 선택된 리드 임베딩 가져오기 [64, 768]
        selected_lead_embeddings = selected_lead_embeddings.unsqueeze(1) ## 더하기 위해서 차원 확장 [64, 1, 768] 
        x = x + selected_lead_embeddings # [64, 32, 768]

        # x = rearrange(x, 'b c n p -> b (c n) p')
        for i in range(self.encoder.depth):
            x = getattr(self.encoder, f'block{i}')(x)
        x = self.encoder.norm(x)
        ## 마스킹 되지 않은 부분만 encoder의 입력으로 들어가서 representation을 뽑음

        # x > [64, 32, 768]
        return x, random_leads
        # 임베딩 값과 각 샘플 내에 선택된 random_leads 값도 반환

    def forward_decoder(self, x, random_leads):        
        # 마스킹 X 하나의 lead embedding (representation)
        x = self.to_decoder_embedding(x) ## Linear 층 태움. [64, 32, 256]
        b, n, d = x.shape
        
        ## 마스킹 부분
        mask_embeddings = self.mask_embedding.unsqueeze(1) # [1, 1, 1, 256]
        mask_embeddings = mask_embeddings.repeat(b, self.num_leads, n, 1) ## [64, 12, 32, 256]
        mask_embeddings = mask_embeddings.to(x.dtype)
        
        # x 변경
        mask_embeddings[torch.arange(b), random_leads, :, :] = x ## (64, 32, 256)
        
        # 위치 임베딩 및 SEP 임베딩 추가
        x = mask_embeddings + self.decoder_pos_embed.unsqueeze(1)
        ## [64, 12, 32, 256]
        
        ## lead embedding 추가
        lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.decoder_lead_embeddings]).unsqueeze(0) ## [1, 12, 256]
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n, -1) ## [64, 12, 32, 256]
        x = x + lead_embeddings ## [64, 12, 32, 256]
        ## lead embedding 추가
        
        # decoding
        x_lead = rearrange(x, 'b c n p -> b (c n) p')
        for block in self.decoder_blocks: ## 각 리드에 디코더 블록 적용
            x_lead = block(x_lead) ## [64, 32, 256]
        x_lead = rearrange(x_lead, 'b (c n) p -> b c n p', c=12) # [64, 12, 32, 256]
        x_lead = x_lead[:, :, 1:-1, :] ## [64, 12, 30, 256]
        x_lead = self.decoder_norm(x_lead) ## [64, 12, 30, 256]
        x_lead = self.decoder_head(x_lead) ## [64, 12, 30, 75]
        
        return x_lead

    def forward_loss(self, series, pred):
        """
        series: (batch_size, num_leads, seq_len)
        pred: (batch_size, num_leads, n, patch_size)
        """
        
        ## series 데이터 패치화
        target = self.patchify(series) ## [64, 12, 30, 75]
        
        ## 픽셀값 정규화 (옵션, default True 같음)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True) ## 패치별 평균
            var = target.var(dim=-1, keepdim=True) ## 패치별 분산
            target = (target - mean) / (var + 1.e-6)**.5 ## 정규화 수행

        ## 예측값과 실제값의 차이 제곱
        loss = (pred - target) ** 2 ## MSE Loss
        
        ## 각 패치에 대한 평균 손실 계산
        loss = loss.mean()
        return loss

    def forward(self,
                series):
        recon_loss = 0
        pred = None
        mask = None

        # latent, mask, ids_restore = self.forward_encoder(series, mask_ratio)
        # pred = self.forward_decoder(latent, ids_restore)
    
        latent, random_leads = self.forward_encoder(series)
        pred = self.forward_decoder(latent, random_leads)
        recon_loss = self.forward_loss(series, pred)

        return {"loss": recon_loss, "pred": pred, "mask": mask}

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_lead_random_small_dec256d4b(**kwargs):
    model = ST_MEM_LEAD_Random(embed_dim=384,
                               depth=12,
                               num_heads=6,
                               decoder_embed_dim=256,
                               decoder_depth=4,
                               decoder_num_heads=4,
                               mlp_ratio=4,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               **kwargs)
    return model


def st_mem_vit_lead_random_base_dec256d4b(**kwargs):
    model = ST_MEM_LEAD_Random(embed_dim=768,
                               depth=12,
                               num_heads=12,
                               decoder_embed_dim=256,
                               decoder_depth=4,
                               decoder_num_heads=4,
                               mlp_ratio=4,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               **kwargs)
    return model
