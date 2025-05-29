# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import List, Union, Optional, Tuple
from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, collect_attn_map=False):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if collect_attn_map:
            # visual cross map start
            L1 = x.size(1)
            L2 = context.size(1)
            q_permuted = q.permute(0, 2, 1, 3) # [B, n, L1, d]
            k_permuted = k.permute(0, 2, 1, 3) # [B, n, L2, d]
            scale_factor = 1.0 / math.sqrt(d)
            k_transposed = k_permuted.transpose(-2, -1) # [B, n, d, L2]
            attn_scores = torch.matmul(q_permuted, k_transposed) * scale_factor # [B, n, L1, L2]
            if context_lens is not None:
                mask = torch.arange(L2, device=q.device)[None, None, None, :] >= context_lens.to(q.device)[:, None, None, None]
                attn_scores = attn_scores.masked_fill(mask, -torch.finfo(attn_scores.dtype).max)
            attn_weights = torch.softmax(attn_scores, dim=-1) # [B, n, L1, L2]


        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        if collect_attn_map:
            return x, attn_weights 
        
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        collect_attn_map=False,
        depth_tensor=None,
        depth_tensor_lens=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, collect_attn_map):
            if collect_attn_map:
                cross_x, attn_scores = self.cross_attn(self.norm3(x), context, context_lens, collect_attn_map) 
            else:
                cross_x = self.cross_attn(self.norm3(x), context, context_lens, collect_attn_map) 
            x = x + cross_x
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            if collect_attn_map:
                return x, attn_scores 
            else:
                return x
            
        if collect_attn_map:
            x, attn_scores = cross_attn_ffn(x, context, context_lens, e, collect_attn_map)
            return x, attn_scores 

        x = cross_attn_ffn(x, context, context_lens, e, collect_attn_map)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        depth_tensor=None,
        clip_fea=None,
        y=None,
        words_indices=None,
        block_id=-1,
        type=None,
        timestep=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        
        # 1, 32760, 1536


        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # e0 1, 6, 1536

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])) # 1, 512, 1536
        
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            depth_tensor=depth_tensor,
            depth_tensor_lens=None,
            collect_attn_map=False)


        save_block_id = block_id 
        attn_map = None
        binary_mask = None
        for i, block in enumerate(self.blocks):
            kwargs["collect_attn_map"] = False
            if i == save_block_id:
                kwargs["collect_attn_map"] = True
                x, attn_map = block(x, **kwargs)
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if save_block_id != -1 and words_indices is not None:
            binary_mask = self.generate_attention_mask(
                            attention_map=attn_map, # [1, 12, 32760, 512] batchsize, head_num, l_x, l_context
                            target_word_indices=words_indices,
                            grid_sizes=grid_sizes, # Make sure grid_sizes covers the full batch
                            target_x_shape=x[0].shape, # channel, frames, h, W
                            batch_index=0, # Process the first item in the batch
                            head_index=None, # Average over heads
                            word_aggregation_method='mean'
                            )

        return [u.float() for u in x], binary_mask

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    @torch.no_grad() # Usually don't need gradients for mask generation
    def generate_attention_mask(
        self,
        attention_map: torch.Tensor,
        grid_sizes: torch.Tensor,
        target_x_shape: Tuple[int, int, int, int], # Target shape: (C, T, H, W)
        batch_index: int = 0,
        target_word_indices: Union[List[int], slice] = None,
        head_index: Optional[int] = None, # Process single head or average
        word_aggregation_method: str = 'mean', # How to combine scores for multiple words
        upsample_mode_spatial: str = 'nearest', # 'nearest', 'bilinear'
        upsample_mode_temporal: str = 'nearest', # 'nearest', 'linear'
        output_dtype: torch.dtype = torch.float32 # or torch.bool for soft mask before threshold
    ) -> torch.Tensor:
        """
        Generates a binary mask from an attention map based on attention towards target words.

        The mask identifies regions in the video (x) that attend strongly to the specified
        context words, exceeding a given threshold. The mask has the same dimensions as x.

        Args:
            attention_map (torch.Tensor): Attention weights [B, Head_num, Lx, Lctx].
                                        Lx = flattened video tokens (patches),
                                        Lctx = context tokens (words).
            target_word_indices (Union[List[int], slice]): Indices or slice for the target
                                                        word(s) in the Lctx dimension.
            grid_sizes (torch.Tensor): Patch grid dimensions [B, 3] -> (F, H_patch, W_patch)
                                    for each batch item, corresponding to Lx.
                                    F, H_patch, W_patch should be integers.
            target_x_shape (Tuple[int, int, int, int]): The desired output shape [C, T, H, W],
                                                        matching the original video tensor x.
            threshold (float): Value between 0 and 1. Attention scores >= threshold become 1 (True),
                            otherwise 0 (False).
            batch_index (int, optional): Batch item to process. Defaults to 0.
            head_index (Optional[int], optional): Specific head to use. If None, average
                                                attention across all heads. Defaults to None.
            word_aggregation_method (str, optional): How to aggregate scores if multiple
                                                    target_word_indices are given ('mean',
                                                    'sum', 'max'). Defaults to 'mean'.
            upsample_mode_spatial (str, optional): PyTorch interpolate mode for H, W dimensions.
                                            Defaults to 'nearest'.
            upsample_mode_temporal (str, optional): PyTorch interpolate mode for T dimension.
                                                Defaults to 'nearest'.
            output_dtype (torch.dtype, optional): Data type of the output mask.
                                                Defaults to torch.bool.

        Returns:
            torch.Tensor: A binary mask tensor of shape target_x_shape [C, T, H, W].

        Raises:
            TypeError: If inputs are not torch.Tensors.
            ValueError: If tensor dimensions or indices are invalid, or if
                        aggregation/upsample modes are unknown.
            IndexError: If batch_index or head_index are out of bounds.
        """
        import torch.nn.functional as F
        
        # --- Input Validation ---
        if not isinstance(attention_map, torch.Tensor):
            raise TypeError("attention_map must be a torch.Tensor")
        if not isinstance(grid_sizes, torch.Tensor):
            raise TypeError("grid_sizes must be a torch.Tensor")
        if attention_map.dim() != 4:
            raise ValueError(f"attention_map must be [B, H, Lx, Lctx], got {attention_map.dim()} dims")
        if grid_sizes.dim() != 2 or grid_sizes.shape[1] != 3:
            raise ValueError(f"grid_sizes must be [B, 3], got {grid_sizes.shape}")
        if len(target_x_shape) != 4:
            raise ValueError(f"target_x_shape must be [C, T, H, W], got length {len(target_x_shape)}")

        B, H, Lx, Lctx = attention_map.shape
        C_out, T_out, H_out, W_out = target_x_shape

        if not 0 <= batch_index < B:
            raise IndexError(f"batch_index {batch_index} out of range for batch size {B}")
        if head_index is not None and not 0 <= head_index < H:
            raise IndexError(f"head_index {head_index} out of range for head count {H}")
        if word_aggregation_method not in ['mean', 'sum', 'max']:
            raise ValueError(f"Unknown word_aggregation_method: {word_aggregation_method}")
        if upsample_mode_spatial not in ['nearest', 'bilinear']:
            raise ValueError(f"Unknown upsample_mode_spatial: {upsample_mode_spatial}")
        if upsample_mode_temporal not in ['nearest', 'linear']:
            raise ValueError(f"Unknown upsample_mode_temporal: {upsample_mode_temporal}")


        # --- Select Head(s) ---
        if head_index is None:
            # Average across heads. Shape -> [Lx, Lctx]
            attn_map_processed = attention_map[batch_index].mean(dim=0)
        else:
            # Select specific head. Shape -> [Lx, Lctx]
            attn_map_processed = attention_map[batch_index, head_index]

        # --- Select and Aggregate Word Attention ---
        # Ensure target_word_indices are valid before slicing
        if isinstance(target_word_indices, slice):
            _slice_indices = range(*target_word_indices.indices(Lctx))
            if not _slice_indices: # Empty slice
                num_words = 0
            elif _slice_indices.start >= Lctx or _slice_indices.stop < -Lctx : # Basic out of bounds check
                num_words = len(_slice_indices) # Proceed cautiously or add stricter check
            else:
                num_words = len(_slice_indices)
            word_indices_str = f"slice({_slice_indices.start}:{_slice_indices.stop}:{_slice_indices.step})"
            word_attn_scores = attn_map_processed[:, target_word_indices] # Shape -> [Lx, num_words]
        elif isinstance(target_word_indices, list):
            # Check indices are within bounds
            valid_indices = [idx for idx in target_word_indices if -Lctx <= idx < Lctx]
            if not valid_indices:
                num_words = 0
                word_attn_scores = torch.empty((Lx, 0), device=attention_map.device, dtype=attention_map.dtype) # Handle empty case
            else:
                word_attn_scores = attn_map_processed[:, valid_indices] # Shape -> [Lx, num_words]
                num_words = len(valid_indices)
            word_indices_str = str(valid_indices) # Report used indices
        else:
            raise TypeError(f"target_word_indices must be list or slice, got {type(target_word_indices)}")

        if num_words > 1:
            if word_aggregation_method == 'mean':
                aggregated_scores = word_attn_scores.mean(dim=-1)
            elif word_aggregation_method == 'sum':
                aggregated_scores = word_attn_scores.sum(dim=-1)
            elif word_aggregation_method == 'max':
                aggregated_scores = word_attn_scores.max(dim=-1).values
            # aggregated_scores shape -> [Lx]
        elif num_words == 1:
            aggregated_scores = word_attn_scores.squeeze(-1) # Shape -> [Lx]
        else: # No valid words selected
            return torch.zeros(target_x_shape, dtype=output_dtype, device=attention_map.device)

        # --- Reshape to Video Patch Grid ---
        # Ensure grid sizes are integers
        f_patch, h_patch, w_patch = map(int, grid_sizes[batch_index].tolist())
        actual_num_tokens = f_patch * h_patch * w_patch

        if actual_num_tokens == 0:
            return torch.zeros(target_x_shape, dtype=output_dtype, device=attention_map.device)

        # Handle mismatch between expected tokens (from grid) and actual attention length (Lx)
        if actual_num_tokens > Lx:
            # Pad aggregated_scores to actual_num_tokens size
            padding_size = actual_num_tokens - aggregated_scores.numel()
            scores_padded = F.pad(aggregated_scores, (0, padding_size), "constant", 0)
            scores_unpadded = scores_padded # Use the padded version for reshaping
            # This scenario is less common than Lx > actual_num_tokens
        elif actual_num_tokens < Lx:
            scores_unpadded = aggregated_scores[:actual_num_tokens]
        else:
            scores_unpadded = aggregated_scores # Shape [actual_num_tokens]

        try:
            # Reshape to [F_patch, H_patch, W_patch]
            attention_patch_grid = scores_unpadded.reshape(f_patch, h_patch, w_patch)
        except RuntimeError as e:
            raise e

        # --- Upsample to Original Video Resolution ---
        # Add batch and channel dims for interpolation: [1, 1, F_patch, H_patch, W_patch]
        # Note: Assuming attention is channel-agnostic here.
        grid_for_upsample = attention_patch_grid.unsqueeze(0).unsqueeze(0).float() # Interpolate needs float

    
        # --- SIMPLIFIED LOGIC: Always use 3D interpolation ---
        target_size_3d = (T_out, H_out, W_out)

        # Determine the 3D interpolation mode.
        # Default to 'nearest' unless temporal dimension changes AND 'linear' is requested.
        if upsample_mode_temporal == 'linear' and f_patch != T_out:
            upsample_mode_3d = 'trilinear'
            align_corners_3d = False # align_corners usually False for non-nearest modes
        else:
            # Use 'nearest' if T isn't changing, or if temporal mode is 'nearest'.
            # 'nearest' is generally safer and handles spatial modes implicitly.
            upsample_mode_3d = 'nearest'
            align_corners_3d = None # align_corners=None for nearest

        upsampled_scores_grid = F.interpolate(grid_for_upsample,
                                        size=target_size_3d,
                                        mode=upsample_mode_3d,
                                        align_corners=align_corners_3d)
        # Expected shape: [1, 1, T_out, H_out, W_out] == [1, 1, 21, 60, 104]

        # --- END SIMPLIFIED LOGIC ---
    
        # Remove batch and channel dims: [T_out, H_out, W_out]
        upsampled_scores = upsampled_scores_grid.squeeze(0).squeeze(0)

        # --- Thresholding ---
        binary_mask_thw = (upsampled_scores / torch.max(upsampled_scores)) # Shape [T_out, H_out, W_out]

        # --- Expand Channel Dimension ---
        # Repeat the mask across the channel dimension C_out
        # Input shape: [T_out, H_out, W_out]
        # After unsqueeze(0): [1, T_out, H_out, W_out]
        # Target shape:      [C_out, T_out, H_out, W_out]
        # This expand operation is valid as explained above.
        final_mask = binary_mask_thw.unsqueeze(0).expand(C_out, T_out, H_out, W_out)

        return final_mask.to(dtype=output_dtype)
