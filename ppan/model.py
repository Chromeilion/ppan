import math
from typing import Optional, Dict, List, Any

import torch
from torch import nn
from torchvision.models.video import MViT_V2_S_Weights, MViT
from torchvision.models.video.mvit import _unsqueeze, _ovewrite_named_param, \
    MSBlockConfig, WeightsEnum

from ppan.config import device


class PPAnMViT(MViT):
    """
    A headless version of the MViT implementation in PyTorch. original code
    found here:
    https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)

        return x


class Encoder(nn.Module):
    def __init__(self,
                 seq_len: int,
                 mvit_weights,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mvit = mvit_v2_s(weights=mvit_weights)
        self.linear = nn.Linear(in_features=393, out_features=seq_len)

    def forward(self, x: torch.Tensor):
        x = self.mvit(x)
        x = torch.swapaxes(x, 1, 2)
        x = self.linear(x)
        x = torch.swapaxes(x, 1, 2)
        return x


class Decoder(nn.Module):
    """
    A decoder that sits on top of the MViT based encoder and makes sequence
    predictions.
    """
    def __init__(self, d_model: int, nhead: int, n_layers: int,
                 n_tokens: int, emb_dim: int, seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(
                in_features=emb_dim,
                out_features=n_tokens
            )
        )
        self.positional_encoding = PositionalEmbedding(d_model=d_model,
                                                       max_len=seq_len)

        self.embedding = nn.Embedding(
            num_embeddings=n_tokens,
            embedding_dim=emb_dim
        )

    def forward(self, tgt, memory,
                tgt_mask: Optional = None,
                tgt_pad_mask: Optional = None):
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        x = self.decoder(tgt=tgt,
                         memory=memory,
                         tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_pad_mask)
        x = self.head(x)
        return x


class PPAnModel(nn.Module):
    def __init__(self,
                 n_tokens: int,
                 emb_dim: int,
                 bos_token: int,
                 eos_token: int,
                 pad_token: int,
                 nhead: int = 10,
                 seq_len: int = 110,
                 encoder_weights: Optional[MViT_V2_S_Weights] = None,
                 n_decoder_layers: Optional[int] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if n_decoder_layers is None:
            n_decoder_layers = 5
        self.seq_len: int = seq_len
        self.bos = bos_token
        self.eos = eos_token
        self.pad = pad_token
        self.n_tokens = n_tokens

        self.encoder = Encoder(
            seq_len=seq_len,
            mvit_weights=encoder_weights
        )
        self.decoder = Decoder(
            d_model=emb_dim,
            n_layers=n_decoder_layers,
            nhead=nhead,
            emb_dim=emb_dim,
            n_tokens=n_tokens,
            seq_len=seq_len
        )
        self.register_buffer(
            "tgt_mask",
            get_tgt_mask(seq_len=seq_len)
        )

    def forward(self, img: torch.Tensor, tgt: torch.Tensor,
                tgt_pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.encoder(img)
        if self.training:
            x = self.decoder(tgt=tgt,
                             memory=x,
                             tgt_mask=self.tgt_mask,
                             tgt_pad_mask=tgt_pad_mask)
        else:
            x = self.evaluate(memory=x)
        return x

    def evaluate(self,
                 memory: torch.Tensor) -> torch.Tensor:
        """
        Greedy iterative decoding for when the model is in eval mode.

        Parameters
        ----------
        memory : torch.Tensor
            output of the encoder

        Returns
        -------
        preds : torch.Tensor
        """
        tgt = torch.fill(torch.zeros(size=(memory.shape[0],
                                           self.seq_len), dtype=torch.long),
                         self.pad).to(device)
        tgt[:, 0] = self.bos
        tgt_pad_mask = torch.ones(size=(memory.shape[0], self.seq_len),
                                  dtype=torch.bool).to(device)
        x = torch.zeros(size=(memory.shape[0], memory.shape[1],
                              self.n_tokens)).to(device)
        x[:, :, self.pad] = 1
        done = [False for _ in range(tgt.shape[0])]
        for i in range(self.seq_len - 1):
            tgt_pad_mask[:, i] = False
            out = self.decoder(
                memory=memory,
                tgt=tgt,
                tgt_mask=self.tgt_mask,
                tgt_pad_mask=tgt_pad_mask
            )
            tgt[:, i+1] = torch.argmax(out, dim=2)[:, i]
            for j in range(tgt.shape[0]):
                if self.eos not in tgt[j, :]:
                    x[j, i, :] = out[j, i, :]
                elif self.eos in tgt[j, :] and not done[j]:
                    x[j, i, :] = out[j, i, :]
                    done[j] = True
            if all(done):
                break
        return x


class PositionalEmbedding(nn.Module):
    """
    A very simple trainable positional embedding.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.Tensor(max_len, d_model))
        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


def _mvit_ppan(
    block_setting: List[MSBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MViT:
    """
    Code originally taken from PyTorch:
    https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py
    """
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "spatial_size", weights.meta["min_size"])
        _ovewrite_named_param(kwargs, "temporal_size",
                              weights.meta["min_temporal_size"])
    spatial_size = kwargs.pop("spatial_size", (224, 224))
    temporal_size = kwargs.pop("temporal_size", 16)

    model = PPAnMViT(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        block_setting=block_setting,
        residual_pool=kwargs.pop("residual_pool", False),
        residual_with_cls_embed=kwargs.pop("residual_with_cls_embed", True),
        rel_pos_embed=kwargs.pop("rel_pos_embed", False),
        proj_after_attn=kwargs.pop("proj_after_attn", False),
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def mvit_v2_s(*, weights: Optional[MViT_V2_S_Weights] = None,
              progress: bool = True, **kwargs: Any) -> MViT:
    """
    Code originally taken from PyTorch:
    https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py

    Constructs a small MViTV2 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V2_S_Weights
        :members:
    """
    weights = MViT_V2_S_Weights.verify(weights)

    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384,
                           384, 384, 384, 384, 768],
        "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384,
                            384, 384, 384, 384, 768, 768],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return _mvit_ppan(
        spatial_size=(224, 224),
        temporal_size=16,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def get_tgt_mask(seq_len):
    """
    Build a diagonal mask with the aim to stop transformer from cheating by
    looking ahead in the given tgt tensor.
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool),
                      diagonal=1)
    return mask
