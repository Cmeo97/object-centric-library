from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from models.base_model import BaseModel
from models.nn_utils import get_conv_output_shape, make_sequential_from_config
from models.shared.nn import PositionalEmbedding
from .group_operations import PerceiverSW, CommAttention

import torch.nn.functional as F

class EncoderConfig(Dict):
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    width: int
    height: int
    input_channels: int = 3


class DecoderConfig(Dict):
    conv_tranposes: List[bool]
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    width: int
    height: int
    input_channels: int = 3

class vmnnConfig(Dict):
    communication: str
    comm_key_size: int
    comm_value_size: int
    num_comm_heads: int
    comm_dropout: int
    z_size: int
    D_num_heads: int
    n_SK_slots: int


class Encoder(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        channels: List[int] = (32, 32, 32, 32),
        kernels: List[int] = (5, 5, 5, 5),
        strides: List[int] = (1, 1, 1, 1),
        paddings: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
        batchnorms: List[bool] = tuple([False] * 4),
    ):
        super().__init__()
        assert len(kernels) == len(strides) == len(paddings) == len(channels)
        self.conv_bone = make_sequential_from_config(
            input_channels,
            channels,
            kernels,
            batchnorms,
            False,
            paddings,
            strides,
            "relu",
            try_inplace_activation=True,
        )
        output_channels = channels[-1]
        output_width, output_height = get_conv_output_shape(
            width, height, kernels, paddings, strides
        )
        self.pos_embedding = PositionalEmbedding(
            output_width, output_height, output_channels
        )
        self.lnorm = nn.GroupNorm(1, output_channels, affine=True, eps=0.001)
        self.conv_1x1 = [
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
        ]
        self.conv_1x1 = nn.Sequential(*self.conv_1x1)

    def forward(self, x: Tensor) -> Tensor:
        conv_output = self.conv_bone(x)
        conv_output = self.pos_embedding(conv_output)
        conv_output = conv_output.flatten(2, 3)  # bs x c x (w * h)
        conv_output = self.lnorm(conv_output)
        return self.conv_1x1(conv_output)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        width: int,
        height: int,
        channels: List[int] = (32, 32, 32, 4),
        kernels: List[int] = (5, 5, 5, 3),
        strides: List[int] = (1, 1, 1, 1),
        paddings: List[int] = (2, 2, 2, 1),
        output_paddings: List[int] = (0, 0, 0, 0),
        conv_transposes: List[bool] = tuple([False] * 4),
        activations: List[str] = tuple(["relu"] * 4),
    ):
        super().__init__()
        self.conv_bone = []
        assert len(channels) == len(kernels) == len(strides) == len(paddings)
        if conv_transposes:
            assert len(channels) == len(output_paddings)
        self.pos_embedding = PositionalEmbedding(width, height, input_channels)
        self.width = width
        self.height = height

        self.conv_bone = make_sequential_from_config(
            input_channels,
            channels,
            kernels,
            False,
            False,
            paddings,
            strides,
            activations,
            output_paddings,
            conv_transposes,
            try_inplace_activation=True,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.pos_embedding(x)
        output = self.conv_bone(x)
        img, mask = output[:, :3], output[:, -1:]
        return img, mask


class SlotAttentionModule(nn.Module):
    def __init__(self, num_slots, channels_enc, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5

        self.slots_mu = nn.Parameter(torch.rand(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))
        with torch.no_grad():
            limit = sqrt(6.0 / (1 + dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(channels_enc, dim, bias=False)
        self.to_v = nn.Linear(channels_enc, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(channels_enc, eps=0.001)
        self.norm_slots = nn.LayerNorm(dim, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=0.001)
        self.dim = dim

    def forward(self, inputs: Tensor, num_slots: Optional[int] = None) -> Tensor:
        b, n, _ = inputs.shape
        if num_slots is None:
            num_slots = self.num_slots

        mu = self.slots_mu.expand(b, num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, num_slots, -1).exp()
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, self.dim), slots_prev.reshape(-1, self.dim)
            )

            slots = slots.reshape(b, -1, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots





class vmnn_cell(nn.Module):
    def __init__(
                self,
                latent_size,                                                            
                communication,
                n_SK_slots,
                comm_key_size = 32,
                comm_value_size = 100, 
                num_comm_heads = 1, 
                comm_dropout= 0.1, 
                z_size = 100, 
                D_num_heads = 1,
                latent_layers= 2,
                hidden_size= 100, 
                num_slots = 6
    ):
        super().__init__()
           
        self.z_size = z_size
        self.dynamics_update_num_heads = D_num_heads 
        self.communication = communication
        self.num_comm_heads = num_comm_heads
        self.comm_key_size = comm_key_size
        self.comm_value_size = comm_value_size
        self.n_SK_slots = n_SK_slots
        self.latent_size = latent_size
        self.latent_layers = latent_layers
        self.hidden_size = hidden_size 
        self.num_slots = num_slots


         ## MLPs Initialization
        layers = []
        layers.append(nn.Linear(self.latent_size, self.latent_size))
        layers.append(nn.Tanh())
        for i in range(self.latent_layers):
            layers.append(nn.Linear(self.latent_size, self.latent_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(self.latent_size, int(2*self.latent_size)))
        
        self.mlp =  nn.Sequential(*layers)
        
        if self.communication == 'CA':
            self.communication_attention = CommAttention(self.latent_size, self.comm_key_size, self.num_comm_heads, self.num_slots, comm_dropout)
        else:
            self.communication_attention = PerceiverSW(self.n_SK_slots, self.latent_size, self.num_comm_heads, self.num_slots)
     

    def forward(self, slots):
        """
        Input : z (batch_size, num_z_inputs, z_size)
                hs (batch_size, num_units, hidden_size)
                
        Output: new hs
               
        """
        # Compute variational latent variables 
        z_ = self.mlp(slots)
        mu, log_var = z_.chunk(2, dim=2) 
        std = torch.exp(0.5 * log_var)
        z = (mu + std * torch.randn_like(std))

        # Communication
        context = self.communication_attention(z)
        z = z + context
        return z, mu, log_var

class beta_VAE_loss(nn.Module):
    """nn.Module for beta_VAE_loss

    """
    def __init__(self, reconstruction_loss, beta):
        super().__init__()
        self.warmup_epoch = 25
        self.C = 50.0
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss

    def forward(self, recon_x, x, mu, log_var, epoch=10):

        if self.reconstruction_loss == "MSE":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.reconstruction_loss == "BCE":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)).sum(1)
        C_factor = min(epoch / (self.warmup_epoch + 1), 1)
        KLD_diff = torch.abs(KLD - self.C * C_factor)

        return (recon_loss + self.beta * KLD_diff).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


@dataclass(eq=False, repr=False)
class vmnnAE(BaseModel):
    latent_size: int
    beta: int
    encoder_params: EncoderConfig
    decoder_params: DecoderConfig
    vmnn_params: vmnnConfig
    input_channels: int = 3
    eps: float = 1e-8
    mlp_size: int = 128
    attention_iters: int = 3
    w_broadcast: Union[int, Literal["dataset"]] = "dataset"
    h_broadcast: Union[int, Literal["dataset"]] = "dataset"

    encoder: Encoder = field(init=False)
    decoder: Decoder = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.loss_fn = beta_VAE_loss('MSE', self.beta)
        if self.w_broadcast == "dataset":
            self.w_broadcast = self.width
        if self.h_broadcast == "dataset":
            self.h_broadcast = self.height
        self.encoder_params.update(
            width=self.width, height=self.height, input_channels=self.input_channels
        )
        self.encoder = Encoder(**self.encoder_params)
        self.slot_attention = SlotAttentionModule(
            self.num_slots,
            self.encoder_params["channels"][-1],
            self.latent_size,
            self.attention_iters,
            self.eps,
            self.mlp_size,
        )

        
        self.vmnn = vmnn_cell(**self.vmnn_params)   

        self.decoder_params.update(
            width=self.w_broadcast,
            height=self.h_broadcast,
            input_channels=self.latent_size,
        )
        self.decoder = Decoder(**self.decoder_params)

    @property
    def slot_size(self) -> int:
        return self.latent_size

    def spatial_broadcast(self, slot: Tensor) -> Tensor:
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, self.w_broadcast, self.h_broadcast)

    def forward(self, x: Tensor) -> dict:
        with torch.no_grad():
            x = x * 2.0 - 1.0
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1)
        z = self.slot_attention(encoded)
        h, mu, log_var = self.vmnn(z)
        bs = z.size(0)
        h = h.flatten(0, 1)
        h = self.spatial_broadcast(h)
        img_slots, masks = self.decoder(h)
        img_slots = img_slots.view(bs, self.num_slots, 3, self.width, self.height)
        masks = masks.view(bs, self.num_slots, 1, self.width, self.height)
        masks = masks.softmax(dim=1)
        recon_slots_masked = img_slots * masks
        recon_img = recon_slots_masked.sum(dim=1)
        loss, _, Dm = self.loss_fn(recon_img, x, mu, log_var)
        with torch.no_grad():
            recon_slots_output = (img_slots + 1.0) / 2.0
        return {
            "loss": loss,  # scalar
            "Dm": Dm, 
            "mask": masks,  # (B, slots, 1, H, W)
            "slot": recon_slots_output,  # (B, slots, 3, H, W)
            "representation": z,  # (B, slots, latent dim)
            #
            "reconstruction": recon_img,  # (B, 3, H, W)
        }
