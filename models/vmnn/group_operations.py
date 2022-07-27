import torch
import torch.nn as nn
import math
from functools import wraps
from typing import Optional
from torch import Tensor
from einops import rearrange
from perceiver.model.core import InputAdapter
from perceiver.model.core import PerceiverEncoder, CrossAttention

class GroupLinearLayer(nn.Module):
    """
    for num_blocks blocks, do linear transformations independently

    self.w: (num_blocks, din, dout)

    x: (batch_size, num_blocks, din)
        -> permute: (num_blocks, batch_size, din)
        -> bmm with self.w: (num_blocks, batch_size, din) (bmm) (num_blocks, din, dout)
                            for each block in range(num_blocks):
                                do (batch_size, din) mat_mul (din, dout)
                                concatenate
                            result (num_blocks, batch_size, dout)
        -> permute: (batch_size, num_blocks, dout)

    """
    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)


class CommAttention(nn.Module):
    """
    Input:  key_var     (N, num_keys, d_k) used to construct keys
            value_var   (N, num_keys, D_v)
            query_var   (N, num_queries, D_key=D_query)

            h (batch_size, num_units, hidden_size)
    Output: context (multi head attention output)
    """

    def __init__(self,
                 hidden_size,
                 kdim,
                 num_heads,
                 num_blocks,
                 dropout
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kdim = kdim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.key = GroupLinearLayer(hidden_size, kdim * num_heads, num_blocks)
        self.query = GroupLinearLayer(
            hidden_size, kdim * num_heads, num_blocks)
        self.value = GroupLinearLayer(
            hidden_size, hidden_size * num_heads, num_blocks)
        self.output_fc = GroupLinearLayer(
            num_heads * hidden_size, hidden_size, num_blocks)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h):
        key = self.key(h)
        query = self.query(h)
        value = self.value(h)

        key = self.transpose_for_scores(key, self.num_heads, self.kdim)
        query = self.transpose_for_scores(query, self.num_heads, self.kdim)
        value = self.transpose_for_scores(
            value, self.num_heads, self.hidden_size)

        scores = torch.matmul(query, key.transpose(-1, -2)
                              ) / math.sqrt(self.kdim)
        probs = nn.Softmax(dim=-1)(scores)


        # inactive modules have zero-value query -> no context for them
        #probs = self.dropout(probs)

        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context.size(
        )[:-2] + (self.num_heads * self.hidden_size,)
        # concatenate all heads
        context = context.view(*new_context_layer_shape)
        context = self.output_fc(context)  # to be add to current h

        return context


# Input adapater for perceiver
class agent_input_adapter(InputAdapter):
    def __init__(self, max_seq_len: int, num_input_channels: int):
        super().__init__(num_input_channels=num_input_channels)

        self.pos_encoding = nn.Parameter(
            torch.empty(max_seq_len, num_input_channels))
        self.scale = math.sqrt(num_input_channels)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.pos_encoding.uniform_(-0.5, 0.5)

    def forward(self, x):
       
        b, l, dim = x.shape  # noqa: E741
        p_enc = rearrange(self.pos_encoding[:l], "... -> () ...")
        return x * self.scale + p_enc


class PerceiverSW(nn.Module):
    def __init__(self, n_SK_slots, h_dim, n_attn_heads, num_hidden):
        super().__init__()
        input_adapter = agent_input_adapter(num_input_channels=h_dim, max_seq_len=num_hidden)
            
        self.n_SW_layers = 2
        self.PerceiverEncoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=n_SK_slots,  # N
            num_latent_channels=h_dim,  # D
            num_cross_attention_qk_channels=h_dim,  # C
            num_cross_attention_heads=n_attn_heads,
            num_self_attention_heads=n_attn_heads,  # small because observational space is small
            num_self_attention_layers_per_block=self.n_SW_layers,
            num_self_attention_blocks=self.n_SW_layers,
            dropout=0.0,
        )
        self.SK_attention_read = CrossAttention(
            num_heads=1,
            num_q_input_channels=h_dim,
            num_kv_input_channels=h_dim,
            num_qk_channels=h_dim,
            num_v_channels=h_dim,
            dropout=0.0,
        )

    def forward(self, h):
        
        Memory = self.PerceiverEncoder(h)
        context = self.SK_attention_read(h, Memory)

        return context

class SCOFFDynamics(nn.Module):
    """Dynamic sharing of parameters (GRU) between hidden state vectors.
    
    Args:
        `input_size`: dimension of a single input
        `hidden_size`: dimension of a single hidden state vector
        `num_hidden`:
        `num_rules`:
        `use_rule_embedding`: bool = False
    Inputs:
        `input`: [N, num_hidden, single_input_size]
        `h`: [N, num_hidden, single_hidden_size]
        
    Outputs:
        `h_new`: [N, num_hidden, single_hidden_size],
        `attn`: [N, num_hidden, num_rules] 
    """
  
    def __init__(self, input_size: int, hidden_size: int, num_hidden: int, num_rules: int, num_head: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size # dimension of each block's hidden state
        self.key_size = 128
        self.num_hidden = num_hidden
        self.num_head = num_head
        self.num_grus = num_rules
        self.pool_gru = nn.ModuleList([nn.GRU(input_size=self.hidden_size, 
                                              hidden_size=self.hidden_size,
                                              num_layers=1,
                                              batch_first=True,
                                              bidirectional=False) for _ in range(self.num_grus)])

        self.grus_keys = torch.nn.Parameter(
            torch.randn(1, self.num_grus, self.num_head*self.key_size))                        # (N, num_hidden, key_size)
        self.grus_attn = nn.MultiheadAttention(
            embed_dim=self.key_size, num_heads=self.num_head, batch_first=False)             # (N, num_hidden, key_size)
     
        self.query_proj = nn.Linear(self.hidden_size, self.num_head*self.key_size, bias=False) # (N*num_hidden, key_size)
        

    def forward(self, z, h):
        """
        Inputs:
            `input`: [N, num_hidden, single_input_size]
            `h`: [N, num_hidden, single_hidden_size]
            
        Outputs:
            `hnext`: [N, num_hidden, single_hidden_size],
            `attn_sm`: [N, num_hidden, num_rules] from softmax
            `attn_gsm`: [N, num_OFs, n_templates] (num_bloccks==k==num_object_files) from gumbel_softmax
        """

        #self.blockify_params()
        bs = h.shape[0]                                                                      # h: previous hidden state  
        # from current hidden to q, shape: [N, num_hidden, key_size]
        query = self.query_proj(h)  # from current hidden to q, shape: [N, num_hidden, key_size]
        h = h.reshape(-1, self.hidden_size)   # h.shape: (N*num_hidden, hidden_size)
        z = z.reshape(-1, self.input_size)  # Shape: [N*num_hidden, input_size]
       
        hnext_stack = []
     
        for gru in self.pool_gru:         # input [N*num_hidden, input_size], h [N*num_hidden, hidden_size]
            hnext_l, _ = gru(
                z.reshape(-1, 1, self.input_size),
                h.reshape(1, -1, self.hidden_size)
                )    # Shape: [N*num_hidden, hidden_size]
            hnext_stack.append(hnext_l.reshape(-1, self.num_hidden, 1, self.hidden_size))
          
        hnext = torch.cat(hnext_stack, 2) # Shape: [N, num_hidden, N_grus, hidden_size]
 
        keys = self.grus_keys.repeat(bs, 1, 1)  # n_ploicies,bsz,Embsz,
     
        attn_logits = torch.matmul(keys, query.transpose(1, 2)) / math.sqrt(self.key_size) # Shape: (N, num_grus, num_hidden)
        attention_score = nn.functional.gumbel_softmax(
            attn_logits, tau=1, hard=True, dim=2)  # (N, num_grus, num_hidden)
        h_new = torch.einsum(
                    'bikd, bki->bid', hnext, attention_score) #[N, num_hidden, hidden_size]
       
        return h_new

class AttentionalDynamicsUpdate(nn.Module):
    """
    Dynamics update function trough attention mechanism:

        - keys: [W_eh h, W_ez z]
        - query: W_q h
        - values: [W_vh h, W_vz z]
. 
    LayerNorm used for stability purposes.

    Args:
        `h_key_size`: hidden variables key size
        `z_key_size`: latent z key size
        'q_size' : query size 
        'hidden_size': hidden vectors size
        'num_heads': num heads

    Inputs:
        `z`: current memory slots of shape [N, K_mem, D_mem]
        `h`: hidden state vectors of shape [N, K_hidden, D_hidden] 
        
        
   Output:
        `h`: updated hidden state vectors
    """
    def __init__(self, h_key_size, z_key_size, hidden_size, z_size, num_heads=1):
        super().__init__()
        self.h_key_size = int(h_key_size/2)
        self.z_key_size = int(z_key_size/2)
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.num_heads = num_heads
        self.z_value_size = int(h_key_size/2)
        self.h_value_size = int(z_key_size/2)
        self.query_size = self.num_heads*self.z_key_size + self.num_heads*self.h_key_size 
   
        self.z_key_transform = nn.Linear(self.z_size, self.num_heads * self.z_key_size, bias=False) # latent_z -> write_key
        self.z_value_transform = nn.Linear(self.z_size, self.num_heads * self.z_value_size, bias=False) # latent_z -> write_value
        self.h_key_transform = nn.Linear(self.hidden_size, self.num_heads * self.h_key_size, bias=False) # hidden -> write_key
        self.h_value_transform = nn.Linear(self.hidden_size, self.num_heads * self.h_value_size, bias=False) # hidden -> write_value
        self.h_query_transform = nn.Linear(self.hidden_size, self.query_size, bias=False) # memory -> write_query

        self.attn_out_transform_output = nn.Linear(self.query_size, self.hidden_size, bias=False) # memory -> write_query
       
        self.LayerNorm = nn.LayerNorm(self.hidden_size)

    def forward(self, h, z):
        """
        Inputs:
            `inputs`: - h, hidden variables 
                      - z, latent variables sampled from MLPs
        
        Returns:
            `output`: - h, updated hidden variables 
        """
        
        # inputs = self.norm_inputs(inputs)            Normalize z and h? May be a good idea! 
   
        h = h.reshape(h.shape[0], -1, self.hidden_size)
       
        k_h = self.h_key_transform(h) # Shape: (batch_size, num_inputs, slot_size).
        v_h = self.h_value_transform(h) # Shape: (batch_size, num_inputs, slot_size).

        k_z = self.z_key_transform(z) # Shape: (batch_size, num_inputs, slot_size).
        v_z = self.z_value_transform(z) # Shape: (batch_size, num_inputs, slot_size).

        q_h = self.h_query_transform(h)
      
        
        attn_logits = torch.matmul(torch.cat((k_h, k_z), dim=2), q_h.transpose(1, 2)) / math.sqrt(self.hidden_size) # Shape: (batch_size, num_inputs, num_slots).
        #print(attn_logits.shape)
        probs = torch.softmax(attn_logits, dim=-1) # Shape: (batch_size, num_inputs, num_slots).
        #print(probs.shape)

        #print(torch.matmul(probs, torch.cat((v_h, v_z), dim=2)).shape)
        h_new = self.LayerNorm(self.attn_out_transform_output(torch.matmul(probs, torch.cat((v_h, v_z), dim=2))))
        #print(h_new.shape)
        return h_new
           
        

if __name__ == "__main__":
    
    h_key_size=100
    z_key_size=300
    hidden_size = 100
    z_size = 100
    Dynamics = AttentionalDynamicsUpdate(h_key_size, z_key_size, hidden_size, z_size, num_heads=4)

    h = torch.rand((64, 16, 100))
    z = torch.rand((64, 16, 100))

    h_new = Dynamics(h,z)

    
