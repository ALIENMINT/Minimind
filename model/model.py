from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch.nn as nn
import torch
import math
from typing import Optional, Tuple
from torch.nn import functional as F

# derived nn.Module
class RMSNorm(nn.Module):
    def __init__ (self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # scale parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        return x * self._norm(x.float()).type_as(x) * self.weight
        

def PreCompute_freqence_cis(dim:int,end:int(32*1024),rope_base,rope_scaling:Optional[dict]=None):
    # __init__
    frequence, attn_factor = (1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)),1.0)

    if rope_scaling is not None:
        original_max,factor,beta_fast,beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )
    
    # greater than original max position embedding
    if end > original_max:
        # b -> i
        inv_dim = lambda b:(dim*math.log(original_max/(b*2*math.pi)))/(2*math.log(rope_base))
        
        low,high = (max(math.floor(inv_dim(beta_fast)),0),
                    min(math.ceil(inv_dim(beta_slow)),dim//2))

        # i < low, ramp = 0; i > high, ramp = 1; else, ramp = (i-low)/(high-low)
        ramp = torch.clamp(
            (torch.arange(dim//2, device=frequence.device).float() -low)/max(high-low,0.001),
            0,
            1)

        frequence = frequence*(1-ramp+ramp/factor)
    
    # positional rank
    t = torch.arange(end, device=frequence.device).float()

    # get rotation angle
    frequence = torch.outer(t,frequence).float()
    frequence_cos = torch.cat([torch.cos(frequence),torch.cos(frequence)],dim=-1)*attn_factor
    frequence_sin = torch.cat([torch.sin(frequence),torch.sin(frequence)],dim=-1)*attn_factor

    return frequence_cos,frequence_sin

# RoPE implementation
def apply_rotory_pos_emb(q,k,frequence_cos,frequence_sin,position_ids=None,unsqueeze_dim=1):
    # [a,b]->[-b,a]
    def rotate_half(x):
        return  torch.cat(
            (-x[..., x.shape[-1]//2:], 
             x[..., :x.shape[-1]//2]),
            dim=-1
        )
    # x_rotated = x*cos+rotate_half(x)*sin
    q_embed = (q*cos.unsqueeze(unsqueeze_dim)+
               (rotate_half(q)*sin.unsqueeze(unsqueeze_dim)))
    k_embed = (k*cos.unsqueeze(unsqueeze_dim)+
                (rotate_half(k)*sin.unsqueeze(unsqueeze_dim)))
    return q_embed,k_embed

# group tensor for GQA
def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:,:,:,None,:]
            .expand(bs,slen,num_key_value_heads,n_rep,head_dim)
            .reshape(bs,slen,num_key_value_heads*n_rep,head_dim)
            )# for memory efficiency, rather than repeating the tensor

class Attention(nn.modules):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads//self.num_key_value_heads
        self.head_dim = args.hidden_size//args.num_attention_heads
        
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
        self.out_proj = nn.Linear(args.num_attention_heads*self.head_dim, args.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout=args.dropout

        self.flash=hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention

    # forward function
    # q,k,v:
    # split input into num_attention_heads and head_dim
    # RoPE
    # kv chache (use repeat_kv for GQA)
    # scaled dot product attention, qxk^T/sqrt(head_dim)
    # integrate heads and project output
    def forward(
            self,
            x:torch.Tensor, 
            position_embeddings:Tuple[torch.Tensor,torch.Tensor],past_key_values:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
            use_cache=False,
            attention_mask:Optional[torch.Tensor]=None,
            ) -> torch.Tensor:
        # calculate q,k,v
        bsz,seq_len,_ = x.shape
        xq,xk,xb = self.q_proj(x),self.k_proj(x),self.v_proj(x)
        # split into num_attention_heads and head_dim
        xq = xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)#[:,:,8,64]
        xk = xk.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        xv = xb.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        # RoPE
        cos,sin = position_embeddings
        xq,xk = apply_rotory_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
        # repeat kv for GQA (kv cache)
        if past_key_values is not None:
            xk = torch.cat([past_key_values[0],xk],dim=1)
            xv = torch.cat([past_key_values[1],xv],dim=1)
        past_kv = (xk,xv) if use_cache else None

        xq,xk,xv = (
            xq.transpose(1,2),#[bsz,n_local_heads,seq_len,head_dim]->[bsz,seq_len,n_local_heads,head_dim]
            repeat_kv(xk,self.n_rep).transpose(1,2),#[bsz,num_key_value_heads,seq_len,head_dim]->[bsz,seq_len,num_key_value_heads*n_rep,head_dim]
            repeat_kv(xv,self.n_rep).transpose(1,2),#[bsz,num_key_value_heads,seq_len,head_dim]->[bsz,seq_len,num_key_value_heads*n_rep,head_dim]
        )
        # scaled dot product attention
        if self.flash and seq_len >1 and (attention_mask is None or torch.all(attention_mask==1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz,1,1,-1).expand(bsz,self.n_local_heads,seq_len,-1).bool()
            )
            output = F.scaled_dot_product_attention(xq,xk,xv,attn_mask=attn_mask,dropout_p=self.dropout if self.training else 0.0,is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            # scores = scores + casual mask
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len),float("-inf"), device=scores.device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0-extended_attention_mask) * torch.finfo(scores.dtype).min
                scores = scores + extended_attention_mask
            # Softmax 
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        #[bsz,n_local_heads,seq_len,head_dim]->[bsz,seq_len,n_local_heads,head_dim]
        output=output.transpose(1,2).reshape(bsz,seq_len,-1)
        output=self.resid_dropout(self.out_proj(output))
        return output,past_kv
    