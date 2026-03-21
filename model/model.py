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
from typing import Optional, Tuple, Union
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

# derived nn.Module
class RMSNorm(nn.Module):
    def __init__ (self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # scale parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
        

def PreCompute_freqence_cis(
        dim:int,
        end:int= int(32*1024),
        rope_base = float(1e6),
        rope_scaling:Optional[dict]=None
        ):
    # __init__
    frequence, attn_factor = (1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)),1.0)

    if rope_scaling is not None:
        original_max,factor,beta_fast,beta_slow = (
            rope_scaling.get("original_max_position_embeddings",2048),
            rope_scaling.get("factor",16),
            rope_scaling.get("beta_fast",32.0),
            rope_scaling.get("beta_slow",1.0),
            rope_scaling.get("attention_factor",1.0),
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
def apply_rotory_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
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

class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_key_value_heads if args.num_key_value_heads is not None 
            else args.num_attention_heads)
        
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
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
    
# FFN
class FeedForward(nn.Module):
    # init
    # to high demension
    # gate
    # SwiGLU
    # to low dimension
    # dropout
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size*8/3) #? experimentally, 8/3 is a good
            args.intermediate_size = 64*((intermediate_size+64-1)//64)# upn to 64x

        self.up_proj = nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.down_proj = nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
        self.gate_proj = nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self,x):
        # return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)*self.up_proj(x)))) # this is false, need to activate gate first to control the info flow
        return self.dropout(
            self.down_proj(
                self.act_fn(self.gate_proj(x))*self.up_proj(x)
                ))

# Transformer block
class Block(nn.Module):
    def __init__(self, layer_id:int, config:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size//self.num_attention_heads
        self.self_attn = Attention(config)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(self.hidden_size,eps= config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size,eps= config.rms_norm_eps)
        self.mlp = FeedForward(config)
    
    def forward(self,hidden_states,position_embeddings,past_key_values=None,use_cache=False,attention_mask=None):
        residual = hidden_states # backup hidden
        hidden_states,present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), # norm -> attn
            position_embeddings,
            past_key_values,
            use_cache,
            attention_mask,
            )
        hidden_states = residual + hidden_states # attn residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) # norm -> FFN
        return hidden_states,present_key_value

class Model(nn.Module):
    def __init__(self, config:MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size,self.num_hidden_layers=(
            config.vocab_size,
            config.num_hidden_layers,
        )
        # Token embeddings
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Block(i,config) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size,eps= config.rms_norm_eps)
        # RoPE precompute (a buffer, reduce repeat computation)
        freqs_cos,freqs_sin = PreCompute_freqence_cis(
            config.hidden_size//config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self,
            input_ids:Optional[torch.Tensor]=None,
            attention_mask:Optional[torch.Tensor]=None,
            past_key_values:Optional[Tuple[torch.Tensor]]=None,
            use_cache=False,
            **kwargs,
    ):
        batch_size, seq_len = input_ids.shape

        # resovle the compatibility issue (cant understand, pass)
        if hasattr(self, "freqs_cos"):
            past_key_values = None

        past_key_values = past_key_values or [None]*len(self.layers)
        
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len],
        )

        presents = []
        for _, (layer, _past_key_values) in enumerate( # _ is layer index
            zip(self.layers, past_key_values)
            ):
            hidden_states,_present=layer(
                hidden_states,
                position_embeddings,
                past_key_values=_past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(_present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
        [layer.mlp.aux_loss for layer in self.layers if hasattr(layer.mlp, 'aux_loss')],
        hidden_states.new_zeros(1).squeeze(),
    )

        return hidden_states,presents,aux_loss
    
class CausalLM(PreTrainedModel, GenerationMixin):# Huggingface/s
    config_class = MokioMindConfig

    def __init__(self, config:MokioMindConfig):
        self.config = config
        super().__init__(config)
        self.model = Model(config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size,bias=False
        )

        self.model.embed_tokens.weight = self.lm_head.weight # share weight for output and next input
        # self.OUT = CausalLMOutputWithPast() # from huggingface
        ## ↑, deleted for uninformed

    def forward(
            self,
            input_ids:Optional[torch.Tensor]=None,
            attention_mask:Optional[torch.Tensor]=None,
            labels: Optional[torch.Tensor] = None,
            past_key_values:Optional[Tuple[Tuple[torch.Tensor]]]=None,
            use_cache:bool=False,
            Logits_to_keep:Union[int,torch.Tensor]=0,
            **args,):
        hidden_states,past_key_values,aux_loss = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            use_cache = use_cache,
            **args,
        )
        # initialize output logits
        slice_indices=(
            slice(-Logits_to_keep,None) if isinstance(Logits_to_keep,int) 
            else Logits_to_keep)
        # [bsz,seq_len,hidden_size] -> [bsz,1(slice_indices),hidden_size]
        # lm_head: [bsz,1,hidden_size] -> [bsz,1,vocab_size]
        logits = self.lm_head(hidden_states[:,slice_indices,:]) 

        # loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # informed output
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output