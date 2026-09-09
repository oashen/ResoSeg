import torch.nn as nn
import torch
from .Encoder import trunc_normal_
from .transformer import TwoWayBlock,SwishGLU,class_attention


class TaggingDecoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_class=3,
                 num_mask=3,
                 num_heads=4,
                 num_layers=2,  # 神经网络层数，多头注意+FFN算一层
                 block_params: dict = None,
                 activation='gelu',
                 block_activation='swiglu',
                 dot_scale=4,
                 mask_mlp:list=[(256,0.1),(256,0.1)],
                 evt_mlp:list=[(256,0.1),(256,0.1)],
                 skip_first_selfattn=False,
                 CDN=False,
                 condition_dim=32,
                 gate_attn=False,
                 evt_level=True
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_mask = num_mask
        self.num_class = num_class
        self.dot_dim=embed_dim // dot_scale

        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads,num_layers=num_layers, ffn_ratio=4,
                           dropout=0.1,attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=block_activation,
                           scale_heads=True,skip_first_selfattn=skip_first_selfattn,no_selfattn=not evt_level,
                           CDN=CDN,condition_dim=condition_dim,gate_attn=gate_attn)
        if block_params is not None:
            default_cfg.update(block_params)

        self.TwoWayAttn=TwoWayBlock(**default_cfg)
        # (B,num_mask,embed_dim)-MLP->(B,num_mask,dot_dim*num_class)-view->(B,num_mask*num_class,dot_dim)-after dot->(B,num_mask,num_class,sqe_len)
        self.mask_token = nn.Parameter(torch.zeros(1, num_mask, embed_dim), requires_grad=True)
        trunc_normal_(self.mask_token, std=.02)

        acts = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swiglu': SwishGLU()
        }
        act = acts[activation]
        # input particles and mask go through same size mlp
        mask_fcs = [[] for _ in range(self.num_mask)]
        particle_fcs=[]
        in_dim = embed_dim
        # !!!! Must be a ModuleList(), or else these part will not be identified as part of model
        self.mask_mlp = nn.ModuleList()
        for j in range(len(mask_mlp)):
            out_dim, drop_rate = mask_mlp[j]
            for i in range(self.num_mask):
                mask_fcs[i].append(nn.Sequential(nn.Linear(in_dim, out_dim), act, nn.Dropout(drop_rate)))
            if j!=len(mask_mlp)-1:
                particle_fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), act,nn.LayerNorm(out_dim), nn.Dropout(drop_rate)))
            else:
                particle_fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), act, nn.Dropout(drop_rate)))
            in_dim = out_dim
        for i in range(self.num_mask):
            mask_fcs[i].append(nn.Linear(in_dim, self.dot_dim*self.num_class))
            self.mask_mlp.append(nn.Sequential(*(mask_fcs[i])))
        particle_fcs.append(nn.Linear(in_dim, self.dot_dim))  # (B,sqe_len,embed_dim)->(B,sqe_len,dot_dim)
        self.particle_mlp = nn.Sequential(*particle_fcs)

        # evt score output
        if evt_level:
            self.evt_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            trunc_normal_(self.evt_token, std=.02)
            evt_fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in evt_mlp:
                evt_fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), act, nn.Dropout(drop_rate)))
                in_dim = out_dim
            evt_fcs.append(nn.Linear(in_dim, 2))
            self.evt_mlp = nn.Sequential(*evt_fcs)
        else:
            self.evt_token = None
            self.evt_mlp = None
    def forward(self, x, etac_token=None, mask=None,attn_mask=None,condition=None):
        # x (batch,sqe_len,embed_dim)
        # etac_token (batch,1,embed_dim) as prompt token, can also add e_cms as prompt
        # mask: (batch,1,sqe_len) -- real particle = 1, padded = 0
        padding_mask = ~mask.squeeze(1)
        batch_size = x.size(0)
        sqe_len = x.size(1)
        if self.evt_token is not None:
            token = torch.cat([self.evt_token, self.mask_token], dim=1)
        else:
            token=self.mask_token
        token = token.expand(batch_size, -1, -1)
        # token (batch,evt_token+num_mask*mask_token+etac_token
        if etac_token is not None:
            token = torch.cat([token, etac_token], dim=1)

        token, x = self.TwoWayAttn(token, x, padding_mask=padding_mask, attn_mask=attn_mask, condition=condition)

        if self.evt_token is not None:
            evt_token = token[:, 0, :] # (batch,dim)
            evt_token = self.evt_mlp(evt_token)  # (batch,2)

            mask_token = token[:, 1:1 + (self.num_mask), :]
        else:
            mask_token=token
            evt_token=None
        mask_list=[]
        for i in range(self.num_mask):
            mask_list.append(self.mask_mlp[i](mask_token[:,i,:]))
        # (B,num_mask,embed_dim)->(B,num_mask,dot_dim*num_class)->(B,num_mask*num_class,dot_dim)
        mask_token=torch.stack(mask_list,dim=1)
        mask_token = mask_token.view(batch_size, self.num_mask * self.num_class, self.dot_dim)
        # (batch, sqe_len, embed_dim)->(batch,  dot_dim, sqe_len)
        x=self.particle_mlp(x).transpose(1, 2)
        #(B, num_mask * num_class, sqe_len)->(B, num_mask,num_class, sqe_len)
        mask_output=(mask_token@x).view(batch_size,self.num_mask,self.num_class,sqe_len)
        # (B, num_mask,num_class, sqe_len)->(B, num_mask, sqe_len,num_class)
        mask_output=mask_output.permute(0,1,3,2)
        return evt_token, mask_output

class PartDecoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads=4,
                 num_layers=2,  # 神经网络层数，多头注意+FFN算一层
                 block_params: dict = None,
                 activation='gelu',
                 block_activation='swiglu',
                 evt_mlp:list=[(256,0.1),(256,0.1)],
                 CDN=False,
                 condition_dim=32,
                 gate_attn=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim

        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1,attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=block_activation,
                           scale_heads=True, scale_resids=True,
                           CDN=CDN,condition_dim=condition_dim,gate_attn=gate_attn)
        if block_params is not None:
            default_cfg.update(block_params)

        self.cls_attn = nn.ModuleList([class_attention(**default_cfg) for _ in range(num_layers)])
        self.evt_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.evt_token, std=.02)

        acts = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swiglu': SwishGLU()
        }
        act = acts[activation]
        # evt score output
        evt_fcs = []
        in_dim = embed_dim
        for out_dim, drop_rate in evt_mlp:
            evt_fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), act, nn.Dropout(drop_rate)))
            in_dim = out_dim
        evt_fcs.append(nn.Linear(in_dim, 2))
        self.evt_mlp = nn.Sequential(*evt_fcs)
    def forward(self, x, etac_token=None,mask=None,attn_mask=None,condition=None):
        # x (batch,sqe_len,embed_dim)
        # etac_token (batch,1,embed_dim) as prompt token, can also add e_cms as prompt
        # mask: (batch,1,sqe_len) -- real particle = 1, padded = 0
        padding_mask = ~mask.squeeze(1)
        batch_size = x.size(0)
        evt_token = self.evt_token.expand(batch_size, -1, -1)
        for attn in self.cls_attn:
            evt_token = attn(x, evt_token, padding_mask=padding_mask, attn_mask=attn_mask, condition=condition)
        evt_token=self.evt_mlp(evt_token).squeeze(1)   # (batch,2)
        return evt_token, None

