''' Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
'''

import copy
import torch
import torch.nn as nn
from weaver.utils.logger import _logger
from .transformer import Block
from .Encoder import Embed


# track level + event level + generate P4 of eta_c
# The cls_fc is merged into fc
# need the index of px in the input features, and the py, pz, energy should be next to px
class GenP4(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads=4,
                 num_layers=2,  # 神经网络层数，多头注意+FFN算一层
                 block_params:dict=None,
                 activation='gelu',
                 block_activation='swiglu',
                 mlp=[(256, 0.1), (256, 0.1)],
                 CDN=False,
                 condition_dim=32,
                 gate_attn=False,
                 # misc
                 **kwargs) -> None:
        super().__init__(**kwargs)
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,scale_heads=True, scale_resids=True,
                           CDN=CDN,condition_dim=condition_dim,gate_attn=gate_attn)
        cfg_block = copy.deepcopy(default_cfg)
        cfg_block['activation']=block_activation
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        self.etac_embed = Embed(4, [embed_dim], activation=activation)

        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])

        # FFN for P4 gen
        gen_fcs = []
        in_dim = embed_dim
        for out_dim, drop_rate in mlp:
            gen_fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        gen_fcs.append(nn.Linear(in_dim, 4))

        self.gen_mlp = nn.Sequential(*gen_fcs)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x,mask=None, std:dict=None,condition=None):
        """
            Args:
                x (batch,sqe_len,embed_dim): is the embedded inputs
                mask (N, 1, P) : real particle = 1, padded = 0
                std (dict): the mean and std of etac truth p4, in the form of {'px':(mean,std),'py':(mean,std),
                'pz':(mean,std),'energy':(mean,std)}. energy is optional. If energy is not in std, will generate
                inv. mass instead of energy

            Returns:
                etac_token(batch,1,embed_dim), gen_p4 (batch,4)
        """
        with torch.no_grad():
            # generate the random p4 of etac
            random_p4 = torch.empty(x.size(0), 1, 4).uniform_(-0.7, 0.7)
            sum_of_squares = torch.sum(random_p4[..., :3] ** 2, dim=2, keepdim=True)
            etac_m = torch.randn(x.size(0), 1, 1) * 0.0305 + 2.9841
            final_value = (etac_m ** 2 + sum_of_squares) ** 0.5
            random_p4[..., 3] = final_value.squeeze(-1)
            if (std is not None):
                random_p4[:, :, 0] = (random_p4[:, :, 0] - std['px'][0]) / std['px'][1]
                random_p4[:, :, 1] = (random_p4[:, :, 1] - std['py'][0]) / std['py'][1]
                random_p4[:, :, 2] = (random_p4[:, :, 2] - std['pz'][0]) / std['pz'][1]
                if 'energy' in std.keys():
                    random_p4[:, :, 3] = (random_p4[:, :, 3] - std['energy'][0]) / std['energy'][1]
                else:
                    # energy not in std means use m to replace energy
                    random_p4[:, :, -1:] = etac_m
            # the etac particle
            etac_token = random_p4.permute(0, 2, 1).to(x.device) # embed need (batch,feature,sqe_len)

        etac_token = self.etac_embed(etac_token)

        padding_mask = ~mask.squeeze(1)  # (N, P)   # mask转变为多头注意力需要的格式
        # cross attn, Q= etac token, K=V=x
        for block in self.blocks:
            etac_token = block(etac_token, kv=x, padding_mask=padding_mask,condition=condition)
        gen_p4 = self.gen_mlp(etac_token).squeeze(1)   # (batch, 4)

        etac_token = self.norm(etac_token)

        return etac_token, gen_p4
