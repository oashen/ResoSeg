''' Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
from weaver.utils.logger import _logger
from .transformer import Block


# 动态修剪？
# 按照target的范围随机取数后添加上限为1，将输入随机修剪掉一部分
# 如果是training，就先将数据随机排序，确保所有mask为true的排在前面
class SequenceTrimmer(nn.Module):
    # **kwargs自动捕获父类需要的参数，不需要一一明确写出来
    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()
        out_perm=None
        maxlen=x.size(-1)
        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))#target范围内均匀取随机数后，上限为1
                    # type_as(x),将mask变量类型变成和x一样
                    # sum(dim=-1)后，最后一个维度消失。如mask=[[1,1,0,0],[1,1,1,0]],sum后变为[[2],[3]]
                    # torch.quantile(tensor，q),在len()*q的位置对tensor取值,分数位置也会取对应差值来的数
                    # 返回的是一个数
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    # ~mask，对mask取否后，为true的地方替换为-1
                    # 也即是一个应用mask的操作，为false则替换为-1
                    rand.masked_fill_(~mask, -1)
                    # 降序排列，argsort返回排列后的索引
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    out_perm=perm.clone().detach()
                    # gather(tensor,dim,idx),按perm的顺序取值，也即将mask按照rand的排序结果排序
                    mask = torch.gather(mask, -1, perm)
                    # perm.expand_as(x) ，把perm从(N,1,P)扩展到(N,C,P)，扩展的维度值和之前的相同。被扩展的维度必须是1
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        # unsqueeze(-1)扩张维度，(N,1,P)变为(N,1,P,1),实际就是把P的每一个值套起来了
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]
        return x, v, mask, uu,out_perm,maxlen


# embed，(batch,features,sqe_len)->(sqe_len,batch,embed_dim)
class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()
        # 标准化操作 (x-mean)/sigma，但是在此基础上引入了可学习参数，进行缩放和偏移，仅使用当前batch数据
        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                # GELU 高斯误差线性单元，ReLu修正线性单元，非线性层
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        # 模块封装
        self.embed = nn.Sequential(*module_list)

    def forward(self, x, condition=False):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            if not condition: x = x.permute(0,2,1).contiguous()
        # x: (batch, seq_len, embed_dim)
        return self.embed(x)


# concat模式直接将pairwise_lv_fts和uu卷积到一起，sum是分别embed之后相加
# forward返回的是对称元素矩阵，（batch,embed_dim,P,P)
class PairEmbed(nn.Module):
    # pairwise_input_dim是uu的dim？ pairwise_lv_dim是pairwise_lv_fts的output dim 也是embed的input dim（sum mod）
    # uu或许是另外的inductive bias

    def __init__(
            self, pairwise_lv_dim: list, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = sum(1 for x in pairwise_lv_dim if x > 0)
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        # partial函数作用为预绑定一些参数，后续调用pairwise_lv_fts时，可以调用self.pairwise_lv_fts(xi,xj)
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        # 两个input_dim直接卷积到一起
        if self.mode == 'concat':
            input_dim = self.pairwise_lv_dim + pairwise_input_dim
            # 等效于 if: = else: =[]
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    # 1维卷积层，input,output,卷积核
                    nn.Conv1d(input_dim, dim, 1),
                    # 卷积后归一化
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            # pre_activation 就不加非线性层了
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)

        # 两个独立的embed？
        elif self.mode == 'sum':
            if self.pairwise_lv_dim > 0:
                input_dim = self.pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    # extend和 + 结果相同，但更高效，+不会直接修改原list，但extend会
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None,ecms=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)

        # 验证输入有效性，x和uu至少有一个不能是none
        assert (x is not None or uu is not None)

        # 不计算梯度
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                # torch.tril_indices(row, col, offset) 生成二维方阵的下三角部分索引，offset=0包含主对角线，否则排除
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    # repeat，复制对应维度n次
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len) # (batch, dim, seq_len, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2) 注意i,j是下三角的索引（数组），所以是seq_len*(seq_len+1)/2
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj,ecms) #（batch，pairwise_lv_dim，seq_len*(seq_len+1)/2）
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2),ecms)
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


# track level + event level + generate P4 of eta_c
# The cls_fc is merged into fc
# need the index of px in the input features, and the py, pz, energy should be next to px
class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 # network configurations
                 pair_input_dim: list = [1, 1, 0, 1, 1, 0, 0, 0, 0],  # pairwise_lv_fts的num_outputs
                 embed_dims=[256,256],
                 pair_embed_dims=[64,64],
                 num_heads=4,
                 num_layers=8,  # 神经网络层数，多头注意+FFN算一层
                 block_params:dict=None,
                 activation='gelu',
                 block_activation='swiglu',
                 # misc
                 trim=True, # 动态修剪
                 CDN=False,
                 condition_input=2,
                 condition_dims=[32,128,32],
                 gate_attn=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        # 动态修剪
        self.trimmer = SequenceTrimmer(enabled=trim)

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        self.embed_dim = embed_dim
        # key为字符串的dict
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_heads=True, scale_resids=True,CDN=CDN,condition_dim=condition_dims[-1],gate_attn=gate_attn)

        cfg_block = copy.deepcopy(default_cfg)
        cfg_block['activation']=block_activation
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity() # 占位符层

        # embed最后是头数
        self.pair_embed = PairEmbed(
            pair_input_dim, 0, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=False, use_pre_activation_pair=True,
            for_onnx=False)

        if CDN:
            self.condition_embed= Embed(condition_input, condition_dims, activation=activation)
        else:
            self.condition_embed = None

        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, condition=None,ecms=None):
        # x: (batch,features,sqe_len)
        # v: (batch,4,sqe_len) [px,py,pz,energy]
        # mask: (batch,1,sqe_len) -- real particle = 1, padded = 0
        # for pytorch: uu (batch, C', num_pairs), uu_idx (batch, 2, num_pairs)
        # for onnx: uu (batch, C', sqe_len, sqe_len), uu_idx=None
        if type(x)==tuple:x,v,mask=x
        with torch.no_grad():
            if uu_idx is not None:
                uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu,perm,max_len = self.trimmer(x, v, mask, uu)   # 动态裁剪,perm是随机剪切后的顺序
            padding_mask = ~mask.squeeze(1)  # (N, P)   # mask转变为多头注意力需要的格式
            if perm is not None:
                perm = perm.squeeze(1)  # (N, P)

        # input embedding
        # (batch,features,sqe_len)->(batch,sqe_len,embed_dim)
        # embedding 后应用mask，填充为0
        x = self.embed(x)

        x=x.masked_fill(~mask.permute(0,2, 1), 0)  # (batch,sqe_len,embed_dim)
        attn_mask = None
        if (v is not None or uu is not None) and self.pair_embed is not None:
            attn_mask = self.pair_embed(v, uu,ecms=ecms).view(-1, v.size(-1), v.size(-1))  # (batch*num_heads, P, P)

        if self.condition_embed is not None:
            embed_condition=self.condition_embed(condition,condition=True)
        else:
            embed_condition=None

        # transform
        # 经过多头注意力和FFN后，x的维度是embed_dims[-1]
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, attn_mask=attn_mask,condition=embed_condition)

        x=self.norm(x)
        return x,mask,embed_condition,perm,max_len


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj,ecms, num_outputs: list=[1,1,0,1,1,0,0,0,0], eps=1e-8, for_onnx=False):
    '''
        calculate the interaction U
        :param xi: (N, 6, num_pairs) px,py,pz,energy,emc_e,q
        :param xj: (N, 6, num_pairs) px,py,pz,energy,emc_e,q
        :param num_outputs: a list that control which features are using, padding mask
    '''
    input_length = xi.size(1)
    if input_length > 4:
        emc_e_i = xi[:, 4:5, :]  # (batch, 1, num_pairs)
        emc_e_j = xj[:, 4:5, :]
    if input_length > 5:
        q_i = xi[:, 5:6, :]
        q_j = xj[:, 5:6, :]
    xi = xi[:, :4, :]   # (batch, 4, num_pairs)
    xj = xj[:, :4, :]   # (batch, 4, num_pairs)
    outputs=[]

    xij = xi + xj
    lnm2 = torch.log(to_m2(xij, eps=eps))
    if num_outputs[0]: outputs.append(lnm2)   # inv. M^2

    if len(num_outputs)>1 and num_outputs[1]:
        if not torch.is_tensor(ecms):
            ecms = torch.tensor(ecms, dtype=xi.dtype, device=xi.device)
        ecms = ecms.to(dtype=xi.dtype, device=xi.device)
        p_cms = torch.stack([
            ecms * 0.011,
            torch.zeros_like(ecms),
            torch.zeros_like(ecms),
            ecms
        ], dim=1).unsqueeze(-1)
        p_recoil = p_cms - (xi + xj)
        lnrm = torch.log(to_m2(p_recoil,eps=eps))
        outputs.append(lnrm)  # recoil mass^2

    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)

    if len(num_outputs)>2 and num_outputs[2]: outputs.append(lndelta)

    if len(num_outputs)>3 and num_outputs[3]:
        if input_length > 5:
            qij = q_i * q_j
            outputs.append(qij)  # charge product
        else:
            raise RuntimeError(
                f'pairwise_lv_fts requires q feature when num_outputs[3] == 1, but no q feature available, xi.size(1)={xi.size(1)}')

    if len(num_outputs)>4 and num_outputs[4]:
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        outputs.append(lnkt)

    if len(num_outputs)>5 and num_outputs[5]:
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs.append(lnz)

    if len(num_outputs)>6 and num_outputs[6]:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(torch.abs(costheta))    # helicity angle


    if len(num_outputs)>7 and num_outputs[7]:
        cospolar = xij[:, 2:3,:] / xij[:, :3, :].norm(dim=1, keepdim=True).clamp(min=eps)
        outputs.append(cospolar)   # polar angle of the pair momentum

    if len(num_outputs)>8 and num_outputs[8]:
        if input_length > 4:
            emc_eij = torch.log(torch.abs(emc_e_i - emc_e_j + eps))
            outputs.append(emc_eij)
        else:
            raise RuntimeError(
                f'pairwise_lv_fts requires emc_e feature when num_outputs[8] == 1, but no emc_e feature available, xi.size(1)={xi.size(1)}')

    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
