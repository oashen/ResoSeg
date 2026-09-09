import torch.nn as nn
import torch
import torch.nn.functional as F

# Adaptive Layer Normalization
class AdaLN(nn.Module):
    def __init__(self, embed_dim:int, condition_dim:int):
        '''
        Adaptive Layer Normalization
        :param embed_dim: the embedding dimension of x
        :param condition_dim: the embedding dimension of condition
        '''
        super().__init__()
        self.GELU   = nn.GELU()
        self.linear = nn.Linear(condition_dim, 2 * embed_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, condition_emb):
        '''
        Args:
        :param x: [batch, seq_len, embed_dim]
        :param condition_emb: [batch, condition_dim] is the embedding of condition

        Returns: condition layer normed input x
        '''
        emb = self.linear(self.GELU(condition_emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        x = F.layer_norm(x, x.shape[-1:], weight=None, bias=None, eps=1e-6)
        return x * (1 + scale) + shift

class SmartNorm(nn.Module):
    def __init__(self, embed_dim:int, condition_dim:int=None):
        '''
        automatically choose norm class, if condition_dim is None use nn.LayerNorm, else use AdaLN
        '''
        super().__init__()
        if condition_dim is None:
            self.norm = nn.LayerNorm(embed_dim)
            self.CDN=False
        else:
            self.norm = AdaLN(embed_dim, condition_dim)
            self.CDN=True
    def forward(self, x, condition = None):
        if self.CDN:
            return self.norm(x, condition)
        else:
            return self.norm(x)

# 多头注意力模块，含两个可学习参数向量，对头输出和FFN输出进行缩放
class Block(nn.Module):
    # 没有使用nn.Sequential()，而是手动控制
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='swiglu',scale_heads=True, scale_resids=True,
                 CDN=False,condition_dim=32,gate_attn=False):
        super().__init__()
        self.attn=Attention(embed_dim=embed_dim, num_heads=num_heads,
                 dropout=attn_dropout, add_bias_kv=add_bias_kv, scale_heads=scale_heads, gate_attn=gate_attn)

        self.dropout = nn.Dropout(dropout)

        # ffn
        self.mlp=MLP(embed_dim=embed_dim, ffn_ratio=ffn_ratio,activation_dropout=activation_dropout,
                     activation=activation)

        # 动态控制 残差贡献
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

        if not CDN: condition_dim=None
        self.attn_norm=SmartNorm(embed_dim, condition_dim)
        self.mlp_norm=SmartNorm(embed_dim, condition_dim)

    def forward(self, q,kv=None, padding_mask=None, attn_mask=None,condition=None):
        """
        Args:
            q (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            kv (Tensor, optional): if is None, K=V=q. else K=V=kv, shape `(KV_seq_len, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, KV_seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = q
        q=self.attn_norm(q,condition)
        q = self.attn(q,kv=kv,padding_mask=padding_mask,attn_mask=attn_mask)
        q = self.dropout(q)
        q += residual   # 残差连接，即输入和输出相加

        residual = q
        q=self.mlp_norm(q,condition)
        q=self.mlp(q)
        q = self.dropout(q)
        if self.w_resid is not None:
            # 可以替换成residual=torch.einsum('c,abc->abc', self.w_resid, residual)
            # mul为逐元素乘法，更高效，应用广播机制
            # pytorch的广播机制，从右向左（低到高）维扩展(4)+(1,2,4)->(1,2,4)+(1,2,4)
            # 维度是1的情况下，可以通过数据复制来扩展
            # (1,4)+(2,4)->(2,4)+(2,4)
            # (2,2,3)+(4,4,4)不能通过广播机制扩展
            residual = torch.mul(self.w_resid, residual)
        q += residual
        return q

class class_attention(nn.Module):
    # 没有使用nn.Sequential()，而是手动控制
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='swiglu',scale_heads=True, scale_resids=True,
                 CDN=False,condition_dim=32,gate_attn=False):
        super().__init__()
        self.attn=Attention(embed_dim=embed_dim, num_heads=num_heads,
                 dropout=attn_dropout, add_bias_kv=add_bias_kv, scale_heads=scale_heads, gate_attn=gate_attn)

        self.dropout = nn.Dropout(dropout)

        # ffn
        self.mlp=MLP(embed_dim=embed_dim, ffn_ratio=ffn_ratio,activation_dropout=activation_dropout,
                     activation=activation)

        # 动态控制 残差贡献
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

        if not CDN: condition_dim=None
        self.attn_norm=SmartNorm(embed_dim, condition_dim)
        self.mlp_norm=SmartNorm(embed_dim, condition_dim)

    def forward(self, x, x_cls=None,kv=None, padding_mask=None, attn_mask=None,condition=None):
        """
        class attention: https://arxiv.org/pdf/2103.17239.pdf
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor): class token, shape '(1,batch,embed_dim)'
            kv (Tensor, optional): if is None, K=V=q. else K=V=kv, shape `(KV_seq_len, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, KV_seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            q=x_cls
            kv = torch.cat((x_cls, x), dim=1)  # ( batch,seq_len+1, embed_dim)
        else:
            q=x
        residual = q
        q = self.attn_norm(q,condition)
        q = self.attn(q,kv=kv,padding_mask=padding_mask,attn_mask=attn_mask)
        q = self.dropout(q)
        q += residual   # 残差连接，即输入和输出相加

        residual = q
        q=self.mlp_norm(q,condition)
        q=self.mlp(q)
        q = self.dropout(q)
        if self.w_resid is not None:
            # 可以替换成residual=torch.einsum('c,abc->abc', self.w_resid, residual)
            # mul为逐元素乘法，更高效，应用广播机制
            # pytorch的广播机制，从右向左（低到高）维扩展(4)+(1,2,4)->(1,2,4)+(1,2,4)
            # 维度是1的情况下，可以通过数据复制来扩展
            # (1,4)+(2,4)->(2,4)+(2,4)
            # (2,2,3)+(4,4,4)不能通过广播机制扩展
            residual = torch.mul(self.w_resid, residual)
        q += residual
        return q


class TwoWayBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8,num_layers=2, ffn_ratio=4,
                 dropout=0.1,attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='swiglu',scale_heads=True,
                 skip_first_selfattn=False,no_selfattn=False,
                 CDN=False,condition_dim=32,gate_attn=False):
        super().__init__()
        if not CDN: condition_dim = None
        self.twowayattn = nn.ModuleList(
            [TwoWayAttention(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio,
                             dropout=dropout,attn_dropout=attn_dropout, activation_dropout=activation_dropout,
                             add_bias_kv=add_bias_kv, activation=activation,
                             scale_heads=scale_heads, skip_self_attn=(((i == 0) and skip_first_selfattn) or no_selfattn),
                             CDN=CDN, condition_dim=condition_dim,gate_attn=gate_attn) for i in range(num_layers)]
        )

        self.token_to_particle_norm = SmartNorm(embed_dim,condition_dim)
        self.token_to_particle_attn = Attention(embed_dim=embed_dim, num_heads=num_heads,
                                                dropout=attn_dropout, add_bias_kv=add_bias_kv,
                                                scale_heads=scale_heads,gate_attn=gate_attn)

    def forward(self, token,particle, padding_mask=None, attn_mask=None,condition=None):
        for attn in self.twowayattn:
            token,particle = attn(token,particle, padding_mask=padding_mask, attn_mask=attn_mask,condition=condition)

        residual = token
        token = self.token_to_particle_norm(token,condition)
        token = self.token_to_particle_attn(token,kv=particle,padding_mask=padding_mask,attn_mask=attn_mask)
        token += residual
        return token,particle


class TwoWayAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,dropout=0.1,
                 attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='swiglu',
                 scale_heads=True,skip_self_attn=False,
                 CDN=False,condition_dim=32,gate_attn=False):
        super().__init__()
        if not CDN:condition_dim = None
        # token: norm + attn
        if not skip_self_attn:
            self.self_norm = SmartNorm(embed_dim, condition_dim)
            self.self_attn = Attention(embed_dim=embed_dim, num_heads=num_heads,
                                       dropout=attn_dropout, add_bias_kv=add_bias_kv,
                                       scale_heads=scale_heads,gate_attn=gate_attn)
        else:
            self.self_attn =None

        self.token_to_particle_norm = SmartNorm(embed_dim,condition_dim)
        self.token_to_particle_attn = Attention(embed_dim=embed_dim, num_heads=num_heads,
                                                dropout=attn_dropout, add_bias_kv=add_bias_kv,
                                                scale_heads=scale_heads,gate_attn=gate_attn)

        # ffn
        self.mlp_norm = SmartNorm(embed_dim,condition_dim)
        self.mlp = MLP(embed_dim=embed_dim, ffn_ratio=ffn_ratio, activation_dropout=activation_dropout,
                       activation=activation)

        self.particle_to_token_norm = SmartNorm(embed_dim,condition_dim)
        self.particle_to_token_attn = Attention(embed_dim=embed_dim, num_heads=num_heads,
                                                dropout=attn_dropout, add_bias_kv=add_bias_kv,
                                                scale_heads=scale_heads,gate_attn=gate_attn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token,particle, padding_mask=None, attn_mask=None,condition=None):
        if self.self_attn is not None:
            residual=token
            token=self.self_norm(token,condition)
            token=self.self_attn(token,kv=token)
            token += residual

        residual = token
        token = self.token_to_particle_norm(token,condition)
        token=self.token_to_particle_attn(token,kv=particle,padding_mask=padding_mask,attn_mask=attn_mask)
        token += residual

        residual = token
        token = self.mlp_norm(token,condition)
        token = self.mlp(token)
        token += residual

        residual = particle
        particle = self.particle_to_token_norm(particle,condition)
        particle = self.particle_to_token_attn(particle,kv=token)
        particle += residual

        token=self.dropout(token)
        particle=self.dropout(particle)
        return token,particle


# norm + multi-head attention + norm (optional)
class Attention(nn.Module):
    # 没有使用nn.Sequential()，而是手动控制
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1, add_bias_kv=False, scale_heads=True, gate_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # heads
        if gate_attn:
            self.attn = GatedAttention(embed_dim, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                add_bias_kv=add_bias_kv,  # 在K和V投影过程中添加可学习的偏置项
                batch_first=True
            )

        # 可学习参数，调整各头的重要性
        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None

    def forward(self, q, kv=None, padding_mask=None, attn_mask=None):
        """
        Args:
            q (Tensor): input to the layer of shape `(batch, seq_len, embed_dim)`
            kv (Tensor, optional): if is None, K=V=q. else K=V=kv, shape `(batch, KV_seq_len, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, KV_seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, seq_len, embed_dim)`
        """
        if kv is None: kv = q
        q = self.attn(q, kv, kv, key_padding_mask=padding_mask,
                      attn_mask=attn_mask)[0]  # (batch, seq_len, embed_dim)

        # 先把x拆开，每个head乘上缩放因子后再拼回去
        if self.c_attn is not None:
            batch_size = q.size(0)
            # 前两个维度不变，最后一个维度拆分成(num_heads,head_dim)
            q = q.view(batch_size, -1, self.num_heads, self.head_dim)
            # einsum 爱因斯坦求和规定，tbhd是x的维度，h是c_attn的维度，h维相乘，然后输出的时候hd互换，维度标记可以是任意字母，区分大小写
            # 将每个头乘上其缩放因子
            q = torch.einsum('tbhd,h->tbdh', q, self.c_attn)
            # view的性能更好，但要求数据在内存上连续，reshape不要求
            q = q.reshape(batch_size, -1, self.embed_dim)
        return q


class MLP(nn.Module):
    def __init__(self, embed_dim=128, ffn_ratio=4,activation_dropout=0.1,activation='swiglu'):
        super().__init__()
        self.ffn_dim = embed_dim * ffn_ratio
        # ffn
        acts = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swiglu': SwishGLU()
        }
        self.fc1 = nn.Linear(
            embed_dim,
            self.ffn_dim * 2 if activation == 'swiglu' else self.ffn_dim
        )
        self.act = acts.get(activation, acts['relu'])
        self.act_dropout = nn.Dropout(activation_dropout)
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)
    def forward(self, q):
        q = self.act(self.fc1(q))
        q = self.act_dropout(q)
        return self.fc2(q)


class SwishGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

class GatedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 分离投影层，支持不同的输入来源
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

        # 门控线性层（基于 query 输入）
        self.gate_linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.ones_(self.gate_linear.bias)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        """
        Args:
            q: (B, N_q, C)
            k: (B, N_k, C)
            v: (B, N_v, C)
            key_padding_mask: padding mask particle=0, padded=1
            attn_mask: (B * num_heads, N_q, N_k)，attn += attn_mask
        Returns:
            输出张量，形状 (B, N_q, C)
        """
        B, N_q, C = q.shape
        B, N_k, C_k = k.shape
        B, N_v, C_v = v.shape

        # 投影并 reshape 为多头格式
        p_q = self.q_proj(q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N_q, D)
        p_k = self.k_proj(k).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N_k, D)
        p_v = self.v_proj(v).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N_v, D)

        # 计算注意力分数 (B, H, N_q, N_k)
        attn = (p_q @ p_k.transpose(-2, -1)) * self.scale

        # 添加浮点数注意力掩码（如果提供）
        if attn_mask is not None:
            # attn_mask 形状: (B * H, N_q, N_k) -> (B, H, N_q, N_k)
            attn_mask = attn_mask.reshape(B, self.num_heads, N_q, N_k)
            attn = attn + attn_mask

        # 应用padding mask（如果提供）
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        # softmax 与 dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权聚合：输出 (B, H, N_q, D) -> (B, N_q, C)
        x_attn = (attn @ p_v).transpose(1, 2).reshape(B, N_q, C)

        # 门控机制（基于 query 输入 x）
        gate_scores = torch.sigmoid(self.gate_linear(q))
        x_out = self.proj(x_attn * gate_scores)
        return x_out, attn
