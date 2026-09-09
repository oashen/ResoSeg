from .Encoder import Encoder
from .GenP4 import GenP4
from .TaggingDecoder import TaggingDecoder,PartDecoder
import torch.nn as nn
import torch


class SamTaggerModel(nn.Module):
    def __init__(self,
                 input_dim,
                 num_class:int=3,
                 num_mask:int=3,
                 pair_input_dim:list=[1,1,0,1,1,0,0,0,0],  # pairwise_lv_fts的num_outputs
                 embed_dims:list=[256, 256],
                 pair_embed_dims:list=[64, 64],
                 dot_scale:int=4,
                 num_heads:int=4,
                 embed_layers:int=8,
                 gen_layers:int=2,
                 decoder_layers:int=2,
                 embed_block_params: dict = None,
                 gen_block_params: dict = None,
                 decoder_block_params: dict = None,
                 activation:str='gelu',
                 block_activation:str='swiglu',
                 gen_mlp:list=[(256, 0.1), (256, 0.1)],
                 mask_mlp: list = [(256, 0.1), (256, 0.1)],
                 evt_mlp: list = [(256, 0.1), (256, 0.1)],
                 trim:bool=True,
                 threshold:float=-1,
                 CDN:bool = False,  # use condition layer norm or not
                 condition_input:int = 2,
                 condition_dims:list = [32, 128, 32],
                 gate_attn=False,
                 mod_only_encoder=False,
                 part=False,
                 sam=True
                 ):
        '''
        :param num_class: num of particle tagging labels
        :param num_mask: num of output tagging results
        :param pair_input_dim: dim of the interaction inout features, check Encoder.pairwise_lv_fts() for detail
        :param dot_scale: dot_dim = embed_dims[-1]//dot_scale, dot_dim is the dim when mask@particle in TaggingDecoder
        :param embed_layers: layers of embed attn
        :param gen_layers: layers of generation attn, if gen_layers<=0, do not generate p4
        :param decoder_layers: layers of decoder block(include self attn + 2 cross attn)
        :param embed_block_params: change other settings of embed attn block here,same for other two
        :param activation: activation of MLPs
        :param block_activation: activation of MLPs in attn blocks
        :param gen_mlp: mlp for output, in the form of (dim,dropout), same for other 3
        :param trim: use the SequenceTrimmer or not
        :param threshold: the eta_c score threshold, if is not None, will output the inv. mass of eta_c.
        if threshold < 0, use a relative cut: mask[:,:,1] is the biggest score.
        if threshold >1 , use a learnable parameter, initial value = 0.5.
        :param CDN: use condition layer norm or not.
        :param condition_input: input dim of condition
        :param condition_dims: embed dim of condition
        :param gate_attn: use gate attention or not
        :param mod_only_encoder: if is True, gate_attn and CDN only apply to encoder
        :param part: use ParT decoder or not
        :param sam: use SAM decoder or not, if both part and sam are activate, sam will not include event token
        '''
        super().__init__()
        embed_dim = embed_dims[-1]
        encoder = Encoder(
            input_dim,
            pair_input_dim=pair_input_dim,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=embed_layers,
            block_params=embed_block_params,
            activation=activation,
            block_activation=block_activation,
            trim=trim,
            CDN=CDN,
            condition_input=condition_input,
            condition_dims=condition_dims,
            gate_attn=gate_attn
        )
        if gen_layers > 0:
            gen = GenP4(
                embed_dim,
                num_heads=num_heads,
                num_layers=gen_layers,  # 神经网络层数，多头注意+FFN算一层
                block_params=gen_block_params,
                activation=activation,
                block_activation=block_activation,
                mlp=gen_mlp,
                CDN=(CDN and not mod_only_encoder),
                condition_dim=condition_dims[-1],
                gate_attn=(gate_attn and not mod_only_encoder)
            )
        else: gen = None
        decoder=nn.ModuleList()
        if part:
            ParT_decoder=PartDecoder(
                embed_dim,
                num_heads=num_heads,
                num_layers=decoder_layers,  # 神经网络层数，多头注意+FFN算一层
                block_params=decoder_block_params,
                activation=activation,
                block_activation=block_activation,
                evt_mlp=evt_mlp,
                CDN=(CDN and not mod_only_encoder),
                condition_dim=condition_dims[-1],
                gate_attn=(gate_attn and not mod_only_encoder)
            )
            decoder.append(ParT_decoder)
        if sam:
            SAM_decoder=TaggingDecoder(
                embed_dim,
                num_class=num_class,
                num_mask=num_mask,
                num_heads=num_heads,
                num_layers=decoder_layers,  # 神经网络层数，多头注意+FFN算一层
                block_params=decoder_block_params,
                activation=activation,
                block_activation=block_activation,
                dot_scale=dot_scale,
                mask_mlp=mask_mlp,
                evt_mlp=evt_mlp,
                skip_first_selfattn= (gen is not None),
                CDN=(CDN and not mod_only_encoder),
                condition_dim=condition_dims[-1],
                gate_attn=(gate_attn and not mod_only_encoder),
                evt_level=not part
            )
            decoder.append(SAM_decoder)
        if not decoder:
            raise ValueError("The model have no decoder, please set at least one decoder (ParT or SAM or both)")
        self.model = SamTagger(encoder=encoder,tagging_decoder=decoder,genP4=gen)
        self.threshold = threshold
        if self.threshold > 1:
            self.threshold = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.num_mask = num_mask
    def cal_invm(self, p4, mask):
        if self.threshold < 0:
            max_indices = torch.argmax(mask, dim=-1)    # (B, seq_len)
            etac_mask = (max_indices == 1)  # (B, seq_len)
        else:
            etac_mask = mask[:, :, 1] > self.threshold  # (B, seq_len)
        etac_mask = etac_mask.unsqueeze(-1).expand_as(p4)  # 形状: (B, seq_len, 4)
        # etac_mask.float(): True->1,False->0
        masked_p4 = p4 * etac_mask.float()  # 形状: (B, seq_len, 4)
        masked_etac_p4 = masked_p4.sum(dim=1)  # 形状: (B, 4)

        px, py, pz, e = masked_etac_p4[:, 0], masked_etac_p4[:, 1], masked_etac_p4[:, 2], masked_etac_p4[:, 3]
        mass_squared = e ** 2 - (px ** 2 + py ** 2 + pz ** 2)
        mass_squared = torch.clamp(mass_squared, min=0)
        mass = torch.sqrt(mass_squared)
        mass = mass.unsqueeze(-1)
        mass = torch.cat([masked_etac_p4, mass], dim=1)
        return mass

    def forward(self, x, v=None, mask=None, std=None, condition=None, ecms=None):
        '''
            :param x: (batch,features,sqe_len)
            :param v: (batch,4,sqe_len) [px,py,pz,energy]
            :param mask: (batch,1,sqe_len) -- real particle = 1, padded = 0
            :param std:  the (mean,std) of etac P4
            :param condition: event level condition, like num of charged tracks, num of photons
            :param ecms: The energy of center-of-mass system, need it to calculate the recoil mass of interaction U

            return:
            evt_score (B, 2)
            tagging_mask (B, num_mask, sqe_len, num_class) is the tagging result
            perm (B, sqe_len) is the sequence after SequenceTrimmer, needed in the loss cal.
            max_len is sqe_len after SequenceTrimmer, needed in the loss cal.
            gen_p4 (B, 4) is the generated p4 of eta_c
            P5 (B, num_mask,5) [px,py,pz,e,m] of eta_c according to the mask, if threshold is None,return None
        '''
        evt_score,tagging_mask,gen_p4,perm,max_len=self.model(x,v=v,mask=mask,std=std,condition=condition,ecms=ecms)

        # with torch.no_grad():  !!!!!!!!!!!! must not be here
        if self.threshold is not None and tagging_mask is not None:
            if perm is not None:
                temp_perm=perm.unsqueeze(1)
                v = torch.gather(v, -1, temp_perm.expand_as(v))
            # -> (B,sqe_len,4)
            v=v[:, :, :max_len].permute(0,2,1).contiguous()
            p5=[]
            for i in range(self.num_mask):
                mask=tagging_mask[:,i,:,:] # (B, sqe_len, num_class)
                p5.append(self.cal_invm(v,mask)) # (B,5) [px.py,pz,e,m]
            p5=torch.stack(p5,dim=1)
        else:
            p5=None
        return evt_score, tagging_mask, perm, max_len, gen_p4, p5


class SamTagger(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 tagging_decoder: nn.ModuleList,
                 genP4: GenP4 = None):
        super().__init__()
        self.encoder = encoder
        self.genP4 = genP4
        self.tagging_decoder = tagging_decoder
        # If do not generate P4, use learnable parameters as eatc token
        # if self.GenP4 is None:
        #     self.etac_token = nn.Parameter(torch.zeros(1, 1, Encoder.embed_dim), requires_grad=True)
        #     trunc_normal_(self.cls_token, std=.02)
    def forward(self,x,v=None, mask=None,std=None,condition=None,ecms=None):
        # mask is updated by SequenceTrimmer, perm is the sequence of SequenceTrimmer
        # x (batch,sqe_len,embed_dim) perm(batch,sqe_len)
        x, mask, embed_condition, perm, max_len = self.encoder(x, v, mask, condition=condition,ecms=ecms)
        gen_p4 = None
        # etac_token (batch,1,embed_dim),gen_p4(batch,4)
        if self.genP4 is not None:
            etac_token, gen_p4 = self.genP4(x, mask=mask, std=std,condition=embed_condition)
        else:
            # self.etac_token.expand(x.size(0),1, -1)
            etac_token=None
        evt_score=None
        tagging_mask=None
        for decoder in self.tagging_decoder:
            output = decoder(x,etac_token,mask,condition=embed_condition)
            if output[0] is not None:
                evt_score=output[0]
            if output[1] is not None:
                tagging_mask=output[1]
        return evt_score, tagging_mask, gen_p4, perm, max_len