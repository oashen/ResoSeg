from .SamTaggerModel import SamTaggerModel
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
import math


class SamTaggerLightning(pl.LightningModule):
    def __init__(self, config:dict,input_dim,std):
        super().__init__()
        torch.cuda.empty_cache()
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        samtagger_config=config['SamTagger']
        samtagger_config['input_dim'] = input_dim

        self.mod = SamTaggerModel(**samtagger_config).to(device)
        self.mloss=config['mloss']
        self.stable_lr = config['stable_lr']
        self.UW = config['UW']
        self.R_loss = config['R_loss']
        self.CDN = config['SamTagger']['CDN']

        if self.UW:
            self.f_tag = nn.Parameter(torch.tensor(0, dtype=torch.float32,device=device))
            self.f_evt = nn.Parameter(torch.tensor(0, dtype=torch.float32,device=device))
            self.f_gen = nn.Parameter(torch.tensor(0, dtype=torch.float32,device=device))
            self.f_R = nn.Parameter(torch.tensor(0, dtype=torch.float32,device=device))
            self.log_vars = [self.f_tag, self.f_evt]
            if samtagger_config['gen_layers']>0:self.log_vars.append(self.f_gen)
            if self.R_loss:self.log_vars.append(self.f_R)
        else:
            self.f_tag = config['f_tag']
            self.f_evt = config['f_evt']
            self.f_gen = config['f_gen']
            self.f_R=config['f_R']

        self.target_m_mean=config['target_m_mean']
        self.LT = config['SamTagger']['threshold'] > 1


        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config['SamTagger']['num_class'], ignore_index=-1)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config['SamTagger']['num_class'], ignore_index=-1)
        self.train_evt_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2, ignore_index=-1)
        self.val_evt_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2, ignore_index=-1)
        self.MSE = nn.MSELoss()
        self.std=std
        self.save_hyperparameters()
    def forward(self, *x):
        return self.mod(*x)
    def process_model_input(self, x):
        masks, features, vectors, condition, ecms = x
        features = features.permute(0, 2, 1)  # permute the features to (batch_size, num_features, num_tokens)
        vectors = vectors.permute(0, 2, 1)

        if self.CDN:
            return features, vectors, masks, self.std, condition, ecms   # x,v,maske,uu,uu_idx,std
        else:
            return features, vectors, masks, self.std, None, ecms
    def process_output(self,batch,output,mode=0):
        '''
        :param batch:
            x
            label (B, sqe_len)
            n_tot (B)
            evt_label (B)
            p4: (B, 4)
            reconstructed target m (B)
        :param output:
            evt_score (B, 2)
            tagging_mask (B, num_mask, sqe_len, num_class) is the tagging result
            perm (B, sqe_len) is the sequence after SequenceTrimmer, needed in the loss cal.
            max_len is sqe_len after SequenceTrimmer, needed in the loss cal.
            gen_p4 (B, 4) is the generated p4 of eta_c
            p5 (B, num_mask,5) [px,py,pz,e,m] of eta_c according to the mask, if threshold is None,return None
        :param mode: 0 is train, 1 is val, 2 is pred
        '''
        evt_score, tagging_mask, perm, max_len, gen_p4, p5 = output
        losses=[]
        if mode in [0, 1]:
            _, label, n_tot, evt_label, p4,target_m = batch
            # tag_loss
            if tagging_mask is not None:
                num_mask=self.mod.num_mask
                n_tot = n_tot.sum()
                if perm is not None:
                    label = torch.gather(label, -1, perm)
                label = (label[:,:max_len]).permute(1, 0)  # (sqe_len, B)
                tag_loss=0
                for i in range(num_mask):
                    mask = (tagging_mask[:,i,:,:]).permute(1, 0, 2)  # (sqe_len, B, num_class)
                    for token, j in zip(mask, label):
                        tag_loss += F.cross_entropy(token, j, reduction='sum', ignore_index=-1)
                        if mode == 0: self.train_acc(token, j)
                        elif mode == 1: self.val_acc(token, j)
                tag_loss = tag_loss/(num_mask*n_tot)
                losses.append(tag_loss)
            else:
                tag_loss=torch.tensor(0.0,device=self.this_device, requires_grad=True)
                losses.append(tag_loss)
            # evt_loss
            if evt_score is not None:
                evt_loss = F.cross_entropy(evt_score, evt_label)  # the event level loss
                if mode == 0:self.train_evt_acc(evt_score, evt_label)
                elif mode == 1:self.val_evt_acc(evt_score, evt_label)
                losses.append(evt_loss)
            else:
                evt_loss=torch.tensor(0.0,device=tag_loss.device, requires_grad=True)
                losses.append(evt_loss)
            # gen_loss
            # mass count in gen_loss
            if gen_p4 is not None:
                if self.mloss:
                    p4_mass = self.compute_invariant_mass(p4).to(p4.device)
                    gen_mass = self.compute_invariant_mass(gen_p4).to(gen_p4.device)
                    gen_p4 = torch.cat((gen_p4, gen_mass.unsqueeze(1)), dim=1)
                    p4 = torch.cat((p4, p4_mass.unsqueeze(1)), dim=1)
                # etac_event_mask = cls_logits[:, 1] > cls_logits[:, 0]
                etac_event_mask = evt_label == 1
                gen_p4 = gen_p4[etac_event_mask]
                p4 = p4[etac_event_mask]
                gen_loss = self.MSE(gen_p4, p4)
                losses.append(gen_loss)
            else:
                # 当 gen_layers=0 时，gen_p4 为 None，需要创建一个零值的 tensor 而不是 Python float
                # 以确保梯度计算正常进行
                gen_loss = torch.tensor(0.0,device=tag_loss.device, requires_grad=True)
                if not self.UW:self.f_gen = 0
            R_loss = torch.tensor(0.0,device=tag_loss.device, requires_grad=True)
            scale_value = target_m.clone()    # target_m now is a tensor
            scale_value[scale_value == 0] = self.target_m_mean    # replace the 0 values of backgrounds to pdg value
            if self.R_loss:
                output_m = p5[:,0,4]
                R_loss = (abs(output_m-target_m)/scale_value).mean()
                losses.append(R_loss)
            losses=torch.stack(losses)
            return losses, tag_loss, evt_loss, gen_loss, R_loss
        else:
            if evt_score is not None:
                evt_score = torch.softmax(evt_score, dim=-1)
            if tagging_mask is not None:
                tagging_mask = torch.softmax(tagging_mask, dim=-1)
            return evt_score, tagging_mask, gen_p4, p5


    def training_step(self, batch, batch_idx):
        # x:(mask, features, vectors)
        x , _, _, _, _, _ = batch
        output=self(*self.process_model_input(x))

        losses, tag_loss, evt_loss, gen_loss, R_loss = self.process_output(batch,output,mode=0)
        if self.UW:
            log_vars = torch.stack(self.log_vars)
            weight = torch.exp(-log_vars)  # 1/sigma^2
            weighted_loss = (weight * losses).sum()
            reg = log_vars.sum()
            loss = 0.5*weighted_loss + 0.5*reg  # Li/2sigma^2 + log(sigma)
            self.log('tag_weight', weight[0], on_step=False, on_epoch=True)
            self.log('evt_weight', weight[1], on_step=False, on_epoch=True)
            if gen_loss!=0:
                self.log('gen_weight', weight[2], on_step=False, on_epoch=True)
                if R_loss!=0:
                    self.log('R_weight', weight[3], on_step=False, on_epoch=True)
            elif R_loss!=0:
                self.log('R_weight', weight[2], on_step=False, on_epoch=True)
        else:
            loss = ((self.f_tag*tag_loss + self.f_evt*evt_loss + self.f_gen*gen_loss) /
                    (self.f_tag+self.f_gen+self.f_evt)) + self.f_R * R_loss
        self.log('train_loss', loss)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('train_Tag_loss', tag_loss, on_step=False, on_epoch=True)
        self.log('train_Evt_loss', evt_loss, on_step=False, on_epoch=True)
        if gen_loss.item() != 0.0:
            self.log('train_Gen_loss', gen_loss, on_step=False, on_epoch=True)
        if self.R_loss:
            self.log('train_R_loss', R_loss, on_step=False, on_epoch=True)
        self.log('train_TagAcc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train_EvtAcc', self.train_evt_acc, on_step=False, on_epoch=True)

        if self.LT:
            threshold = self.mod.threshold
            self.log('threshold', threshold, on_step=False, on_epoch=True)

        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x:(mask, features, vectors)
        x, _, _, _, _, _ = batch
        output = self(*self.process_model_input(x))

        losses, tag_loss, evt_loss, gen_loss, R_loss = self.process_output(batch, output, mode=1)
        if self.UW:
            log_vars = torch.stack(self.log_vars)
            weight = torch.exp(-log_vars)  # 1/sigma^2
            weighted_loss = (weight * losses).sum()
            reg = log_vars.sum()
            loss = 0.5 * weighted_loss + 0.5 * reg  # Li/2sigma^2 + log(sigma)
        else:
            loss = ((self.f_tag * tag_loss + self.f_evt * evt_loss + self.f_gen * gen_loss) /
                    (self.f_tag + self.f_gen + self.f_evt)) + self.f_R * R_loss
        self.log('val_loss', loss)
        self.log('val_loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('val_Tag_loss', tag_loss, on_step=False, on_epoch=True)
        self.log('val_Evt_loss', evt_loss, on_step=False, on_epoch=True)
        if gen_loss.item() != 0.0:
            self.log('val_Gen_loss', gen_loss, on_step=False, on_epoch=True)
        if self.R_loss:
            self.log('val_R_loss', R_loss, on_step=False, on_epoch=True)
        self.log('val_TagAcc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_EvtAcc', self.val_evt_acc, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch  # x: a tuple of (points, features, vectors)
        output = self(*self.process_model_input(x))
        evt_score, tagging_mask, gen_p4, mass=self.process_output(batch, output, mode=2)
        return evt_score, tagging_mask, gen_p4, mass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.stable_lr)
        max_epochs = self.trainer.max_epochs

        def lr_lambda(epoch):  # PyTorch的LambdaLR传入step而非epoch
            warmup_ratio = 0.20  # 20%的周期用于预热
            decay_ratio = 0.10  # 10%的周期用于衰减

            if epoch < warmup_ratio * max_epochs:
                # 线性预热：从0到初始学习率
                return min((epoch + 0.1) / (warmup_ratio * max_epochs), 1.0)  # 避免除零
            elif epoch < (1 - decay_ratio) * max_epochs:
                # 保持阶段：100%初始学习率
                return 1.0
            else:
                # 余弦衰减阶段：平滑过渡到最低学习率
                decay_steps = epoch - (1 - decay_ratio) * max_epochs
                decay_max = decay_ratio * max_epochs
                return 0.5 * (1 + math.cos(math.pi * decay_steps / decay_max))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]

    def compute_invariant_mass(self,p4):
        px, py, pz,e = p4[:, 0], p4[:, 1], p4[:, 2], p4[:, 3]
        mass_squared = e ** 2 - (px ** 2 + py ** 2 + pz ** 2)
        # 使用 clamp 来确保不会出现负数从而导致 sqrt 计算复数
        mass_squared = torch.clamp(mass_squared, min=0)
        mass = torch.sqrt(mass_squared)
        return mass
