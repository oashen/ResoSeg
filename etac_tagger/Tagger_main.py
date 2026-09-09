from .sam_tagger import __version__ as model_version
from .default_setting import base_setting
from .utils import wlog, NaNDataDetector, m_calc, deep_update
from .Dataset import TaggerDataset,max_token
from .sam_tagger import SamTaggerLightning
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import uproot
import traceback
import os
import re
import copy


__version__ = '3.3.6'
tagger_version=__version__
DEFAULT_SEED = 1110111


class EtacTagger():
    def __init__(self,config:dict=None,testrun=False):
        self.config=copy.deepcopy(base_setting)
        self.update_config=config
        if config is not None:self.config=deep_update(self.config,config)

        self.config['lightning_model']['SamTagger']['condition_input']=len(self.config['dataset']['conditions'])
        self.sigcut = self.config['sigcut']
        self.gamcut = self.config['gamcut']
        self.pic_path = os.path.normpath(self.config['train_pic_path'])+'/'
        self.root_path = os.path.normpath(self.config['train_root_path'])+'/'

        self.etac_p4_row = self.config['dataset']['etac_p4']
        self.p_col = self.config['dataset']['interactions']
        self.p_col_raw = [i + '_raw' for i in self.p_col][:4]
        self.ccolumn = self.config['dataset']['charged_feat']
        self.bcolumn = self.config['dataset']['both_feat']
        self.column_notrs = self.config['dataset']['no_stander_feat']
        self.label = self.config['dataset']['tag_label']
        self.evt_label = self.config['dataset']['evt_label']
        self.n_tot_col = self.config['dataset']['n_tot']
        self.n_charge_col = self.config['dataset']['n_charge']

        self.config['dataset']['path'] = os.path.normpath(self.config['dataset']['path'])
        self.datasetpath = self.config['dataset']['path']
        self.dataset = TaggerDataset(self.config['dataset'])
        self.pipihc=self.config['dataset']['pipihc']

        if self.config['lightning_model']['SamTagger']['gen_layers'] > 0:
            self.gen_model=True
        else:
            self.gen_model=False
        if self.config['lightning_model'].get('R_loss',False):
            self.R_loss=True
        else:
            self.R_loss=False
        self.CDN = self.config['lightning_model']['SamTagger']['CDN']

        self.testrun=testrun
        print(f'EtacTagger version: {tagger_version}\nSamTagger version: {model_version}')

    # return logname, trainset_predict root, testset root and summary of the model
    def training(self,train_file=None,val_file=None,test_file=None,predict=True,train_ckpt:str=None,versionseed:bool=True):
        """
        auto predict the trainset and testset with the best val_loss ckpt
        :param train_file: if is None, default is {datasetpath}/train_df.root
        :param val_file: if is None, default is {datasetpath}/val_df.root
        :param test_file: if is None, default is {datasetpath}/test_df.root
        :param train_ckpt: if is not None, will use this ckpt to train
        :param predict: if True, will predict the trainset and testset with the best val_loss ckpt
        :param versionseed: if True, will use the version number to set the seed, else will use the default seed
        :return: logname, train_root:list, test_root:list, summary, ckpt
        """
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']

        logname = self.config.get('name','')
        ckptpath = 'ckpt'
        if logname == '' and (self.update_config is not None):
            logname += json.dumps(self.update_config).replace('dataset','').replace('lightning_model','').replace('SamTagger','').replace('{','').replace('}','').replace('\"', '').replace(' ', '').replace(':', '').replace(',','_').replace('true', 'Y').replace('false', 'N').replace('[', '').replace(']', '')

        ckptpath = ckptpath + f'/{logname}/'
        existing_versions = []
        if os.path.exists(ckptpath):
            for item in os.listdir(ckptpath):
                # 匹配 v数字 格式的版本号
                match = re.match(r'ver(\d+)', item)
                if match and os.path.isdir(os.path.join(ckptpath, item)):
                    existing_versions.append(int(match.group(1)))
        # 确定新版本号
        if existing_versions:
            new_version = max(existing_versions) + 1
        else:
            new_version = 0
        if versionseed:
            seed = DEFAULT_SEED + new_version
        else:
            seed = DEFAULT_SEED
        pl.seed_everything(seed, workers=True)
        ckptpath += f'ver{new_version}/'

        logfile = logname+f'_ver{new_version}.log'
        wlog(f'CONFIG:\n{json.dumps(self.config,indent=4)}', logfile)
        wlog(f'Effective seed: {seed}', logfile)

        if train_file is not None:
            wlog(f'train dataset: {train_file}', logfile)
        if val_file is not None:
            wlog(f'validation dataset: {val_file}', logfile)
        if test_file is not None:
            wlog(f'test dataset: {test_file}', logfile)

        if self.testrun:
            start=0
            end=100
        else:
            start=None
            end=None
        wlog('Loading train dataframe', logfile)
        if train_file is not None:
            T_name = train_file
        else:
            T_name = f"{self.datasetpath}/train_df.root"
        wlog(f'Loading {T_name}', logfile)
        _, T_b, T_num, train_dataset = self.dataset.get_dataset(T_name, start=start, end=end,CDN=self.CDN)

        std = {}
        temp_data=T_num[T_num[self.evt_label]==1]['truth_target_px']
        std['px'] = (temp_data.mean(),temp_data.std())
        temp_data = T_num[T_num[self.evt_label] == 1]['truth_target_py']
        std['py'] = (temp_data.mean(),temp_data.std())
        temp_data = T_num[T_num[self.evt_label] == 1]['truth_target_pz']
        std['pz'] = (temp_data.mean(),temp_data.std())
        temp_data = T_num[T_num[self.evt_label] == 1]['truth_target_e']
        std['energy'] = (temp_data.mean(),temp_data.std())

        wlog('Loading val dataframe', logfile)
        if val_file is not None:
            V_name=val_file
        else:
            V_name = f"{self.datasetpath}/val_df.root"
        wlog(f'Loading {V_name}', logfile)
        _, _, _, val_dataset = self.dataset.get_dataset(V_name, start=start, end=end,CDN=self.CDN)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        input_dims = len(self.ccolumn + self.bcolumn)
        if train_ckpt is not None:
            model_part_pl = SamTaggerLightning.load_from_checkpoint(checkpoint_path=train_ckpt)
        else:
            model_part_pl = SamTaggerLightning(self.config['lightning_model'],input_dims,std=std)
            wlog(model_part_pl,logfile)

        from pytorch_lightning.callbacks import ModelCheckpoint
        val_checkpoint_callback = ModelCheckpoint(
            dirpath=ckptpath+'val',  # 保存路径
            filename='{epoch:02d}-{train_loss_epoch:.2f}-{val_loss_epoch:.2f}',  # 文件名格式
            save_top_k=3,  # 保存最佳3个模型
            monitor='val_loss',  # 监控的指标
            mode='min'  # 越小越好
        )

        train_checkpoint_callback = ModelCheckpoint(
            dirpath=ckptpath+'train',  # 保存路径
            filename='{epoch:02d}-{train_loss_epoch:.2f}-{val_loss_epoch:.2f}',  # 文件名格式
            save_top_k=3,  # 保存最佳3个模型
            monitor='train_loss',  # 监控的指标
            mode='min'  # 越小越好
        )

        trainer = pl.Trainer(
            max_epochs=epochs if not self.testrun else 20,
            logger=pl.loggers.TensorBoardLogger('tb_logs', name=logname),
            accelerator='gpu' if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[val_checkpoint_callback,train_checkpoint_callback,NaNDataDetector()]
        )


        wlog('////////////////////////////////////Training////////////////////////////////////////', logfile)
        wlog(f'Log saved as : {logname}', logfile)
        os.makedirs(f'tb_logs/{logname}', exist_ok=True)
        # Fit the model
        trainer.fit(model_part_pl, train_loader, val_loader)
        ckpt=val_checkpoint_callback.best_model_path
        wlog('///////////////////////////////////Train done///////////////////////////////////////', logfile)
        from pytorch_lightning.utilities.model_summary import ModelSummary
        summary = ModelSummary(model_part_pl)
        if predict:
            wlog(f'///////////////////////////////////Predict with {ckpt}///////////////////////////////////',logfile)
            # set dataset to predict mode
            train_dataset.setmode()
            os.makedirs(self.root_path+logname+f'/ver{new_version}', exist_ok=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size,  num_workers=0)
            train_root=self.predict(ckpt,T_name,self.root_path+logname+f'/ver{new_version}'+'/train.root',
                                    from_train=(T_b,T_num,train_loader))
            wlog('Train set done',logfile)

            if test_file is not None:
                Te_name=test_file
            else:
                Te_name = f"{self.datasetpath}/test_df.root"
            wlog(f'Loading {Te_name}', logfile)
            test_root = self.predict(ckpt, Te_name, self.root_path+logname+f'/ver{new_version}' + '/test.root')
            wlog('Test set done', logfile)
            return logname,train_root,test_root,summary,ckpt
        else: return logname,None,None,summary,ckpt

    def training_iter(self, train_ckpt:str=None,start_from:float=0.1,step:float=0.1,end=1,predict=True,mail:bool=True,versionseed:bool=True):
        """
        fine-tuning the ckpt with multi dataset, auto predict the trainset and testset with the best val_loss ckpt
        :return: logname, train_root:list, test_root:list, summary, ckpt
        """
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        iter_num = np.round(np.arange(start_from,end,step)*100).astype(int)
        for train_step in iter_num:
            logname = self.config.get('name','')
            ckptpath = 'ckpt'
            if logname == '' and (self.update_config is not None):
                logname += json.dumps(self.update_config).replace('dataset','').replace('lightning_model','').replace('SamTagger','').replace('{','').replace('}','').replace('\"', '').replace(' ', '').replace(':', '').replace(',','_').replace('true', 'Y').replace('false', 'N').replace('[', '').replace(']', '')
            logname += f'_{int(train_step)}set'
            ckptpath = ckptpath + f'/{logname}/'
            existing_versions = []
            if os.path.exists(ckptpath):
                for item in os.listdir(ckptpath):
                    # 匹配 v数字 格式的版本号
                    match = re.match(r'ver(\d+)', item)
                    if match and os.path.isdir(os.path.join(ckptpath, item)):
                        existing_versions.append(int(match.group(1)))
            # 确定新版本号
            if existing_versions:
                new_version = max(existing_versions) + 1
            else:
                new_version = 0
            if versionseed:
                seed = DEFAULT_SEED + new_version
            else:
                seed = DEFAULT_SEED
            pl.seed_everything(seed, workers=True)
            ckptpath += f'ver{new_version}/'

            logfile = logname+f'_ver{new_version}.log'
            wlog(f'CONFIG:\n{json.dumps(self.config,indent=4)}', logfile)
            wlog(f'Effective seed: {seed}', logfile)

            stdfile = self.datasetpath + f'/std_{int(train_step)}.json'

            if self.testrun:
                start=0
                end=100
            else:
                start=None
                end=None
            wlog('Loading train dataframe', logfile)
            T_name = f"{self.datasetpath}/train_df_{int(train_step)}.root"
            wlog(f'Loading {T_name}', logfile)
            _, T_b, T_num, train_dataset = self.dataset.get_dataset(T_name, start=start, end=end,CDN=self.CDN,stdfile=stdfile)

            std = {}
            temp_data=T_num[T_num[self.evt_label]==1]['truth_target_px']
            std['px'] = (temp_data.mean(),temp_data.std())
            temp_data = T_num[T_num[self.evt_label] == 1]['truth_target_py']
            std['py'] = (temp_data.mean(),temp_data.std())
            temp_data = T_num[T_num[self.evt_label] == 1]['truth_target_pz']
            std['pz'] = (temp_data.mean(),temp_data.std())
            temp_data = T_num[T_num[self.evt_label] == 1]['truth_target_e']
            std['energy'] = (temp_data.mean(),temp_data.std())

            wlog('Loading val dataframe', logfile)
            V_name = f"{self.datasetpath}/val_df_{int(train_step)}.root"
            wlog(f'Loading {V_name}', logfile)
            _, _, _, val_dataset = self.dataset.get_dataset(V_name, start=start, end=end,CDN=self.CDN)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            input_dims = len(self.ccolumn + self.bcolumn)
            if train_ckpt is not None:
                model_part_pl = SamTaggerLightning.load_from_checkpoint(checkpoint_path=train_ckpt)
            else:
                model_part_pl = SamTaggerLightning(self.config['lightning_model'], input_dims, std=std)
                wlog(model_part_pl, logfile)

            from pytorch_lightning.callbacks import ModelCheckpoint
            val_checkpoint_callback = ModelCheckpoint(
                dirpath=ckptpath+'val',  # 保存路径
                filename='{epoch:02d}-{train_loss_epoch:.2f}-{val_loss_epoch:.2f}',  # 文件名格式
                save_top_k=3,  # 保存最佳3个模型
                monitor='val_loss',  # 监控的指标
                mode='min'  # 越小越好
            )

            train_checkpoint_callback = ModelCheckpoint(
                dirpath=ckptpath+'train',  # 保存路径
                filename='{epoch:02d}-{train_loss_epoch:.2f}-{val_loss_epoch:.2f}',  # 文件名格式
                save_top_k=3,  # 保存最佳3个模型
                monitor='train_loss',  # 监控的指标
                mode='min'  # 越小越好
            )

            trainer = pl.Trainer(
                max_epochs=epochs if not self.testrun else 20,
                logger=pl.loggers.TensorBoardLogger('tb_logs', name=logname),
                accelerator='gpu' if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=[val_checkpoint_callback,train_checkpoint_callback,NaNDataDetector()]
            )


            wlog('////////////////////////////////////Training////////////////////////////////////////', logfile)
            wlog(f'Log saved as : {logname}', logfile)
            os.makedirs(f'tb_logs/{logname}', exist_ok=True)
            # Fit the model
            trainer.fit(model_part_pl, train_loader, val_loader)
            ckpt=val_checkpoint_callback.best_model_path
            wlog('///////////////////////////////////Train done///////////////////////////////////////', logfile)
            from pytorch_lightning.utilities.model_summary import ModelSummary
            summary = ModelSummary(model_part_pl)
            if predict:
                wlog(f'///////////////////////////////////Predict with {ckpt}///////////////////////////////////',logfile)
                # set dataset to predict mode
                train_dataset.setmode()
                os.makedirs(self.root_path+logname+f'/ver{new_version}', exist_ok=True)
                train_loader = DataLoader(train_dataset, batch_size=batch_size,  num_workers=0)
                train_root=self.predict(ckpt,T_name,self.root_path+logname+f'/ver{new_version}'+'/train.root',
                                        from_train=(T_b,T_num,train_loader))
                wlog('Train set done',logfile)

                Te_name = f"{self.datasetpath}/test_df_{int(train_step)}.root"
                wlog(f'Loading {Te_name}', logfile)
                test_root = self.predict(ckpt, Te_name, self.root_path+logname+f'/ver{new_version}' + '/test.root')
                wlog('Test set done', logfile)

                if mail:
                    textopt = json.dumps(self.update_config, indent=4)
                    text = f'EtacTagger version: {tagger_version}\n Model version: {model_version}\n' + f'{textopt} done'
                    wlog(f'Train Root:{train_root}', logfile)
                    wlog(f'Test Root:{test_root}', logfile)
                    wlog(f'Model size:{summary}', logfile)
                    text += str(summary)
                    text += f'BEST CKPT:\n{ckpt}\n'
                    text += f'root and pic ver:{model_version}\n'

                    picpath = self.pic_path + logname.rstrip('/') + f'/{new_version}/'
                    os.makedirs(picpath, exist_ok=True)
                    pics=[]
                    try:
                        pics += self.draw_log(logname, picpath)
                        wlog('Draw loss done', logfile)
                    except Exception:
                        e_out = str(traceback.format_exc())
                        wlog(f'Tensorboard Draw log Error: {e_out}', logfile)
                        text += f'\n Tensorboard drawlog error:{e_out}\n Use json to draw the loss'
                    try:
                        roc, f1text = self.drawROC(test_root, train_root, picpath)
                        wlog('Draw ROC done', logfile)
                        pics += roc
                        text += f1text
                    except Exception:
                        e_out = str(traceback.format_exc())
                        wlog(f'EtacTagger Draw pic Error: {e_out}', logfile)
                        text += f'\n drawpic error:{e_out}'
                    wlog(text, logfile)

    # if dataloader is not none, use dataloader, else make the dataset and dataloader
    # if training, do not need the standard and the output will be little different
    # return the resfile root path
    def __predict_single_move(self,ckpt,file,output,start=None,end=None,from_train=None, cut_idx=2, logfile='predict.log'):
        batch_size = self.config['batch_size']
        model = SamTaggerLightning.load_from_checkpoint(ckpt)
        model.eval()
        trainer = pl.Trainer(accelerator='gpu', devices=1)
        if from_train is not None:
            Te_b, Te_num, test_loader = from_train
        else:
            _, Te_b, Te_num, test_dataset = self.dataset.get_dataset(file, start, end, mode=0,CDN=self.CDN)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
        model_threshold=model.mod.threshold
        resfile=output

        output= trainer.predict(model, dataloaders=test_loader)
        evt_score = torch.cat([i[0] for i in output])      # (B, 2)
        if output[0][1] is not None:
            tagging_mask = [i[1] for i in output]      # N* (B, num_mask, sqe_len, num_class) could be None
        else:
            tagging_mask = None
        if output[0][2] is not None:
            gen_p4 = torch.cat([i[2] for i in output])         # (B, 4) could be None
        else:
            gen_p4 = None
        if output[0][3] is not None:
            p5 = torch.cat([i[3] for i in output])   # (B, num_mask, 5) could be None
        else:
            p5 = None
        device = evt_score.device
        padded_tensors = []
        wlog('///////////////////////////////////Predict done///////////////////////////////////////',logfile)

        # 预分配容器
        container = {
            'px': [], 'py': [], 'pz': [], 'e': [],
            'sigevent_score': evt_score[:, 1].tolist(), 'bkgevent_score': evt_score[:, 0].tolist()
        }
        def df_to_tensor(df, cols, dtype=torch.float32):
            return torch.stack([torch.tensor(df[col].values, dtype=dtype) for col in cols], dim=1).to(device)

        # 四动量转换到GPU（假设设备已定义）
        p4s = df_to_tensor(Te_b, self.p_col_raw)  # (total_tracks, 4)
        iGood = torch.tensor(Te_b['iGood'].values, dtype=torch.int,requires_grad=False).to(device)
        Ecms = torch.tensor(Te_num['ecms'].values, dtype=torch.float32,requires_grad=False).to(device)
        n_tot_tensor = torch.tensor(Te_num[self.n_tot_col].values, dtype=torch.int,requires_grad=False).to(device)

        # 事件索引计算优化
        cum_n_tot = torch.cumsum(n_tot_tensor, dim=0)
        event_starts = torch.cat([torch.tensor([0], device=device), cum_n_tot[:-1]])  # 根据n_tot计算每个event的初始位置
        n_tot_tensor[n_tot_tensor > max_token] = max_token  # 根据模型中的max_token,对n_tot进行截断,每个event只取前max_token

        # 修改后的训练标签处理
        if self.label in Te_b.columns:
            label_tensor = torch.tensor(Te_b[self.label].values, dtype=torch.int).to(device)
            # 使用预转换的tensor加速
            container['label'] = [
                label_tensor[s:s + n].cpu().tolist()
                for s, n in zip(event_starts.cpu().numpy(), n_tot_tensor.cpu().numpy())
            ]
        else:
            label_tensor = None
        evt_num = len(Te_num[self.n_tot_col])  # num of event
        for i in range(evt_num):
            n_tot = n_tot_tensor[i]
            # 四动量切片
            p4_event = p4s[event_starts[i]:event_starts[i] + n_tot]
            container['px'].append(p4_event[:, 0].tolist())
            container['py'].append(p4_event[:, 1].tolist())
            container['pz'].append(p4_event[:, 2].tolist())
            container['e'].append(p4_event[:, 3].tolist())

        if tagging_mask is not None:
            max_tokenlen = max([i.size(2) for i in tagging_mask])
            # merge track score
            # logits(tokens,batch,label)
            for i in tagging_mask:
                a = i.size(2)
                pad_size = max_tokenlen - a
                # 在seq_len维度上补齐
                padded = torch.cat([i, torch.zeros(i.size(0), i.size(1), pad_size, i.size(3),  device=i.device)], dim=2)
                padded_tensors.append(padded)
            tagging_mask = torch.cat(padded_tensors, dim=0)    # (B, num_mask, sqe_len, num_class)
            num_mask = tagging_mask.size(1)
            for i in range(num_mask):
                container[f'sig_score_{i}'] = []
                container[f'bkg_score_{i}'] = []
                container[f'gam_score_{i}'] = []
            if gen_p4 is not None:
                for i in range(len(self.etac_p4_row)):
                    container['gen_'+self.etac_p4_row[i]] = gen_p4[:,i].tolist()
                temp_col = (gen_p4[:, 3] ** 2-gen_p4[:, 0] ** 2 - gen_p4[:, 1] ** 2 - gen_p4[:, 2] ** 2) ** 0.5
                container['gen_' + self.etac_p4_row[0].replace('_px','_m')] = temp_col.tolist()

            sigcut = [-1] + self.sigcut
            gamcut = [-1] + self.gamcut
            if label_tensor is not None:
                container['sig_iou'] = []
                container['gam_iou'] = []
                container['PMR'] = []
            # [0] is the threshold output
            for level in range(len(sigcut)):
                container[f'res_m_{level}'] = []
                container[f'res_psipm_{level}'] = []
                container[f'res_px_{level}'] = []
                container[f'res_py_{level}'] = []
                container[f'res_pz_{level}'] = []
                container[f'res_e_{level}'] = []
                container[f'res_hc_px_{level}'] = []
                container[f'res_hc_py_{level}'] = []
                container[f'res_hc_pz_{level}'] = []
                container[f'res_hc_e_{level}'] = []
                container[f'res_hc_m_{level}'] = []

            for i in range(evt_num):
                n_tot = n_tot_tensor[i]
                Ecms_event = Ecms[i]
                scores = tagging_mask[i,:, :n_tot,:]  # 当前事件的scores,(num_mask,tracks,scores)
                # 四动量切片
                p4_event = p4s[event_starts[i]:event_starts[i] + n_tot]
                iGood_event = iGood[event_starts[i]:event_starts[i] + n_tot]

                if label_tensor is not None:
                    label_event=label_tensor[event_starts[i]:event_starts[i] + n_tot]
                    iou_list=[[],[],[]]
                    for score_idx in [1, 2]:
                        gt_foreground = label_event == score_idx  # (seq_len, B)
                        if score_idx==1:
                            model_threshold=self.sigcut[cut_idx]
                        else:
                            model_threshold=self.gamcut[cut_idx]
                        for mask_i in range(num_mask):
                            mask_event=scores[mask_i]
                            pred_foreground = mask_event[:, score_idx] > model_threshold

                            intersection = (gt_foreground & pred_foreground).sum(dim=0).float()  # (B,)
                            union = (gt_foreground | pred_foreground).sum(dim=0).float()  # (B,)
                            iou_value = torch.where(union == 0,torch.ones_like(union),intersection / union)
                            iou_list[score_idx].append(iou_value.item())
                    for mask_i in range(num_mask):
                        iou_list[0].append(int(iou_list[1][mask_i] == 1 and iou_list[2][mask_i] == 1))
                    container['sig_iou'].append(iou_list[1])
                    container['gam_iou'].append(iou_list[2])
                    container['PMR'].append(iou_list[0])

                # 按三个阈值级别处理
                for level in range(0,len(sigcut)):
                    res_m=[]
                    res_px=[]
                    res_py=[]
                    res_pz=[]
                    res_e=[]
                    res_hc_m = []
                    res_hc_px = []
                    res_hc_py = []
                    res_hc_pz = []
                    res_hc_e = []
                    res_psip_m=[]
                    for mask_i in range(num_mask):
                        ntrk_mask = iGood_event < 0
                        if level != 0:
                            sig_mask = scores[mask_i, :, 1] > sigcut[level]  # a list of bool,len=n_tot
                            etac_p4 = p4_event[sig_mask].sum(dim=0)
                            m_val = m_calc(etac_p4) if etac_p4.shape[0] > 0 else 0.0
                            if not self.pipihc:
                                # select 3 gamma according to their energy
                                gam0_mask = (scores[mask_i, :, 2] > gamcut[level]) & (p4_event[:, 3] < 0.07) & (~sig_mask) & (ntrk_mask)
                                gam1_mask = (scores[mask_i, :, 2] > gamcut[level]) & (p4_event[:, 3] < 0.16) & (p4_event[:, 3] > 0.08) & (~sig_mask) & (ntrk_mask)
                                gam2_mask = (scores[mask_i, :, 2] > gamcut[level]) & (p4_event[:, 3] > 0.2) & (~sig_mask) & (ntrk_mask)
                                # tensor of 3 gammas' p4, [num,4]
                                gam0_tensor = p4_event[gam0_mask]
                                gam1_tensor = p4_event[gam1_mask]
                                gam2_tensor = p4_event[gam2_mask]
                            else:
                                label2_mask = (scores[mask_i, :, 2] > gamcut[level]) & (~sig_mask)
                                pi_mask = label2_mask & (~ntrk_mask)
                                gam_mask = label2_mask & (ntrk_mask)
                                pi_tensor = p4_event[pi_mask]
                                gam_tensor = p4_event[gam_mask]

                        # level==0 is the output using a relative cut: mask[:,:,1/2] is the biggest score
                        else:
                            max_indices = torch.argmax(scores[mask_i], dim=-1)  # (tracks,scores)
                            label2_mask = (max_indices == 2)
                            if not self.pipihc:
                                gam0_mask = label2_mask & (p4_event[:, 3] < 0.07) & (ntrk_mask)
                                gam1_mask = label2_mask & (p4_event[:, 3] < 0.16) & (p4_event[:, 3] > 0.08) & (ntrk_mask)
                                gam2_mask = label2_mask & (p4_event[:, 3] > 0.2) & (ntrk_mask)
                                # tensor of 3 gammas' p4, [num,4]
                                gam0_tensor = p4_event[gam0_mask]
                                gam1_tensor = p4_event[gam1_mask]
                                gam2_tensor = p4_event[gam2_mask]
                            else:
                                pi_mask = label2_mask & (~ntrk_mask)
                                gam_mask = label2_mask & (ntrk_mask)
                                pi_tensor = p4_event[pi_mask]
                                gam_tensor = p4_event[gam_mask]

                            this_p5=p5[i,mask_i]
                            etac_p4=this_p5[:4]
                            m_val=this_p5[4]

                        # at least one condidate for each gamma
                        # for pipihc, at least one gamma and two pi
                        if (not self.pipihc and (gam0_tensor.size(0) < 1 or gam1_tensor.size(0) < 1 or gam2_tensor.size(0) < 1))\
                                or (self.pipihc and (gam_tensor.size(0) < 1 or pi_tensor.size(0) < 2)):
                            res_px.append(etac_p4[0].detach().cpu().item())
                            res_py.append(etac_p4[1].detach().cpu().item())
                            res_pz.append(etac_p4[2].detach().cpu().item())
                            res_e.append(etac_p4[3].detach().cpu().item())
                            res_m.append(m_val.detach().cpu().item() if isinstance(m_val, torch.Tensor) else float(m_val))
                            res_psip_m.append(0)
                            res_hc_m.append(0)
                            res_hc_px.append(0)
                            res_hc_py.append(0)
                            res_hc_pz.append(0)
                            res_hc_e.append(0)
                            continue
                        # 质量计算
                        # loop photons to select the best psip
                        if (not self.pipihc):
                            temp_diff=99999.9
                            psip_val=0
                            hc_val = etac_p4.clone()
                            for gam0_p4 in gam0_tensor:
                                for gam1_p4 in gam1_tensor:
                                    for gam2_p4 in gam2_tensor:
                                        hc_p4 = etac_p4.clone()
                                        cms_p4 = etac_p4.clone()
                                        cms_p4 += gam0_p4+gam1_p4+gam2_p4
                                        this_val=m_calc(cms_p4)
                                        diff=abs(this_val-Ecms_event)
                                        if diff<temp_diff:
                                            temp_diff=diff
                                            psip_val=this_val
                                            hc_val = hc_p4 + gam2_p4
                        else:
                            temp_diff = 99999.9
                            psip_val = 0
                            hc_val = etac_p4.clone()
                            for gam_p4 in gam_tensor:
                                for pi_i in range(0,pi_tensor.size(0)):
                                    for pi_j in range(pi_i+1,pi_tensor.size(0)):
                                        pi1 = pi_tensor[pi_i]
                                        pi2 = pi_tensor[pi_j]
                                        hc_p4 = etac_p4.clone()
                                        cms_p4 = etac_p4.clone()
                                        cms_p4 += pi1+pi2+gam_p4
                                        this_val = m_calc(cms_p4)
                                        diff = abs(this_val - Ecms_event)
                                        if diff < temp_diff:
                                            temp_diff = diff
                                            psip_val = this_val
                                            hc_val = hc_p4 + gam_p4
                        hc_m = m_calc(hc_val)
                        res_px.append(etac_p4[0].detach().cpu().item())
                        res_py.append(etac_p4[1].detach().cpu().item())
                        res_pz.append(etac_p4[2].detach().cpu().item())
                        res_e.append(etac_p4[3].detach().cpu().item())
                        res_hc_px.append(hc_val[0].detach().cpu().item())
                        res_hc_py.append(hc_val[1].detach().cpu().item())
                        res_hc_pz.append(hc_val[2].detach().cpu().item())
                        res_hc_e.append(hc_val[3].detach().cpu().item())
                        res_hc_m.append(hc_m)
                        res_m.append(m_val.detach().cpu().item() if isinstance(m_val, torch.Tensor) else float(m_val))
                        res_psip_m.append(psip_val.detach().cpu().item() if isinstance(psip_val, torch.Tensor) else float(psip_val))
                    # 存储不同级别的结果
                    container[f'res_px_{level}'].append(res_px)
                    container[f'res_py_{level}'].append(res_py)
                    container[f'res_pz_{level}'].append(res_pz)
                    container[f'res_e_{level}'].append(res_e)
                    container[f'res_m_{level}'].append(res_m)
                    container[f'res_hc_px_{level}'].append(res_hc_px)
                    container[f'res_hc_py_{level}'].append(res_hc_py)
                    container[f'res_hc_pz_{level}'].append(res_hc_pz)
                    container[f'res_hc_e_{level}'].append(res_hc_e)
                    container[f'res_hc_m_{level}'].append(res_hc_m)
                    container[f'res_psipm_{level}'].append(res_psip_m)

                # 收集四动量和分数
                for mask_i in range(num_mask):
                    container[f'sig_score_{mask_i}'].append(scores[mask_i,:, 1].tolist())
                    container[f'bkg_score_{mask_i}'].append(scores[mask_i,:, 0].tolist())
                    container[f'gam_score_{mask_i}'].append(scores[mask_i,:, 2].tolist())


        wlog('DF to dict....',logfile)
        num_dict = {col: Te_num[col].values for col in Te_num.columns}
        container.update(num_dict)
        del Te_num,Te_b
        wlog('DF to dict.... done', logfile)

        wlog(f'writing {resfile}',logfile)
        with uproot.recreate(resfile) as f:
            f["all"] = container
        wlog(f'\n\n{resfile} saved\n\n',logfile)
        return resfile

    def predict(self, ckpt: str, file: str, output: str,
                idx: int = 0, size: int = 1500000,
                sigcut: list = None, gamcut: list = None, cut_idx:int = 2,
                from_train=None, logfile: str = 'predict.log') -> list[str]:
        """
        If the file is too big, will split it into several part
        :param ckpt: ckpt of model
        :param file: input file name
        :param output: output root name
        :param idx: the start index of the small part
        :param size: size of each part
        :param from_train: (df_b,df_num,dataloder) from training
        :param sigcut: score cuts, will output the etac P4 according to these cuts
        :param gamcut: score cuts, will output the psip P4 according to thest cuts
        :param logfile: logfile
        :return: list of output files
        """
        if sigcut is not None:self.sigcut=sigcut
        if gamcut is not None:self.gamcut=gamcut
        if from_train is not None:
            return [self.__predict_single_move(ckpt,file,output,from_train=from_train,cut_idx=cut_idx)]

        totlen=len(uproot.concatenate(f"{file}:num", library="pd", filter_name=['n_tot']))
        start=idx*size
        end=min((idx+1)*size,totlen)
        resfiles=[]
        while start<end:
            if idx == 0 and end == totlen:
                resfile=output
                wlog(f'start predict {file}', file=logfile)
            else:
                resfile=output.replace('.root',f'_{idx}.root')
                wlog(f'start predict {file}: {start} - {end}', file=logfile)
            resfiles.append(
                self.__predict_single_move(ckpt, file, resfile, start, end,
                                           logfile=logfile)
            )
            idx+=1
            start = idx * size
            end = min((idx + 1) * size, totlen)
        return resfiles