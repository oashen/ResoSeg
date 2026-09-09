from torch.utils.data import Dataset
import numpy as np
import torch
from .utils import wlog
import uproot
import pandas as pd
import random
import json

max_token=50
def _p4(px, py, pz, e):
    import vector
    vector.register_awkward()
    return vector.zip({'px': px, 'py': py, 'pz': pz, 'energy': e})


class TaggerDataset():
    def __init__(self,config:dict=None):
        self.config = config
        self.etac_p4_row = self.config['etac_p4']
        self.ccolumn = self.config['charged_feat']
        self.bcolumn = self.config['both_feat']
        self.column_notrs = self.config['no_stander_feat']
        self.label = self.config['tag_label']
        self.evt_label = self.config['evt_label']
        self.topobranches = self.config['topobranches']
        self.datasetpath = self.config['path']
        self.stdfile = self.config['norm_file']
        self.incmc_bkg=self.config['incmc_bkg']
        self.interactions_col = self.config['interactions']
        if self.stdfile is None:
            self.stdfile = self.datasetpath+'/std.json'
    
    def get_dataset(self,file:str,start: int = None, end: int = None, mode: int = 1, CDN=False,stdfile:str=None):
        """
        :param mode: 0 = predict, 1 = training
        :param CDN: if is conditional layer norm
        :param stdfile: the normalization parameter file, if is None, use the default file in config
        :return: df_c, df_b, df_num, dataset
        """
        if start is not None and end is not None:
            df_num = uproot.concatenate(f"{file}:num", library="pd", filter_name=['n_tot', 'n_charge'])
            totlen = len(df_num)
            end = min(end, totlen)
            n_c = (df_num[:start]['n_charge'].sum(), df_num[:end]['n_charge'].sum())
            n_b = (df_num[:start]['n_tot'].sum(), df_num[:end]['n_tot'].sum())
            with uproot.open(file) as f:
                # 读取 num 树，指定条目范围
                tree_num = f['num']
                df_num = tree_num.arrays(library='pd', entry_start=start, entry_stop=end)
                # 读取 c 树
                tree_c = f['c']
                df_c = tree_c.arrays(library='pd', entry_start=n_c[0], entry_stop=n_c[1])
                # 读取 b 树
                tree_b = f['b']
                df_b = tree_b.arrays(library='pd', entry_start=n_b[0], entry_stop=n_b[1])
                del tree_b, tree_c, tree_num
        else:
            df_num = uproot.concatenate(f"{file}:num", library="pd")
            df_c = uproot.concatenate(f"{file}:c", library="pd")
            df_b = uproot.concatenate(f"{file}:b", library="pd")
        for col in self.interactions_col[:5]:
            df_b[f'{col}_raw'] = df_b[col]
        df_c[f'{self.interactions_col[-1]}_raw']=df_c[self.interactions_col[-1]]
        df_c, df_b = self.norm(df_c, df_b,stdfile)

        dataset = DfToDataset(df_c, df_b, df_num, self.config, mode, CDN)
        return df_c, df_b, df_num, dataset

    # split num first,then split c and b according to n_tot
    def __dfsplit(self, df_num, df_c, df_b, testsize):
        p = int(len(df_num) * testsize)
        print(f'test len: {p}/{len(df_num)}')
        testdf_num = df_num[:p]
        traindf_num = df_num[p:]
        test_ntot = testdf_num['n_tot'].sum()
        test_ncharge = testdf_num['n_charge'].sum()
        testdf_c = df_c[:test_ncharge]
        traindf_c = df_c[test_ncharge:]
        testdf_b = df_b[:test_ntot]
        traindf_b = df_b[test_ntot:]
        return testdf_num, testdf_c, testdf_b, traindf_num, traindf_c, traindf_b

    # add random truth P4 for background
    def rand_bkgtruth(self, df: pd.DataFrame = None, rootfile: str = None):
        if rootfile is not None and df is None:
            df = uproot.concatenate(f"{rootfile}:num", library="pd")
            df_c = uproot.concatenate(f"{rootfile}:c", library="pd")
            df_b = uproot.concatenate(f"{rootfile}:b", library="pd")
        zero_e_mask = df[self.etac_p4_row[3]] == 0  # self.etac_p4_row[3] 应该是 'e' 列名
        zero_e_indices = df[zero_e_mask].index

        if len(zero_e_indices) > 0:
            # 生成新的 px, py, pz (-0.7 到 0.7 范围内的随机数)
            num_to_modify = len(zero_e_indices)
            new_px_py_pz = np.random.uniform(-0.7, 0.7, size=(num_to_modify, 3))

            # 生成随机不变质量 m (2 到 3.5 范围内)
            random_m = np.random.uniform(2.0, 3.5, size=num_to_modify)

            # 更新 DataFrame 中的 px, py, pz
            # 假设 truth_p4 是包含 ['px', 'py', 'pz', 'e'] 的列表
            df.loc[zero_e_indices, self.etac_p4_row[0]] = new_px_py_pz[:, 0]  # px
            df.loc[zero_e_indices, self.etac_p4_row[1]] = new_px_py_pz[:, 1]  # py
            df.loc[zero_e_indices, self.etac_p4_row[2]] = new_px_py_pz[:, 2]  # pz

            # 计算新的能量 e = sqrt(m^2 + px^2 + py^2 + pz^2)
            px = df.loc[zero_e_indices, self.etac_p4_row[0]].values
            py = df.loc[zero_e_indices, self.etac_p4_row[1]].values
            pz = df.loc[zero_e_indices, self.etac_p4_row[2]].values
            new_e = np.sqrt(random_m ** 2 + px ** 2 + py ** 2 + pz ** 2)

            # 更新 DataFrame 中的能量 e
            df.loc[zero_e_indices, self.etac_p4_row[3]] = new_e
        if rootfile is None:
            return df
        else:
            dict = {col: df[col].values for col in df.columns}
            dict_c = {col: df_c[col].values for col in df_c.columns}
            dict_b = {col: df_b[col].values for col in df_b.columns}
            with uproot.recreate(rootfile) as file:
                file["c"] = dict_c
                file["b"] = dict_b
                file["num"] = dict
            return None

    # pre-process for root files got from EtacTagger ana program,output files are for training
    def ana_to_train(self, trainset_ratio=0.4, filename: str = None):
        bcolumn = ['px', 'py', 'pz', 'energy',
                   'emc_x', 'emc_y', 'emc_z', 'emc_theta', 'emc_dtheta', 'emc_phi', 'emc_dphi', 'emc_energy',
                   'emc_time', 'emc_numHits', 'emc_e3x3',
                   'emc_e5x5', 'emc_dE', 'emc_a20moment', 'emc_a42moment',
                   'emc_secondmoment', 'emc_latmoment', 'iGood']+ [self.label]
        ccolumn = ['q','vz0', 'vr0', 'mdc_theta', 'mdc_phi',
                   'd0', 'phi0', 'kappa', 'z0', 'tanlambda',
                   'd0_err', 'phi0_err', 'kappa_err', 'z0_err', 'tanlambda_err',
                   'prob_pid','candy_pid',  'pid_prob_e', 'pid_prob_mu', 'pid_prob_pi', 'pid_prob_k', 'pid_prob_proton',
                   'muc_dpt', 'muc_numLayer', 'muc_maxHitsInlater','muc_chisq', 'muc_dof', 'muc_numHits']
        numcolumn = ['n_charge_track', 'n_charge', 'n_neutral', 'n_tot', 'kmchisq', 'n_res', 'res_pdg', 'res_dm',
                     'res_m', 'R',
                     'target_m', 'kmfit_target_m', 'recfit_target_m',
                     'cms_m', 'kmfit_cms_m', 'recfit_cms_m',
                     'Run', 'Event', 'ecms',
                     'truth_target_px', 'truth_target_py', 'truth_target_pz', 'truth_target_e', 'truth_target_m',
                     'recfit_target_px', 'recfit_target_py', 'recfit_target_pz', 'recfit_target_e',
                     'target_px', 'target_py', 'target_pz', 'target_e',
                     'recfit_cms_px', 'recfit_cms_py', 'recfit_cms_pz', 'recfit_cms_e',
                     'cms_px', 'cms_py', 'cms_pz', 'cms_e'
                     ] + self.topobranches + [self.evt_label]

        otherfile=False
        if filename is None:
            filename = [f'{self.datasetpath}/sig.root', f'{self.datasetpath}/hcbkg.root']
            if self.incmc_bkg: filename.append(f'{self.datasetpath}/incmc_bkg.root')
        else:
            filename=[filename]
            otherfile=True

        drop_cut = {
            'px': [-3.686, 3.686], 'py': [-3.686, 3.686], 'pz': [-3.686, 3.686], 'energy': [0, 3.686],
            'd0': [-10, 10], 'kappa': [-30, 30], 'z0': [-25, 25], 'tanlambda': [-4, 4]
        }
        ndrop_cut = {
            'emc_x': [-100, 100], 'emc_y': [-100, 100], 'emc_z': [-150, 150], 'emc_energy': [0, 3.686],
            'emc_a20moment': [0, 1],
            'emc_a42moment': [0, 1], 'emc_secondmoment': [0, 200], 'emc_latmoment': [0, 1]
        }
        setzero_cut = {
            'muc_dpt': [0, 200]
        }
        name_len = len(filename)
        dfdict_c = {'rand_idx': []}
        dfdict_b = {'rand_idx': []}
        dfdict_num = {'rand_idx': [], 'root_index': [],'n_charge_noemc': [],'n_charge_withemc': []}

        for key in ccolumn:
            dfdict_c[key] = []
        for key in bcolumn :
            dfdict_b[key] = []
        for key in numcolumn :
            dfdict_num[key] = []
        lastN = 0
        lastsumN = 0
        rand_idx = random.sample(range(0, 5000000), 5000000)
        rand_st = 0
        for n in range(name_len):
            file = uproot.open(filename[n])
            tree = file['all']
            # 读取所有分支到字典中
            bd = tree.arrays(ccolumn + bcolumn + numcolumn)
            N = len(bd)
            printN = int(N / 20)
            sum_ntot = 0

            for i in range(0, N):
                if (i % printN == 0):
                    wlog(f'{filename[n]}: loading dataframe: {i}/{N}', 'pre_train.log')
                event_data = {}
                drop = False
                drop_idx = []
                for key in numcolumn:
                    event_data[key] = bd[key][i]
                event_data['root_index'] = i if n == 0 else -i  # sig.root, root_index>0; bkg, root_index<0
                n_cfix = event_data['n_charge']
                n_bfix = event_data['n_neutral']
                n_totfix = event_data['n_tot']
                sum_ntot += n_totfix

                for key in ccolumn + bcolumn :
                    event_data[key] = list(bd[key][i])

                    # 处理nan数据，暂时认为不会有label=1,2的nan数据
                    for j in range(len(event_data[key])):
                        if np.isnan(event_data[key][j]) and j not in drop_idx:
                            drop_idx.append(j)
                            if j < event_data['n_charge']:
                                n_cfix -= 1
                            elif j < event_data['n_tot']:
                                n_bfix -= 1
                            n_totfix -= 1
                            wlog(f"NaN Warning: {filename[n]},entry {i}, {key}, label is {bd['label'][i][j]}",
                                 'pre_train.log')

                # track level loop
                for idx in range(event_data['n_tot']):
                    if idx in drop_idx: continue  # 防止重复处理nan数据行
                    track_drop = False
                    for key, cut in drop_cut.items():
                        if key in event_data.keys() and idx < len(event_data[key]):
                            if not ((cut[0] <= event_data[key][idx] <= cut[1]) or event_data[key][idx] == 9999):
                                if event_data[self.label][idx] in [1, 2]:
                                    drop = True
                                    break
                                else:
                                    drop_idx.append(idx)
                                    if idx < event_data['n_charge']:
                                        n_cfix -= 1
                                    elif idx < event_data['n_tot']:
                                        n_bfix -= 1
                                    n_totfix -= 1
                                    track_drop = True
                                    break
                    if track_drop: continue

                    for key, cut in ndrop_cut.items():
                        if (key in event_data.keys() and
                                not ((cut[0] <= event_data[key][idx] <= cut[1]) or event_data[key][idx] == 9999)):
                            if idx < event_data['n_charge']:
                                event_data[key][idx] = 9999
                            else:
                                if event_data[self.label][idx] == 2:
                                    drop = True
                                    break
                                else:
                                    drop_idx.append(idx)
                                    n_bfix -= 1
                                    n_totfix -= 1
                                    track_drop = True
                                    break
                    if track_drop: continue

                    if idx < event_data['n_charge']:
                        for key, cut in setzero_cut.items():
                            if (key in event_data.keys() and
                                    not ((cut[0] <= event_data[key][idx] <= cut[1]) or event_data[key][idx] == 9999)):
                                event_data[key][idx] = 9999
                    if drop: break
                # if any true track can't pass the cut, drop the event
                if drop: continue
                # else, drop the false track
                for idx in sorted(drop_idx, reverse=True):
                    for key in event_data.keys():
                        if type(event_data[key]) == list and idx < len(event_data[key]):
                            del event_data[key][idx]
                # save to dict
                event_data['n_charge'] = n_cfix
                event_data['n_neutral'] = n_bfix
                event_data['n_tot'] = n_totfix
                n_noemc=0
                for idx in range(n_cfix):
                    if event_data['emc_energy'][idx] == 9999:
                        n_noemc += 1
                event_data['n_charge_noemc'] = n_noemc
                event_data['n_charge_withemc'] = n_cfix - n_noemc
                if n_totfix < 1: continue

                for key in event_data.keys():
                    if key in ccolumn:
                        dfdict_c[key] += event_data[key]
                    elif key in bcolumn:
                        dfdict_b[key] += event_data[key]
                    elif key in numcolumn + ['n_charge_noemc','n_charge_withemc']:
                        dfdict_num[key].append(event_data[key])
                # random index for shuffle, c,b,num must have a same rand_idx,+0.001*track_idx to make sure track will not be shuffled in each event
                dfdict_num['rand_idx'].append(float(rand_idx[rand_st]))
                dfdict_c['rand_idx'] += [rand_idx[rand_st] + 0.001 * track_idx for track_idx in range(n_cfix)]
                dfdict_b['rand_idx'] += [rand_idx[rand_st] + 0.001 * track_idx for track_idx in range(n_totfix)]
                rand_st += 1
                dfdict_num['root_index'].append(event_data['root_index'])
            evtnum = len(dfdict_num['n_tot']) - lastN
            tracknum = sum(dfdict_num['n_tot']) - lastsumN
            wlog(f'{filename[n]}: loading dataframe done\nevent rate:{evtnum}/{N},track rate:{tracknum}/{sum_ntot}',
                 'pre_train.log')
            lastN += evtnum
            lastsumN += tracknum
            del bd, tree, file
        del rand_idx
        df_c = pd.DataFrame(dfdict_c)
        df_b = pd.DataFrame(dfdict_b)
        df_num = pd.DataFrame(dfdict_num)
        del dfdict_b, dfdict_c, dfdict_num

        df_b = df_b.sort_values(by='rand_idx').reset_index(drop=True)
        df_c = df_c.sort_values(by='rand_idx').reset_index(drop=True)
        df_num = df_num.sort_values(by='rand_idx').reset_index(drop=True)
        wlog('Shuffle done', 'pre_train.log')

        # add the random truth P4 for bkg
        df_num = self.rand_bkgtruth(df_num)

        if otherfile:
            num_dict = {col: df_num[col].values for col in df_num.columns}
            c_dict = {col: df_c[col].values for col in df_c.columns}
            b_dict = {col: df_b[col].values for col in df_b.columns}

            df_name = filename[0].replace('.root','_df.root')
            with uproot.recreate(df_name) as file:
                file["num"] = num_dict
                file["c"] = c_dict
                file["b"] = b_dict
            wlog(f'{df_name} saved', 'pre_train.log')
            return df_name

        testdf_num, testdf_c, testdf_b, traindf_num, traindf_c, traindf_b = self.__dfsplit(df_num, df_c, df_b,
                                                                                           1 - trainset_ratio)
        testdf_num, testdf_c, testdf_b, valdf_num, valdf_c, valdf_b = self.__dfsplit(testdf_num, testdf_c, testdf_b,
                                                                                     0.5)
        standard_dict = {}

        for key in ccolumn:
            if 'muc' in key:
                mean = traindf_c[traindf_c[key] != 9999][key].mean()
                std = traindf_c[traindf_c[key] != 9999][key].std()
                standard_dict[key] = (mean, std)
            else:
                mean = traindf_c[key].mean()
                std = traindf_c[key].std()
                standard_dict[key] = (mean, std)
        for key in bcolumn:
            if 'emc' in key:
                mean = traindf_b[traindf_b[key] != 9999][key].mean()
                std = traindf_b[traindf_b[key] != 9999][key].std()
                standard_dict[key] = (mean, std)
            else:
                mean = traindf_b[key].mean()
                std = traindf_b[key].std()
                standard_dict[key] = (mean, std)

        with open(self.datasetpath + '/std.json', 'w', encoding='utf-8') as f:
            json.dump(standard_dict, f, ensure_ascii=False)
        wlog('Standard done', 'pre_train.log')

        T_num_dict = {col: traindf_num[col].values for col in traindf_num.columns}
        T_c_dict = {col: traindf_c[col].values for col in traindf_c.columns}
        T_b_dict = {col: traindf_b[col].values for col in traindf_b.columns}

        traindf_name = f"{self.datasetpath}/train_df.root"
        with uproot.recreate(traindf_name) as file:
            file["num"] = T_num_dict
            file["c"] = T_c_dict
            file["b"] = T_b_dict
        wlog(f'{traindf_name} saved', 'pre_train.log')
        del T_num_dict, T_c_dict, T_b_dict

        Te_num_dict = {col: testdf_num[col].values for col in testdf_num.columns}
        Te_c_dict = {col: testdf_c[col].values for col in testdf_c.columns}
        Te_b_dict = {col: testdf_b[col].values for col in testdf_b.columns}
        test_df_name = f"{self.datasetpath}/test_df.root"
        with uproot.recreate(test_df_name) as file:
            file["num"] = Te_num_dict
            file["c"] = Te_c_dict
            file["b"] = Te_b_dict
        wlog(f'{test_df_name} saved', 'pre_train.log')
        del Te_num_dict, Te_c_dict, Te_b_dict

        V_num_dict = {col: valdf_num[col].values for col in valdf_num.columns}
        V_c_dict = {col: valdf_c[col].values for col in valdf_c.columns}
        V_b_dict = {col: valdf_b[col].values for col in valdf_b.columns}
        val_df_name = f"{self.datasetpath}/val_df.root"
        with uproot.recreate(val_df_name) as file:
            file["num"] = V_num_dict
            file["c"] = V_c_dict
            file["b"] = V_b_dict
        wlog(f'{val_df_name} saved', 'pre_train.log')
        del V_num_dict, V_c_dict, V_b_dict

        wlog(f'pre-process for training done', 'pre_train.log')
        return 0

    # pre-process for root files got from EtacTagger ana program,output files are for training
    def ana_to_train_iter(self, start_from:float=0.1 ,step=0.1, end=1,trainset_ratio:float=0.4):
        '''
        make multi dataset from ana Alg, ratio start from 0+step, stop at end.
        '''
        bcolumn = ['px', 'py', 'pz', 'energy',
                   'emc_x', 'emc_y', 'emc_z', 'emc_theta', 'emc_dtheta', 'emc_phi', 'emc_dphi', 'emc_energy',
                   'emc_time', 'emc_numHits', 'emc_e3x3',
                   'emc_e5x5', 'emc_dE', 'emc_a20moment', 'emc_a42moment',
                   'emc_secondmoment', 'emc_latmoment', 'iGood'] + [self.label]
        ccolumn = ['q', 'vz0', 'vr0', 'mdc_theta', 'mdc_phi',
                   'd0', 'phi0', 'kappa', 'z0', 'tanlambda',
                   'd0_err', 'phi0_err', 'kappa_err', 'z0_err', 'tanlambda_err',
                   'prob_pid', 'candy_pid', 'pid_prob_e', 'pid_prob_mu', 'pid_prob_pi', 'pid_prob_k',
                   'pid_prob_proton',
                   'muc_dpt', 'muc_numLayer', 'muc_maxHitsInlater', 'muc_chisq', 'muc_dof', 'muc_numHits']
        numcolumn = ['n_charge_track', 'n_charge', 'n_neutral', 'n_tot', 'kmchisq', 'n_res', 'res_pdg', 'res_dm',
                     'res_m', 'R',
                     'target_m', 'kmfit_target_m', 'recfit_target_m',
                     'cms_m', 'kmfit_cms_m', 'recfit_cms_m',
                     'Run', 'Event', 'ecms',
                     'truth_target_px', 'truth_target_py', 'truth_target_pz', 'truth_target_e', 'truth_target_m',
                     'recfit_target_px', 'recfit_target_py', 'recfit_target_pz', 'recfit_target_e',
                     'target_px', 'target_py', 'target_pz', 'target_e',
                     'recfit_cms_px', 'recfit_cms_py', 'recfit_cms_pz', 'recfit_cms_e',
                     'cms_px', 'cms_py', 'cms_pz', 'cms_e'
                     ] + self.topobranches + [self.evt_label]

        filename = [f'{self.datasetpath}/sig.root', f'{self.datasetpath}/hcbkg.root']
        if self.incmc_bkg: filename.append(f'{self.datasetpath}/incmc_bkg.root')

        drop_cut = {
            'px': [-3.686, 3.686], 'py': [-3.686, 3.686], 'pz': [-3.686, 3.686], 'energy': [0, 3.686],
            'd0': [-10, 10], 'kappa': [-30, 30], 'z0': [-25, 25], 'tanlambda': [-4, 4]
        }
        ndrop_cut = {
            'emc_x': [-100, 100], 'emc_y': [-100, 100], 'emc_z': [-150, 150], 'emc_energy': [0, 3.686],
            'emc_a20moment': [0, 1],
            'emc_a42moment': [0, 1], 'emc_secondmoment': [0, 200], 'emc_latmoment': [0, 1]
        }
        setzero_cut = {
            'muc_dpt': [0, 200]
        }
        name_len = len(filename)
        dfdict_c = {'rand_idx': []}
        dfdict_b = {'rand_idx': []}
        dfdict_num = {'rand_idx': [], 'root_index': [], 'n_charge_noemc': [], 'n_charge_withemc': []}

        for key in ccolumn:
            dfdict_c[key] = []
        for key in bcolumn:
            dfdict_b[key] = []
        for key in numcolumn:
            dfdict_num[key] = []
        lastN = 0
        lastsumN = 0
        rand_idx = random.sample(range(0, 5000000), 5000000)
        rand_st = 0
        for n in range(name_len):
            file = uproot.open(filename[n])
            tree = file['all']
            # 读取所有分支到字典中
            bd = tree.arrays(ccolumn + bcolumn + numcolumn)
            N = len(bd)
            printN = int(N / 20)
            sum_ntot = 0

            for i in range(0, N):
                if (i % printN == 0):
                    wlog(f'{filename[n]}: loading dataframe: {i}/{N}', 'pre_train.log')
                event_data = {}
                drop = False
                drop_idx = []
                for key in numcolumn:
                    event_data[key] = bd[key][i]
                event_data['root_index'] = i if n == 0 else -i  # sig.root, root_index>0; bkg, root_index<0
                n_cfix = event_data['n_charge']
                n_bfix = event_data['n_neutral']
                n_totfix = event_data['n_tot']
                sum_ntot += n_totfix

                for key in ccolumn + bcolumn:
                    event_data[key] = list(bd[key][i])

                    # 处理nan数据，暂时认为不会有label=1,2的nan数据
                    for j in range(len(event_data[key])):
                        if np.isnan(event_data[key][j]) and j not in drop_idx:
                            drop_idx.append(j)
                            if j < event_data['n_charge']:
                                n_cfix -= 1
                            elif j < event_data['n_tot']:
                                n_bfix -= 1
                            n_totfix -= 1
                            wlog(f"NaN Warning: {filename[n]},entry {i}, {key}, label is {bd['label'][i][j]}",
                                 'pre_train.log')

                # track level loop
                for idx in range(event_data['n_tot']):
                    if idx in drop_idx: continue  # 防止重复处理nan数据行
                    track_drop = False
                    for key, cut in drop_cut.items():
                        if key in event_data.keys() and idx < len(event_data[key]):
                            if not ((cut[0] <= event_data[key][idx] <= cut[1]) or event_data[key][idx] == 9999):
                                if event_data[self.label][idx] in [1, 2]:
                                    drop = True
                                    break
                                else:
                                    drop_idx.append(idx)
                                    if idx < event_data['n_charge']:
                                        n_cfix -= 1
                                    elif idx < event_data['n_tot']:
                                        n_bfix -= 1
                                    n_totfix -= 1
                                    track_drop = True
                                    break
                    if track_drop: continue

                    for key, cut in ndrop_cut.items():
                        if (key in event_data.keys() and
                                not ((cut[0] <= event_data[key][idx] <= cut[1]) or event_data[key][idx] == 9999)):
                            if idx < event_data['n_charge']:
                                event_data[key][idx] = 9999
                            else:
                                if event_data[self.label][idx] == 2:
                                    drop = True
                                    break
                                else:
                                    drop_idx.append(idx)
                                    n_bfix -= 1
                                    n_totfix -= 1
                                    track_drop = True
                                    break
                    if track_drop: continue

                    if idx < event_data['n_charge']:
                        for key, cut in setzero_cut.items():
                            if (key in event_data.keys() and
                                    not ((cut[0] <= event_data[key][idx] <= cut[1]) or event_data[key][
                                        idx] == 9999)):
                                event_data[key][idx] = 9999
                    if drop: break
                # if any true track can't pass the cut, drop the event
                if drop: continue
                # else, drop the false track
                for idx in sorted(drop_idx, reverse=True):
                    for key in event_data.keys():
                        if type(event_data[key]) == list and idx < len(event_data[key]):
                            del event_data[key][idx]
                # save to dict
                event_data['n_charge'] = n_cfix
                event_data['n_neutral'] = n_bfix
                event_data['n_tot'] = n_totfix
                n_noemc = 0
                for idx in range(n_cfix):
                    if event_data['emc_energy'][idx] == 9999:
                        n_noemc += 1
                event_data['n_charge_noemc'] = n_noemc
                event_data['n_charge_withemc'] = n_cfix - n_noemc
                if n_totfix < 1: continue

                for key in event_data.keys():
                    if key in ccolumn:
                        dfdict_c[key] += event_data[key]
                    elif key in bcolumn:
                        dfdict_b[key] += event_data[key]
                    elif key in numcolumn + ['n_charge_noemc', 'n_charge_withemc']:
                        dfdict_num[key].append(event_data[key])
                # random index for shuffle, c,b,num must have a same rand_idx,+0.001*track_idx to make sure track will not be shuffled in each event
                dfdict_num['rand_idx'].append(float(rand_idx[rand_st]))
                dfdict_c['rand_idx'] += [rand_idx[rand_st] + 0.001 * track_idx for track_idx in range(n_cfix)]
                dfdict_b['rand_idx'] += [rand_idx[rand_st] + 0.001 * track_idx for track_idx in range(n_totfix)]
                rand_st += 1
                dfdict_num['root_index'].append(event_data['root_index'])
            evtnum = len(dfdict_num['n_tot']) - lastN
            tracknum = sum(dfdict_num['n_tot']) - lastsumN
            wlog(f'{filename[n]}: loading dataframe done\nevent rate:{evtnum}/{N},track rate:{tracknum}/{sum_ntot}',
                 'pre_train.log')
            lastN += evtnum
            lastsumN += tracknum
            del bd, tree, file
        del rand_idx
        df_c = pd.DataFrame(dfdict_c)
        df_b = pd.DataFrame(dfdict_b)
        df_num = pd.DataFrame(dfdict_num)
        del dfdict_b, dfdict_c, dfdict_num

        df_b = df_b.sort_values(by='rand_idx').reset_index(drop=True)
        df_c = df_c.sort_values(by='rand_idx').reset_index(drop=True)
        df_num = df_num.sort_values(by='rand_idx').reset_index(drop=True)
        wlog('Shuffle done', 'pre_train.log')

        # add the random truth P4 for bkg
        df_num = self.rand_bkgtruth(df_num)

        ratio=np.arange(start_from,end,step)
        for r in ratio:
            slice_num, slice_c, slice_b, _, _, _ = self.__dfsplit(df_num,df_c, df_b,r)
            testdf_num, testdf_c, testdf_b, traindf_num, traindf_c, traindf_b = self.__dfsplit(slice_num, slice_c, slice_b,
                                                                                               1 - trainset_ratio)
            testdf_num, testdf_c, testdf_b, valdf_num, valdf_c, valdf_b = self.__dfsplit(testdf_num, testdf_c, testdf_b,
                                                                                         0.5)
            standard_dict = {}

            for key in ccolumn:
                if 'muc' in key:
                    mean = traindf_c[traindf_c[key] != 9999][key].mean()
                    std = traindf_c[traindf_c[key] != 9999][key].std()
                    standard_dict[key] = (mean, std)
                else:
                    mean = traindf_c[key].mean()
                    std = traindf_c[key].std()
                    standard_dict[key] = (mean, std)
            for key in bcolumn:
                if 'emc' in key:
                    mean = traindf_b[traindf_b[key] != 9999][key].mean()
                    std = traindf_b[traindf_b[key] != 9999][key].std()
                    standard_dict[key] = (mean, std)
                else:
                    mean = traindf_b[key].mean()
                    std = traindf_b[key].std()
                    standard_dict[key] = (mean, std)

            with open(self.datasetpath + f'/std_{int(round(r*100))}.json', 'w', encoding='utf-8') as f:
                json.dump(standard_dict, f, ensure_ascii=False)
            wlog('Standard done', 'pre_train.log')

            T_num_dict = {col: traindf_num[col].values for col in traindf_num.columns}
            T_c_dict = {col: traindf_c[col].values for col in traindf_c.columns}
            T_b_dict = {col: traindf_b[col].values for col in traindf_b.columns}

            traindf_name = f"{self.datasetpath}/train_df_{int(round(r*100))}.root"
            with uproot.recreate(traindf_name) as file:
                file["num"] = T_num_dict
                file["c"] = T_c_dict
                file["b"] = T_b_dict
            wlog(f'{traindf_name} saved', 'pre_train.log')
            del T_num_dict, T_c_dict, T_b_dict

            Te_num_dict = {col: testdf_num[col].values for col in testdf_num.columns}
            Te_c_dict = {col: testdf_c[col].values for col in testdf_c.columns}
            Te_b_dict = {col: testdf_b[col].values for col in testdf_b.columns}
            test_df_name = f"{self.datasetpath}/test_df_{int(round(r*100))}.root"
            with uproot.recreate(test_df_name) as file:
                file["num"] = Te_num_dict
                file["c"] = Te_c_dict
                file["b"] = Te_b_dict
            wlog(f'{test_df_name} saved', 'pre_train.log')
            del Te_num_dict, Te_c_dict, Te_b_dict

            V_num_dict = {col: valdf_num[col].values for col in valdf_num.columns}
            V_c_dict = {col: valdf_c[col].values for col in valdf_c.columns}
            V_b_dict = {col: valdf_b[col].values for col in valdf_b.columns}
            val_df_name = f"{self.datasetpath}/val_df_{int(round(r*100))}.root"
            with uproot.recreate(val_df_name) as file:
                file["num"] = V_num_dict
                file["c"] = V_c_dict
                file["b"] = V_b_dict
            wlog(f'{val_df_name} saved', 'pre_train.log')
            del V_num_dict, V_c_dict, V_b_dict

            wlog(f'pre-process for training done', 'pre_train.log')
        return 0

    def norm(self,df_c, df_b,stdfile:str = None):
        if stdfile is not None:
            with open(stdfile, 'r') as f:
                self.norm_dict = json.load(f)
        else:
            with open(self.stdfile, 'r') as f:
                self.norm_dict = json.load(f)

        for key in self.ccolumn:
            if (key in self.column_notrs) or (key not in df_c.columns):
                continue
            mean = self.norm_dict[key][0]
            std = self.norm_dict[key][1]
            if 'muc' in key:
                df_c.loc[df_c[key] != 9999, key] = (df_c[df_c[key] != 9999][key] - mean) / std
                df_c.loc[df_c[key] == 9999, key] = 0
            else:
                df_c[key] = (df_c[key] - mean) / std
        for key in self.bcolumn:
            if (key in self.column_notrs) or (key not in df_b.columns): 
                continue
            mean = self.norm_dict[key][0]
            std = self.norm_dict[key][1]
            if 'emc' in key:
                df_b.loc[df_b[key] != 9999, key] = (df_b[df_b[key] != 9999][key] - mean) / std
                df_b.loc[df_b[key] == 9999, key] = 0
            else:
                df_b[key] = (df_b[key] - mean) / std
        return df_c, df_b

    def denorm(self,df_c, df_b):
        with open(self.stdfile, 'r') as f:
            self.norm_dict = json.load(f)
        for key in self.ccolumn:
            if (key in self.column_notrs) or (key not in df_c.columns):
                continue
            if 'muc' in key:
                cutkey = 'muc_numHits'
            else:
                cutkey = False
            mean = self.norm_dict[key][0]
            std = self.norm_dict[key][1]
            if cutkey:
                df_c.loc[df_c[cutkey] != 0, key] = (df_c[df_c[cutkey] != 0][key] * std) + mean
            else:
                df_c[key] = (df_c[key] * std) + mean
        for key in self.bcolumn:
            if (key in self.column_notrs) or (key not in df_b.columns): 
                continue
            if 'emc' in key:
                cutkey = 'emc_energy'
            else:
                cutkey = False
            mean = self.norm_dict[key][0]
            std = self.norm_dict[key][1]
            if cutkey:
                df_b.loc[df_b[cutkey] != 0, key] = (df_b[df_b[cutkey] != 0][key] * std) + mean
            else:
                df_b[key] = (df_b[key] * std) + mean
        return df_c, df_b


class DfToDataset(Dataset):
    def __init__(self, df_c: pd.DataFrame, df_b: pd.DataFrame, df_num: pd.DataFrame, config: dict
                 , mode: int = 1,CDN=False, maxstep=1000000, token_num=max_token, logfile: str = None):
        """
        If n_tot > token_num, cut to token_num else fill to token_num with zero
        :param df_c: dataframe that have charged tracks feat.
        :param df_b: dataframe that have both_feats feat.
        :param df_num: dataframe that have event level feat.
        :param mode: 0 = predict, 1 = training
        """
        print('Initializing DataFrameTokenizedDataset...')
        self.logfile = logfile
        self.mode = mode
        self.CDN = CDN

        self.interactions_col = config['interactions']
        self.interactions_col = [i + '_raw' for i in self.interactions_col]
        c_columns = config['charged_feat']
        b_columns = config['both_feat']

        self.n_c = df_num[config['n_charge']].values
        self.n_tot = df_num[config['n_tot']].values
        if self.mode: self.target_m = df_num[config['target_m']].values
        if type(config['ecms']) == str:
            self.ecms=df_num[config['ecms']].values
        elif type(config['ecms']) == float:
            self.ecms = config['ecms']
        # With inputs, we define three sets of tokenized data: features, vectors, and points
        self.interactions = df_b[self.interactions_col[:5]].values
        self.c_interactions = df_c[self.interactions_col[-1]].values
        self.inputs_c_feat = df_c[c_columns].values
        self.inputs_b_feat = df_b[b_columns].values
        self.N = len(df_num)

        self.zeros_forgamma = np.array([0. for _ in range(0, len(c_columns))])
        self.zeros_forall = np.array([0. for _ in range(0, len(c_columns + b_columns))])
        self.features = None
        self.vectors = None
        self.mask = None
        self.conditions = torch.tensor(df_num[config['conditions']].values, dtype=torch.float32) if self.CDN else None

        if self.mode:
            self.tag_label = df_b[config['tag_label']].values
            self.evt_y = torch.tensor(df_num[config['evt_label']].values, dtype=torch.long)
            self.truth_p4 = torch.tensor(df_num[config['etac_p4']].values, dtype=torch.float32)
            self.tag_y = None

        self.max_token = token_num
        if maxstep < 0: maxstep = self.N
        bn = 0
        en = maxstep
        self.index=0
        self.c_index=0
        self.u_index=0
        while bn < self.N:
            self.make_dataset(bn, en)
            bn += maxstep
            en += maxstep
        wlog(f'  features: {self.features.shape}', self.logfile)
        wlog(f'  mask: {self.mask.shape}', self.logfile)
        wlog(f'  vectors: {self.vectors.shape}', self.logfile)
        if self.CDN: wlog(f'  conditions: {self.conditions.shape}', self.logfile)
        if self.mode:
            wlog(f'  Tag Label: {self.tag_y.shape}', self.logfile)
            wlog(f'  Evt Label: {self.evt_y.shape}', self.logfile)
            wlog(f'  Truth P4: {self.truth_p4.shape}', self.logfile)
            del self.tag_label

        del self.interactions, self.inputs_c_feat, self.inputs_b_feat
        del self.zeros_forgamma, self.zeros_forall, self.n_c

    def make_dataset(self, bn, en):
        if en > self.N: en = self.N
        wlog(f"Makeing datasets from {bn} to {en}", self.logfile)
        # Define tokenized input "features" with dim (N, L=5, d=7)
        # [N,jet/l/met,feature]
        features = []  # (N,tokens,features)
        tmask = []
        ty = []
        U_q = []
        for i in range(bn, en):
            if en - bn > 20:
                if (i % ((en - bn) // 20)) == 0: wlog(f'Making datasets: {i}/{en}', self.logfile)
            tokens = []
            y = []
            mask = []
            n_c = self.n_c[i]
            n_tot = self.n_tot[i]
            t_q = []
            for j in range(self.index, self.index + n_tot):  # 先取带电/中性都有的features，按照n_tot切片，逐个粒子添加为token
                tokens.append(self.inputs_b_feat[j, :])
                mask.append(1)
                if self.mode: y.append(self.tag_label[j])  # 每个粒子一个class token,填充

            for j in range(0, n_c):  # 对只有带电粒子有的features，按照n_c切片，逐个补充进去
                tokens[j] = np.concatenate((tokens[j], self.inputs_c_feat[self.c_index + j, :]), axis=0)
                t_q.append(self.c_interactions[self.c_index + j ])
            for j in range(n_c, n_tot):  # 中性径迹添加对应个数的0
                tokens[j] = np.concatenate((tokens[j], self.zeros_forgamma), axis=0)
                t_q.append(0)
            if n_tot < self.max_token:
                for j in range(n_tot, self.max_token):
                    tokens.append(self.zeros_forall)  # token数补足到最大
                    mask.append(0)
                    t_q.append(0)
                    if self.mode: y.append(-1)  # 填充数据的label为-1,须在后续交叉熵计算中忽视
            else:
                tokens = tokens[:self.max_token]
                mask = mask[:self.max_token]
                t_q =t_q[:self.max_token]
                if self.mode: y = y[:self.max_token]

            if self.mode: ty.append(y)

            tmask.append([mask])
            features.append(tokens)
            self.index += n_tot
            self.c_index += n_c
            U_q.append(t_q)

        if self.features is None:
            self.features = torch.tensor(np.array(features), dtype=torch.float32)
            self.mask = torch.tensor(np.array(tmask), dtype=torch.float32)
        else:
            features = torch.tensor(np.array(features), dtype=torch.float32)
            mask = torch.tensor(np.array(tmask), dtype=torch.float32)
            self.features = torch.cat((self.features, features), dim=0)
            self.mask = torch.cat((self.mask, mask), dim=0)

        if self.mode:
            if self.tag_y is None:
                self.tag_y = torch.tensor(np.array(ty), dtype=torch.long)
            else:
                y = torch.tensor(np.array(ty), dtype=torch.long)
                self.tag_y = torch.cat((self.tag_y, y), dim=0)
        # Define tokenized "vectors" with dim (N, L=5, d=4), d=4 for (px, py, pz, energy)
        # 需要用没有经过标准化处理的原始数据计算四动量（理所当然）
        px = []
        py = []
        pz = []
        energy = []
        U_emc_e = []
        for i in range(bn, en):
            t_px = []
            t_py = []
            t_pz = []
            t_e = []
            t_emc_e = []
            n_tot = self.n_tot[i]
            for j in range(self.u_index, self.u_index + n_tot):  # 按顺序添加四动量
                t_px.append(self.interactions[j, 0])
                t_py.append(self.interactions[j, 1])
                t_pz.append(self.interactions[j, 2])
                t_e.append(self.interactions[j, 3])
                if len(self.interactions_col) > 4:
                    t_emc_e.append(self.interactions[j, 4])

            if n_tot < self.max_token:
                for j in range(n_tot, self.max_token):  # 不足补0
                    t_px.append(0)
                    t_py.append(0)
                    t_pz.append(0)
                    t_e.append(0)
                    t_emc_e.append(0)
            else:
                t_px = t_px[:self.max_token]
                t_py = t_py[:self.max_token]
                t_pz = t_pz[:self.max_token]
                t_e = t_e[:self.max_token]
                t_emc_e = t_emc_e[:self.max_token]
            px.append(t_px)
            py.append(t_py)
            pz.append(t_pz)
            energy.append(t_e)
            if len(self.interactions_col) > 4: U_emc_e.append(t_emc_e)
            self.u_index += n_tot

        interactions_U = [px, py, pz, energy]
        if len(self.interactions_col) > 4: interactions_U.append(U_emc_e)
        if len(self.interactions_col) > 5: interactions_U.append(U_q)
        interactions_U = torch.tensor(np.stack(interactions_U, axis=-1), dtype=torch.float32)
        if self.vectors is None:
            self.vectors = interactions_U
        else:
            self.vectors = torch.cat((self.vectors, interactions_U), dim=0)

    def setmode(self):
        """
        If in training mode, switch to predict mode
        """
        self.mode = 0

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        mask = self.mask[idx]
        features = self.features[idx]
        vectors = self.vectors[idx]
        conditions = self.conditions[idx] if self.CDN else torch.empty(0)
        ecms = self.ecms[idx] if type(self.ecms) is not float else self.ecms
        if self.mode:
            y = self.tag_y[idx]
            n_tot = self.n_tot[idx]
            cls_y = self.evt_y[idx]
            truth_p4 = self.truth_p4[idx]
            target_m=self.target_m[idx]
            return (mask, features, vectors,conditions,ecms), y, n_tot, cls_y, truth_p4, target_m
        else:
            return mask, features, vectors,conditions,ecms
