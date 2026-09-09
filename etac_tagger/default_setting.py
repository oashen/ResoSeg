base_setting={
    'name': 'default',
    'epochs': 100,
    'batch_size': 1024,
    'sigcut': [0.45,0.5,0.55,0.7,0.8,0.9,0.95],
    'gamcut': [0.35,0.40,0.45,0.7,0.8,0.9,0.95],
    # the train output path
    'train_pic_path': 'train_pic/',
    'train_root_path': 'test_root/',
    # dataset setting
    'dataset': {
        'path': '/aifs/user/home/lichunkai1/datasets',
        'pipihc': False,
        'incmc_bkg': False,
        'norm_file': None, # None = path + std.json
        # branches related with topoana, make sure they are in your root files
        'topobranches': ['iDcyTr', 'iDcyIFSts', 'nSigP_hc', 'nSigP_etac',
                         'nIncDcyBr_hc2etac', 'iDcyBrIncDcyBr_hc2etac_0'],
        'etac_p4': ['target_px', 'target_py', 'target_pz', 'target_e'],  # the generation target
         # should be [px,py,pz,e,emc_energy,q]
        'interactions': ['px', 'py', 'pz', 'energy','emc_energy','q'],
        # features that only charged tracks have
        'charged_feat': ['vz0', 'vr0', 'd0', 'phi0', 'kappa', 'z0', 'tanlambda', 'q', 'candy_pid'],
        # features that both charged tracks and neutral tracks have
        'both_feat': ['emc_theta', 'emc_dtheta', 'emc_phi', 'emc_dphi', 'emc_energy', 'emc_dE',
                      'emc_a20moment','emc_a42moment', 'emc_secondmoment', 'emc_latmoment'],
        'ecms': 'ecms',
        'conditions': ['n_charge_noemc', 'n_charge_withemc', 'n_neutral'],
        # features that only charged tracks have
        # 'charged_feat': ['vz0', 'vr0', 'mdc_costhe', 'mdc_theta', 'mdc_phi', 'd0', 'phi0', 'kappa',
        #                  'z0', 'tanlambda', 'q', 'pid_prob_pi', 'pid_prob_k', 'pid_prob_proton'],
        # features that both charged tracks and neutral tracks have
        # 'both_feat': ['emc_x', 'emc_y', 'emc_z', 'emc_theta', 'emc_dtheta', 'emc_phi', 'emc_dphi',
        #               'emc_energy','emc_time', 'emc_numHits', 'emc_e3x3', 'emc_e5x5', 'emc_dE',
        #               'emc_a20moment','emc_a42moment', 'emc_secondmoment', 'emc_latmoment',
        #               'px', 'py', 'pz', 'energy'],
        # features will not be standard
        'no_stander_feat': ['q', 'prob_pid','candy_pid', 'm'],
        'tag_label': 'label',
        'evt_label': 'sig_event',
        'n_charge': 'n_charge',
        'n_tot': 'n_tot',
        'target_m': 'target_m'
    },
    'lightning_model':  {
        'stable_lr': 1e-4,
        'UW': False, # if True, use learnable f parameters
        'f_tag': 1,
        'f_evt': 1,
        # gen_p4 config
        'f_gen': 1,
        'mloss': False,
        # whether use the mass of eta_c as loss
        'R_loss': False,
        'target_m_mean':2.9841, # the PDG value for events not with eta_c
        'f_R':1,
        'SamTagger': {
            'num_class': 3,
            'num_mask': 1,
            # list that control which features are using, padding mask
            # [ln(m^2), ln(RM^2), ln(delta), qi*qj, ln(kt), ln(z), |helicity angle|, Pij_z/|Pij|, ln(|emc_ei - emc_ej|)]
            # default setting means use [ln(m^2), ln(RM^2),qi*qj, ln(kt)]
            'pair_input_dim': [1,1,0,1,1,0,0,0,0],
            'embed_dims': [64,256,64],
            'pair_embed_dims': [64,64,64],
            'dot_scale': 4,
            'num_heads': 4,
            'embed_layers': 4,
            'gen_layers': 0, # 0 means don't gen p4
            'decoder_layers': 2,
            'embed_block_params':  None,
            'gen_block_params':  None,
            'decoder_block_params':  None,
            'activation': 'gelu',
            'block_activation': 'swiglu',
            'gen_mlp': [(64, 0.1), (64, 0.1)],
            'mask_mlp':  [(64, 0.1), (64, 0.1)],
            'evt_mlp':  [(64, 0.1), (64, 0.1)],
            'trim': True,
            'threshold': -1,
            'CDN': False,   # use condition layer norm or not
            'condition_dims':[32,128,32],
            'gate_attn': False,
            'mod_only_encoder':False,
            'part':False,   # use ParT decoder
            'sam':True      # use SAM decoder
        }
    }
}