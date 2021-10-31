import os
from glob import glob
from tqdm import tqdm

import torch

def run_train_scripts(train_path, train_feat_path, val_path, val_feat_path, save_dir, epoch_checkpoint):
    
    epochs = 1
    for epoch in range(1, epochs+1):
        try:
            already_done = (torch.load(epoch_checkpoint))['epoch']
        except:
            already_done = 0

        print(f'Epoch {epoch+already_done}: ')

        for i, datapath in tqdm(enumerate(glob(train_path+'*'))):
            file_name = datapath.split('/')[-1]
            file_name = file_name.split('.')[0]
            feat_path = train_feat_path+file_name+'.npy'
            
            # Train with combined features
            # os.system(f'python train.py --data_path {datapath} --separate_val_path {val_path} --separate_val_features_path {val_feat_path} --save_dir {save_dir} --target_columns r_i_docking_score --dataset_type classification --epochs 1 --class_balance --num_folds 1 --features_generator rdkit_2d_normalized --features_path {feat_path} --attention --no_features_scaling')
            
            # Train with RDKit features
            # os.system(f'python train.py --data_path {datapath} --separate_val_path {val_path} --save_dir {save_dir} --target_columns r_i_docking_score --dataset_type classification --epochs 1 --class_balance --num_folds 1 --features_generator rdkit_2d_normalized --attention --no_features_scaling')

            # Train with NCBS features
            os.system(f'python train.py --data_path {datapath} --separate_val_path {val_path} --separate_val_features_path {val_feat_path} --save_dir {save_dir} --target_columns r_i_docking_score --dataset_type classification --epochs 1 --class_balance --num_folds 1 --features_path {feat_path} --attention --no_features_scaling')

            torch.save({'epoch': epoch+already_done}, epoch_checkpoint)
            print(f'File {i} done. ({file_name})')

if __name__ == "__main__":
    save_dir = './model_checkpoints/'
    val_path = './data/new_data/val/'
    val_feat_path = './data/new_data_norm_feats/val/'
    train_path = './data/new_data/train/'
    train_feat_path = './data/new_data_norm_feats/train/'
    epoch_checkpoint = './model_checkpoints/epoch.pt'

    run_train_scripts(train_path, train_feat_path, val_path, val_feat_path, save_dir, epoch_checkpoint)