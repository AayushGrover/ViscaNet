import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# Prepare a dictionary mapping SN_IDs and SMILES
df = pd.read_csv('./data/nsp1_supernaturaldb_sift_data/sn_id_smiles.tsv', sep='\t')
sn_id = df.id.tolist()
# sn_id = df.SNID.tolist()
smiles = df.smiles.tolist()
id2smiles = dict(zip(sn_id, smiles))

# Convert continuous target to binary
def f(x):
    if x > -3.15066:
        return 0
    return 1

def check(x):
    if 'SN' in x:
        return 1
    return 0

# sn_id_list = ['SN00220639', 'SN00103215', 'SN00003832', 'SN00216190']
idx = 0
l = list()
for fname in tqdm(glob('./data/nsp1_supernaturaldb_sift_data/*')):
    if 'smiles' in fname:
        continue
    else:
        df = pd.read_csv(fname, sep='\t', header=1)
        # df['temp'] = df['Title'].apply(lambda x: check(x))
        # df = df[df['temp'] == 1]
        l = ['smiles'] + list(df.columns)
        # l_copy = ['temp', 'Title']
        l_copy = ['Title']
        for i in l:
            if ":" in i:
                l_copy.append(i)
        df['smiles'] = df['Title'].apply(lambda x: id2smiles[x])
        for i in l_copy:
            l.remove(i)
        df = df[l]
        
        target_col = 'r_i_docking_score'
        smiles_col = 'smiles'
        rem_cols = [
            'i_i_glide_lignum',
            'i_i_glide_posenum',
            'i_lp_mmshare_version',
            'i_m_Source_File_Index',
            'r_i_glide_gscore'
            ]
        df.drop(columns=rem_cols, inplace=True)

        df[target_col] = df[target_col].apply(lambda x: f(x))
        idx += 1
        df.dropna(inplace=True)

        # df_new = df[df['Title'].isin(sn_id_list)]
        # print(df_new.shape)
        
        # if df_new.shape[0] > 0:
        #     try:
        #         df_final = df_final.append(df_new)
        #     except:    
        #         df_final = df_new
        df.to_csv(f'./data/new_data/df{idx}.csv', index=None)

# df_final = df_final.drop_duplicates(subset=['Title'])
# df_final.to_csv('./data/new_data/test/sp_new.csv', index=None)
