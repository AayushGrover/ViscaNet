# ViscaNet

This is the official implementation of Virtual Screening Assistant Network (ViscaNet). The motivation of various modules in this repository is from [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) and [A self‐attention based message passing neural network for predicting molecular lipophilicity and aqueous solubility](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-0414-z). The basecode was obtained from [chemprop](https://github.com/chemprop/chemprop).

## Installation

This is common for all the experiments.
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/).
2. Download the github repository including “viscanet.yml”.
3. Open the terminal in your system and go to the location where viscanet.yml is downloaded. Then, run `conda env create -f environment.yml`. For more details, check the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
4. Once the environment is created, enter the environment using the command `conda activate viscanet`.

----------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------

## Running the Code

### For FDA Data 

The following steps can be used for any experiment where data is not split into multiple files. 

#### Data Processing

1. Add the data files to `./data/` directory. 
2. Run all the cells of **preprocess_fda.ipynb**.

#### Model Training & Testing

1. Run `python train_fda.py --data_path (path to data file) --dataset_type classification --smiles_column s_sd_SMILES --target_columns r_i_docking_score --epochs 30 --num_folds 1 --features_path (path to features) --attention --separate_test_path ./data/fda.csv --separate_test_features_path ./data/fda.npy`
   - Sample data_path = `./data/fda.csv`
   - Sample features_path = `./data/fda.npy`
   - If you want to split one data file into train/val/test, do not use `--separate_test_path` and `--separate_test_features_path`
   - If you have a separate test or val file, use `--separate_val_path` and `--separate_val_features_path`, and `--separate_test_path` and `--separate_test_features_path`
   - If you need attention images, use `--viz_dir` and give the path to the location where you want to store those images.
2. This will give the test scores and create TruePositives.csv and FalsePositives.csv in the `./inference/` directory. 
3. Run all the cells of **get_fdaid.ipynb**. This will add the drugbank_ID to the above mentioned generated files. 

----------------------------------------------------------------------------------------------------------------------------

### For Supernatural Data

The following steps can be used for any experiment where data is split into multiple files.

#### Data Processing

1. Add the data files to `./data/` directory. Add them in `./data/nsp1_supernaturaldb_sift_data/`. Also, create directories `new_data`, `new_data_feats`, and `new_data_norm_feats` inside the `data` directory.
2. Run `python preprocess_new_data.py` to preprocess the supernatural data. The processed data will store in the `./data/new_data/` directory. 
3. Use `python feature.py` to generate the feature (.npy) files for train as well as test data. These will be stored in `./data/new_data_feats/` directory.
4. Generate normalized features for both train and test data by executing `python feature_normalize.py`. These will be stored in `./data/new_data_norm_feats/` directory.
5. Split the data and the obtained normalized features into train, val, and test in directories `./data/new_data/` and `./data/new_data_norm_feats/` respectively.

#### Training

1. Run `python os_train.py`. This will train the model on all the datafiles available in `/data/new_data/train/` directory.
2. The model will be stored in `./model_checkpoints/` directory which is created automatically. Every time you run point number 1, it will run the new epoch for model training. For example, if you run `python os_train.py` 5 times in sequence, it will mean that the model is now trained for 5 epochs.

#### Testing

1. To test the model, you can run `python os_test.py --target_columns r_i_docking_score --dataset_type classification --epochs 1 --num_folds 1 --no_features_scaling --data_path ./`.
    - The paths are hard-coded inside the **os_test.py** and therefore, `--data_path` can be anything.
    - If you need attention images, use `--viz_dir` and give the path to the location where you want to store those images.

**NOTE** - Kindly fix the paths in case it throws an error or feel free to contact [me](https://github.com/AayushGrover) or raise an issue.

