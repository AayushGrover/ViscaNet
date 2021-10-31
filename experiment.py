"Runs the complete experiment"
import os

from os_train import run_train_scripts
from os_test import run_test_scripts

def run_experiment(train_path, train_feat_path, val_path, val_feat_path, test_path, \
    test_feat_path, save_dir, epoch_checkpoint, best_model_dir, epochs=1):
    
    '''
    train_path : path of training data
    train_feat_path : path of features for training data
    val_path : path of validation data
    val_feat_path : path of features for validation data
    test_path : path of testing data
    test_feat_path : path of features for testing data
    save_dir : directory path to save model checkpoint
    epoch_checkpoint : path to save epoch
    best_model_dir : directory path to save best model
    epochs : total number of epochs before the experiment terminates
    '''

    best_score = 0.0
    for _ in range(epochs):
        
        run_train_scripts(train_path, train_feat_path, val_path, val_feat_path, save_dir, epoch_checkpoint)
        mcc_score = run_test_scripts(test_path, test_feat_path, save_dir)
        
        if not os.path.isdir(best_model_dir):
            os.mkdir(best_model_dir)

        if mcc_score > best_score:
            best_score = mcc_score
            model_path = (save_dir+'*')
            os.system(f'cp -r {model_path} {best_model_dir}')

if __name__ == "__main__":
    epochs = 35
    save_dir = './model_checkpoint/'
    val_path = './data/new_data/val/'
    val_feat_path = './data/new_data_norm_feats/val/'
    train_path = './data/new_data/train/'
    train_feat_path = './data/new_data_norm_feats/train/'
    epoch_checkpoint = './model_checkpoint/epoch.pt'
    best_model_dir = './best_model_checkpoint/'
    test_path = './data/new_data/test/'
    test_feat_path = './data/new_data_norm_feats/test/'

    run_experiment(
        train_path=train_path,
        train_feat_path=train_feat_path,
        val_path=val_path,
        val_feat_path=val_feat_path,
        test_path=test_path,
        test_feat_path=test_feat_path,
        save_dir=save_dir,
        epoch_checkpoint=epoch_checkpoint,
        best_model_dir=best_model_dir,
        epochs=epochs
    )
