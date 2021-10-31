"""Tests a model on a dataset."""
from glob import glob
import numpy as np

from chemprop.args import TrainArgs
from chemprop.train.run_testing import run_testing
from chemprop.utils import create_logger

def check(x):           # To sum over TP, FP, TN, FN
    if 'False' in x or 'True' in x:
        return True
    return False

def run_test_scripts(test_path, test_feat_path, save_dir):
    args = TrainArgs().parse_args()
    args.target_columns = ['r_i_docking_score']
    args.dataset_type = 'classification'
    args.epochs = 1
    args.attention = True
    args.viz_dir = './attention-fda-new/'
    args.num_folds = 1
    args.no_features_scaling = True
    args.save_dir = save_dir
    args.save_dir += 'fold_0/'
    logger = create_logger(name='test', save_dir=args.save_dir, quiet=args.quiet)

    scores = list()
    for i, fname in enumerate(glob(test_path+'*')):
        args.data_path = fname
        fname = (fname.split('/')[-1]).split('.')[0]
        args.features_path = test_feat_path + fname + '.npy'
        scores.append(run_testing(args, logger))
        print(f'File {i} done. ({fname})')

    all_score = list()
    for i in range(len(scores[0])):
        score_l = list()
        for score in scores:
            score_l.append(score[i])
        all_score.append(np.array(score_l))

    metrics = [
        'AUC-ROC', 
        'True Negative', 
        'False Positive', 
        'False Negative', 
        'True Positive', 
        'Precision', 
        'Recall',
        'Specificity',
        'F1 Score',
        'MCC'
        ]

    d = dict()
    for score, metric in zip(all_score, metrics):
        if check(metric):
            sum_val = np.sum(score)
            d[metric] = sum_val
        
    tp = d['True Positive']
    fp = d['False Positive']
    tn = d['True Negative']
    fn = d['False Negative']

    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    specificity = (tn/(tn+fp))
    f1 = (2*tp/(2*tp+fn+fp))
    mcc = ((tp*tn)-(fp*fn))/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
    print('True Negative = ', tn)
    print('False Positive = ', fp)
    print('False Negative = ', fn)
    print('True Positive = ', tp)
    print('Precision = ', precision)
    print('Recall = ', recall)
    print('Specificity = ', specificity)
    print('F1-score = ', f1)
    print('MCC = ', mcc)

    return mcc     # retuns MCC value

if __name__ == "__main__":
    test_path = './data/new_data/test/'
    save_dir = './best_model_checkpoint/'
    test_feat_path = './data/new_data_norm_feats/test/'

    run_test_scripts(test_path, test_feat_path, save_dir)