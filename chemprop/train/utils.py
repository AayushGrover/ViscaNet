import pandas as pd

def save(smiles, pred, tar, rows):
    '''
    smiles : List of SMILES representations of drugs
    pred : List of predicted outputs
    tar : List of target outputs
    rows : List of row features
    '''
    assert len(smiles) == len(pred) == len(tar) == len(rows)
    
    binarized_preds = [[1.0] if x[0] > 0.5 else [0.0] for x in pred]
    
    tp_rows = []
    fp_rows = []

    for i in range(len(smiles)):
        p = binarized_preds[i][0]
        t = tar[i][0]

        if p == 1:
            # Corresponding to True Positives
            if t == 1:
                # tp_smile.append(smiles[i])
                # tp_pred.append(pred[i][0])
                # tp_tar.append(tar[i][0])
                tp_rows.append([pred[i][0]] + list(rows[i].values()))

            # Corresponding to False Positives
            else:
                # fp_smile.append(smiles[i])
                # fp_pred.append(pred[i][0])
                # fp_tar.append(tar[i][0])
                fp_rows.append([pred[i][0]] + list(rows[i].values()))

    cols = ['Pred'] + list(rows[0].keys())

    # tp_df = pd.DataFrame(list(zip(tp_smile,tp_pred,tp_tar)), columns=['Smiles','Pred','Target'])
    # fp_df = pd.DataFrame(list(zip(fp_smile,fp_pred,fp_tar)), columns=['Smiles','Pred','Target'])

    tp_df = pd.DataFrame(tp_rows, columns=cols)
    fp_df = pd.DataFrame(fp_rows, columns=cols)

    try:
        tp_df1 = pd.read_csv('./inference/supernatural/deepsite/TruePositives.csv')
        fp_df1 = pd.read_csv('./inference/supernatural/deepsite/FalsePositives.csv')
        tp_df1 = pd.concat([tp_df, tp_df1],ignore_index=True)
        fp_df1 = pd.concat([fp_df, fp_df1],ignore_index=True)
        tp_df1 = tp_df1.sort_values(by='Pred', ascending=False)
        fp_df1 = fp_df1.sort_values(by='Pred', ascending=True)

    except:
        tp_df1 = tp_df.sort_values(by='Pred', ascending=False)
        fp_df1 = fp_df.sort_values(by='Pred', ascending=True)

    tp_df1.to_csv(f'./inference/supernatural/deepsite/TruePositives.csv',index=None)
    fp_df1.to_csv(f'./inference/supernatural/deepsite/FalsePositives.csv',index=None)
    
