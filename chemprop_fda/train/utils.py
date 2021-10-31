import pandas as pd

def save_clas(smiles, pred, tar, rows):
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
    tn_rows = []
    fn_rows = []

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
        else:
            if t == 1:
                fn_rows.append([pred[i][0]] + list(rows[i].values()))
            else:
                tn_rows.append([pred[i][0]] + list(rows[i].values()))

    cols = ['Pred'] + list(rows[0].keys())

    # tp_df = pd.DataFrame(list(zip(tp_smile,tp_pred,tp_tar)), columns=['Smiles','Pred','Target'])
    # fp_df = pd.DataFrame(list(zip(fp_smile,fp_pred,fp_tar)), columns=['Smiles','Pred','Target'])
    print(len(tp_rows), len(tp_rows[0]))

    tp_df = pd.DataFrame(tp_rows, columns=cols)
    fp_df = pd.DataFrame(fp_rows, columns=cols)
    tn_df = pd.DataFrame(tn_rows, columns=cols)
    fn_df = pd.DataFrame(fn_rows, columns=cols)

    try:
        tp_df1 = pd.read_csv('./inference/fda/deepsite/TruePositives.csv')
        fp_df1 = pd.read_csv('./inference/fda/deepsite/FalsePositives.csv')
        tn_df1 = pd.read_csv('./inference/fda/deepsite/TrueNegatives.csv')
        fn_df1 = pd.read_csv('./inference/fda/deepsite/FalseNegatives.csv')

        tp_df1 = pd.concat([tp_df, tp_df1],ignore_index=True)
        fp_df1 = pd.concat([fp_df, fp_df1],ignore_index=True)
        tn_df1 = pd.concat([tn_df, tn_df1],ignore_index=True)
        fn_df1 = pd.concat([fn_df, fn_df1],ignore_index=True)

        tp_df1 = tp_df1.sort_values(by='Pred', ascending=False)
        fp_df1 = fp_df1.sort_values(by='Pred', ascending=True)
        tn_df1 = tn_df1.sort_values(by='Pred', ascending=False)
        fn_df1 = fn_df1.sort_values(by='Pred', ascending=True)

    except:
        tp_df1 = tp_df.sort_values(by='Pred', ascending=False)
        fp_df1 = fp_df.sort_values(by='Pred', ascending=True)
        tn_df1 = tn_df.sort_values(by='Pred', ascending=False)
        fn_df1 = fn_df.sort_values(by='Pred', ascending=True)

    tp_df1.to_csv(f'./inference/fda/deepsite/TruePositives.csv',index=None)
    fp_df1.to_csv(f'./inference/fda/deepsite/FalsePositives.csv',index=None)
    tn_df1.to_csv(f'./inference/fda/deepsite/TrueNegatives.csv',index=None)
    fn_df1.to_csv(f'./inference/fda/deepsite/FalseNegatives.csv',index=None)

def save_reg(smiles, pred, tar, rows):
    '''
    smiles : List of SMILES representations of drugs
    pred : List of predicted outputs
    tar : List of target outputs
    rows : List of row features
    '''
    assert len(smiles) == len(pred) == len(tar) == len(rows)
    
    # binarized_preds = [[1.0] if x[0] > 0.5 else [0.0] for x in pred]
    
    # tp_smile = []
    # tp_pred = []
    # tp_tar = []
    final_rows = []

    # fp_smile = []
    # fp_pred = []
    # fp_tar = []
    # fp_rows = []

    for i in range(len(smiles)):
        # p = binarized_preds[i][0]
        # t = tar[i][0]

        # if p == 1:
            # Corresponding to True Positives
            # if t == 1:
                # tp_smile.append(smiles[i])
                # tp_pred.append(pred[i][0])
                # tp_tar.append(tar[i][0])
        final_rows.append([pred[i][0]] + list(rows[i].values()))

        
    cols = ['Pred'] + list(rows[0].keys())

    # tp_df = pd.DataFrame(list(zip(tp_smile,tp_pred,tp_tar)), columns=['Smiles','Pred','Target'])
    # fp_df = pd.DataFrame(list(zip(fp_smile,fp_pred,fp_tar)), columns=['Smiles','Pred','Target'])
    # print(len(tp_rows), len(tp_rows[0]))

    # tp_df = pd.DataFrame(tp_rows, columns=cols)
    # fp_df = pd.DataFrame(fp_rows, columns=cols)

    final_df = pd.DataFrame(final_rows, columns=cols)

    try:
        final_df1 = pd.read_csv('./inference/fda/deepsite/PredictionsReg.csv')
        final_df1 = pd.concat([final_df, final_df1],ignore_index=True)
        final_df1 = final_df1.sort_values(by='Pred', ascending=True)

    except:
        final_df1 = final_df.sort_values(by='Pred', ascending=True)
        
    final_df1.to_csv(f'./inference/fda/deepsite/PredictionsReg.csv',index=None)
    
