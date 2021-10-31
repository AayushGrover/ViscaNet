from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from chemprop_fda.data import MoleculeDataLoader, MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            val: bool = False,
            fold_num: int = -1,
            model_idx: int = -1) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data_loader: A MoleculeDataLoader.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    smiles = []
    rows = []

    for batch in tqdm(data_loader, disable=disable_progress_bar):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()
        smile_batch = batch.smiles()
        row = batch.rows()

        # Visualize attention weights
        if not val:
            encoder = model.encoder
            encoder.viz_attention(mol_batch, features_batch, fold_num, model_idx)

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # if np.isnan(np.sum(batch_preds)):
            # print(torch.isnan(torch.sum(mol_batch)))
            # print(torch.isnan(torch.sum(features_batch)))

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
        smiles.extend(smile_batch)
        rows.extend(row)

    return preds, smiles, rows
