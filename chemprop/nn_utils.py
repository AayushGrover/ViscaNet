import math
from typing import List, Union
import os
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps

from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.features import MolGraph

def smiles2id():
    smiles = 's_sd_SMILES'
    db_id = 's_sd_DRUGBANK\\_ID'
    
    try:    
        df = pd.read_csv('./data/nsp1_deep_fda_htvs_sift.tsv', sep='\t', header=1)
        df_smiles = df[smiles].tolist()
        df_db = df[db_id].tolist()
        smiles2db = dict(zip(df_smiles, df_db))
        return smiles2db
    except:
        return None
    
def smiles2sn():
    
    try:    
        df = pd.read_csv('./data/nsp1_supernaturaldb_sift_data/sn_id_smiles.tsv', sep='\t')
        sn_id = df.id.tolist()
        smiles = df.smiles.tolist()
        smiles2id = dict(zip(smiles, sn_id))
        return smiles2id
    except:
        return None

def visualize_atom_attention(viz_dir: str,
                             mol_graph: MolGraph,
                             num_atoms: int,
                             attention_weights: torch.FloatTensor,
                             fold_num: int,
                             model_idx: int):
    """
    Saves figures of attention maps between atoms. Note: works on a single molecule, not in batch
    :param viz_dir: Directory in which to save attention map figures.
    :param smiles: Smiles string for molecule.
    :param num_atoms: The number of atoms in this molecule.
    :param attention_weights: A num_atoms x num_atoms PyTorch FloatTensor containing attention weights.
    """
    # mol = Chem.MolFromSmiles(smiles)
    if viz_dir is None:
        return
    mol = mol_graph.mol
    smiles = mol_graph.smiles

    smiles_viz_dir = os.path.join(viz_dir, f'Fold-{fold_num},Model-{model_idx}')
    os.makedirs(smiles_viz_dir, exist_ok=True)
    atomSum_weights=np.zeros(num_atoms)
    for a in range(num_atoms):
        a_weights = attention_weights[a].cpu().data.numpy()
        atomSum_weights+=a_weights
    Amean_weight=atomSum_weights/num_atoms
    nanMean=np.nanmean(Amean_weight)

    smiles2db = smiles2id()

    interest = ['DB01603', 'DB00355']

    if smiles2db[smiles] in interest:

        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                            Amean_weight-nanMean,
                                                            colorMap='Blues',
                                                            contourLines=10)
        
        # for atom, val in zip(mol.GetAtoms(), (Amean_weight-nanMean)):
        #     print(atom.GetAtomicNum(), val)
        
        if smiles2db is not None:
                save_path = os.path.join(smiles_viz_dir, f'{smiles2db[smiles]}.png')
                fig.savefig(save_path, bbox_inches='tight')

        else:
            save_path = os.path.join(smiles_viz_dir, f'{smiles}.png')
            
        plt.close(fig)
    

def compute_pnorm(model: nn.Module) -> float:
    """Computes the norm of the parameters of a model."""
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """Computes the norm of the gradients of a model."""
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


def compute_molecule_vectors(model: nn.Module,
                             data: MoleculeDataset,
                             batch_size: int,
                             num_workers: int = 8) -> List[np.ndarray]:
    """
    Computes the molecule vectors output from the last layer of a MoleculeModel.

    :param model: A MoleculeModel.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param num_workers: Number of parallel data loading workers.
    :return: A list of 1D numpy arrays of length hidden_size containing
    the molecule vectors generated by the model for each molecule provided.
    """
    training = model.training
    model.eval()
    data_loader = MoleculeDataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    vecs = []
    for batch in tqdm(data_loader, total=len(data_loader)):
        # Apply model to batch
        with torch.no_grad():
            batch_vecs = model.featurize(batch.mols(), batch.features())

        # Collect vectors
        vecs.extend(batch_vecs.data.cpu().numpy())

    if training:
        model.train()

    return vecs


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]
