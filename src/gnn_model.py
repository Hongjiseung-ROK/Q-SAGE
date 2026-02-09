import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

# Feature definitions
ATOM_TYPES = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I']
# Add unknown type for robustness
ATOM_TYPES_MAP = {a: i for i, a in enumerate(ATOM_TYPES)}

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

def mol_to_graph_data(mol, y_val=None):
    """
    Converts RDKit Mol object to PyG Data object.
    Features: Atom Type (One-Hot)
    """
    if mol is None: return None
    
    # Node Features
    x = []
    for atom in mol.GetAtoms():
        # Feature: Atom Type
        sym = atom.GetSymbol()
        feats = []
        # One-hot encoding for atom type
        feats += [int(sym == t) for t in ATOM_TYPES]
        # Append unknown if not in list (though list covers basics)
        if sym not in ATOM_TYPES:
             feats.append(1)
        else:
             feats.append(0)
             
        # Feature: Degree
        feats.append(atom.GetDegree())
        # Feature: Implicit Valence
        feats.append(atom.GetImplicitValence())
        # Feature: Aromaticity
        feats.append(int(atom.GetIsAromatic()))
        
        x.append(feats)
        
    x = torch.tensor(x, dtype=torch.float)
    
    # Edge Index (Adjacency)
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Undirected graph
        edge_index.append([start, end])
        edge_index.append([end, start])
        
    if not edge_index: # Single atom or disconnected
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
    # Target value
    y = torch.tensor([y_val], dtype=torch.float) if y_val is not None else None
    
    return Data(x=x, edge_index=edge_index, y=y)


class UncertaintyGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, heads=4, dropout_rate=0.5):
        super(UncertaintyGCN, self).__init__()
        # GATv2 is even better, but let's stick to standard GAT for stability
        # Multi-head attention
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads)
        # Second layer input dim = hidden * heads
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1) # Final single head
        
        self.dropout_rate = dropout_rate
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        # 1. GAT Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU is often used with GAT
        x = F.dropout(x, p=self.dropout_rate, training=True) 
        
        # 2. GAT Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=True)
        
        # 3. GAT Layer 3
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        
        # 4. Readout
        x = global_mean_pool(x, batch) 

        # 5. Predict
        x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.out(x)
        
        return x

    def predict_with_uncertainty(self, data, num_samples=20):
        """
        Runs the model multiple times with dropout enabled to estimate uncertainty.
        Returns:
            mean_pred: The average prediction.
            std_dev: The standard deviation (uncertainty).
        """
        self.train() # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                out = self(data)
                preds.append(out.item()) # Assuming single graph inference here
        
        preds = np.array(preds)
        return np.mean(preds), np.std(preds)

if __name__ == "__main__":
    # Test
    smiles = "C1=CC=CC=C1"
    mol = Chem.MolFromSmiles(smiles)
    data = mol_to_graph_data(mol)
    
    # Calculate feature size
    num_features = data.x.shape[1]
    
    model = UncertaintyGCN(num_node_features=num_features)
    mean, std = model.predict_with_uncertainty(data)
    
    print(f"Test Molecule: {smiles}")
    print(f"Prediction: {mean:.4f} +/- {std:.4f}")
