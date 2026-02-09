import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os

def load_smiles(filepath):
    """Loads SMILES from a CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return []
    
    df = pd.read_csv(filepath)
    if 'SMILES' not in df.columns:
         # Fallback: assume first column is SMILES if no header matches
         print("Warning: 'SMILES' column not found. Using first column.")
         return df.iloc[:, 0].tolist()
    
    print(f"Loaded {len(df)} SMILES from {filepath}")
    return df['SMILES'].tolist()

def process_molecule(smiles, mol_id=None):
    """
    1. Converts SMILES to Mol
    2. Adds Hydrogens
    3. Generates 3D Conformer (Random Coordinates -> Distance Geometry)
    4. Optimizes Geometry (MMFF94)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to parse SMILES: {smiles}")
        return None

    # Add Hydrogens (Crucial for 3D and ORCA)
    mol = Chem.AddHs(mol)

    # Embed Molecule (Generate 3D coords)
    # useRandomCoords=True can be helpful for difficult geometries, 
    # but standard EmbedMolecule is usually fine for small organics.
    params = AllChem.ETKDGv3()
    embed_stat = AllChem.EmbedMolecule(mol, params)
    
    if embed_stat != 0:
        print(f"Embedding failed for {smiles}, retrying with random coords...")
        embed_stat = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if embed_stat != 0:
            print(f"Structure generation failed for {smiles}")
            return None

    # MMFF94 Optimization
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"MMFF optimization failed for {smiles}: {e}")
        # We might still return the unoptimized 3D structure, but usually optimization is preferred.
        
    return mol

def main():
    # Test run
    data_path = os.path.join(os.path.dirname(__file__), '../data/raw_smiles.csv')
    smiles_list = load_smiles(data_path)
    
    print(f"Processing {len(smiles_list)} molecules...")
    
    valid_mols = []
    for i, smi in enumerate(smiles_list):
        mol = process_molecule(smi)
        if mol:
            valid_mols.append(mol)
            # Optional: Print progress
            # print(f"Prepared {smi} ({i+1}/{len(smiles_list)})")
            
    print(f"Successfully processed {len(valid_mols)}/{len(smiles_list)} molecules.")

if __name__ == "__main__":
    main()
