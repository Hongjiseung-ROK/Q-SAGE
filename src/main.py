import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import random

# Add src to path if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import load_smiles, process_molecule
from src.orca_manager import OrcaManager
from src.gnn_model import UncertaintyGCN, mol_to_graph_data

# Configuration
DATA_PATH = "data/raw_smiles.csv"
WORK_DIR = "data/orca_work"
RESULTS_FILE = "data/results.csv"
SEED_SIZE = 5
AL_ITERATIONS = 5
K_SAMPLES = 3  # Molecules to acquire per iteration (Small number for testing)
EPOCHS = 50

class ActiveLearner:
    def __init__(self):
        self.smiles_pool = load_smiles(DATA_PATH)
        self.pool_data = [] # List of {'smiles': s, 'mol': m, 'graph': g, 'y': val, 'labeled': bool}
        self.orca = OrcaManager(WORK_DIR)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = []

    def initialize_pool(self):
        print(f"Initializing pool with {len(self.smiles_pool)} molecules...")
        for i, smi in enumerate(self.smiles_pool):
            print(f"Processing {i+1}/{len(self.smiles_pool)}: {smi}", end='\r')
            mol = process_molecule(smi)
            if mol:
                # Graph data without y for now
                graph = mol_to_graph_data(mol)
                self.pool_data.append({
                    'id': i,
                    'smiles': smi,
                    'mol': mol,
                    'graph': graph,
                    'y': None, # Target: HOMO-LUMO Gap
                    'uncertainty': 0.0,
                    'pred': 0.0,
                    'labeled': False
                })
        print(f"\nPool initialized. Valid molecules: {len(self.pool_data)}")
        
        # Initialize Model
        if len(self.pool_data) > 0:
            num_features = self.pool_data[0]['graph'].num_features
            self.model = UncertaintyGCN(num_features).to(self.device)

    def query_oracle(self, indices):
        """
        Runs ORCA for the specified indices in pool_data using Parallel Processing.
        Updates 'y' and 'labeled'.
        """
        import concurrent.futures
        
        # Helper function for parallel execution
        def run_single_calculation(idx):
            item = self.pool_data[idx]
            if item['labeled']: return None
            
            # Generate Input
            name = f"mol_{item['id']}"
            self.orca.write_input(item['mol'], name)
            
            # Run ORCA
            out_file = self.orca.run_orca(name)
            
            res_gap = None
            if out_file:
                res = self.orca.parse_output(name)
                if res:
                    res_gap = res['gap']
            
            return idx, res_gap

        print(f"Querying Oracle for {len(indices)} molecules (Parallel Execution)...")
        
        # Parallel Execution
        # ORCA itself is parallel via %pal, but usually limited by total cores.
        # If we have 100 small molecules, running 4 at a time (each using 1-2 cores) is efficient.
        MAX_WORKERS = min(4, os.cpu_count() or 1)
        
        newly_labeled_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {executor.submit(run_single_calculation, idx): idx for idx in indices}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, gap = future.result()
                if gap is not None:
                    item = self.pool_data[idx]
                    item['y'] = gap
                    item['labeled'] = True
                    item['graph'].y = torch.tensor([gap], dtype=torch.float)
                    newly_labeled_count += 1
                    print(f"  [Job {idx}] Finished: Gap = {gap:.4f} eV")
                else:
                    print(f"  [Job {idx}] Failed or skipped.")
                
        return newly_labeled_count

    def train_model(self):
        """Trains the GNN on labeled data."""
        labeled_data = [d['graph'] for d in self.pool_data if d['labeled']]
        if not labeled_data:
            print("No labeled data to train on.")
            return 0.0
            
        loader = DataLoader(labeled_data, batch_size=4, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        self.model.train()
        print(f"Training on {len(labeled_data)} samples for {EPOCHS} epochs...")
        
        final_loss = 0.0
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out.view(-1), batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            final_loss = total_loss / len(loader)
            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}: Loss {final_loss:.4f}")
        return final_loss

    def evaluate_pool(self):
        """
        Runs inference on UNLABELED data with MC-Dropout to get Uncertainty.
        """
        unlabeled_indices = [i for i, d in enumerate(self.pool_data) if not d['labeled']]
        
        self.model.train() # Keeping train mode for dropout!
        
        print("Evaluating uncertainty on unlabeled pool...")
        with torch.no_grad():
            for idx in unlabeled_indices:
                data = self.pool_data[idx]['graph'].to(self.device)
                
                # Manual MC Dropout Loop
                preds = []
                for _ in range(20): # 20 forward passes
                    out = self.model(data)
                    preds.append(out.item())
                
                mean_pred = np.mean(preds)
                std_dev = np.std(preds)
                
                self.pool_data[idx]['pred'] = mean_pred
                self.pool_data[idx]['uncertainty'] = std_dev

    def acquisition_step(self, k):
        """
        Selects top k molecules with highest uncertainty.
        """
        unlabeled = [(i, d['uncertainty']) for i, d in enumerate(self.pool_data) if not d['labeled']]
        # Sort by uncertainty descending
        unlabeled.sort(key=lambda x: x[1], reverse=True)
        
        top_k_indices = [x[0] for x in unlabeled[:k]]
        return top_k_indices

    def run(self):
        self.initialize_pool()
        
        # 1. Random Seed
        all_indices = list(range(len(self.pool_data)))
        seed_indices = random.sample(all_indices, min(SEED_SIZE, len(all_indices)))
        
        print("\n--- Phase 4.1: Seed Sampling ---")
        self.query_oracle(seed_indices)
        
        for iteration in range(AL_ITERATIONS):
            print(f"\n--- AL Iteration {iteration + 1}/{AL_ITERATIONS} ---")
            
            # 2. Train
            loss = self.train_model()
            
            # 3. Predict & Uncertainty
            self.evaluate_pool()
            
            # 4. Acquisition
            candidates = self.acquisition_step(K_SAMPLES)
            if not candidates:
                print("No more candidates to label.")
                break
                
            print(f"Selected {len(candidates)} candidates for labeling.")
            
            # 5. Label
            new_count = self.query_oracle(candidates)
            
            # Log stats
            labeled_cnt = sum(1 for d in self.pool_data if d['labeled'])
            print(f"Total Labeled: {labeled_cnt}, Training Loss: {loss:.4f}")
            self.history.append({
                'iteration': iteration,
                'labeled_count': labeled_cnt,
                'newly_labeled': new_count,
                'train_loss': loss
            })

        print("\nActive Learning Loop Complete!")
        # Save results
        pd.DataFrame(self.history).to_csv(RESULTS_FILE, index=False)
        print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    learner = ActiveLearner()
    learner.run()
