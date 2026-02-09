import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(results_file="data/results.csv"):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return

    df = pd.read_csv(results_file)
    
    # Set style
    plt.style.use('bmh') # Clean, professional style
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Training Loss vs Iteration
    ax1 = axes[0]
    ax1.plot(df['iteration'], df['train_loss'], 'o-', color='tab:red', linewidth=2, markersize=8)
    ax1.set_title('Model Convergence\n(Loss vs Iteration)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('AL Iteration')
    ax1.set_ylabel('Training Loss (MSE)')
    ax1.grid(True, alpha=0.5)
    
    # Panel 2: Data Acquisition
    ax2 = axes[1]
    bars = ax2.bar(df['iteration'], df['labeled_count'], color='tab:blue', alpha=0.7, label='Total Labeled')
    ax2.plot(df['iteration'], df['labeled_count'], 's--', color='tab:blue', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
                
    ax2.set_title('Data Acquisition Rate\n(Labeled Count vs Iteration)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('AL Iteration')
    ax2.set_ylabel('Total Labeled Molecules')
    ax2.grid(True, alpha=0.5, axis='y')
    
    # Panel 3: Sample Efficiency (The "True" Learning Curve)
    ax3 = axes[2]
    ax3.plot(df['labeled_count'], df['train_loss'], 'D-', color='tab:purple', linewidth=2, markersize=8)
    ax3.set_title('Sample Efficiency\n(Loss vs Dataset Size)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Labeled Molecules')
    ax3.set_ylabel('Training Loss (MSE)')
    ax3.invert_xaxis() # Optional: sometimes people like to see x axis increasing. Let's keep it normal.
    # Actually standard is x increasing.
    # Let's annotate the iteration numbers
    for i, txt in enumerate(df['iteration']):
        ax3.annotate(f"Iter {txt}", (df['labeled_count'][i], df['train_loss'][i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.grid(True, alpha=0.5)

    plt.suptitle('Project ALCHEMIST: Active Learning Performance Report', fontsize=16, y=1.05)
    plt.tight_layout()
    
    output_path = "analysis/learning_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced plot saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    plot_results()
