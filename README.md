# Project ALCHEMIST: Uncertainty-Aware Active Learning for Molecular Property Discovery

## Abstract
**Project ALCHEMIST** is an autonomous research agent designed to accelerate the discovery of molecular properties by integrating **Graph Neural Networks (GNNs)** with **Quantum Chemical Calculations (DFT)**. By employing an **Active Learning** strategy based on epistemic uncertainty, the system autonomously selects and labels only the most informative molecules from a large candidate pool. This approach significantly reduces the computational cost associated with high-fidelity simulations (ORCA) while maintaining high predictive accuracy.

---

## 1. Methodology

### 1.1 Architecture Overview
The system operates as a closed-loop agent consisting of three core modules:
1.  **Surrogate Model (AI Brain)**: A **Graph Attention Network (GAT)** that predicts molecular properties and estimates uncertainty using **MC-Dropout**.
2.  **Oracle (Experiment)**: A parallelized interface for **ORCA (Quantum Chemistry Package)** that performs Density Functional Theory (DFT) calculations on demand.
3.  **Active Learning Strategy**: An acquisition function that selects top-$k$ candidates with the highest predictive variance ($\sigma^2$) for labeling.

### 1.2 Uncertainty Quantification
To enable the agent to "know what it doesn't know," we implement **Monte Carlo Dropout** during inference. The uncertainty $\sigma$ for a molecule $x$ is approximated by the variance of $T$ stochastic forward passes:
$$ \sigma^2(x) \approx \frac{1}{T} \sum_{t=1}^{T} (\hat{y}_t - \bar{y})^2 $$
where $\hat{y}_t$ is the prediction of the $t$-th stochastic pass.

### 1.3 Parallel Oracle Execution
Molecular labeling is the computational bottleneck. We mitigate this by implementing a **Parallel Oracle** using `ThreadPoolExecutor`, allowing multiple DFT calculations to run concurrently on available CPU cores (`src/main.py`).

---

## 2. Repository Structure
The repository is organized to separate source code, data, and analysis scripts:

```
project_root/
├── src/                # Source Code
│   ├── build_env.py    # Environment Verification
│   ├── preprocess.py   # RDKit Pre-processing (3D Conformer Gen)
│   ├── gnn_model.py    # GAT Model Architecture
│   ├── orca_manager.py # ORCA Automation Interface
│   └── main.py         # Active Learning Loop Entry Point
├── data/               # Datasets
│   ├── raw_smiles.csv  # Candidate Pool
│   └── results.csv     # Training Logs
├── analysis/           # Visualization Tools
│   └── plot_efficiency.py
└── README.md           # Documentation
```

---

## 3. Usage

### 3.1 Prerequisites
Ensure the environment is configured with `rdkit`, `torch`, and `cclib`.
```bash
python src/build_env.py
```

### 3.2 Data Preparation
Generate 3D conformers for the candidate molecules.
```bash
python src/preprocess.py
```

### 3.3 Execution
Launch the autonomous agent. The system will iteratively select molecules, run ORCA calculations, and retrain the model.
```bash
python src/main.py
```

### 3.4 Analysis
 visualize the learning efficiency and model convergence.
```bash
python analysis/plot_efficiency.py
```

---

## 4. Results
The system demonstrates successful autonomous learning behavior:
- **Efficiency**: The agent rapidly reduces prediction error by prioritizing high-uncertainty samples.
- **Robustness**: The automation pipeline includes rigorous checks for SCF convergence, ensuring data integrity.
- **Scalability**: Parallel execution allows for high-throughput screening on local workstations.

![Learning Curve](analysis/learning_curve.png)

---

**Author**: AI Agent & User (Computational Chemistry Team)
**Version**: 1.0.0
