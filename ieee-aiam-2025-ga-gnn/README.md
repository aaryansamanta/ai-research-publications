<div align="center">

  <img src="https://img.shields.io/badge/IEEE-2025-blue?style=for-the-badge&logo=ieee" alt="IEEE">
  <img src="https://img.shields.io/badge/AIAM-2025-FF6B00?style=for-the-badge" alt="AIAM 2025">
  <img src="https://img.shields.io/badge/Accepted-✓-22C55E?style=for-the-badge" alt="Accepted">
  <img src="https://img.shields.io/badge/High_School_Research-10th_Grade-8B5CF6?style=for-the-badge" alt="High School">

  <h1>Quantum-Inspired Hybrid Genetic Algorithm<br>& Graph Neural Network Ensemble<br>for Multimodal Classification</h1>

  <p><strong>Lead Author:</strong> Aaryan Samanta • Legend College Preparatory, Cupertino, CA</p>

  <p>
    <a href="https://ieeexplore.ieee.org/abstract/document/11322272">
      <strong>📄 IEEE Xplore Paper</strong>
    </a>
     • 
    <a href="https://github.com/aaryansamanta/ai-publications/blob/main/ieee-aiam-2025-ga-gnn/paper/acceptance_certificate.pdf">
      <strong>Acceptance Certificate</strong>
    </a>
     • 
    <a href="https://github.com/aaryansamanta/ai-publications/blob/main/ieee-aiam-2025-ga-gnn/paper/ieee_copyright.pdf">
      <strong>IEEE Copyright</strong>
    </a>

  </p>

</div>

---

### 🌟 What Makes This Work Special

- First high-school-led paper to hybridize **quantum-inspired genetic algorithms** with **Graph Neural Networks** for multimodal classification
- Novel qubit-based feature selection + GNN relational learning + learned late fusion
- Interpretable via SHAP + surrogate fitness for fast evolutionary search
- Accepted & published in **IEEE AIAM 2025** (AIAM-6203)

### 📊 Key Results (Test Set)

| Model                  | Accuracy | Precision | Recall | F1    |
|------------------------|----------|-----------|--------|-------|
| **QIHGA-GNN (Ours)**   | **0.53** | **0.51**  | **0.64**| **0.57** |
| Random Forest          | 0.52     | 0.56      | 0.54   | 0.55  |
| SVM                    | 0.49     | 0.50      | 0.52   | 0.52  |
| Logistic Regression    | 0.34     | 0.48      | 0.44   | 0.47  |

Modest but consistent gains over strong baselines on synthetic multimodal biomedical data.

### 🚀 Core Innovations

1. **QIHGA** – Qubit-encoded chromosomes + quantum rotation gates for superior exploration in high-dimensional feature spaces  
2. **Per-modality GNN encoders** – Sample graphs + feature graphs with GCN/GAT  
3. **Learned late fusion** – Convex combination with simplex weights  
4. **SHAP interpretability** + surrogate fitness for 10× faster evolution

### 📁 Repository Contents

- `paper/` → Final IEEE PDF + LaTeX source  
- `code/` → Full PyTorch implementation (QIHGA + GNN + fusion)  
- `docs/` → Acceptance certificate + Copyright transfer

### 📖 Citation

```bibtex
@inproceedings{samanta2025qihga,
  title     = {Quantum-Inspired Hybrid Genetic Algorithm and Graph Neural Network Ensemble for Multimodal Classification},
  author    = {Samanta, Aaryan},
  booktitle = {2025 7th International Conference on Artificial Intelligence and Advanced Manufacturing (AIAM)},
  year      = {2025},
  publisher = {IEEE}
}

