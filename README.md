# Generative Inverse Design of Allele-Specific sgRNAs for Autosomal Dominant Disorders

An end-to-end computational therapeutics framework leveraging a **Conditional Variational Autoencoder (C-VAE)** to solve the inverse problem of CRISPR-Cas9 single-guide RNA (sgRNA) engineering.

---

## 🗺️ System Pipeline Architecture

Our framework coordinates discrete biological constraints across four distinct operational phases, actively bridging macro-genomic foundation layers with deep generative modeling:

```text
[Phase 0: Context Extraction] ──> [Phase 1: Inverse Design] ──> [Phase 2: Physics Filter] ──> [Phase 3: Biological Oracle]
 (Evo 2 Foundation Embeds)        (C-VAE Generates sgRNA)        (ViennaRNA Fold Verification)   (STRAND + sgDesigner Verification)
 Phase 0 (Feature Extraction): Extracts deep genomic and chromatin context embeddings from upstream foundation models (such as Evo 2).

Phase 1 (Inverse Design): Uses a specialized Hybrid CNN-BiLSTM C-VAE to read context condition vectors and mathematically generate de novo 20-bp single-guide RNA sequences.

Phase 2 (Geometric Physics Filtering): Assesses minimum free energy (MFE) structural constraints (inspired by gRNAde) to verify stable hairpin formations required for Cas9 assembly.

Phase 3 (Biological Oracle Validation): Routes generated candidate structures through predictive validation systems (inspired by STRAND and sgDesigner) to score single-cell transcriptomic shifts, off-target liabilities, and strict allele discrimination indices.

⚙️ Infrastructure & Workflow Strategy
To maximize reproducibility and hardware utilization, this project splits code orchestration and compute execution into a hybrid workflow:

Local Environment: Repository management, documentation, data schemas, and pipeline layout.

Compute Environment (Kaggle Accelerator): Large-scale structural training of the C-VAE and multi-oracle evaluation loops using free cloud-hosted NVIDIA T4 GPUs.

Running Training on Kaggle:
Pull code files from this repository directly into a Kaggle Notebook environment using the GitHub integration or via:
!git clone https://github.com/capybarawombat/lhp_sef_sgRNA.git

Upload target genomic datasets as a zipped Kaggle Dataset asset.

Run the training script to export optimized checkpoints (.pth) back into cloud storage.

📚 Core Literature Matrix & Reference Hub
The theoretical foundations of this project are directly built upon, contrasted with, and validated by the following peer-reviewed works:

1. STRAND: Sequence-Conditioned Transport for Single-Cell Perturbations
Authors: Fu et al. (2026)

Paper Link: arXiv:2602.10156v1

Project Utility: The Forward Evaluation Oracle. STRAND implements sequence-conditioned optimal transport to map baseline cell state distributions to perturbed distributions.

2. gRNAde: Geometric Deep Learning for 3D RNA Inverse Design
Authors: Joshi et al. (2024)

Paper Link: PMC11142113

Project Utility: Structural Featurization & Multi-State Logic. We utilize gRNAde’s multi-state conformation constraints to ensure our generated sgRNAs maintain stable backbone structural integrity.

3. sgDesigner: Generalizable sgRNA Design for Improved CRISPR/Cas9 Editing
Authors: Hiranniramol et al. (2020)

Paper Link: Bioinformatics (Vol. 36, Issue 9)

Project Utility: Benchmarking Framework & Validation Design. We mirror their robust validation methodology by generating 1,000 completely random, shuffled negative control sequences to establish an experimental baseline.
