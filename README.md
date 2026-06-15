# 🧬 MAGE-CRISPR: Energy-Guided Discrete Diffusion & Dual-Oracle Validation for Allele-Specific Therapeutics (Pharma-Grade)

[![CRISPR-Cas9](https://img.shields.io/badge/CRISPR--Cas9-Therapeutics-crimson?style=for-the-badge&logo=dna&logoColor=white)](https://en.wikipedia.org/wiki/CRISPR)
[![Discrete Diffusion](https://img.shields.io/badge/Model-C--DDM%20(D3PM)-indigo?style=for-the-badge&logo=pytorch&logoColor=white)](#phase-1-generative-inverse-design-via-discrete-diffusion-the-c-ddm-engine)
[![Dual Oracle](https://img.shields.io/badge/Validation-Dual--Oracle%20(Evo2%20%26%20STRAND)-darkgreen?style=for-the-badge)](#phase-3-the-dual-oracle-biological-verification-safety--efficacy)
[![Kaggle GPU](https://img.shields.io/badge/Kaggle-GPU%20Training-blue?style=for-the-badge&logo=kaggle&logoColor=white)](#️-infrastructure--workflow-strategy)

An end-to-end, pharma-grade computational therapeutics pipeline designed for the **de novo generation of structurally viable, allele-specific sgRNAs** targeting autosomal dominant single-nucleotide polymorphisms (SNPs) (e.g., KRAS G12D). The architecture leverages a **Conditional Discrete Denoising Diffusion Probabilistic Model (C-DDM)** guided by real-time thermodynamic gradients, coupled with robust 3D structural filters and a dual-oracle biological validation framework.

---

## 🗺️ System Pipeline Architecture

The workflow translates raw multi-scale genomic context into highly-optimized discrete guide sequences through four distinct operational phases. Unlike traditional post-generation filtering, our model utilizes mathematical force vectors to steer the AI during the generation process itself.

```mermaid
flowchart TD
    subgraph P0["🧬 Phase 0: Conditioning (Multi-Modal Feature Extraction)"]
        direction LR
        In0["Macro-Context (1-Mb DNA)"] ---> E2["Evo 2 Genomic Encoder"]
        In1["Micro-Context (100-bp One-hot)"] ---> CNN["CNN Spatial Motif Extractor"]
        CNN ---> LSTM["BiLSTM Sequential Dependency Extractor"]
        E2 & LSTM ---> Conc["Concatenate Embeddings"]
        Conc ---> Cond["Condition Vector (c)"]
    end

    subgraph P1["🤖 Phase 1: Generative Inverse Design (C-DDM Engine)"]
        direction TB
        Cond & Noise["Uniform Categorical Noise (X_T)"] ---> Markov["Discrete D3PM Denoising Loop (Markov Transition)"]
        Markov ---> CFG["Real-Time Gradient Steering ∇x E"]
        CFG ---> Penalty["Thermodynamic Force Vector<br>(Repels Wild-Type Trajectory)"]
        Penalty ---> Mismatch["Intentional Compensatory Mismatch Engineering"]
        Mismatch ---> sgRNA["Discrete 20-bp sgRNA Candidates"]
    end

    subgraph P2["🌡️ Phase 2: Hierarchical Structural Validation"]
        direction TB
        sgRNA ---> Filter2D["2D Thermodynamic Pre-Filter<br>(ViennaRNA / RNAfold)"]
        Filter2D ---> Hairpins["Check Seed Hairpins & Scaffold Cross-Binding"]
        Hairpins --->|Pass| Folding3D["3D Geometric Filter<br>(RhoFold / AlphaFold 3)"]
        Folding3D ---> Alignment["Structural Alignment & RMSD Score Check"]
        Alignment ---> PDB["PDB Conformation Filter"]
    end

    subgraph P3["🔮 Phase 3: Dual-Oracle Verification (Safety & Efficacy)"]
        direction TB
        PDB ---> OA["Oracle A: Global Safety (Evo 2 40B)<br>- 1-Mb zero-shot cleavage likelihood<br>- Catastrophic off-target proof"]
        PDB ---> OB["Oracle B: Efficacy (STRAND)<br>- Paired Control Verification<br>- Single-cell transcriptomic shifts"]
        OA & OB ---> Out["🏆 Pharma-Grade, Clinically-Ready sgRNA"]
    end

    P0 ---> P1
    P1 ---> P2
    P2 ---> P3

    style P0 fill:#101423,stroke:#3b82f6,stroke-width:2px,color:#f3f4f6
    style P1 fill:#101423,stroke:#10b981,stroke-width:2px,color:#f3f4f6
    style P2 fill:#101423,stroke:#f59e0b,stroke-width:2px,color:#f3f4f6
    style P3 fill:#101423,stroke:#8b5cf6,stroke-width:2px,color:#f3f4f6
```
