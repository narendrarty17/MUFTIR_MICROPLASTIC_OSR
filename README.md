
# µFTIR Microplastic Open-Set Recognition (OSR)

### Contrastive Embedding + OSR Baselines

## Project Overview
This project implements an end-to-end replication-style pipeline inspired by:

Smolen et al., “Adaptable microplastic classification using similarity learning on µFTIR spectra”, PNAS (2025)

The goal is to:
1. Learn embedding representations of µFTIR spectra using contrastive learning
2. Perform open-set recognition (OSR) to:
3. accept known polymers (closed set)
4. reject unknown polymers, degraded samples, and non-plastics
5. Analyze failure modes, especially under stress conditions
6. This README documents everything completed up to Step 9 (baseline OSR + analysis).

## Dataset Summary
1. Input format: HDF5 µFTIR mapping files
2. Preprocessed format: .npz files (one per physical sample)
3. Spectral resolution:
    a. Original: 882 wavenumbers
    b. After zero-padding: 896 points

4. Each .npz file contains:
    a. spectra: shape (N, 896) — normalized spectra
    b. polymer: string (e.g. "PE")
    c. condition: string (clean, fibers, UV_deg, etc.)
    d. wavenumbers: shape (882,) (reference axis)

## Important design principle
    Labels (polymer, condition) are file-level metadata, not spectrum-level.
    Each spectrum inherits its label from the file it came from.

## Project Structure (Key Directories)

    MUFTIR_MICROPLASTIC_OSR/
    │
    ├── data/
    │   ├── raw/                          # Original HDF5 files
    │   ├── standardized/
    │   │   ├── spectra_npz/              # Extracted spectra (Step 3)
    │   │   ├── spectra_npz_norm/         # Normalized + padded spectra (Step 4)
    │   │   └── embeddings_npz/           # Learned embeddings (Step 8)
    │   └── splits/
    │       └── split_definition.json     # Sample-wise splits (Step 5)
    │
    ├── src/
    │   ├── audit/                        # Dataset inspection utilities
    │   ├── preprocessing/               # Extraction, normalization, splitting
    │   ├── datasets/                    # PyTorch Dataset
    │   ├── models/                      # CNN + embedding network
    │   ├── losses/                      # Contrastive loss
    │   ├── training/                    # Training loops
    │   ├── evaluation/                  # OSR methods
    │   └── analysis/                    # Visualization & stress analysis
    │
    └── results/
        └── figures/                      # UMAPs, boxplots, etc.


## Step-by-Step Pipeline Documentation

### Step 0 - Dataset Audit & Mapping
1. Goal: Understand dataset composition and experimental roles.
2. Scripts: src/audit/inspect_hdf5_structure.py
3. Key outcomes
    a. Verified
        x. Key outcomes
        y. Each HDF5 file contains many spectra (mapping data)
    b. Categorized Samples into:- 
        x. Clean (Big-11 polymers)
        y. Open-set polymers
        z. Stress tests (UV, thermal, high background, fibers)
        xx. Non-plastics


### Step 1 - Project Structure Creation
1. Goal: Create a clean, reproducible directory layout.
2. Scripts: Manual + OS-based directory creation
3. Outcome: Stable folder structure used consistently throughout the project

### Step 2 - HDF5 Inspection
1. Goal: Confirm spectral shape, datatype, and consistency across files.
2. Findings
    a. Each dataset inside HDF5 = one spectrum
    b. Shape (882,), dtype float64
    c. Wavenumbers shared across spectra

### Step 3 - Spectra Extraction
1. Goal: Convert raw HDF5 files into structured .npz files.
2. Script: src/preprocessing/extract_spectra.py
3. Output: data/standardized/spectra_npz/*.npz
4. Each file corresponds to one physical sample, containing:
    a. all spectra from that sample
    b. polymer and condition metadata

### Step 4 - Normalization and Zero Padding 
1. Goal: Prepare spectra for CNN input.
2. Operations: 
    a. Min–max normalization to [0, 1]
    b. Zero-padding from 882 → 896 points
3. Script: src/preprocessing/normalize_and_pad.py
3. Output: data/standardized/spectra_npz_norm/*.npz

### Step 5 - Sample-wise Dataset Splitting 
1. Goal: Prevent data leakage by splitting by sample, not by spectrum.
2. Script: src/preprocessing/define_splits.py 
3. Design
    a. Splits are file-wise, not spectrum-wise
    b. Polymers in Big-11 distributed across:
        x. train
        y. validation
        z. test_closed_set
    c. Seperate splits for: 
        x. open_set
        y. stress_tests
        z. non_plastic
4. Output: data/splits/split_definition.json

### Step 6 - Embedding Model

#### Step 6.1 - Dataset & DataLoader
1. Script: src/datasets/spectral_dataset.py
2. Key design: Each __getitem__ returns:
    a. x: one spectrum (896,)
    b. y: label (if enabled)
    c. sample_id: string identifier (critical for OSR)

#### Step 6.2 - CNN + ANN Embedding Network
1. Script: src/models/embedding_net.py
2. Model: 
    a. 1D CNN feature extractor
    b. Fully connected projection head
    c. Output embedding dimension: 128

#### Step 6.3 - Contrastive Training Loop
1. Scripts:
    a. src/training/train_embedding.py
    b. src/losses/contrastive_loss.py
2. Loss: 
    a. Contrastive loss with positive and negative pairs
    b. Polymer-balanced batch sampling (to avoid degenerate batches)

#### Step 7 - Closed-Set Classifier (Sanity Check)
1. Goal: Verify embeddings are meaningful.
2. Results: Closed-set test accuracy ≈ 0.87
This establishes that embeddings encode polymer information.

#### Step 8 - Open-Set Recognition (OSR)
#### Step 8.1 - Embedding Extraction
1. Goal: Freeze the network and extract embeddings for all splits.
2. Script: src/evaluation/extract_embeddings.py
3. Critical Fix: Ensured sample_id is preserved as a string, not a numeric value
4. Output: data/standardized/embeddings_npz/*.npz
5. Each File Contains:- 
    a. embeddings: (N, 128)
    b. sample_id
    c. split

#### Step 8.2 - Distance-to-Centroid OSR (Baseline)
1. Script: src/evaluation/osr_distance_baseline.py
2. Method: 
    a. Compute centroids per polymer (train set)
    b. Reject samples beyond a global distance threshold (95th percentile)
3. Observed behavior
    a. Moderate open-set rejection
    b. Poor stress-test rejection
    c. Conservative but limited OSR

#### Step 8.3 - One-Class SVM OSR
1. Script: src/evaluation/osr_oneclass_svm.py
2. Method: 
    a. One OC-SVM per polymer
    b. Accept if any polymer model accepts
3. Key Insight
    a. OC-SVM exposed embedding overlap
    b. Performed worse than centroid OSR due to weak separation

#### Step 9 - Embedding Visualization & Stress-Test Analysis
#### Step 9.1 - UMAP Visualization
1. Script: src/analysis/visualize_embeddings.py
2. Outputs: 
    a. results/figures/umap_by_polymer.png
    b. results/figures/umap_by_split.png
3. Revealed
    a. Significant inter-polymer overlap
    b. Stress samples largely remain within clean clusters

#### Step 9.2 (Extension) - Visualization
1. Script: src/analysis/plot_stress_distances.py
2. Outputs: results/figures/stress_distance_boxplot.png
3. Key insight
    a. Some polymers (PP, PMMA, SA-ABS) are highly stress-sensitive
    b. Others (EVA, PVC) show limited drift
    c. Drift often remains within OSR acceptance regions

#### Key Scientific Conclusions (So Far)
1. Contrastive embeddings provide moderate class separation
2. OSR performance is representation-limited, not threshold-limited
3. Stress degradation does not reliably push spectra out of known regions
4. Decision models (centroids, OC-SVM) cannot compensate for weak embeddings
This justifies moving to Step 10: improving similarity learning.

#### What Is NOT Done Yet (By Design)
1. Triplet loss / hard negative mining
2. Adaptive or hierarchical similarity learning
3. Paper-specific adaptive boundary methods
4. Final performance claims

# Next Step
### Step 10 - Improving embeddings (beyond contrastive loss)