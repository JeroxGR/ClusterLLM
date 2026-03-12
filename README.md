# ClusterLLM: Large Language Models as a Guide for Text Clustering

This repository contains the implementation of **ClusterLLM**, based on the paper [ClusterLLM: Large Language Models as a Guide for Text Clustering (EMNLP 2023)](https://arxiv.org/abs/2305.14871) by Yuwei Zhang, Zihan Wang, and Jingbo Shang.

## Overview

ClusterLLM is a novel framework that leverages Large Language Models (LLMs) to guide and improve text clustering. Traditional text clustering methods struggle with two fundamental challenges:

1. **Perspective**: How to obtain good text representations (embeddings) that capture the right semantic relationships for a given clustering task.
2. **Granularity**: How to determine the optimal number of clusters.

ClusterLLM addresses both challenges by using LLMs as a "teacher" to provide supervision signals, while keeping costs low through a **Proxy Teacher** strategy that routes easy examples to a fast CrossEncoder and only sends hard examples to the LLM.

## Repository Structure

```
.
├── README.md                  # This file
├── Proposed/                  # Our proposed solution with modifications and improvements
│   ├── perspective/           # Perspective module (embedding improvement)
│   │   ├── 1_predict_triplet/ # Triplet sampling and LLM-based prediction
│   │   └── 2_finetune/        # Contrastive fine-tuning of embeddings
│   ├── granularity/           # Granularity module (cluster number prediction)
│   │   ├── hierarchy/         # Custom hierarchical clustering implementation
│   │   └── scripts/           # Shell scripts for running experiments
│   ├── image/                 # Architecture diagrams
│   └── requirements.txt       # Python dependencies
├── Reproduction/              # Reproduction of the original paper's experiments
│   ├── perspective/           # Same structure as Proposed
│   └── granularity/           # Same structure as Proposed
└── .gitignore
```

## How It Works

### High-Level Architecture

```
Raw Text Data + Initial Embeddings
          |
          |--------------------------------------+
          v                                      v
   +------------------+                 +------------------+
   |   PERSPECTIVE     |                 |   GRANULARITY     |
   |  (Better Embeds)  |                 | (Cluster Count)   |
   +--------+---------+                 +--------+---------+
            |                                     |
   1. Sample hard triplets               1. Build dendrogram
   2. LLM predicts similarity            2. Sample pairs at
   3. Fine-tune embeddings                  multiple levels
      with contrastive loss              3. LLM predicts if
            |                               pairs belong together
            v                            4. F-beta score to find
   Improved Embeddings                      optimal cluster count
                                                  |
                                                  v
                                         Predicted K (number
                                          of clusters)
```

### Module 1: Perspective -- Learning Better Embeddings

The Perspective module improves text embeddings through LLM-guided triplet learning:

1. **Triplet Sampling** (`perspective/1_predict_triplet/triplet_sampling.py`):
   - Performs initial clustering on existing embeddings.
   - For each data point (the "query"), selects two comparison texts from different clusters.
   - Focuses on **hard examples** -- cases where the query sits near the boundary between clusters (measured by entropy of cluster assignment probabilities).

2. **Triplet Prediction** (`perspective/1_predict_triplet/predict_triplet.py`):
   - Uses the **Proxy Teacher** strategy to minimize LLM API costs:
     - **Easy samples** (low entropy): Predicted by a fast CrossEncoder model (~80% of samples).
     - **Hard samples** (high entropy): Sent to the LLM (e.g., GPT-3.5-turbo) for prediction (~20% of samples).
   - The LLM is asked: *"Which choice is the query more similar to?"*

3. **Embedding Fine-tuning** (`perspective/2_finetune/finetune.py`):
   - Uses the LLM-annotated triplets to fine-tune an embedding model (INSTRUCTOR) with contrastive loss.
   - The loss pushes the query embedding closer to the LLM-chosen positive and farther from the negative.
   - Uses task-specific temperature scaling (e.g., 0.08 for domain clustering, 0.01 for intent clustering).

4. **Embedding Generation** (`perspective/2_finetune/get_embedding.py`):
   - Generates improved embeddings using the fine-tuned model.
   - Evaluates clustering quality with standard metrics (NMI, ARI, ACC) when labels are available.

### Module 2: Granularity -- Predicting the Number of Clusters

The Granularity module determines the optimal number of clusters:

1. **Pair Sampling** (`granularity/sample_pairs.py`):
   - Builds a hierarchical clustering dendrogram over the data.
   - At each merge step in the dendrogram, samples pairs of texts from the two clusters being merged.
   - Creates labeled pairs across multiple granularity levels (e.g., from 2 clusters to 60 clusters).
   - Labels each pair as "Yes" (same cluster) or "No" (different clusters).

2. **Pair Prediction** (`granularity/predict_pairs.py`):
   - Again uses the **Proxy Teacher** strategy:
     - **Confident predictions** (CrossEncoder score > 0.8): Use the CrossEncoder directly.
     - **Uncertain predictions** (score between 0.5 and 0.8): Route to the LLM.
   - The LLM is asked: *"Do these two sentences belong to the same category?"*

3. **Cluster Number Prediction** (`granularity/predict_num_clusters.py`):
   - For each candidate number of clusters K, evaluates how well the LLM predictions match the hierarchical structure.
   - Uses the F-beta score (beta = 0.92) to balance precision and recall.
   - The K with the highest F-beta score is selected as the predicted optimal number of clusters.

### Key Innovation: Proxy Teacher Strategy

The most important design choice in ClusterLLM is the **Proxy Teacher** strategy, which dramatically reduces LLM API costs:

- A lightweight CrossEncoder model handles the majority of easy examples where the answer is clear from the embeddings alone.
- The expensive LLM is only called for genuinely ambiguous cases where embedding-based methods are uncertain.
- This results in ~80% cost reduction while maintaining high-quality supervision on the examples that matter most.

## Proposed Solution (Proposed/ Folder)

The `Proposed/` folder contains our team's modified and improved version of the ClusterLLM pipeline. Key aspects of the proposed solution include:

- **Enhanced Hierarchical Clustering**: Custom Cython-optimized hierarchical clustering (`hierarchy/`) based on Ward's method with efficient distance update rules, providing faster agglomerative clustering for large datasets.
- **Multi-Granularity Pair Sampling**: Extended pair sampling across a wider range of cluster numbers (configurable `min_clusters` to `max_clusters`) to better capture the clustering structure.
- **Improved Proxy Teacher Routing**: Refined thresholds for CrossEncoder confidence-based routing to optimize the trade-off between cost and prediction quality.
- **E5 Embedding Support**: Added support for E5 embeddings (`finetune_e5.py`, `get_embedding_e5.py`) as an alternative to the INSTRUCTOR embedding model.
- **Self-Training Pipeline**: Includes a self-training variant (`convert_triplet_self.py`) where the model's own predictions are used to generate additional training data.

## Installation

```bash
pip install -r Proposed/requirements.txt
```

### Dependencies

- Python 3.8+
- PyTorch
- Transformers 4.20.0
- Sentence-Transformers >= 2.2.0
- scikit-learn 1.2.0
- OpenAI API (for LLM predictions)
- h5py 3.8.0 (for embedding storage)

### Datasets

Download the datasets from [Google Drive](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view?usp=sharing) and unzip into the project directory.

## Usage

### Running the Perspective Pipeline

```bash
# Step 1: Generate initial embeddings
cd Proposed/perspective/2_finetune
bash scripts/get_embedding.sh

# Step 2: Sample triplets
cd Proposed/perspective/1_predict_triplet
bash scripts/triplet_sampling.sh

# Step 3: Predict triplets using LLM
# (Edit scripts/predict_triplet.sh to add your OpenAI API key first)
bash scripts/predict_triplet.sh

# Step 4: Convert triplet format
cd Proposed/perspective/2_finetune
bash scripts/convert_triplet.sh
bash scripts/convert_triplet_self.sh

# Step 5: Fine-tune embeddings
bash scripts/finetune.sh

# Step 6: Generate improved embeddings
bash scripts/get_embedding.sh  # Point to finetuned checkpoint
```

### Running the Granularity Pipeline

```bash
# Step 1: Sample pairs at multiple granularities
cd Proposed/granularity
bash scripts/sample_pairs.sh

# Step 2 (optional): Sample pairs for in-context examples
bash scripts/sample_pairs_for_prompt.sh

# Step 3: Predict pairs using LLM
# (Edit scripts/predict_pairs.sh to add your OpenAI API key first)
bash scripts/predict_pairs.sh

# Step 4: Predict optimal number of clusters
bash scripts/predict_num_clusters.sh
```

## Citation

```bibtex
@misc{zhang2023clusterllm,
      title={ClusterLLM: Large Language Models as a Guide for Text Clustering},
      author={Yuwei Zhang and Zihan Wang and Jingbo Shang},
      year={2023},
      eprint={2305.14871},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgments

- Original paper and codebase by Yuwei Zhang, Zihan Wang, and Jingbo Shang (UC San Diego).
- Some code adapted from [instructor-embedding](https://github.com/xlang-ai/instructor-embedding).

## Contact

For questions about this repository, please open an issue.
