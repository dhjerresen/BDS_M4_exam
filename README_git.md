# Green Patent Detection – Agentic Workflow

This repository contains the code for my **M4 Advanced AI Systems final project**, which builds a pipeline for detecting **green patent claims** using PatentSBERTa and agent-based labeling.

## Overview

The project implements a full machine learning pipeline:

1. **Dataset construction**
   - Balanced dataset of 50k patent claims derived from CPC Y02 climate technology indicators.

2. **Baseline model**
   - Frozen PatentSBERTa embeddings + Logistic Regression classifier.

3. **Uncertainty sampling**
   - Selects the 100 most uncertain claims from the unlabeled pool.

4. **Agentic labeling pipeline**
   - QLoRA-adapted LLM used in a **Multi-Agent System (MAS)**:
     - Advocate agent
     - Skeptic agent
     - Judge agent

5. **Targeted Human-in-the-Loop (HITL) review**
   - Human review is only applied when agents disagree or confidence is low.

6. **Final model training**
   - PatentSBERTa fine-tuned on **silver labels + gold labels (100 reviewed claims)**.

## Outputs

- Final model:  
  https://huggingface.co/danielhjerresen/BDS_M4_exam_final_model

- Gold dataset (100 reviewed claims):  
  https://huggingface.co/datasets/danielhjerresen/BDS_M4_exam_gold_dataset

## Author

Daniel Hjerresen  
MSc Business Data Science – Final Project