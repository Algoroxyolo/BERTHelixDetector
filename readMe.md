## Transmembrane Helix Detection

### Overview

This section of the code focuses on detecting transmembrane helices (TMHs) within protein sequences. Transmembrane helices are essential structural elements of membrane proteins, and accurate detection of these regions is crucial for understanding their function and structure.

### Contents

1. **Code Files**:
    - `CRF.py` and `bert-seg_tag.py`: Python script for training and evaluating a model for TMH detection.
    - `README.md`: Documentation file providing an overview of the TMH detection process and instructions for usage.

2. **Datasets**:
    - The training and testing datasets are provided in separate text files (`IO.txt` and `test-noi.txt`, respectively).

3. **Dependencies**:
    - The code relies on several Python libraries, including transformers, torch, datasets, and torchcrf. Ensure that these libraries are installed in your environment before running the code.

### Instructions


1. **Training and Evaluation**:
    - Run the `bert-seg_tag.py` script to train and evaluate the TMH detection model using BERT.
    - Run the `CRF.py` script to train and evaluate the TMH detection model using CRF-BERT approch.
    - The script includes instructions for specifying model parameters, training epochs, and evaluation strategies.

2. **Output**:
    - After training and evaluation, the script generates evaluation results, including precision, recall, F1 score, and Qok.

### Note

- Ensure that the input protein sequences are properly formatted and provided in the training and testing datasets.
- Fine-tuning the model parameters and adjusting the training strategy may improve the accuracy of TMH detection.
