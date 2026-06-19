# ARAHFL-Medical-Federated-Learning

## Adaptive Reliability-Aware Robust Heterogeneous Federated Learning for Medical Image Analysis

### Project Description

ARAHFL is a federated learning framework designed for medical image classification under heterogeneous client environments. The framework integrates reliability-aware aggregation, robust local training, and dynamic client reliability assessment to improve global model performance while preserving data privacy.

### Features

* Federated Learning Framework
* Reliability-Aware Aggregation
* MixUp Augmentation
* Consistency Regularization
* Multiple CNN Architectures
* Dynamic Reliability Tracking
* PathMNIST Medical Image Dataset Support

### Installation Instructions

1. Clone the repository:

git clone https://github.com/yourusername/ARAHFL-Medical-Federated-Learning.git

2. Create a virtual environment:

python -m venv arahfl_env

3. Activate the environment:

Windows:
arahfl_env\Scripts\activate

4. Install dependencies:

pip install -r requirements.txt

### How to Run the Prototype

1. Activate the virtual environment.
2. Open Jupyter Notebook.
3. Navigate to:

notebooks/prototype_v1.ipynb

4. Run all notebook cells sequentially.

### Folder Structure

ARAHFL-Medical-Federated-Learning/

├── README.md

├── requirements.txt

├── environment_setup.md

├── src/

│ ├── models.py

│ ├── client.py

│ ├── server.py

│ ├── reliability.py

│ └── utils.py

├── notebooks/

│ └── prototype_v1.ipynb

├── data/

├── experiments/

├── results/

└── docs/

### Citation

If you use this work, please cite:

Adaptive Reliability-Aware Robust Heterogeneous Federated Learning for Medical Image Analysis (ARAHFL).
