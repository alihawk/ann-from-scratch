# Artificial Neural Networks from Scratch - Classification and Regression

This project implements a fully-connected Artificial Neural Network (ANN) from scratch using NumPy, supporting classification, regression, gradient checking, and real-world competition predictions.

---

## 📁 Project Structure

- `nn.py` — Main ANN model code.
- `competition.ipynb` — Training and submitting for FTIR tissue classification.
- `test_nn.py` — Unit tests for verifying ANN.
- `datasets/` — Contains small datasets: `doughnut.tab`, `squares.tab`.
- `submission.npy` — Example FTIR competition submission.
- `doughnut_loss.svg` — Training loss curve for visualization.

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/alihawk/ann-from-scratch.git
cd ann-from-scratch


Install the required Python packages:

pip install -r requirements.txt


How to Run
1. Train Neural Networks

python nn.py --units 8,4 --activations relu,tanh --lambda 0.01 --epochs 3000


2. Run Unit Tests

python test_nn.py


3. Competition Task

jupyter notebook competition.ipynb
