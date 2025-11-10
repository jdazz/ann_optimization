# ANN Optimization

This repository optimizes an Artificial Neural Network (ANN) architecture using Optune to improve predictions using configurable input and output variables.  
It automatically optimizes hyperparameters, evaluates model performance, and saves the best model.

---

## Environment Setup

Install all required dependencies:

```bash
pip install -r requirements.txt

---

## Configuration

Before running the script, **update the `config.yaml` file** to match your data setup:

- Set correct paths for your training and testing files:
```yaml
training_path: "data/training.json"
testing_path: "data/testing.json"

- Define the input features and the output target:
```yaml
input_variables: ['feature1', 'feature2', 'feature3']
output_variables: ['target']