# ANN Optimization

This repository optimizes an Artificial Neural Network (ANN) architecture using Optuna to improve predictions using configurable input and output variables.  
It automatically optimizes hyperparameters, evaluates model performance, and saves the best model.

---

## Environment Setup

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Before running the script, **update the `config.yaml` file** to match your data setup:

- Set correct paths for your training and testing files:
```yaml
training_path: "data/training.json"
testing_path: "data/testing.json"
```

- Define the input features and the output target:
```yaml
input_variables: ['feature1', 'feature2', 'feature3']
output_variables: ['target']
```

> ATTENTION: Accepted data types: JSON (.json), CSV (.csv), Excel (.xls/.xlsx), Parquet (.parquet), Pandas DataFrame or numpy.ndarray.
> Data must have no missing values. Depending on the dataset size, this program may take a long time to run.
---

## Run the Program

Execute the main script:

```bash
python main.py
```

---

## Output

Model performance metrics printed in the console include:
- Test Accuracy
- NMAE (Normalized Mean Absolute Error)
- R² Score

The best model is saved in: 

```bash
models/ANN_best_model.pt
```

