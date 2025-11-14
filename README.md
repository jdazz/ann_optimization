# ANN Optimization

This repository provides a fully automated framework for optimizing an Artificial Neural Network (ANN) using Optuna-based hyperparameter search.
The system is designed to identify the best-performing network configuration for user-defined regression tasks.


Workflow Overview

1. Configurable Inputs & Outputs

You can define which dataset columns act as input features and target variables.
All model settings—including architecture and search space—are controlled via a config.yaml file.


2. Automated Hyperparameter Optimization

Optuna searches through a wide range of hyperparameters, such as:
	•	Learning rate
	•	Batch size
	•	Number of hidden layers
	•	Neurons per layer
	•	Activation functions
	•	Training epochs

Each trial is trained and validated using K-fold cross-validation, ensuring robust and unbiased model evaluation.


3. Model Selection & Final Training

After all trials finish:
	•	Optuna selects the best-performing hyperparameter set based on validation loss.
	•	The ANN is rebuilt using these optimal hyperparameters.
	•	The final model is re-trained on the full training dataset to maximize predictive performance.


4. Model Saving & Checkpointing

The best model (architecture + weights) is saved for:
	•	Reproducibility
	•	Deployment
	•	Future inference


5. Testing on Unseen Data

The final model is evaluated on a separate test dataset.
The system reports:
	•	R² Score
	•	Normalized Mean Absolute Error (NMAE)
	•	Mean Relative Error (MRE) distribution
	•	Accuracy threshold metrics (e.g., percentage of predictions within X% error)


Summary

This pipeline provides a systematic, reproducible, and fully automated approach to ANN training and optimization—from hyperparameter search to final performance reporting.

---

## Environment Setup

To install all requirements run:

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

> ATTENTION: Accepted data types: JSON (.json), CSV (.csv), Excel (.xls/.xlsx) and Parquet (.parquet).
> Data must have no missing values.
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

---

## Web App integration

This project includes a web interface built with Streamlit that allows users to upload their training and testing datasets, as well as a configuration file, directly through a browser. The web app automates the ANN optimization process, trains the model, evaluates performance on unseen data, and displays the results including test accuracy, NMAE, R² score, and the model architecture. This provides an accessible, user-friendly way to run the ANN optimization without needing to interact directly with the code. You can find this Web App at this location : ???

```bash
streamlit run app.py
```

---

## Acknowledgements

This program was developped at the Institute for Dynamic Systems and Control at ETHZ, ???
