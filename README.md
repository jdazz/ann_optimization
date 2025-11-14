# ANN Optimization

This repository provides a fully automated framework for optimizing an Artificial Neural Network (ANN) using Optuna-based hyperparameter search.
The system is designed to identify the best-performing network configuration for a user-defined regression problem.

The workflow consists of the following stages:

	- 1.	Configurable Inputs & Outputs
You define which variables from your dataset serve as inputs and outputs. The model, search space, and architecture are all derived from a YAML configuration file.

	- 2.	Automated Hyperparameter Optimization
Optuna explores a wide range of hyperparameter combinations—such as learning rate, batch size, number of layers, neurons per layer, and activation functions.
Each combination is trained and evaluated using K-fold cross-validation, ensuring robust and unbiased performance estimation.

	- 3.	Model Selection & Final Training
After evaluating all trials, Optuna selects the best hyperparameter set based on validation loss.
The system then rebuilds the ANN using these optimal hyperparameters and re-trains it on the full training dataset to obtain the final model.

	- 4.	Model Saving & Weight Checkpointing
The best model is saved (architecture + weights) for reproducibility and later inference.

	- 5.	Testing on Unseen Data
The trained model is finally evaluated on a separate unseen dataset, producing metrics such as:
	•	Coefficient of determination (R²)
	•	Normalized Mean Absolute Error (NMAE)
	•	Mean Relative Error (MRE) distribution
	•	Percentage of predictions within a specified error threshold

Together, this pipeline ensures a systematic, reproducible, and well-tuned ANN modeling process, from hyperparameter search to final performance reporting.

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
