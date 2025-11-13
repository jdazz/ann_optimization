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

This project includes a web interface built with Streamlit that allows users to upload their training and testing datasets, as well as a configuration file, directly through a browser. The web app automates the ANN optimization process, trains the model, evaluates performance on unseen data, and displays the results including test accuracy, NMAE, R² score, and the model architecture. If no config.yaml is uploaded, the app automatically uses the default config.yaml from the project directory. This provides an accessible, user-friendly way to run the ANN optimization without needing to interact directly with the code. You can find this Web App at this location : ???

```bash
streamlit run app.py
```

---

## Acknowledgements

This program was developped at the Institute for Dynamic Systems and Control at ETHZ, ???
