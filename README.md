# ANN Optimization Dashboard (Streamlit)

Interactive Streamlit app for hyperparameter optimization of ANN regressors using Optuna. Upload data, tune the search space, run trials, and download trained models from the browser.

---

## Features
- Web UI for loading tabular data, selecting features/targets, and setting train/test handling (split slider or separate test file).
- Optuna-powered search for ANN hyperparameters with live metrics (CV loss, R², NMAE).
- Parity plots and Optuna visualizations for completed runs.
- One-click downloads for models, predictions, and zipped run artifacts.
- Runs and config persisted to disk so you can reload and keep progress.

## Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Run the app
```bash
streamlit run app.py
```
Open the URL shown in the terminal (usually http://localhost:8501).

## Using the dashboard
1) **Upload data**
   - Upload a training file (`csv`, `xlsx`, or `json`).
   - Optional: click **“Upload test data (optional)”** to provide a separate test file; otherwise use the train/test split slider.
2) **Select columns**  
   Choose feature columns (X) and a target column (Y) from the detected headers.
3) **Configure training**  
   Adjust trials, learning rate/epochs, batch size, hidden layers/neurons, activation functions, optimizer, and CV settings. Enable feature standardization if needed.
4) **Run + monitor**  
   Start the run from the control panel and watch live metrics/logs update.
5) **Review results**  
   Inspect best metrics, parity plot, and Optuna charts, then download artifacts directly from the UI.

## Configuration
- Working config lives in `config.yaml`; UI changes persist automatically and seed the next reload.
- Key sections:
  - `variables`: feature/target names.
  - `cross_validation`: split ratio, k-folds, and standardization toggle.
  - `hyperparameter_search_space` and `network`: bounds/choices for the Optuna search.
- You can edit `config.yaml` manually before launching if you prefer.

## Outputs & folders
- `runs/<dataset>__<status>`: per-run folders with best model (`.pt`), predictions CSV, metrics, logs, and a zipped archive.
- `optuna_study.db`: Optuna study storage (SQLite).
- `temp_data/`: staging area for uploads (ignored by git).


## Acknowledgements
Developed by the Institute for Dynamic Systems and Control at ETH Zürich.

## License
MIT. See `LICENSE` for details.
