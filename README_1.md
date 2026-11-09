# Part B – AutoML (open source)

- **Platform AutoML (Steps 1–8):** PyCaret, with every model logged to a local MLflow tracking server → `01_pycaret_automl.ipynb`
- **H2O AutoML (Step 9):** → `02_h2o_automl.ipynb`
- Both notebooks load `~/Downloads/athletes.csv` through `data_prep.py`, which applies the same cleaning and encoding as Part A.

## Setup (macOS, VS Code)

1. **Java (needed by H2O):**
   ```bash
   brew install openjdk@17
   sudo ln -sfn "$(brew --prefix)/opt/openjdk@17/libexec/openjdk.jdk" /Library/Java/JavaVirtualMachines/openjdk-17.jdk
   java -version
   ```
2. **Python 3.11 environment** (from this folder):
   ```bash
   brew install python@3.11
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. In VS Code, open this folder, open a notebook, and choose the `.venv` kernel (top right → Select Kernel).

## Run

1. `01_pycaret_automl.ipynb` – Run All (about 5–15 minutes).
2. `02_h2o_automl.ipynb` – Run All (two 5-minute AutoML runs plus explanations).
3. MLflow UI: `source .venv/bin/activate && mlflow ui`, then open http://127.0.0.1:5000.

Plots, leaderboards and the Part A comparison are saved to `results/`.
