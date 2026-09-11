# Predictive Maintenance AI

A beginner-friendly Data Science capstone project that predicts whether an industrial machine is likely to fail soon using sensor and operating-condition data.

## Objective
Predict `machine_failure` (0 = No Failure, 1 = Failure) from:
- temperature
- vibration
- pressure
- rotational speed
- operating hours
- load percentage
- maintenance history
- machine age

## Technology
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit.

## Project Structure
```text
predictive_maintenance_ai/
├── app.py
├── train_model.py
├── generate_dataset.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
└── data/
    └── predictive_maintenance.csv
```

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The Streamlit dashboard allows a user to enter machine sensor readings and receive a predicted failure risk.

## Important note
The included dataset is a synthetic educational dataset created for this project. It is intended for demonstration and learning, not for real industrial safety decisions.
