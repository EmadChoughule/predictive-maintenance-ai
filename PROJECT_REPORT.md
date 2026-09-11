# CAPSTONE PROJECT REPORT
## Predictive Maintenance AI

### 1. Project Overview
Predictive maintenance uses historical machine and sensor data to estimate whether equipment may fail. This project builds a machine-learning classification system that predicts machine failure risk from temperature, vibration, pressure, RPM, operating hours, load, maintenance history and machine age.

### 2. Problem Statement
Unexpected machine failures can cause downtime, repair costs and production delays. The goal is to build a simple AI system that identifies machines with higher failure risk so maintenance can be planned earlier.

### 3. Objectives
1. Generate/prepare a structured machine sensor dataset.
2. Explore the relationship between sensor readings and failures.
3. Train a machine-learning classification model.
4. Evaluate the model using accuracy, precision, recall and confusion matrix.
5. Create an interactive dashboard for predictions.

### 4. Dataset
The included dataset contains 1,200 synthetic machine records. Each row represents a machine observation.

Features:
- temperature
- vibration
- pressure
- rpm
- operating_hours
- load_percentage
- maintenance_count
- machine_age_years

Target:
- machine_failure: 0 = No Failure, 1 = Failure

### 5. Data Science Methodology
1. Dataset generation
2. Data inspection
3. Train/test split (80/20)
4. Random Forest classification
5. Model evaluation
6. Interactive prediction dashboard

### 6. Algorithm
Random Forest Classifier was selected because it works well for tabular data, can model non-linear relationships, and provides feature importance.

### 7. Expected Result
The system classifies a machine as low or high failure risk and displays the estimated probability. Users can change sensor values in the Streamlit dashboard and immediately obtain a prediction.

### 8. Business Use
A real implementation could help maintenance teams prioritize inspections, reduce unplanned downtime and support preventive maintenance scheduling.

### 9. Limitations
This educational version uses synthetic data. Real industrial deployment would require validated sensor data, domain expertise, monitoring infrastructure and extensive safety validation.

### 10. Future Scope
- Use real IoT sensor streams.
- Add time-series forecasting.
- Send automatic maintenance alerts.
- Add anomaly detection.
- Deploy the dashboard to a cloud platform.
