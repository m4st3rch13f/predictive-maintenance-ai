# Predictive Maintenance System - Capstone Project
GitHub Repository:
https://github.com/m4st3rch13f/predictive-maintenance-ai

Installation & Setup Instructions:
----------------------------------

1. Clone the repository:
   git clone https://github.com/m4st3rch13f/predictive-maintenance-ai.git

2. Navigate into the folder:
   cd predictive-maintenance-ai

3. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  (Linux/Mac)
   venv\Scripts\activate     (Windows)

4. Install dependencies:
   pip install -r requirements.txt

5. Run the app locally:
   streamlit run app.py

System Requirements:
--------------------
- Python 3.10+
- scikit-learn 1.6.1 (or compatible)
- TensorFlow 2.12+
- Streamlit
- pandas, numpy, matplotlib, seaborn

Required Files:
---------------
Place the following files in the `models/` folder before running the app:
- anomaly_scaler.pkl
- isolation_forest_model.pkl
- dbscan_model.pkl
- lstm_autoencoder_model.keras
- final_rf_classifier.pkl
- rf_label_encoder.pkl
- rf_scaler.pkl
