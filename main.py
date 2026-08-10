import os, sys
from config import DATA_RAW_DIR, DATA_PROC_DIR, MODEL_DIR

def run_pipeline():
    print("\n" + "="*55)
    print("   AI-Based Log Analysis & Threat Detection Pipeline")
    print("="*55 + "\n")

    from src.data_generator  import run as run_data
    run_data()

    from src.preprocess      import run as run_preprocess
    run_preprocess()

    from src.features        import run as run_features
    run_features()

    from src.model           import run as run_model
    run_model()

    from src.threat_detector import run as run_threats
    run_threats()

   
    from src.alerts          import run as run_alerts
    run_alerts()

if __name__ == "__main__":
    run_pipeline()