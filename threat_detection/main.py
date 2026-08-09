import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_RAW_DIR, DATA_PROC_DIR, MODEL_DIR

def create_directories():
    dirs = [DATA_RAW_DIR, DATA_PROC_DIR, MODEL_DIR, "notebooks", "tests"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[✔] Project directories verified / created.")

def run_pipeline():
    print("\n" + "="*55)
    print("   AI-Based Log Analysis & Threat Detection Pipeline")
    print("="*55 + "\n")

    # from src.data_generator  import run as run_data
    # run_data()

    # from src.preprocess      import run as run_preprocess
    # run_preprocess()

    # from src.features        import run as run_features
    # run_features()

    # from src.model           import run as run_model
    # run_model()

    # from src.threat_detector import run as run_threats
    # run_threats()

    # from dashboard.app import   ← launched separately: streamlit run
    # ── Phase 7 ───────────────────────────────────────────────
    from src.alerts          import run as run_alerts
    run_alerts()

    print("\n" + "="*55)
    print("  ✅ Full pipeline complete!")
    print("  👉 Run: streamlit run dashboard/app.py")
    print("="*55 + "\n")
if __name__ == "__main__":
    run_pipeline()