import os
import pickle
import matplotlib.pyplot as plt
from config import Config
from environment import Environment
from evaluation import Evaluator

def redraw_histogram():
    timestamp = "20251218_172404"
    output_dir = os.path.join("results", timestamp)
    pkl_path = os.path.join(output_dir, "results.pkl")
    
    print(f"Loading results from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
        
    config = Config()
    config.n_sessions = 100
    env = Environment(config)
    
    evaluator = Evaluator(config, results, env)
    
    # Redraw histogram with larger fonts (code updated in evaluation.py)
    hist_path = os.path.join(output_dir, "histogram_price.png")
    evaluator.plot_price_histogram(hist_path)
    print("Done. Histogram redrawn.")

if __name__ == "__main__":
    redraw_histogram()

