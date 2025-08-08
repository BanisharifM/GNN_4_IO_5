#!/usr/bin/env python3
import os
import sys
import yaml
import joblib
import torch
import shap
import pandas as pd
import numpy as np

# make sure your project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import IODataProcessor
from src.models.gnn import TabGNNRegressor
from src.models.tabular import TabGNNTabularModel

# === Configurable paths ===
CONFIG_PATH         = "configs/experiment7.yml"
INPUT_CSV           = "/u/mbanisharifdehkordi/Github/IOR_Benchmark/data/darshan_csv_log_L2/darshan_parsed_output_6-29-V5_norm_log_L2.csv"
GNN_MODEL_PATH      = "logs/training/all/Experiment7/combined/tabgnn_part.pt"
TAB_MODEL_PATH      = "logs/training/all/Experiment7/combined/tabular_part.joblib"
CALIB_COEFFS_CSV    = "scripts/data_analyze/calibration_coeffs_V5.csv"
OUTPUT_CSV_DIR      = "/u/mbanisharifdehkordi/Github/GNN_4_IO_4/shap"
OUTPUT_CSV_PATH     = os.path.join(OUTPUT_CSV_DIR, "darshan_parsed_output_6-29-V5_norm_log_scaled_with_shap_calib.csv")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    cfg = load_config(CONFIG_PATH)

    # 1) Load your fully–normalized (log10+L2) CSV
    df = pd.read_csv(INPUT_CSV)
    print(f"🔹 Loaded {len(df)} rows from {INPUT_CSV}")

    # Extract and temporarily hold test_id if present
    if "test_id" in df.columns:
        test_id_column = df["test_id"].copy()
        df = df.drop(columns=["test_id"])
    else:
        test_id_column = None

    # 1a) Load calibration coefficients
    cb = pd.read_csv(CALIB_COEFFS_CSV).iloc[0]
    a, b = cb["a"], cb["b"]
    print(f"🔹 Calibration: y_calib = {a:.6f}·y_raw + {b:.6f}")

    # 2) Create and run the data processor
    dp = IODataProcessor(
        data_path=None,
        important_features=None,
        similarity_thresholds=None,
        precomputed_similarity_path=None
    )
    dp.data = df
    dp.preprocess_data()
    data = dp.create_combined_pyg_data(target_column=cfg["target_column"])

    # 3) Load your trained GNN
    gnn = TabGNNRegressor(
        in_channels     = data.x.shape[1],
        hidden_channels = cfg["hidden_dim"],
        gnn_out_channels= cfg["hidden_dim"],
        mlp_hidden_channels=[cfg["hidden_dim"], cfg["hidden_dim"]//2],
        num_layers      = cfg["num_layers"],
        num_graph_types = 1,
        model_type      = "gcn",
        dropout         = cfg["dropout"]
    )
    ckpt = torch.load(GNN_MODEL_PATH, map_location="cpu")
    gnn.load_state_dict(ckpt["model_state_dict"])
    gnn.eval()

    # 4) Load your tabular model
    tab_ckpt = joblib.load(TAB_MODEL_PATH)
    tab_model = tab_ckpt["model"]
    tab_model.model_type = "lightgbm"

    # 5) Combine into the full ensemble
    combined = TabGNNTabularModel(
        gnn_model             = gnn,
        tabular_model         = tab_model,
        use_original_features = True
    )

    # 6) Make predictions for all rows
    with torch.no_grad():
        raw_pred = combined.predict(
            x=data.x,
            edge_indices=[data.edge_index],
            batch=None
        )
    if isinstance(raw_pred, torch.Tensor):
        y_pred_raw = raw_pred.cpu().numpy().flatten()
    else:
        y_pred_raw = raw_pred.flatten()

    # 7) Apply calibration
    y_pred_calib = a * y_pred_raw + b

    # 8) True labels and errors
    y_true = data.y.cpu().numpy().flatten()
    err_calib = y_pred_calib - y_true

    print("🔹 Predictions (raw & calibrated) done.")

    # 9) Build the SHAP explainer once on the combined-tabular side
    X_raw = data.x.cpu().numpy()
    X_emb = combined.extract_embeddings(data.x, [data.edge_index])
    X_all = combined.combine_features(X_raw, X_emb)

    print("🔹 Computing SHAP values...")
    explainer   = shap.Explainer(tab_model)
    shap_values = explainer(X_all)  # shape = (n_rows, n_features)

    # 10) Assemble results
    raw_feats = dp.data.drop(columns=[cfg["target_column"]]).columns.tolist()
    emb_feats = [f"gnn_emb_{i}" for i in range(X_emb.shape[1])]
    all_feats = raw_feats + emb_feats

    rows = []
    for i in range(len(df)):
        row = {
            "y_true":    float(y_true[i]),
            "y_pred":    float(y_pred_calib[i]),
            "error":     float(err_calib[i]),
        }
        # add SHAP for this sample
        for j, feat in enumerate(all_feats):
            row[feat] = float(shap_values.values[i, j])
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Add back test_id column if it was present
    if test_id_column is not None:
        out_df["test_id"] = test_id_column.values

    out_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Saved SHAP + calibrated predictions to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
