#!/usr/bin/env python3
import os
import sys
import yaml
import joblib
import torch
import pandas as pd

# ensure your project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data import IODataProcessor
from src.models.gnn import TabGNNRegressor
from src.models.tabular import TabGNNTabularModel

# === Configurable paths ===
CONFIG_PATH     = "configs/experiment7.yml"
INPUT_CSV       = "/u/mbanisharifdehkordi/Github/IOR_Benchmark/data/" \
                  "darshan_csv_log_L2/darshan_parsed_output_6-29-V5_norm_log_L2.csv"
GNN_CKPT_PATH   = "logs/training/all/Experiment7/combined/tabgnn_part.pt"
TAB_CKPT_PATH   = "logs/training/all/Experiment7/combined/tabular_part.joblib"
OUTPUT_CSV_PATH = "scripts/data_analyze/calibration_V5.csv"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1) load config & ensure output dir
    cfg = load_config(CONFIG_PATH)
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # 2) load your normalized features + tags
    df = pd.read_csv(INPUT_CSV)
    print(f"🔹 Loaded {len(df)} rows from {INPUT_CSV}")

    # 3) build PyG graph + feature matrix
    proc = IODataProcessor(
        data_path=None,
        important_features=None,
        similarity_thresholds=None,
        precomputed_similarity_path=None
    )
    proc.data = df  # override so it uses our pre-normalized DataFrame
    proc.preprocess_data()
    if "test_id" in proc.data.columns:
        proc.data = proc.data.drop(columns=["test_id"])
    data = proc.create_combined_pyg_data(target_column=cfg["target_column"])

    # 4) instantiate & load GNN
    gnn = TabGNNRegressor(
        in_channels      = data.x.shape[1],
        hidden_channels  = cfg["hidden_dim"],
        gnn_out_channels = cfg["hidden_dim"],
        mlp_hidden_channels = [cfg["hidden_dim"], cfg["hidden_dim"] // 2],
        num_layers       = cfg["num_layers"],
        num_graph_types  = 1,
        model_type       = "gcn",
        dropout          = cfg["dropout"],
    )
    ckpt = torch.load(GNN_CKPT_PATH, map_location="cpu")
    gnn.load_state_dict(ckpt["model_state_dict"])
    gnn.eval()

    # 5) load tabular model and patch model_type
    tab_ckpt = joblib.load(TAB_CKPT_PATH)
    tab_model = tab_ckpt["model"]
    # TabGNNTabularModel expects tabular_model.model_type to exist:
    tab_model.model_type = "lightgbm"

    # 6) build ensemble
    ensemble = TabGNNTabularModel(
        gnn_model               = gnn,
        tabular_model           = tab_model,
        use_original_features   = True,
    )

    # 7) predict
    with torch.no_grad():
        raw_pred = ensemble.predict(
            x            = data.x,
            edge_indices = [data.edge_index],
            batch        = None
        )
    # ensure numpy array
    y_pred = raw_pred.cpu().numpy().flatten() if hasattr(raw_pred, "cpu") else raw_pred.flatten()
    y_true = data.y.cpu().numpy().flatten()

    # 8) save for calibration
    out = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    out.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Saved calibration pairs to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
