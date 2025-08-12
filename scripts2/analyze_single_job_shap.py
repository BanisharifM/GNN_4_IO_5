import torch
import joblib
import pandas as pd
import numpy as np
import os
import shap
import json
import yaml
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import IODataProcessor
from src.models.gnn import TabGNNRegressor
from src.models.tabular import TabGNNTabularModel

# === Configuration ===
CONFIG_PATH = "configs/experiment7.yml"
SINGLE_JOB_PATH = "data/IOR/Test/outpu_1.csv"
GNN_MODEL_PATH = "logs/training/all/Experiment7/combined/tabgnn_part.pt"
TABULAR_MODEL_PATH = "logs/training/all/Experiment7/combined/tabular_part.joblib"
OUTPUT_DIR = "logs/shap_single_job"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = load_config(CONFIG_PATH)

    # Load and preprocess the single-row CSV
    data_processor = IODataProcessor(
        data_path=SINGLE_JOB_PATH,
        important_features=None,
        similarity_thresholds=None,
        precomputed_similarity_path=None
    )
    # Step 1: Load raw data
    data_processor.load_data()

    # Step 2: Apply log10 + L2 normalization
    print("🔹 Applying log10 + L2 normalization...")
    df = data_processor.data
    features = df.drop(columns=[config["target_column"]])
    
    # Apply log10(x + 1) transformation
    features_log = np.log10(features + 1.0)
    
    # Convert to tensor and normalize
    data_tensor = torch.tensor(features_log.values, dtype=torch.float32)
    normalized_tensor = torch.nn.functional.normalize(data_tensor, p=2, dim=1)
    
    # Reconstruct DataFrame
    normalized_df = pd.DataFrame(normalized_tensor.numpy(), columns=features.columns)
    normalized_df[config["target_column"]] = df[config["target_column"]].values
    data_processor.data = normalized_df

    # Step 3: Continue preprocessing
    data_processor.preprocess_data()
    data = data_processor.create_combined_pyg_data(target_column=config["target_column"])

    # Load trained GNN model
    gnn_model = TabGNNRegressor(
        in_channels=data.x.shape[1],
        hidden_channels=config["hidden_dim"],
        gnn_out_channels=config["hidden_dim"],
        mlp_hidden_channels=[config["hidden_dim"], config["hidden_dim"] // 2],
        num_layers=config["num_layers"],
        num_graph_types=1,
        model_type="gcn",
        dropout=config["dropout"]
    )
    checkpoint = torch.load(GNN_MODEL_PATH, map_location="cpu")
    gnn_model.load_state_dict(checkpoint["model_state_dict"])
    gnn_model.eval()

    # Load trained tabular model
    tabular_checkpoint = joblib.load(TABULAR_MODEL_PATH)
    tabular_model = tabular_checkpoint["model"]
    tabular_model.model_type = "lightgbm"

    # Create combined model
    combined_model = TabGNNTabularModel(
        gnn_model=gnn_model,
        tabular_model=tabular_model,
        use_original_features=True
    )

    # Predict tag
    with torch.no_grad():
        pred = combined_model.predict(x=data.x, edge_indices=[data.edge_index], batch=None)
        y_pred = pred[0].item()
        y_true = data.y[0].item()
        error = y_pred - y_true

    print(f"\nPredicted tag: {y_pred:.4f} | Ground truth: {y_true:.4f} | Error: {error:.4f}")

    # Prepare SHAP input
    X_raw = data.x.cpu().numpy()
    X_gnn = combined_model.extract_embeddings(data.x, [data.edge_index])
    X_combined = combined_model.combine_features(X_raw, X_gnn)

    # Run SHAP
    explainer = shap.Explainer(tabular_model)
    shap_values = explainer(X_combined)

    # Feature names
    raw_features = data_processor.data.drop(columns=[config["target_column"]]).columns.tolist()
    emb_features = [f"gnn_emb_{i}" for i in range(X_gnn.shape[1])]
    all_features = raw_features + emb_features

    # Format SHAP output in DataFrame style
    shap_dict = dict(zip(all_features, shap_values.values[0].tolist()))
    shap_row = pd.DataFrame([{
        "y_true": y_true,
        "y_pred": y_pred,
        "error": error,
        **{k: float(v) for k, v in shap_dict.items()}
    }])

    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, "single_prediction_with_shap_log10_l2.csv")
    shap_row.to_csv(out_path, index=False)
    print(f"\n✅ Saved SHAP CSV: {out_path}")

if __name__ == "__main__":
    main()
