import torch
import joblib
import pandas as pd
import numpy as np
import os
import sys
import yaml
import shap
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import IODataProcessor
from src.models.gnn import TabGNNRegressor
from src.models.tabular import LightGBMModel, TabGNNTabularModel

def load_config(config_path="configs/experiment7.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nRMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")
    return rmse, mae, r2

def save_top_shap_json(shap_df, top_k=5, output_path="."):
    os.makedirs(output_path, exist_ok=True)
    for i, row in shap_df.iterrows():
        sorted_feats = row.abs().sort_values(ascending=False)
        top_feats = sorted_feats.head(top_k).to_dict()
        json_path = os.path.join(output_path, f"row_{i}_top_features.json")
        with open(json_path, 'w') as f:
            json.dump(top_feats, f, indent=2)

def plot_shap_bar(shap_df, feature_names, out_path, title, filename, top_n=15):
    mean_vals = shap_df[feature_names].mean().sort_values(key=abs, ascending=False).head(top_n)
    colors = ['blue' if v > 0 else 'red' for v in mean_vals]

    plt.figure(figsize=(10, 6))
    plt.barh(mean_vals.index[::-1], mean_vals.values[::-1], color=colors[::-1])
    plt.xlabel("Contribution")
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(out_path, filename))
    plt.close()

def plot_embedding_correlation(embedding_df, raw_df, top_embedding, output_path):
    corr_series = raw_df.corrwith(embedding_df[top_embedding])
    corr_series = corr_series.dropna().sort_values(key=np.abs, ascending=False).head(10)
    colors = ['blue' if v > 0 else 'red' for v in corr_series]

    plt.figure(figsize=(10, 6))
    plt.barh(corr_series.index[::-1], corr_series.values[::-1], color=colors[::-1])
    plt.xlabel("Pearson Correlation")
    plt.title(f"Top I/O Counters Correlated with {top_embedding}")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{top_embedding}_correlation.png"))
    plt.close()

def main():
    config = load_config("configs/experiment7.yml")
    base_output = "logs/training/all/Experiment7/combined"
    shap_dir = os.path.join(base_output, "shap_analysis")
    plot_dir = os.path.join(shap_dir, "plots")

    gnn_model_path = os.path.join(base_output, "tabgnn_part.pt")
    tabular_model_path = os.path.join(base_output, "tabular_part.joblib")

    data_processor = IODataProcessor(
        data_path=config["data_path"],
        important_features=None,
        similarity_thresholds=None,
        precomputed_similarity_path=config["precomputed_similarity_path"]
    )
    data_processor.load_data()
    data_processor.preprocess_data()
    data = data_processor.create_combined_pyg_data(target_column=config["target_column"])
    data = data_processor.train_val_test_split(data, random_state=config["seed"])

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
    checkpoint = torch.load(gnn_model_path, map_location="cpu")
    gnn_model.load_state_dict(checkpoint["model_state_dict"])
    gnn_model.eval()

    tabular_checkpoint = joblib.load(tabular_model_path)
    tabular_model = tabular_checkpoint["model"]
    tabular_model.model_type = "lightgbm"

    combined_model = TabGNNTabularModel(
        gnn_model=gnn_model,
        tabular_model=tabular_model,
        use_original_features=True
    )

    with torch.no_grad():
        output = combined_model.predict(x=data.x, edge_indices=[data.edge_index], batch=None)
        test_mask = data.test_mask
        y_pred = output[test_mask]
        y_true = data.y[test_mask].cpu().numpy()
        X_raw = data.x[test_mask].cpu().numpy()

    df_results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'error': y_pred - y_true})
    evaluate_predictions(y_true, y_pred)

    print("\nPreparing SHAP input using combined GNN + tabular features...")
    X_gnn = combined_model.extract_embeddings(x=data.x, edge_indices=[data.edge_index])[test_mask]
    X_combined = combined_model.combine_features(X_raw, X_gnn)

    explainer = shap.Explainer(tabular_model)
    shap_values = explainer(X_combined)

    raw_features = data_processor.data.columns.tolist()
    raw_features.remove(config["target_column"])
    emb_dim = X_gnn.shape[1]
    emb_features = [f"gnn_emb_{i}" for i in range(emb_dim)]
    all_features = raw_features + emb_features

    shap_df = pd.DataFrame(shap_values.values, columns=all_features)
    full_results = pd.concat([df_results.reset_index(drop=True), shap_df], axis=1)
    os.makedirs(shap_dir, exist_ok=True) 
    full_results.to_csv(os.path.join(shap_dir, "predictions_with_shap.csv"), index=False)

    plot_shap_bar(shap_df, raw_features, plot_dir, "Top SHAP Contributions (Raw Features)", "shap_raw.png")
    plot_shap_bar(shap_df, all_features, plot_dir, "Top SHAP Contributions (All Features)", "shap_all.png")

    # === Find all gnn_emb_* that have non-zero SHAP contribution
    mean_shap_emb = shap_df[emb_features].mean().abs()
    top_shap_features = shap_df[all_features].mean().abs().sort_values(ascending=False).head(15)
    significant_embs = [f for f in top_shap_features.index if f.startswith("gnn_emb_")]

    print("\nEmbeddings with non-zero SHAP contributions:")
    for emb in significant_embs:
        print(f"  {emb} ({mean_shap_emb[emb]:.6f})")

    # === Create plots for each embedding
    emb_df = pd.DataFrame(X_gnn, columns=emb_features)
    raw_df = pd.DataFrame(X_raw, columns=raw_features)
    for emb in significant_embs:
        plot_embedding_correlation(emb_df, raw_df, emb, plot_dir)

    # emb_df = pd.DataFrame(X_gnn, columns=emb_features)
    # raw_df = pd.DataFrame(X_raw, columns=raw_features)

    ## ❗ ⚠️ ❗ This is shap for each row so each job can be analyzed by LLM
    # save_top_shap_json(shap_df, top_k=5, output_path=os.path.join(shap_dir, "llm_json"))

if __name__ == "__main__":
    main()
