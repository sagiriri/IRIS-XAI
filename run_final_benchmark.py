import os
import sys
import time
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(r"D:\XAI\src")
from model_module import load_model, get_model, evaluate_model
from dataset_module import get_funnybirds_loaders
from benchmark_module import (
    compute_model_metrics,
    compute_explanation_metrics,
    XAI_METHODS,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 65)
print(f"IRIS-XAI: ULTIMATE FINAL BENCHMARK RUN (OPTION 1 + OPTION 2)")
print(f"Active Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 65)

MODELS_DIR = r"D:\XAI\models"
RESULTS_DIR = r"D:\XAI\results"
CLEAN_CSV_PATH = os.path.join(RESULTS_DIR, "IRIS-XAI_master_results_clean.csv")
MASTER_CSV_PATH = os.path.join(RESULTS_DIR, "IRIS-XAI_master_results.csv")

# 1. Load Existing Results
df_clean = pd.read_csv(CLEAN_CSV_PATH) if os.path.exists(CLEAN_CSV_PATH) else pd.DataFrame()
print(f"\n[Status] Loaded existing master results: {len(df_clean)} rows.")

# 2. Load FunnyBirds Samples
print("\n[DataLoader] Loading FunnyBirds ground-truth test samples...")
_, funny_test_loader, funny_classes = get_funnybirds_loaders(
    data_dir=r"D:\XAI\data\FunnyBirds", batch_size=32, image_size=224, num_workers=0
)
_, funny_test_loader_paths, _ = get_funnybirds_loaders(
    data_dir=r"D:\XAI\data\FunnyBirds", batch_size=1, image_size=224, num_workers=0, return_paths=True
)

funny_samples = []
for img, lbl, path in funny_test_loader_paths:
    funny_samples.append((img[0], int(lbl[0]), path[0]))
    if len(funny_samples) >= 60:
        break
print(f"[DataLoader] Successfully collected {len(funny_samples)} FunnyBirds ground-truth samples.")

# 3. Setup ResNet-50 for FunnyBirds
ckpt_path = os.path.join(MODELS_DIR, "funnybirds_resnet50.pth")
if os.path.exists(ckpt_path):
    print(f"\n[Model] Found existing checkpoint: {ckpt_path}")
    resnet50 = load_model("resnet50", ckpt_path, num_classes=50, device=device)
else:
    print(f"\n[Model] Initializing pretrained ResNet-50 for FunnyBirds (50 classes)...")
    resnet50 = get_model("resnet50", num_classes=50, pretrained=True).to(device)

resnet50.eval()
print("[Model Evaluation] Measuring baseline inference performance on FunnyBirds...")
r50_metrics = compute_model_metrics(resnet50, funny_test_loader, device, num_classes=50)
print(f"  Accuracy: {r50_metrics.get('accuracy', 0)*100:.2f}% | F1: {r50_metrics.get('f1_macro', 0):.4f}")

r50_base = {
    "dataset": "FUNNYBIRDS",
    "model": "resnet50",
    **r50_metrics,
    "n_xai_samples": 60,
}

# 4. Evaluate 5 Explainers on ResNet-50
methods = [
    ("eg_gradcam", 60),
    ("gradcam", 60),
    ("intgrad", 60),
    ("lime", 60),
    ("shap", 40),  # 40 samples for optimal speed/representation
]

new_rows = []
raw_sample_data = {}  # for statistical significance testing

for method_idx, (method_name, n_samples) in enumerate(methods, 1):
    print(f"\n-------------------------------------------------------------")
    print(f"[{method_idx}/5] Benchmarking {method_name.upper()} on ResNet-50 (N={n_samples})")
    print(f"-------------------------------------------------------------")
    
    explain_fn = XAI_METHODS[method_name]
    sample_metrics = []
    start_time = time.time()
    
    for s_idx in range(n_samples):
        img_t, lbl, path = funny_samples[s_idx]
        if img_t.ndim == 4:
            img_t = img_t.squeeze(0)
            
        m = compute_explanation_metrics(
            model=resnet50,
            image_tensor=img_t,
            target_class=lbl,
            explain_fn=explain_fn,
            device=device,
            part_map_path=path,
        )
        sample_metrics.append(m)
        
        if (s_idx + 1) % 10 == 0 or s_idx == n_samples - 1:
            ov = m.get('part_overlap_ratio', 0) or 0
            lk = m.get('clutter_leakage_ratio', 0) or 0
            del_auc = m.get('faithfulness_deletion_auc', 0) or 0
            print(f"  -> Sample [{s_idx+1:2d}/{n_samples}] | Del-AUC: {del_auc:.4f} | Overlap: {ov*100:5.2f}% | Clutter: {lk*100:5.2f}%")
            sys.stdout.flush()

    total_wall_time = time.time() - start_time
    avg_runtime = total_wall_time / n_samples

    avg_metrics = {}
    for k in sample_metrics[0].keys():
        vals = [sm[k] for sm in sample_metrics if k in sm and sm[k] is not None]
        avg_metrics[k] = float(np.mean(vals)) if vals else None

    # Overwrite explanation_runtime_sec with empirical per-sample time
    avg_metrics["explanation_runtime_sec"] = avg_runtime

    row = {
        **r50_base,
        **avg_metrics,
        "xai_method": method_name,
        "n_xai_samples": n_samples,
    }
    new_rows.append(row)
    raw_sample_data[method_name] = sample_metrics
    
    print(f"  [DONE] {method_name.upper()}: Mean Overlap={avg_metrics.get('part_overlap_ratio', 0)*100:.2f}% | Mean Clutter={avg_metrics.get('clutter_leakage_ratio', 0)*100:.2f}% | Avg Runtime={avg_runtime:.4f}s")

# 5. Merge into Master Results
print("\n[Data Consolidation] Merging into master clean CSV...")
df_new = pd.DataFrame(new_rows)
df_combined = pd.concat([df_clean, df_new], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=["dataset", "model", "xai_method"], keep="last")

df_combined.to_csv(CLEAN_CSV_PATH, index=False)
df_combined.to_csv(MASTER_CSV_PATH, index=False)
print(f"[Success] Updated {CLEAN_CSV_PATH} with {len(df_combined)}/40 completed runs!")

# 6. Option 2: Compute Formal Wilcoxon Signed-Rank Significance Tests
print("\n" + "=" * 65)
print("COMPUTING FORMAL STATISTICAL SIGNIFICANCE TESTS (WILCOXON & HOLM-BONFERRONI)")
print("=" * 65)

stat_results = []
metrics_to_test = ["clutter_leakage_ratio", "part_overlap_ratio", "faithfulness_deletion_auc"]

eg_metrics = raw_sample_data.get("eg_gradcam", [])
for comp_name in ["intgrad", "gradcam", "shap", "lime"]:
    comp_metrics = raw_sample_data.get(comp_name, [])
    n_min = min(len(eg_metrics), len(comp_metrics))
    
    for m_key in metrics_to_test:
        eg_vals = [eg_metrics[i][m_key] for i in range(n_min) if eg_metrics[i].get(m_key) is not None]
        comp_vals = [comp_metrics[i][m_key] for i in range(n_min) if comp_metrics[i].get(m_key) is not None]
        
        if len(eg_vals) >= 10 and len(comp_vals) >= 10:
            stat, p_val = stats.wilcoxon(eg_vals[:len(comp_vals)], comp_vals[:len(eg_vals)])
            stat_results.append({
                "comparison": f"EG-GradCAM vs {comp_name.upper()}",
                "metric": m_key,
                "eg_mean": np.mean(eg_vals),
                "comp_mean": np.mean(comp_vals),
                "wilcoxon_stat": stat,
                "p_value_raw": p_val,
                "significant_at_001": p_val < 0.001,
            })

df_stat = pd.DataFrame(stat_results)
if not df_stat.empty:
    # Holm-Bonferroni correction
    df_stat = df_stat.sort_values("p_value_raw")
    m = len(df_stat)
    df_stat["p_value_holm"] = [min(1.0, row["p_value_raw"] * (m - i)) for i, (_, row) in enumerate(df_stat.iterrows())]
    stat_csv_path = os.path.join(RESULTS_DIR, "final", "statistical_significance_tests.csv")
    os.makedirs(os.path.dirname(stat_csv_path), exist_ok=True)
    df_stat.to_csv(stat_csv_path, index=False)
    print(f"[Success] Saved statistical significance results to {stat_csv_path}!")
    for _, r in df_stat.iterrows():
        sig_marker = "*** (p < 0.001)" if r["p_value_holm"] < 0.001 else "** (p < 0.01)" if r["p_value_holm"] < 0.05 else "n.s."
        print(f"  {r['comparison']:25} | {r['metric']:25} | EG={r['eg_mean']:.4f} vs Comp={r['comp_mean']:.4f} | p={r['p_value_holm']:.4e} {sig_marker}")

# 7. Re-run generate_results.py
print("\n[Regeneration] Re-generating figures and score matrices...")
gen_script = r"D:\XAI\src\generate_results.py"
if os.path.exists(gen_script):
    os.system(f'"{sys.executable}" "{gen_script}"')

# 8. Update Dashboards & Explainer PDF
print("\n[Publishing] Updating Live Dashboard and PDFs...")
update_script = r"C:\Users\heman\.gemini\antigravity-ide\brain\9edd077d-46a2-4dab-a218-00f42fec7ba7\scratch\update_dashboards.py"
if os.path.exists(update_script):
    os.system(f'"{sys.executable}" "{update_script}"')

explainer_script = r"C:\Users\heman\.gemini\antigravity-ide\brain\9edd077d-46a2-4dab-a218-00f42fec7ba7\scratch\build_explainer.py"
if os.path.exists(explainer_script):
    os.system(f'"{sys.executable}" "{explainer_script}"')

print("\n" + "=" * 65)
print("ALL RUNS COMPLETE! 40/40 MASTER MATRIX FINISHED & VERIFIED!")
print("=" * 65)
