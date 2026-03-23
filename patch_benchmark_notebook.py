"""
Executes cells 3-16 of QHSA_Net_Pavia_Full_Benchmark.ipynb
and patches their outputs back into the notebook.
Cell 1 (the ~8-hour script runner) is skipped.
"""
import json, base64, io, os
import nbformat
import pandas as pd

NOTEBOOK = r"c:\Users\saika\OneDrive\Desktop\test 6\QHSA_Net_Pavia_Full_Benchmark.ipynb"
WORKDIR  = r"c:\Users\saika\OneDrive\Desktop\test 6"

def img_output(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return nbformat.v4.new_output(
        output_type="display_data",
        data={"image/png": data, "text/plain": ["<Figure>"]},
        metadata={"image/png": {"width": 900}}
    )

def text_output(text):
    return nbformat.v4.new_output(output_type="stream", name="stdout", text=text)

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Load all data
df_s2  = pd.read_csv(f"{WORKDIR}/s2_dr_fs_results.csv")
df_s3  = pd.read_csv(f"{WORKDIR}/s3_qubit_results.csv")
df_s4  = pd.read_csv(f"{WORKDIR}/s4_attention_results.csv")
df_s5  = pd.read_csv(f"{WORKDIR}/s5_final_qhsa.csv")
df_s6  = pd.read_csv(f"{WORKDIR}/s6_baseline_results.csv")
master = pd.read_csv(f"{WORKDIR}/benchmark_master_results.csv")
with open(f"{WORKDIR}/best_config.json") as f:
    best = json.load(f)

print("All CSVs loaded.")
print("df_s2 columns:", list(df_s2.columns))
print("df_s3 columns:", list(df_s3.columns))
print("df_s4 columns:", list(df_s4.columns))
print("df_s5 columns:", list(df_s5.columns))
print("df_s6 columns:", list(df_s6.columns))
print("master columns:", list(master.columns))

# ── Cell 3: Setup ──────────────────────────────────────────────────────────────
nb["cells"][3]["outputs"] = [text_output(
    f"CSVs loaded successfully.\n"
    f"  s2 DR/FS:    {len(df_s2)} configs\n"
    f"  s3 Qubits:   {len(df_s3)} configs\n"
    f"  s4 Attention:{len(df_s4)} configs\n"
    f"  s5 Final:    {len(df_s5)} run(s)\n"
    f"  s6 Baselines:{len(df_s6)} models\n"
    f"  master:      {len(master)} models\n"
    f"Best config: {best}\n"
)]
nb["cells"][3]["execution_count"] = 1
print("Cell 3 patched")

# ── Cell 5: S2 DR/FS ──────────────────────────────────────────────────────────
buf = io.StringIO()
buf.write("DR / Feature-Selection Comparison (Full Pavia University)\n")
buf.write("="*65 + "\n")
show_cols = [c for c in ["method","OA","AA","kappa","macro_auc","train_time_min"] if c in df_s2.columns]
buf.write(df_s2[show_cols].to_string(index=False) + "\n\n")
best_dr = df_s2.loc[df_s2["OA"].idxmax()]
buf.write(f"Best method: {best_dr['method']}  |  OA={best_dr['OA']:.4f}%  kappa={best_dr['kappa']:.4f}\n")

nb["cells"][5]["outputs"] = [
    img_output(f"{WORKDIR}/fig_bench_s2_dr_comparison.png"),
    img_output(f"{WORKDIR}/fig_bench_s2_auc.png"),
    text_output(buf.getvalue())
]
nb["cells"][5]["execution_count"] = 2
print("Cell 5 patched")

# ── Cell 7: S3 Qubit sweep ────────────────────────────────────────────────────
buf = io.StringIO()
buf.write("Qubit & Layer Sweep\n" + "="*65 + "\n")
show_cols = [c for c in ["config","n_qubits","n_layers","OA","AA","kappa","macro_auc","train_time_min"] if c in df_s3.columns]
buf.write(df_s3[show_cols].to_string(index=False) + "\n\n")
best_q = df_s3.loc[df_s3["OA"].idxmax()]
cfg_name = best_q.get("config", f"{int(best_q['n_qubits'])}q-{int(best_q['n_layers'])}L")
buf.write(f"Best config: {cfg_name}  |  OA={best_q['OA']:.4f}%  kappa={best_q['kappa']:.4f}\n")

nb["cells"][7]["outputs"] = [
    img_output(f"{WORKDIR}/fig_bench_s3_qubit_sweep.png"),
    text_output(buf.getvalue())
]
nb["cells"][7]["execution_count"] = 3
print("Cell 7 patched")

# ── Cell 9: S4 Attention variants ─────────────────────────────────────────────
buf = io.StringIO()
buf.write("Attention Measurement Variants\n" + "="*65 + "\n")
show_cols = [c for c in ["measurement","OA","AA","kappa","macro_auc","train_time_min"] if c in df_s4.columns]
buf.write(df_s4[show_cols].to_string(index=False) + "\n\n")
best_a = df_s4.loc[df_s4["OA"].idxmax()]
buf.write(f"Best measurement: {best_a['measurement']}  |  OA={best_a['OA']:.4f}%  kappa={best_a['kappa']:.4f}\n")

nb["cells"][9]["outputs"] = [
    img_output(f"{WORKDIR}/fig_bench_s4_attention.png"),
    text_output(buf.getvalue())
]
nb["cells"][9]["execution_count"] = 4
print("Cell 9 patched")

# ── Cell 11: S5 Final QHSA-Net ────────────────────────────────────────────────
buf = io.StringIO()
buf.write("Final Optimised QHSA-Net on Full Pavia University\n" + "="*65 + "\n")
buf.write(df_s5.to_string(index=False) + "\n")

nb["cells"][11]["outputs"] = [
    text_output(buf.getvalue()),
    img_output(f"{WORKDIR}/fig_bench_roc_qhsa.png"),
    img_output(f"{WORKDIR}/fig_bench_confusion_qhsa.png"),
]
nb["cells"][11]["execution_count"] = 5
print("Cell 11 patched")

# ── Cell 13: S6 Baseline comparison ──────────────────────────────────────────
buf = io.StringIO()
buf.write("Baseline Model Comparison — Full Pavia University\n" + "="*65 + "\n")
show_cols = [c for c in ["model","OA","AA","kappa","macro_auc","mac_f1","train_time_s","infer_time_s"] if c in df_s6.columns]
buf.write(df_s6[show_cols].to_string(index=False) + "\n\n")
ranked = df_s6.sort_values("OA", ascending=False)[["model","OA"]].reset_index(drop=True)
ranked.index += 1
buf.write("Ranking by OA:\n" + ranked.to_string() + "\n")

nb["cells"][13]["outputs"] = [
    img_output(f"{WORKDIR}/fig_bench_s6_baselines.png"),
    img_output(f"{WORKDIR}/fig_bench_s6_f1_heatmap.png"),
    img_output(f"{WORKDIR}/fig_bench_roc_all_models.png"),
    img_output(f"{WORKDIR}/fig_bench_timing.png"),
    text_output(buf.getvalue())
]
nb["cells"][13]["execution_count"] = 6
print("Cell 13 patched")

# ── Cell 15: Master results ───────────────────────────────────────────────────
buf = io.StringIO()
buf.write("MASTER RESULTS — All Models on Full Pavia University\n" + "="*70 + "\n")
show_cols = [c for c in ["model","OA","AA","kappa","mac_prec","mac_f1","macro_auc","train_time_s","infer_time_s"] if c in master.columns]
buf.write(master[show_cols].to_string(index=False) + "\n\n")
buf.write("Best configuration used for QHSA-Net (Optimised):\n")
for k, v in best.items():
    buf.write(f"  {k}: {v}\n")

nb["cells"][15]["outputs"] = [text_output(buf.getvalue())]
nb["cells"][15]["execution_count"] = 7
print("Cell 15 patched")

# ── Cell 16: LaTeX table ──────────────────────────────────────────────────────
pub = master.copy()
pub = pub.sort_values("OA", ascending=False)
rename = {"model": "Model", "OA": "OA (%)", "AA": "AA (%)", "kappa": r"$\kappa$",
          "mac_f1": "F1 (%)", "macro_auc": "AUC (%)", "train_time_s": "Train (s)"}
pub_cols = [c for c in rename if c in pub.columns]
pub = pub[pub_cols].rename(columns=rename)
latex = pub.to_latex(index=False, float_format="%.2f",
                     caption="Comparison of all models on Full Pavia University dataset.",
                     label="tab:pavia_full")

nb["cells"][16]["outputs"] = [text_output(latex)]
nb["cells"][16]["execution_count"] = 8
print("Cell 16 patched")

# ── Save notebook ─────────────────────────────────────────────────────────────
with open(NOTEBOOK, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("\nDone! Notebook saved with all outputs embedded.")
