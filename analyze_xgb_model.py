"""
XGBoost Long Model Rule Analysis
Usage: conda run -n BMEM python analyze_xgb_model.py
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

MODEL_PATH = "outputs/models/XGBoost/long/xgb_trading_model.json"
OUTPUT_DIR = "outputs/models/XGBoost/long"

FEATURE_NAMES = [
    "z_t", "c_t", "a_t", "s_t", "m_t",
    "bias_60d", "net_buy_amt_60d",
    "prob_S0", "prob_S1", "prob_S2", "prob_S3",
    "prob_S4", "prob_S5", "prob_S6", "prob_S7",
    "prob_S8", "prob_S9"
]

def load_model_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_booster(path):
    booster = xgb.Booster()
    booster.load_model(path)
    return booster

def extract_trees(model):
    learner = model.get("learner", {})
    trees = learner.get("gradient_booster", {}).get("model", {}).get("trees", [])
    return trees

def parse_tree_splits(tree):
    split_indices = tree.get("split_indices", [])
    split_conditions = tree.get("split_conditions", [])
    left_children = tree.get("left_children", [])
    right_children = tree.get("right_children", [])

    splits = []
    for i, (feat_idx, cond, lc, rc) in enumerate(
        zip(split_indices, split_conditions, left_children, right_children)
    ):
        if lc != -1:
            feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"f{feat_idx}"
            splits.append({
                "node": i,
                "feature": feat_name,
                "threshold": cond,
            })
    return splits

def analyze_weight(trees):
    split_counter = Counter()
    threshold_by_feature = defaultdict(list)
    for tree in trees:
        for s in parse_tree_splits(tree):
            split_counter[s["feature"]] += 1
            threshold_by_feature[s["feature"]].append(s["threshold"])
    return split_counter, threshold_by_feature

def analyze_root_nodes(trees):
    root_counter = Counter()
    for tree in trees:
        split_indices = tree.get("split_indices", [])
        left_children = tree.get("left_children", [])
        if len(split_indices) > 0 and left_children[0] != -1:
            feat_idx = split_indices[0]
            feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"f{feat_idx}"
            root_counter[feat_name] += 1
    return root_counter

def analyze_leaf_values(trees):
    all_leaves = []
    for tree in trees:
        left_children = tree.get("left_children", [])
        base_weights = tree.get("base_weights", [])
        for lc, w in zip(left_children, base_weights):
            if lc == -1:
                all_leaves.append(w)
    return np.array(all_leaves)

def get_top_rules(trees, top_n=20):
    rule_counter = Counter()
    for tree in trees:
        for s in parse_tree_splits(tree):
            bucket = round(s["threshold"], 3)
            rule_counter[(s["feature"], bucket)] += 1
    return rule_counter.most_common(top_n)

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def build_importance_df(split_counter, threshold_by_feature, gain_scores, cover_scores, total_splits):
    rows = []
    all_feats = set(split_counter.keys()) | set(gain_scores.keys())
    for feat in all_feats:
        w = split_counter.get(feat, 0)
        thresholds = threshold_by_feature.get(feat, [])
        rows.append({
            "Feature": feat,
            "Weight": w,
            "Weight %": f"{w/total_splits*100:.1f}%" if total_splits > 0 else "0%",
            "Gain": round(gain_scores.get(feat, 0.0), 4),
            "Cover": round(cover_scores.get(feat, 0.0), 4),
            "Threshold Mean": round(np.mean(thresholds), 4) if thresholds else None,
            "Threshold Std":  round(np.std(thresholds), 4)  if thresholds else None,
        })
    df = pd.DataFrame(rows).sort_values("Gain", ascending=False).reset_index(drop=True)
    return df

def main():
    print("Loading model...")
    model_json = load_model_json(MODEL_PATH)
    booster = load_booster(MODEL_PATH)
    trees = extract_trees(model_json)
    print(f"  Loaded {len(trees)} trees")

    # --- Hyperparameters ---
    print_section("MODEL HYPERPARAMETERS")
    attrs = model_json.get("learner", {}).get("attributes", {})
    for k, v in attrs.items():
        print(f"  {k}: {v}")
    train_params = model_json.get("learner", {}).get("learner_model_param", {})
    print(f"  num_feature: {train_params.get('num_feature')}")

    # --- Feature Importance (Weight, Gain, Cover) ---
    print_section("FEATURE IMPORTANCE COMPARISON: Weight vs Gain vs Cover")

    split_counter, threshold_by_feature = analyze_weight(trees)
    total_splits = sum(split_counter.values())

    gain_scores  = booster.get_score(importance_type="gain")
    cover_scores = booster.get_score(importance_type="cover")

    df_imp = build_importance_df(split_counter, threshold_by_feature, gain_scores, cover_scores, total_splits)
    print(df_imp.to_string(index=False))

    # Rank comparison
    print_section("RANK COMPARISON (Weight vs Gain)")
    weight_rank = {feat: i+1 for i, feat in enumerate(
        sorted(split_counter, key=split_counter.get, reverse=True))}
    gain_rank = {feat: i+1 for i, feat in enumerate(
        sorted(gain_scores, key=gain_scores.get, reverse=True))}
    all_feats = sorted(set(weight_rank) | set(gain_rank))
    print(f"  {'Feature':<22} {'Weight Rank':>12} {'Gain Rank':>10} {'Δ Rank':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*8}")
    for feat in sorted(all_feats, key=lambda f: gain_rank.get(f, 99)):
        wr = weight_rank.get(feat, "-")
        gr = gain_rank.get(feat, "-")
        delta = (wr - gr) if isinstance(wr, int) and isinstance(gr, int) else "-"
        arrow = f"+{delta}" if isinstance(delta, int) and delta > 0 else str(delta)
        print(f"  {feat:<22} {str(wr):>12} {str(gr):>10} {arrow:>8}")

    # --- Root Node Analysis ---
    print_section("ROOT NODE FEATURES")
    root_counter = analyze_root_nodes(trees)
    for feat, count in root_counter.most_common():
        print(f"  {feat:<20} {count:4d} trees  ({count/len(trees)*100:.1f}%)")

    # --- Leaf Values ---
    print_section("LEAF VALUE DISTRIBUTION")
    leaves = analyze_leaf_values(trees)
    print(f"  Total leaves : {len(leaves)}")
    print(f"  Min / Max    : {leaves.min():.4f} / {leaves.max():.4f}")
    print(f"  Mean / Std   : {leaves.mean():.6f} / {leaves.std():.4f}")
    print(f"  Positive     : {(leaves > 0).sum()} ({(leaves > 0).mean()*100:.1f}%)")
    print(f"  Negative     : {(leaves < 0).sum()} ({(leaves < 0).mean()*100:.1f}%)")

    # --- Top Rules ---
    print_section("TOP 20 MOST FREQUENT SPLIT RULES")
    top_rules = get_top_rules(trees, top_n=20)
    print(f"  {'Feature':<22} {'Threshold':>12} {'Count':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*8}")
    for (feat, thresh), count in top_rules:
        print(f"  {feat:<22} {thresh:>12.4f} {count:>8}")

    # --- Plot ---
    print_section("SAVING CHARTS")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, col, color, title in zip(
        axes,
        ["Weight", "Gain", "Cover"],
        ["steelblue", "coral", "mediumseagreen"],
        ["Weight (Split Count)", "Gain (Avg Info Gain)", "Cover (Avg Samples)"]
    ):
        sub = df_imp[df_imp[col] > 0].sort_values(col)
        ax.barh(sub["Feature"], sub[col], color=color)
        ax.set_title(title)
        ax.set_xlabel(col)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/xgb_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Saved to: {out_path}")
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
