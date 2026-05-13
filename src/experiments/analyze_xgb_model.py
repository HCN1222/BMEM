"""
XGBoost Long Model Rule Analysis
Usage: conda run -n BMEM python analyze_xgb_model.py
"""

import json
import numpy as np
import pandas as pd
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

def load_model(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_trees(model):
    learner = model.get("learner", {})
    gradient_booster = learner.get("gradient_booster", {})
    model_data = gradient_booster.get("model", {})
    trees = model_data.get("trees", [])
    return trees

def parse_tree_splits(tree):
    """Extract all split rules from a single tree."""
    split_indices = tree.get("split_indices", [])
    split_conditions = tree.get("split_conditions", [])
    left_children = tree.get("left_children", [])
    right_children = tree.get("right_children", [])
    base_weights = tree.get("base_weights", [])

    splits = []
    for i, (feat_idx, cond, lc, rc) in enumerate(
        zip(split_indices, split_conditions, left_children, right_children)
    ):
        if lc != -1:  # not a leaf
            feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"f{feat_idx}"
            splits.append({
                "node": i,
                "feature": feat_name,
                "threshold": cond,
                "left_child": lc,
                "right_child": rc,
            })
    return splits

def analyze_feature_importance(trees):
    split_counter = Counter()
    threshold_by_feature = defaultdict(list)

    for tree in trees:
        splits = parse_tree_splits(tree)
        for s in splits:
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
        for i, (lc, w) in enumerate(zip(left_children, base_weights)):
            if lc == -1:  # leaf
                all_leaves.append(w)
    return np.array(all_leaves)

def get_top_rules(trees, top_n=20):
    """Find the most common split conditions (feature + threshold buckets)."""
    rule_counter = Counter()
    for tree in trees:
        splits = parse_tree_splits(tree)
        for s in splits:
            bucket = round(s["threshold"], 3)
            rule_counter[(s["feature"], bucket)] += 1
    return rule_counter.most_common(top_n)

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    trees = extract_trees(model)
    print(f"  Loaded {len(trees)} trees")

    # --- Hyperparameters ---
    print_section("MODEL HYPERPARAMETERS")
    attrs = model.get("learner", {}).get("attributes", {})
    for k, v in attrs.items():
        print(f"  {k}: {v}")

    train_params = model.get("learner", {}).get("learner_model_param", {})
    print(f"  num_feature: {train_params.get('num_feature')}")
    print(f"  num_class: {train_params.get('num_class')}")

    # --- Feature Importance ---
    print_section("FEATURE IMPORTANCE (by split count)")
    split_counter, threshold_by_feature = analyze_feature_importance(trees)
    total_splits = sum(split_counter.values())

    rows = []
    for feat, count in split_counter.most_common():
        thresholds = threshold_by_feature[feat]
        rows.append({
            "Feature": feat,
            "Split Count": count,
            "% of Splits": f"{count/total_splits*100:.1f}%",
            "Threshold Mean": f"{np.mean(thresholds):.4f}",
            "Threshold Std": f"{np.std(thresholds):.4f}",
            "Threshold Min": f"{np.min(thresholds):.4f}",
            "Threshold Max": f"{np.max(thresholds):.4f}",
        })

    df_importance = pd.DataFrame(rows)
    print(df_importance.to_string(index=False))

    # --- Root Node Analysis ---
    print_section("ROOT NODE FEATURES (first split per tree)")
    root_counter = analyze_root_nodes(trees)
    for feat, count in root_counter.most_common():
        print(f"  {feat:<20} {count:4d} trees  ({count/len(trees)*100:.1f}%)")

    # --- Leaf Value Analysis ---
    print_section("LEAF VALUE DISTRIBUTION")
    leaves = analyze_leaf_values(trees)
    print(f"  Total leaf nodes : {len(leaves)}")
    print(f"  Min              : {leaves.min():.4f}")
    print(f"  Max              : {leaves.max():.4f}")
    print(f"  Mean             : {leaves.mean():.6f}")
    print(f"  Std              : {leaves.std():.4f}")
    print(f"  Positive (long)  : {(leaves > 0).sum()} ({(leaves > 0).mean()*100:.1f}%)")
    print(f"  Negative (flat)  : {(leaves < 0).sum()} ({(leaves < 0).mean()*100:.1f}%)")

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n  Percentiles:")
    for p in percentiles:
        print(f"    P{p:2d}: {np.percentile(leaves, p):.4f}")

    # --- Top Rules ---
    print_section("TOP 20 MOST FREQUENT SPLIT RULES")
    top_rules = get_top_rules(trees, top_n=20)
    print(f"  {'Feature':<22} {'Threshold':>12} {'Count':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*8}")
    for (feat, thresh), count in top_rules:
        print(f"  {feat:<22} {thresh:>12.4f} {count:>8}")

    # --- Tree Size Distribution ---
    print_section("TREE SIZE DISTRIBUTION")
    node_counts = [len(t.get("left_children", [])) for t in trees]
    size_counter = Counter(node_counts)
    print(f"  Min nodes per tree: {min(node_counts)}")
    print(f"  Max nodes per tree: {max(node_counts)}")
    print(f"  Most common sizes:")
    for size, cnt in size_counter.most_common(5):
        print(f"    {size} nodes: {cnt} trees")

    # --- Plot Feature Importance ---
    print_section("SAVING FEATURE IMPORTANCE CHART")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    feats = [r["Feature"] for r in rows]
    counts = [r["Split Count"] for r in rows]
    axes[0].barh(feats[::-1], counts[::-1], color="steelblue")
    axes[0].set_xlabel("Split Count")
    axes[0].set_title("Feature Importance (Split Count)")
    axes[0].axvline(x=total_splits / len(feats), color="red", linestyle="--", label="Average")
    axes[0].legend()

    # Leaf value histogram
    axes[1].hist(leaves, bins=80, color="coral", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0, color="black", linestyle="--")
    axes[1].set_xlabel("Leaf Value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Leaf Value Distribution")

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/xgb_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Saved to: {out_path}")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
