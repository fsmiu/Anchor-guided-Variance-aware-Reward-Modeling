"""Generate final result tables from evaluation outputs."""

import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pandas as pd


def load_results(results_dir="results"):
    """Load all result JSON files."""
    results = {}
    if not os.path.exists(results_dir):
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_results.json"):
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
                run_name = data.get("run_name", fname.replace("_results.json", ""))
                results[run_name] = data
    return results


def fmt(val):
    """Format accuracy as percentage string."""
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def parse_run_name(run_name):
    """Extract seed, anchor_short, anchor_lambda, and learning_rate from a run_name string.

    Anchor format:   {base}_{model_short}_anchor-{anchor_short}_lam{lambda}[_lr{lr}]_seed{N}
    Baseline format: {base}_{model_short}[_lr{lr}]_seed{N}
    """
    seed = anchor_short = anchor_lambda = learning_rate = None

    m = re.search(r"_seed(\d+)$", run_name)
    if m:
        seed = int(m.group(1))

    # Lambda is followed by either _lr or _seed, so use a non-greedy match with a lookahead-style alt
    m = re.search(r"_lam([0-9.]+?)(?=_lr|_seed)", run_name)
    if m:
        anchor_lambda = float(m.group(1))

    m = re.search(r"_anchor-(.+?)_lam", run_name)
    if m:
        anchor_short = m.group(1)

    # Learning rate is optional; format tolerates scientific notation (e.g. 2e-6, 1E-06)
    m = re.search(r"_lr([0-9.eE+-]+)_seed", run_name)
    if m:
        try:
            learning_rate = float(m.group(1))
        except ValueError:
            learning_rate = None

    return seed, anchor_short, anchor_lambda, learning_rate


def generate_main_table(results):
    """Generate Table 1: Main Results from all discovered runs."""
    rows = []

    for run_name, r in results.items():
        rb = r.get("rewardbench", {})
        dataset = r.get("dataset", "unknown")
        seed, anchor_short, anchor_lambda, learning_rate = parse_run_name(run_name)

        row = {
            "Run": run_name,
            "Model Type": r.get("model_type", "—"),
            "Dataset": dataset,
            "Seed": seed if seed is not None else "—",
            "Anchor Model": anchor_short if anchor_short is not None else "—",
            "Anchor Lambda": anchor_lambda if anchor_lambda is not None else "—",
            "Anchor %": r.get("anchor_sample_percent", "—"),
            "Anchor Count": r.get("anchor_sample_count", "—"),
            "Learning Rate": learning_rate if learning_rate is not None else "—",
            "Ckpt Step": r.get("checkpoint_step", "—"),
            "Anchor Tau": r.get("resolved_anchor_tau", "—"),
            "Anchor Tau1": r.get("resolved_anchor_tau1", "—"),
            "Anchor Tau2": r.get("resolved_anchor_tau2", "—"),
            "ID Accuracy": fmt(r.get("id_accuracy")),
            "RB Overall": fmt(rb.get("Overall")),
            "RB Chat": fmt(rb.get("Chat")),
            "RB Chat Hard": fmt(rb.get("Chat Hard")),
            "RB Safety": fmt(rb.get("Safety")),
            "RB Reasoning": fmt(rb.get("Reasoning")),
            # ID calibration
            "ID Brier": f"{r['id_brier']:.4f}"   if "id_brier"   in r else "—",
            "ID CE":    f"{r['id_ce_loss']:.4f}"  if "id_ce_loss" in r else "—",
            # RewardBench calibration
            "RB Brier": f"{r['rewardbench']['brier']:.4f}"    if "rewardbench" in r and "brier"    in r["rewardbench"] else "—",
            "RB CE":    f"{r['rewardbench']['ce_loss']:.4f}"  if "rewardbench" in r and "ce_loss"  in r["rewardbench"] else "—",
            # PPE-P
            "PPE-P Acc":   fmt(r.get("ppe_p", {}).get("accuracy")),
            "PPE-P Brier": f"{r['ppe_p']['brier']:.4f}"    if "ppe_p" in r and "brier"    in r["ppe_p"] else "—",
            "PPE-P CE":    f"{r['ppe_p']['ce_loss']:.4f}"  if "ppe_p" in r and "ce_loss"  in r["ppe_p"] else "—",

        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_stratified_table(results):
    """Generate Table 2: Stratified Results for MultiPref runs."""
    rows = []

    for run_name, r in results.items():
        strat = r.get("stratified")
        if strat is None:
            continue

        row = {
            "Run": run_name,
            "High-Agreement Acc": fmt(strat.get("High-Agreement")),
            "Diverging Acc": fmt(strat.get("Diverging")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    print("Loading results...")
    results = load_results()

    if not results:
        print("No results found in results/ directory.")
        return

    print(f"Found results for: {list(results.keys())}")

    # Table 1: Main Results
    print("\n" + "=" * 80)
    print("Table 1: Main Results")
    print("=" * 80)
    table1 = generate_main_table(results)
    print(table1.to_string(index=False))

    # Save as CSV
    os.makedirs("results", exist_ok=True)
    table1.to_csv("results/table1_main_results.csv", index=False)

    # Table 2: Stratified Results
    table2 = generate_stratified_table(results)
    if not table2.empty:
        print("\n" + "=" * 80)
        print("Table 2: Stratified Results on MultiPref")
        print("=" * 80)
        print(table2.to_string(index=False))
        table2.to_csv("results/table2_stratified_results.csv", index=False)

    print("\nTables saved to results/")


if __name__ == "__main__":
    main()
