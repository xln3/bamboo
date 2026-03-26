#!/usr/bin/env python3
"""Analyze paper_claims_v2 data: statistics, distributions, and samples."""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("/home/xln/panda2026/bamboo/data/paper_claims_v2")


def load_all_files():
    """Load all JSON files, return dict of filename -> list of claims."""
    papers = {}
    errors = []
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            papers[f.name] = data
        except Exception as e:
            errors.append((f.name, str(e)))
    return papers, errors


def main():
    papers, errors = load_all_files()

    print("=" * 80)
    print("BAMBOO paper_claims_v2  --  COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # ── 0. Loading summary ──────────────────────────────────────────────────
    print(f"\nFiles found:  {len(papers)}")
    if errors:
        print(f"Files with load errors: {len(errors)}")
        for name, err in errors:
            print(f"  {name}: {err}")

    # ── 1. Basic statistics ─────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("1. BASIC STATISTICS")
    print("─" * 80)

    claim_counts = {name: len(claims) for name, claims in papers.items()}
    total_claims = sum(claim_counts.values())
    n_papers = len(papers)
    avg_claims = total_claims / n_papers if n_papers else 0

    min_paper = min(claim_counts, key=claim_counts.get)
    max_paper = max(claim_counts, key=claim_counts.get)

    print(f"  Total papers:            {n_papers}")
    print(f"  Total claims:            {total_claims}")
    print(f"  Average claims/paper:    {avg_claims:.2f}")
    print(f"  Median  claims/paper:    {sorted(claim_counts.values())[n_papers // 2]}")
    print(f"  Min claims:              {claim_counts[min_paper]}  ({min_paper})")
    print(f"  Max claims:              {claim_counts[max_paper]}  ({max_paper})")

    # Distribution histogram
    buckets = Counter()
    for c in claim_counts.values():
        if c == 0:
            buckets["0"] += 1
        elif c <= 5:
            buckets["1-5"] += 1
        elif c <= 10:
            buckets["6-10"] += 1
        elif c <= 20:
            buckets["11-20"] += 1
        elif c <= 30:
            buckets["21-30"] += 1
        elif c <= 50:
            buckets["31-50"] += 1
        else:
            buckets["51+"] += 1

    print("\n  Claims-per-paper distribution:")
    for bucket in ["0", "1-5", "6-10", "11-20", "21-30", "31-50", "51+"]:
        cnt = buckets.get(bucket, 0)
        bar = "#" * cnt
        print(f"    {bucket:>6s}: {cnt:4d}  {bar}")

    # ── 2. Category distribution ────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("2. CLAIM CATEGORY DISTRIBUTION")
    print("─" * 80)

    category_counter = Counter()
    category_per_paper = defaultdict(lambda: Counter())
    for name, claims in papers.items():
        for cl in claims:
            cat = cl.get("category", "<missing>")
            category_counter[cat] += 1
            category_per_paper[name][cat] += 1

    for cat, cnt in category_counter.most_common():
        pct = 100.0 * cnt / total_claims if total_claims else 0
        print(f"  {cat:20s}  {cnt:5d}  ({pct:5.1f}%)")

    # How many papers have only main, only ablation, or both?
    only_main = sum(1 for p in category_per_paper.values() if set(p.keys()) == {"main"})
    only_ablation = sum(1 for p in category_per_paper.values() if set(p.keys()) == {"ablation"})
    both = sum(1 for p in category_per_paper.values() if "main" in p and "ablation" in p)
    other_combos = n_papers - only_main - only_ablation - both
    # Also count papers with 0 claims (no categories)
    zero_claim_papers = sum(1 for name, claims in papers.items() if len(claims) == 0)

    print(f"\n  Papers with only 'main' claims:       {only_main}")
    print(f"  Papers with only 'ablation' claims:   {only_ablation}")
    print(f"  Papers with both main + ablation:     {both}")
    if other_combos:
        print(f"  Papers with other category combos:    {other_combos}")
    if zero_claim_papers:
        print(f"  Papers with 0 claims:                 {zero_claim_papers}")

    # ── 3. Metric types ─────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("3. METRIC NAME DISTRIBUTION (top 40)")
    print("─" * 80)

    metric_counter = Counter()
    for claims in papers.values():
        for cl in claims:
            metric_counter[cl.get("metric_name", "<missing>")] += 1

    for metric, cnt in metric_counter.most_common(40):
        pct = 100.0 * cnt / total_claims
        bar = "#" * max(1, int(cnt / total_claims * 200))
        print(f"  {metric:30s}  {cnt:5d}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Total unique metric names: {len(metric_counter)}")

    # ── 4. Dataset distribution ─────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("4. DATASET DISTRIBUTION (top 50)")
    print("─" * 80)

    dataset_counter = Counter()
    for claims in papers.values():
        for cl in claims:
            dataset_counter[cl.get("dataset", "<missing>")] += 1

    for ds, cnt in dataset_counter.most_common(50):
        pct = 100.0 * cnt / total_claims
        print(f"  {ds:40s}  {cnt:5d}  ({pct:5.1f}%)")

    print(f"\n  Total unique datasets: {len(dataset_counter)}")

    # How many datasets appear in only 1 paper?
    ds_per_paper = defaultdict(set)
    for name, claims in papers.items():
        for cl in claims:
            ds_per_paper[cl.get("dataset", "<missing>")].add(name)
    single_paper_ds = sum(1 for ds, ps in ds_per_paper.items() if len(ps) == 1)
    print(f"  Datasets appearing in only 1 paper: {single_paper_ds}")

    # ── 5. Tolerance analysis ───────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("5. TOLERANCE ANALYSIS")
    print("─" * 80)

    tolerance_values = []
    papers_with_tolerance = set()
    papers_without_tolerance = set()
    for name, claims in papers.items():
        has_tol = False
        for cl in claims:
            tol = cl.get("tolerance")
            if tol is not None:
                tolerance_values.append(tol)
                has_tol = True
        if has_tol:
            papers_with_tolerance.add(name)
        elif len(claims) > 0:
            papers_without_tolerance.add(name)

    print(f"  Papers with tolerance specified:    {len(papers_with_tolerance)} / {n_papers}")
    print(f"  Papers without tolerance:           {len(papers_without_tolerance)}")
    if tolerance_values:
        tol_counter = Counter(tolerance_values)
        print(f"  Claims with tolerance:             {len(tolerance_values)} / {total_claims}")
        print(f"  Tolerance range:                   [{min(tolerance_values)}, {max(tolerance_values)}]")
        print(f"  Most common tolerance values:")
        for val, cnt in tol_counter.most_common(10):
            print(f"    {val:10.4f}  x {cnt}")

    # ── 6. Empty / zero-claim papers ────────────────────────────────────────
    print("\n" + "─" * 80)
    print("6. PAPERS WITH 0 CLAIMS / EMPTY FILES")
    print("─" * 80)

    zero_papers = [name for name, claims in papers.items() if len(claims) == 0]
    if zero_papers:
        print(f"  Found {len(zero_papers)} paper(s) with 0 claims:")
        for p in zero_papers:
            print(f"    - {p}")
    else:
        print("  None -- all 184 files have at least 1 claim.")

    # Also check for suspiciously low (1-2 claims)
    low_papers = [(name, len(cl)) for name, cl in papers.items() if 0 < len(cl) <= 2]
    if low_papers:
        print(f"\n  Papers with very few claims (1-2):")
        for name, cnt in sorted(low_papers):
            print(f"    {name}: {cnt} claim(s)")

    # ── 7. metric_unit distribution ─────────────────────────────────────────
    print("\n" + "─" * 80)
    print("7. METRIC UNIT DISTRIBUTION")
    print("─" * 80)
    unit_counter = Counter()
    for claims in papers.values():
        for cl in claims:
            unit_counter[cl.get("metric_unit", "<missing>")] += 1
    for unit, cnt in unit_counter.most_common():
        pct = 100.0 * cnt / total_claims
        print(f"  {str(unit):20s}  {cnt:5d}  ({pct:5.1f}%)")

    # ── 8. higher_is_better distribution ────────────────────────────────────
    print("\n" + "─" * 80)
    print("8. HIGHER_IS_BETTER DISTRIBUTION")
    print("─" * 80)
    hib_counter = Counter()
    for claims in papers.values():
        for cl in claims:
            hib_counter[cl.get("higher_is_better", "<missing>")] += 1
    for val, cnt in hib_counter.most_common():
        pct = 100.0 * cnt / total_claims
        print(f"  {str(val):20s}  {cnt:5d}  ({pct:5.1f}%)")

    # ── 9. Sample diverse papers ────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("9. SAMPLE PAPERS (diverse domains)")
    print("─" * 80)

    # Pick papers with different primary metrics to get domain diversity
    # Gather each paper's dominant metric
    paper_primary_metric = {}
    for name, claims in papers.items():
        if claims:
            mc = Counter(cl.get("metric_name", "") for cl in claims)
            paper_primary_metric[name] = mc.most_common(1)[0][0]

    # Pick one paper for each of 4 very different metric families
    target_metrics = ["Accuracy", "BLEU", "mAP", "PSNR", "FID", "mIoU", "AUC", "F1", "ROUGE-L", "Dice"]
    chosen = []
    seen_metrics = set()
    for tm in target_metrics:
        if len(chosen) >= 4:
            break
        for name, pm in paper_primary_metric.items():
            if pm == tm and pm not in seen_metrics:
                chosen.append(name)
                seen_metrics.add(pm)
                break

    # If we don't have 4 yet, fill with different ones
    if len(chosen) < 4:
        for name in sorted(papers.keys()):
            if name not in chosen and len(chosen) < 4:
                pm = paper_primary_metric.get(name, "")
                if pm not in seen_metrics:
                    chosen.append(name)
                    seen_metrics.add(pm)

    for name in chosen:
        claims = papers[name]
        print(f"\n  --- {name} ({len(claims)} claims) ---")
        print(f"  Primary metric: {paper_primary_metric.get(name, 'N/A')}")
        datasets_in_paper = set(cl.get("dataset", "") for cl in claims)
        categories_in_paper = set(cl.get("category", "") for cl in claims)
        print(f"  Datasets: {', '.join(sorted(datasets_in_paper))}")
        print(f"  Categories: {', '.join(sorted(categories_in_paper))}")
        print(f"  Full claims:")
        print(json.dumps(claims, indent=2))

    # ── 10. Additional field completeness check ─────────────────────────────
    print("\n" + "─" * 80)
    print("10. FIELD COMPLETENESS CHECK")
    print("─" * 80)
    expected_fields = ["claim_id", "description", "metric_name", "metric_value",
                       "source_location", "metric_unit", "higher_is_better",
                       "tolerance", "dataset", "category"]
    field_present = Counter()
    field_missing = Counter()
    extra_fields = Counter()
    for claims in papers.values():
        for cl in claims:
            for ef in expected_fields:
                if ef in cl:
                    field_present[ef] += 1
                else:
                    field_missing[ef] += 1
            for k in cl:
                if k not in expected_fields:
                    extra_fields[k] += 1

    print(f"  {'Field':<20s}  {'Present':>8s}  {'Missing':>8s}  {'%Present':>10s}")
    for ef in expected_fields:
        p = field_present.get(ef, 0)
        m = field_missing.get(ef, 0)
        pct = 100.0 * p / total_claims if total_claims else 0
        print(f"  {ef:<20s}  {p:8d}  {m:8d}  {pct:9.1f}%")

    if extra_fields:
        print(f"\n  Unexpected extra fields found:")
        for k, cnt in extra_fields.most_common():
            print(f"    {k}: {cnt}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
