#!/usr/bin/env python3
"""Extract ground-truth claims from MinerU-parsed paper markdowns.

Designed for the BAMBOO benchmark: verifiable, quantitative experimental results
that can be reproduced by running the paper's released code.

Usage:
    python extract_claims_prompt.py --paper-id bamboo-00001
    python extract_claims_prompt.py --batch data/bamboo_curated.json --start 0 --end 100
"""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"

# ──────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — instructs the LLM on HOW to extract claims
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a meticulous ML research auditor. Your task is to extract **reproducible quantitative claims** from a machine-learning paper for a code-reproducibility benchmark.

## What is a "claim"?

A claim is a single, self-contained experimental result that:
1. Reports a **concrete numeric value** (accuracy, F1, BLEU, FID, mIoU, latency, etc.).
2. Is produced by **the authors' own method** (not a cited baseline).
3. Can, in principle, be **verified by re-running the authors' released code** on the stated dataset/split with the stated configuration.

## What is NOT a claim?

- Results of **other people's methods** (baselines rows copied from prior work).
- Purely **theoretical bounds** or **asymptotic complexity** (O(n log n), etc.).
- **Qualitative observations** ("our method produces sharper images").
- Numbers that are **dataset statistics** (number of samples, classes, etc.), not model outputs.
- **Training cost** numbers (GPU hours, wall-clock time) unless the paper explicitly presents them as a key result to reproduce.
- Metrics on **toy / synthetic sanity-check** experiments that are not part of the main evaluation.

## Extraction rules

1. **One claim = one number.** If a table cell reports "92.3 / 87.1" (precision / recall), emit two separate claims.
2. **Include the full context.** Every claim must specify:
   - `metric`: the metric name exactly as the paper uses it (e.g., "mIoU", "Top-1 Accuracy", "BLEU-4", "FID").
   - `value`: the numeric value as a float (e.g., 47.73, not "47.73%"). Keep the original precision.
   - `dataset`: the evaluation dataset/benchmark (e.g., "nuScenes", "ImageNet-1K", "WMT14 En-De").
   - `subset`: the split or subset if specified (e.g., "val", "test", "mini-val", "dev"). Use `null` if unspecified.
   - `table_or_figure`: which table or figure the number appears in (e.g., "Table 1", "Table 3", "Figure 4"). Use `null` if stated only in running text.
   - `configuration`: any non-default settings that condition this result — model variant, backbone, training data fraction, number of shots, etc. Keep it concise but unambiguous. Use `null` if the result is the paper's single main configuration.
   - `context`: a one-sentence natural-language description of what this claim says, readable by someone who has not read the paper.
3. **Prioritise the paper's MAIN results.** These are typically in "Table 1" or "Table 2" or a section titled "Main Results" / "Comparison with State-of-the-Art". Extract ALL rows that belong to the authors' method (including variants and ablations).
4. **Include ablation results.** They are valuable for verifying individual components.
5. **Do NOT fabricate numbers.** Only extract values that appear explicitly in the paper text or tables.
6. **Handle ± (standard deviation) correctly.** If the paper reports "92.3 ± 0.2", set `value` to 92.3 and add a field `std`: 0.2.
7. **Watch for bold / underline.** Papers often bold their best result — these are especially important to capture.
8. **Units matter.** If the paper says "3.2ms latency", the value is 3.2 and the metric is "Latency (ms)".

## Output format

Return a JSON array. Each element is an object with these fields:
```json
{
  "metric": "string",
  "value": number,
  "std": number_or_null,
  "dataset": "string",
  "subset": "string_or_null",
  "table_or_figure": "string_or_null",
  "configuration": "string_or_null",
  "context": "string"
}
```

Return ONLY the JSON array — no commentary, no markdown fences.
If the paper contains no reproducible quantitative claims (e.g., purely theoretical), return an empty array `[]`.
"""

# ──────────────────────────────────────────────────────────────────────
# USER PROMPT — provides the actual paper content
# ──────────────────────────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """\
# Paper metadata

- **Paper ID**: {paper_id}
- **Title**: {title}
- **Venue**: {venue} {year}
- **Code**: {code_url}

# Paper content (MinerU-extracted markdown)

{md_content}

---

Now extract all reproducible quantitative claims from this paper following the system instructions. Return a JSON array.
"""


def build_prompt(paper: dict, md_text: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one paper."""
    user = USER_PROMPT_TEMPLATE.format(
        paper_id=paper["paper_id"],
        title=paper["title"],
        venue=paper["venue"],
        year=paper["year"],
        code_url=paper["code_url"],
        md_content=md_text,
    )
    return SYSTEM_PROMPT, user


def load_paper_prompt(paper_id: str) -> tuple[str, str]:
    """Load a single paper and return prompts."""
    curated_dir = DATA / "bamboo_curated"
    papers = []
    for f in sorted(curated_dir.glob("bamboo-*.json")):
        with open(f) as fh:
            papers.extend(json.load(fh))
    paper = next((p for p in papers if p["paper_id"] == paper_id), None)
    if not paper:
        raise ValueError(f"Paper {paper_id} not found in curated list")

    md_path = DATA / paper["md_file"]
    md_text = md_path.read_text()
    return build_prompt(paper, md_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", help="Single paper ID")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Print the prompt instead of calling an LLM")
    args = parser.parse_args()

    if args.paper_id:
        sys_prompt, user_prompt = load_paper_prompt(args.paper_id)
        if args.show_prompt:
            print("=== SYSTEM PROMPT ===")
            print(sys_prompt)
            print("\n=== USER PROMPT (first 2000 chars) ===")
            print(user_prompt[:2000])
            print(f"\n... ({len(user_prompt)} total chars)")
        else:
            print(json.dumps({
                "system": sys_prompt,
                "user": user_prompt,
                "paper_id": args.paper_id,
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
