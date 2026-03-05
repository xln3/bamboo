# Curated Paper Data

Human-readable and editable TSV files for each venue. These are the working copies for review and annotation.

## Workflow

```bash
cd bamboo/scripts/collect

# Export from JSON → TSV (safe to re-run, preserves human edits)
python sync_curated.py export

# Edit TSV files (Excel, Google Sheets, text editor, etc.)
# Editable columns: code_url, arxiv_id, status, notes, domain

# Import human edits back to JSON
python sync_curated.py import
```

## Columns

| Column | Editable | Description |
|--------|----------|-------------|
| title | no | Paper title |
| venue | no | Venue name |
| code_url | **yes** | GitHub/GitLab/HF repo URL |
| arxiv_id | **yes** | arXiv ID (e.g. 2501.12345) |
| paper_url | no | Link to paper page |
| code_commit | no | Pinned git commit (auto-filled by validation) |
| status | **yes** | `auto` / `verified` / `excluded` / `needs_review` |
| notes | **yes** | Free-form annotation |
| repo_valid | no | Repo validation result (True/False) |
| stars | no | GitHub stars |
| domain | **yes** | Research domain (vision/nlp/multimodal/...) |
| venue_track | no | main/oral/poster/spotlight/workshop/findings |

## Status values

- `auto` — default, not yet reviewed
- `verified` — human confirmed code URL is correct
- `excluded` — excluded from benchmark (with reason in notes)
- `needs_review` — flagged for human review
