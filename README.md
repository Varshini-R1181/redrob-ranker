# Redrob Hackathon — Candidate Ranker

Rule-based, multi-component scorer for the Intelligent Candidate Discovery & Ranking Challenge.
Pure Python standard library. No downloads, no GPU, no network during ranking.

---

## Reproduce the submission

```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

That single command produces the ranked CSV from scratch.  
Runtime: ~40 seconds on CPU. Peak memory: ~450 MB. No pre-computation required.

---

## Requirements

- Python 3.9 or later
- No third-party packages (pure stdlib: `json`, `csv`, `re`, `math`, `argparse`, `datetime`)

Verify:
```bash
python3 --version   # must be 3.9+
```

---

## Validate before submitting

```bash
python3 validate_submission.py submission.csv
# Expected: Submission is valid.
```

---

## Approach summary

**Two-stage ranker:**

**Stage 1 — Primary scoring (all 100K candidates):**

| Component | Weight | Signal |
|---|---|---|
| Title gate (multiplicative) | — | Closes vocabulary of 47 known titles into Tier-A / Tier-B / Tier-C; irrelevant titles suppress the entire score, not just one additive term |
| Shipped evidence | 34% of rest | Regex over career history for production deployment verbs + IR/ML nouns co-occurrence; ownership verbs ("owned the ranking layer") weighted 2× |
| Trusted skills | 14% | Core AI/IR skills gated by duration_months × endorsements × proficiency; skills section alone worth little without corroboration |
| Eval framework | 10% | Explicit mentions of NDCG, MRR, A/B testing, offline evaluation in career descriptions |
| YoE bell curve | 13% | Asymmetric Gaussian (peak 7y): steep left tail (under-experienced is a real risk), gentle right tail (16y at Flipkart/Adobe is still strong) |
| Location fit | 20% | India hub cities = 1.0; other India = 0.7; outside India + willing to relocate = 0.45; outside India + not relocating = 0.05 |
| Education tiebreak | 3% | Institution tier from schema; tiebreaker only, not load-bearing |
| Behavioral multiplier | × 0.55–1.15 | last_active_date recency, notice_period_days, recruiter_response_rate, interview_completion_rate, open_to_work_flag, verified signals |
| Honeypot penalty | × 0.05 | 5 structural consistency checks (expert skill + zero duration, overlapping tenures, end_date before start_date, duration_months vs actual date span, total months vs stated YoE) |
| Consulting-only penalty | × 0.30 | All career history in IT-services industries with no product-company stint |

**Stage 2 — Composite re-ranking (top 150 only):**

Normalizes `saved_by_recruiters_30d` and `github_activity_score` within the top-150 pool, adds as calibrated tiebreakers (max bonus = 0.013). Weights chosen to be smaller than the natural score gap between top-tier bands so tier structure is preserved, but large enough to meaningfully differentiate within the compressed bottom of the top-100.

**Key engineering decisions:**

- **Title as multiplicative gate, not additive term.** Found via 50-sample testing that an irrelevant title (Civil Engineer) could outrank a relevant one (Software Engineer) purely on behavioral signal luck when title was additive. Now Tier-C (HR Mgr, Civil Eng, etc.) suppresses the whole profile to ×0.03.
- **Description deduplication before scoring.** 36% of the pool has identical description templates assigned to different company names (synthetic data artifact). Without dedup, candidates with the same block at two companies scored higher than those with one — double-counting evidence that doesn't exist.
- **Honeypot checks calibrated on full pool.** Originally 7 checks flagged 640 candidates (8× the ~80 documented honeypots). Two checks fired on normal data (salary min/max field noise, mid-career graduate study). Removed; remaining 5 checks flag ~59 candidates pool-wide at OR logic.
- **Asymmetric YoE bell.** Symmetric Gaussian was scoring a 16y Flipkart/Adobe/Glance candidate (three separate production IR systems) at 0.056, putting them at rank 100. Fixed with σ=3.5 left, σ=8.0 right.
- **Behavioral floor proportional to base score.** Flat 0.55 floor was causing quality inversions: 96/100 top-100 candidates had lower base scores than a rank-111 candidate (Microsoft + Saarthi 50M-QPM RAG pipeline) that the flat floor was suppressing. Floor now = max(0.55, 0.45 + 0.15 × base).

**Runtime:** ~40 seconds / 450 MB on a single CPU core for 100K candidates.  
**Validator:** passes `validate_submission.py` with 0 errors.  
**Honeypot rate:** 0 of 100 top candidates flagged.

---

## Files

| File | Purpose |
|---|---|
| `rank.py` | Main ranker — single command produces submission CSV |
| `validate_submission.py` | Official format validator from hackathon bundle |
| `submission_metadata.yaml` | Team metadata mirroring portal submission |
| `requirements.txt` | Dependency list (none — pure stdlib) |
| `team_xxx.csv` | Our submission CSV (rename to your participant ID) |

---

## Sandbox

[**→ Run on HuggingFace Spaces**](https://huggingface.co/spaces/Varshini-R1181/redrob-ranker)

The sandbox accepts a small candidate sample (≤100 candidates as JSONL), runs the full ranker, and returns a ranked CSV. Runs within the 5-minute CPU budget on sample inputs.

---

## AI tools declaration

See `submission_metadata.yaml` for full declaration.
Claude was used for architectural discussion and code review throughout development.
No candidate data was fed to any external LLM. All scoring logic runs locally, offline.
