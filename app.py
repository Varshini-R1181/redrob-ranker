"""
Redrob Hackathon — Sandbox Demo
Runs the full ranker on a small candidate sample (≤100 candidates).
Deploy to HuggingFace Spaces as a Gradio app.

Requirements for Spaces (requirements.txt in same folder):
    gradio>=4.0.0
"""

import gradio as gr
import json
import csv
import io
import sys
import os

# Add parent dir to path so we can import rank.py directly
sys.path.insert(0, os.path.dirname(__file__))
from rank import score_candidate, build_reasoning, composite_score

SAMPLE_JSONL = """\
{"candidate_id":"CAND_0000001","profile":{"anonymized_name":"Demo Candidate","headline":"ML Engineer","summary":"ML engineer with 6 years experience.","location":"Bangalore, Karnataka","country":"India","years_of_experience":6,"current_title":"ML Engineer","current_company":"Demo Corp","current_company_size":"201-500","current_industry":"Internet"},"career_history":[{"company":"Demo Corp","title":"ML Engineer","start_date":"2021-01-01","end_date":null,"duration_months":42,"is_current":true,"industry":"Internet","company_size":"201-500","description":"Built and deployed a semantic search pipeline using FAISS and sentence-transformers. Launched a RAG-based Q&A system to production serving 1M+ queries per month. Designed offline evaluation framework using NDCG and MRR metrics."}],"education":[{"institution":"IIT Bombay","degree":"B.Tech","field_of_study":"Computer Science","start_year":2014,"end_year":2018,"grade":"8.5","tier":"tier_1"}],"skills":[{"name":"Python","proficiency":"expert","endorsements":45,"duration_months":72},{"name":"Machine Learning","proficiency":"advanced","endorsements":30,"duration_months":60},{"name":"elasticsearch","proficiency":"advanced","endorsements":20,"duration_months":36}],"certifications":[],"languages":[{"language":"English","proficiency":"native"}],"redrob_signals":{"profile_completeness_score":92,"signup_date":"2023-01-15","last_active_date":"2026-06-01","open_to_work_flag":true,"profile_views_received_30d":45,"applications_submitted_30d":3,"recruiter_response_rate":0.8,"avg_response_time_hours":4,"skill_assessment_scores":{"Python":88,"Machine Learning":82},"connection_count":350,"endorsements_received":95,"notice_period_days":30,"expected_salary_range_inr_lpa":{"min":25,"max":40},"preferred_work_mode":"hybrid","willing_to_relocate":true,"github_activity_score":72,"search_appearance_30d":120,"saved_by_recruiters_30d":18,"interview_completion_rate":0.9,"offer_acceptance_rate":0.7,"verified_email":true,"verified_phone":true,"linkedin_connected":true}}
"""


def run_ranker(jsonl_text: str, jsonl_file) -> tuple:
    """Accept either pasted JSONL text or an uploaded file."""
    try:
        # Prefer file upload if provided
        if jsonl_file is not None:
            with open(jsonl_file.name, "r", encoding="utf-8") as f:
                raw = f.read()
        elif jsonl_text and jsonl_text.strip():
            raw = jsonl_text
        else:
            raw = SAMPLE_JSONL

        candidates = []
        for i, line in enumerate(raw.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                return None, f"JSON parse error on line {i+1}: {e}"

        if len(candidates) == 0:
            return None, "No candidates found in input."
        if len(candidates) > 200:
            return None, f"Sample too large ({len(candidates)} candidates). Please use ≤200 for the sandbox."

        # Score
        scored = []
        for c in candidates:
            try:
                s, parts = score_candidate(c)
                reasoning = build_reasoning(c, parts)
                sig = c.get("redrob_signals", {})
                scored.append((c["candidate_id"], s, reasoning, parts, sig))
            except Exception as e:
                return None, f"Error scoring {c.get('candidate_id','?')}: {e}"

        scored.sort(key=lambda x: (-x[1], x[0]))

        # Stage 2: composite re-rank (same as production ranker)
        import numpy as np
        saves = [x[4].get("saved_by_recruiters_30d", 0) for x in scored]
        githubs = [max(0.0, x[4].get("github_activity_score", -1)) for x in scored]
        save_p95 = float(np.percentile(saves, 95)) if len(saves) > 1 else max(saves) if saves else 1.0
        github_p95 = float(np.percentile(githubs, 95)) if len(githubs) > 1 else max(githubs) if githubs else 1.0

        composited = []
        for cid, primary, reasoning, parts, sig in scored:
            comp = composite_score(primary, sig, save_p95, github_p95)
            composited.append((cid, comp, primary, reasoning, parts))
        composited.sort(key=lambda x: (-x[1], x[0]))

        n_out = min(100, len(composited))
        top_n = composited[:n_out]

        # Write CSV
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (cid, comp, primary, reasoning, parts) in enumerate(top_n, 1):
            w.writerow([cid, i, round(comp, 6), reasoning])

        # Summary table for display
        display_rows = []
        for i, (cid, comp, primary, reasoning, parts) in enumerate(top_n[:20], 1):
            display_rows.append({
                "rank": i,
                "candidate_id": cid,
                "score": round(comp, 4),
                "title_gate": round(parts["title_score"], 2),
                "shipped": round(parts["shipped_score"], 2),
                "eval": round(parts["eval_score"], 2),
                "honeypot_flags": parts["honeypot_flags"],
                "reasoning_preview": reasoning[:80] + "..." if len(reasoning) > 80 else reasoning,
            })

        csv_content = buf.getvalue()
        summary = f"✅ Ranked {len(top_n)} candidates. Top 20 shown below.\n\n"
        summary += f"{'Rank':>5} {'CandidateID':>15} {'Score':>7} {'Gate':>5} {'Ship':>5} {'Eval':>5} {'HP':>3}\n"
        summary += "-" * 65 + "\n"
        for r in display_rows:
            summary += f"{r['rank']:>5} {r['candidate_id']:>15} {r['score']:>7.4f} {r['title_gate']:>5.2f} {r['shipped']:>5.2f} {r['eval']:>5.2f} {r['honeypot_flags']:>3}\n"

        # Save CSV for download
        tmp_path = "/tmp/redrob_output.csv"
        with open(tmp_path, "w") as f:
            f.write(csv_content)

        return tmp_path, summary

    except Exception as e:
        import traceback
        return None, f"Unexpected error: {e}\n{traceback.format_exc()}"


with gr.Blocks(title="Redrob Ranker — Sandbox") as demo:
    gr.Markdown("""
# Redrob Hackathon — Candidate Ranker Sandbox

Upload a JSONL file (≤200 candidates) or paste JSONL text.  
Runs the full production ranker and returns a ranked CSV.

**Reproduce command (local):**
```
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
""")

    with gr.Row():
        with gr.Column():
            jsonl_file = gr.File(label="Upload candidates JSONL (optional)", file_types=[".jsonl", ".json"])
            jsonl_text = gr.Textbox(
                label="Or paste JSONL here (one candidate per line)",
                placeholder='{"candidate_id": "CAND_0000001", ...}\n{"candidate_id": "CAND_0000002", ...}',
                lines=6,
            )
            run_btn = gr.Button("Run Ranker", variant="primary")

        with gr.Column():
            output_file = gr.File(label="Download ranked CSV")
            output_text = gr.Textbox(label="Results summary", lines=25)

    run_btn.click(
        fn=run_ranker,
        inputs=[jsonl_text, jsonl_file],
        outputs=[output_file, output_text],
    )

    gr.Markdown("*Leave both inputs empty to run on a built-in demo candidate.*")

if __name__ == "__main__":
    demo.launch()
