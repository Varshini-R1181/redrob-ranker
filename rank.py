#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker
Rule-based, multi-component scorer over precomputed features.
CPU-only, no network, single pass over candidates.jsonl.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from datetime import date, datetime

TODAY = date(2026, 6, 19)

# ---------------------------------------------------------------------------
# Title taxonomy — built empirically from the actual 47 distinct titles
# observed in the candidate pool (closed vocabulary, not free text).
# ---------------------------------------------------------------------------
TIER_A_TITLES = {
    "ml engineer", "ai research engineer", "senior software engineer (ml)",
    "computer vision engineer", "junior ml engineer", "ai specialist",
    "recommendation systems engineer", "machine learning engineer",
    "applied ml engineer", "search engineer", "ai engineer", "nlp engineer",
    "senior nlp engineer", "senior machine learning engineer",
    "staff machine learning engineer", "senior ai engineer",
    "senior applied scientist", "lead ai engineer", "data scientist",
    "senior data scientist",
}
TIER_B_TITLES = {
    "software engineer", "full stack developer", "cloud engineer",
    "java developer", ".net developer", "devops engineer",
    "mobile developer", "frontend engineer", "qa engineer",
    "analytics engineer", "data engineer", "data analyst",
    "backend engineer", "senior data engineer", "senior software engineer",
}
TIER_C_TITLES = {
    "business analyst", "hr manager", "mechanical engineer", "accountant",
    "project manager", "customer support", "operations manager",
    "content writer", "sales executive", "civil engineer",
    "graphic designer", "marketing manager",
}
SENIORITY_BONUS_WORDS = ("senior", "staff", "lead", "principal", "founding")

# ---------------------------------------------------------------------------
# Skill taxonomy — the ~50 low-frequency AI/IR/ML-specific tags vs. the
# ~60 generic high-frequency tags. Built from a 20K-row frequency survey.
# ---------------------------------------------------------------------------
CORE_AI_SKILLS = {
    "sentence transformers", "opencv", "llms", "recommendation systems",
    "langchain", "embeddings", "pinecone", "vector search",
    "prompt engineering", "hugging face transformers", "fine-tuning llms",
    "rag", "feature engineering", "computer vision", "information retrieval",
    "semantic search", "cnn", "image classification", "faiss", "mlops",
    "gans", "statistical modeling", "weights & biases",
    "reinforcement learning", "mlflow", "bentoml", "forecasting", "tts",
    "asr", "kubeflow", "time series", "speech recognition",
    "diffusion models", "yolo", "object detection", "data science",
    "llamaindex", "milvus", "nlp", "weaviate", "learning to rank",
    "python", "tensorflow", "qlora", "haystack", "scikit-learn",
    "opensearch", "pytorch", "pgvector", "lora", "peft", "qdrant",
    "machine learning", "deep learning", "elasticsearch", "bm25",
}
# Subset that maps directly to "production embeddings / hybrid retrieval"
# (the JD's "things you absolutely need")
RETRIEVAL_CORE = {
    "sentence transformers", "embeddings", "pinecone", "vector search",
    "faiss", "semantic search", "information retrieval", "rag",
    "langchain", "llamaindex", "milvus", "weaviate", "opensearch",
    "pgvector", "qdrant", "elasticsearch", "bm25", "learning to rank",
    "hugging face transformers",
}
PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}

# ---------------------------------------------------------------------------
# Regex patterns for career-history "shipped evidence" and eval-framework
# experience. Compiled once.
# ---------------------------------------------------------------------------
SHIP_VERB_RE = re.compile(
    r"\b(launched|deployed|shipped|scaled|productioni[sz]ed|rolled out|"
    r"built and deployed|took .* to production|in production|"
    r"model deployment|inference service|model.?serving|"
    r"prediction api|feature store|serving layer|serving pipeline|"
    r"ended up shipping|went to production|runs in production|"
    r"built.{0,30}production|set up.{0,30}inference)\b", re.I
)
# Ownership verbs are stronger than ship verbs — they imply end-to-end
# accountability for a production system, not just contributing to one.
# Scored separately with a higher per-hit weight.
OWNERSHIP_RE = re.compile(
    r"\b(owned the .{0,50}(layer|pipeline|system|model|ranker|stack|service)|"
    r"led the .{0,50}(layer|pipeline|system|ranker)|"
    r"responsible for .{0,40}(production|ranking|retrieval|search|recsys))\b", re.I
)
IR_NOUN_RE = re.compile(
    r"\b(retrieval|ranking|recommendation(s)?|embeddings?|vector search|"
    r"search engine|RAG|semantic search|relevance|matching engine|"
    r"recsys|recommender|hybrid search|re-?rank(ing)?)\b", re.I
)
SCALE_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(%|percent)\b|"
    r"\b\d+(\.\d+)?\s*[kKmMbB]\+?\s*(users|requests|queries|qps|records|events|daily)\b",
    re.I
)
EVAL_RE = re.compile(
    r"\b(NDCG|MRR|MAP@|precision@|recall@|P@\d|A/?B test(ing)?|offline eval"
    r"(uation)?|online eval(uation)?|click-?through|CTR|eval(uation)? framework|"
    r"benchmark(ing)?|offline-to-online)\b", re.I
)
CONSULTING_INDUSTRIES = {"it services", "consulting"}
HUB_CITIES = ("pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr")


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def title_gate_and_bonus(profile, career_history):
    """Returns (gate, seniority_bonus). Gate is multiplicative over the rest
    of the base score -- an irrelevant title should suppress the WHOLE
    profile's relevance, not just lose a fixed number of additive points
    that location/education luck can offset. This was a real bug found via
    sample testing: a Civil Engineer with great location/education stats
    was edging out a Full Stack Developer with average ones, because title
    was originally just one additive term among several."""
    t = profile.get("current_title", "").strip().lower()
    if t in TIER_A_TITLES:
        gate = 1.0
    elif t in TIER_B_TITLES:
        gate = 0.55
    elif t in TIER_C_TITLES:
        gate = 0.05
    else:
        gate = 0.35  # unseen title, neutral fallback

    bonus = 0.0
    if any(w in t for w in SENIORITY_BONUS_WORDS) and gate >= 0.55:
        bonus = 0.05

    n_jobs = len(career_history)
    durations = [h.get("duration_months") or 0 for h in career_history]
    avg_tenure = (sum(durations) / n_jobs) if n_jobs else 0
    if n_jobs >= 3 and avg_tenure < 18:
        gate *= 0.85  # title-hopping penalty

    return max(0.0, min(1.0, gate)), bonus


def shipped_and_eval_components(career_history):
    # Deduplicate descriptions before joining: 36% of the pool has identical
    # description blocks assigned to different company names (synthetic data
    # artifact). Without dedup, a candidate with "owned the ranking layer"
    # at two jobs (same text) scores higher than one who has it at one job
    # — double-counting evidence that's actually the same single template.
    # Use first-80-chars as dedup key (enough to identify template repeats).
    seen = set()
    unique_descs = []
    for h in career_history:
        desc = h.get("description") or ""
        key = desc[:80]
        if key not in seen:
            seen.add(key)
            unique_descs.append(desc)

    text = " ".join(unique_descs)
    ship_hits = len(SHIP_VERB_RE.findall(text))
    ownership_hits = len(OWNERSHIP_RE.findall(text))
    ir_hits = len(IR_NOUN_RE.findall(text))
    scale_hits = len(SCALE_RE.findall(text))
    eval_hits = len(EVAL_RE.findall(text))

    # Ownership hits count double — "owned the ranking layer" signals
    # end-to-end accountability, stronger than "deployed the ranking layer".
    effective_ship_hits = ship_hits + 2 * ownership_hits
    shipped_score = min(1.0, 0.28 * min(effective_ship_hits, ir_hits) + 0.14 * scale_hits)
    eval_score = min(1.0, 0.35 * eval_hits)
    return shipped_score, eval_score, ir_hits


def skills_component(skills, skill_assessment_scores):
    if not skills:
        return 0.0
    total = 0.0
    core_hits = 0
    for s in skills:
        name = (s.get("name") or "").strip().lower()
        if name not in CORE_AI_SKILLS:
            continue
        core_hits += 1
        prof_w = PROFICIENCY_WEIGHT.get(s.get("proficiency"), 0.4)
        dur_w = min(1.0, (s.get("duration_months") or 0) / 18.0)
        endorse_w = min(1.0, 0.3 + (s.get("endorsements") or 0) / 15.0)
        weight = 1.6 if name in RETRIEVAL_CORE else 1.0
        # corroborate against the candidate's own Redrob assessment score if present
        assess = skill_assessment_scores.get(s.get("name"), None)
        assess_w = 0.5 + (assess / 200.0) if isinstance(assess, (int, float)) else 0.85
        total += weight * prof_w * dur_w * endorse_w * assess_w
    if core_hits == 0:
        return 0.0
    return max(0.0, min(1.0, total / 6.0))


def yoe_component(yoe):
    if yoe is None:
        return 0.3
    # Asymmetric: steeper left tail (under-experienced is real risk),
    # gentler right tail (over-experienced at senior role is not).
    # Audit: symmetric gaussian was scoring 16y Flipkart/Adobe candidate at 0.056.
    sigma = 8.0 if yoe >= 7.0 else 3.5
    return math.exp(-((yoe - 7.0) ** 2) / (2 * sigma ** 2))


def location_component(profile, willing_to_relocate):
    country = (profile.get("country") or "").strip().lower()
    location = (profile.get("location") or "").strip().lower()
    if country == "india":
        return 1.0 if any(h in location for h in HUB_CITIES) else 0.7
    return 0.45 if willing_to_relocate else 0.05


def education_component(education):
    if not education:
        return 0.0
    tier_w = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.4, "tier_4": 0.2, "unknown": 0.3}
    best = max((tier_w.get(e.get("tier"), 0.3) for e in education), default=0.0)
    return best


def behavioral_multiplier(sig):
    # NOTE: this range is intentionally narrow (0.55x-1.15x). Behavioral
    # signals should modulate ranking *within* a competency tier (e.g. break
    # ties between two similarly-qualified candidates), not be powerful
    # enough to flip an irrelevant-title candidate above a relevant one.
    # An earlier wider range (0.3x-1.2x) let a passive, inactive Frontend
    # Engineer fall behind an actively-engaged but completely unrelated
    # Civil Engineer in testing — caught via sample sniff test, fixed here.
    last_active = parse_date(sig.get("last_active_date"))
    if last_active is None:
        recency = 0.6
    else:
        days = (TODAY - last_active).days
        if days <= 30:
            recency = 1.0
        elif days <= 90:
            recency = 1.0 - 0.2 * (days - 30) / 60.0
        elif days <= 180:
            recency = 0.8 - 0.15 * (days - 90) / 90.0
        else:
            recency = 0.5

    notice = sig.get("notice_period_days", 60)
    if notice <= 30:
        notice_w = 1.0
    elif notice <= 60:
        notice_w = 0.9
    elif notice <= 90:
        notice_w = 0.75
    else:
        notice_w = 0.55

    resp = sig.get("recruiter_response_rate", 0.5) or 0.0
    interview = sig.get("interview_completion_rate", 0.5) or 0.0
    open_flag = 1.04 if sig.get("open_to_work_flag") else 1.0
    trust = 0.93 + 0.04 * sum([
        bool(sig.get("verified_email")),
        bool(sig.get("verified_phone")),
        bool(sig.get("linkedin_connected")),
    ]) / 3.0

    resp_w = 0.85 + 0.15 * resp
    interview_w = 0.9 + 0.1 * interview

    mult = recency * notice_w * resp_w * interview_w * open_flag * trust
    return max(0.55, min(1.15, mult))


def honeypot_flags(candidate):
    """Returns count of fired checks. Each of these 5 checks individually
    fires on well under 0.05% of the pool and rarely co-occurs with another
    -- found via full-pool testing that two originally-included checks
    (salary min>max, and years-of-experience vs. earliest-education-year)
    were each firing on thousands of completely normal candidates (salary
    field ordering noise, and people who did a graduate degree mid-career)
    and were dropped. With only the 5 checks below, ANY single hit is a
    meaningful anomaly signal -- OR logic across them flags ~59 candidates
    pool-wide, in the right ballpark vs. the documented ~80 honeypots."""
    flags = 0
    career_history = candidate["career_history"]
    skills = candidate.get("skills", [])
    profile = candidate["profile"]

    # 1. expert proficiency with near-zero duration
    for s in skills:
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) < 6:
            flags += 1
            break

    # 2. total career duration grossly exceeds stated years_of_experience
    total_months = sum(h.get("duration_months") or 0 for h in career_history)
    yoe = profile.get("years_of_experience")
    if yoe is not None and total_months > (yoe * 12 + 18):
        flags += 1

    # 3. overlapping full-time roles
    intervals = []
    for h in career_history:
        sd = parse_date(h.get("start_date"))
        ed = parse_date(h.get("end_date")) or TODAY
        if sd:
            intervals.append((sd, ed))
    intervals.sort()
    for i in range(len(intervals) - 1):
        if intervals[i][1] > intervals[i + 1][0]:
            overlap_days = (intervals[i][1] - intervals[i + 1][0]).days
            if overlap_days > 45:
                flags += 1
                break

    # 4. end_date before start_date
    for h in career_history:
        sd, ed = parse_date(h.get("start_date")), parse_date(h.get("end_date"))
        if sd and ed and ed < sd:
            flags += 1
            break

    # 5. duration_months wildly inconsistent with actual date span
    for h in career_history:
        sd = parse_date(h.get("start_date"))
        ed = parse_date(h.get("end_date")) or TODAY
        if sd:
            actual_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            stated = h.get("duration_months") or 0
            if abs(actual_months - stated) > 6:
                flags += 1
                break

    return flags


def consulting_only(career_history):
    if not career_history:
        return False
    return all((h.get("industry") or "").strip().lower() in CONSULTING_INDUSTRIES for h in career_history)


def build_reasoning(candidate, parts):
    p = candidate["profile"]
    title = p.get("current_title")
    company = p.get("current_company")
    yoe = p.get("years_of_experience")
    sig = candidate["redrob_signals"]

    bits = [f"{title} ({yoe}y exp) at {company}, {p.get('location')}"]

    if parts["shipped_score"] > 0.25:
        bits.append("career history shows hands-on retrieval/ranking/ML systems work")
    elif parts["title_score"] >= 0.35:
        bits.append("adjacent engineering background, limited direct retrieval/ranking evidence")
    else:
        bits.append("role and history are not engineering-aligned with this JD")

    if parts["eval_score"] > 0.2:
        bits.append("mentions evaluation/A-B testing experience")

    if parts["honeypot_flags"] >= 1:
        bits.append("profile has internal inconsistencies (flagged, heavily down-weighted)")

    if parts["consulting_only"]:
        bits.append("entire career at IT-services firms (down-weighted per JD)")

    notice = sig.get("notice_period_days")
    if notice is not None:
        if notice <= 30:
            bits.append(f"short {notice}-day notice")
        elif notice > 90:
            bits.append(f"long {notice}-day notice (down-weighted)")

    days_inactive = None
    la = parse_date(sig.get("last_active_date"))
    if la:
        days_inactive = (TODAY - la).days
        if days_inactive > 120:
            bits.append(f"inactive on platform for {days_inactive}d")

    reasoning = "; ".join(bits)
    return reasoning[:300]


def score_candidate(candidate):
    profile = candidate["profile"]
    career_history = candidate["career_history"]
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    sig = candidate["redrob_signals"]

    title_gate, title_bonus = title_gate_and_bonus(profile, career_history)
    shipped_score, eval_score, ir_hits = shipped_and_eval_components(career_history)
    skills_score = skills_component(skills, sig.get("skill_assessment_scores", {}) or {})
    yoe_score = yoe_component(profile.get("years_of_experience"))
    loc_score = location_component(profile, sig.get("willing_to_relocate", False))
    edu_score = education_component(education)

    # title_gate is multiplicative over everything else: an irrelevant title
    # (Tier C) suppresses the WHOLE profile's relevance rather than losing a
    # fixed number of additive points that loc/edu luck could offset.
    rest = (
        0.34 * shipped_score +
        0.14 * skills_score +
        0.10 * eval_score +
        0.13 * yoe_score +
        0.20 * loc_score +
        0.09 * edu_score
    )
    base = title_gate * (rest + title_bonus)
    title_score = title_gate  # kept for reasoning text below

    n_honeypot = honeypot_flags(candidate)
    is_consulting_only = consulting_only(career_history)
    behave_mult = behavioral_multiplier(sig)

    # Proportional behavioral floor: a flat 0.55 floor (set in behavioral_multiplier)
    # is appropriate for average base scores, but creates quality inversions when
    # a high-base-score candidate hits it. Audit found CAND_0092278 (base=0.95,
    # Microsoft/Saarthi 50M-QPM RAG pipeline) sitting at rank 111 behind 96 weaker
    # technical profiles purely due to behavioral suppression.
    # Fix: let exceptional base scores partially offset the behavioral floor.
    # At base=0.95 -> effective floor = 0.59. At base=0.60 -> floor stays 0.56.
    # Cap at 0.72 so no candidate escapes behavioral penalty entirely.
    effective_behave_floor = min(0.72, 0.45 + 0.15 * base)
    behave_mult = max(effective_behave_floor, behave_mult)

    penalty = 1.0
    if n_honeypot >= 1:
        penalty *= 0.05
    if is_consulting_only:
        penalty *= 0.3

    final = base * penalty * behave_mult

    # Secondary tiebreak signal for within-band ordering (NDCG@10 sensitive):
    # recency-weighted "scope of impact" — more IR/ML hits + higher assessment
    # scores among already-high scorers nudges ordering without changing tiers.
    assess_scores = list((sig.get("skill_assessment_scores") or {}).values())
    avg_assess = sum(assess_scores) / len(assess_scores) if assess_scores else 0.0
    tiebreak = 0.001 * ir_hits + 0.0001 * avg_assess

    parts = {
        "title_score": title_score,
        "shipped_score": shipped_score,
        "eval_score": eval_score,
        "skills_score": skills_score,
        "yoe_score": yoe_score,
        "loc_score": loc_score,
        "edu_score": edu_score,
        "honeypot_flags": n_honeypot,
        "consulting_only": is_consulting_only,
        "behave_mult": behave_mult,
        "base": base,
    }
    return final + tiebreak, parts


def composite_score(primary_score, sig, save_p95, github_p95):
    """Secondary composite for within-band re-ranking after primary selection.

    Two-stage design rationale:
    - Primary score selects top-150 candidates (buffer of 50 above cutoff)
    - Composite re-ranks those 150 to final 100 using recruiter-behavior signals
      as genuine independent tiebreakers within the high-quality band

    saved_by_recruiters_30d: revealed preference — recruiters who've seen this
    profile and actively saved it. Tier-A candidates average 27 saves vs Tier-C
    at 6 (4.58x ratio), low correlation with vanity metrics (0.18 vs connections),
    0.47 correlation with primary score pool-wide but only 0.18 within top-1000
    (genuine independent signal in the high-quality band).

    github_activity_score: corroborates that shipped ML evidence in career text
    corresponds to real active coding. 64.7% of pool has no GitHub (-1 = 0 here).

    Weight calibration: max bonus = 0.008 + 0.005 = 0.013, which is:
    - Less than the 0.015 gap between tier-2 and tier-3 (ranks 9->10) — tier
      structure at the top is preserved
    - Much less than the 0.056 gap after rank 3 — top-3 ordering is immovable
    - Larger than the 0.000077 gap at rank 100/101 — intentional: we WANT to
      allow boundary candidates with better recruiter signals to enter top-100
      (that's the point of the 150-candidate buffer)
    """
    saved = sig.get("saved_by_recruiters_30d", 0)
    github = max(0.0, sig.get("github_activity_score", -1))

    saved_norm = min(saved, save_p95) / save_p95 if save_p95 > 0 else 0.0
    github_norm = min(github, github_p95) / github_p95 if github_p95 > 0 else 0.0

    return primary_score + 0.008 * saved_norm + 0.005 * github_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug-dump", default=None)
    args = ap.parse_args()

    # Stage 1: score all candidates, keep top-150 by primary score
    # (50-candidate buffer above the 100 cutoff absorbs boundary effects
    # from the composite re-ranking in stage 2)
    all_scored = []
    candidate_sigs = {}
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            s, parts = score_candidate(c)
            reasoning = build_reasoning(c, parts)
            all_scored.append((c["candidate_id"], s, reasoning, parts))
            candidate_sigs[c["candidate_id"]] = c["redrob_signals"]

    all_scored.sort(key=lambda x: (-x[1], x[0]))
    top150 = all_scored[:150]

    # Stage 2: calibrate composite normalization on the top-150 pool,
    # then re-rank to final 100 using composite score
    saves_150 = [candidate_sigs[cid].get("saved_by_recruiters_30d", 0) for cid,*_ in top150]
    githubs_150 = [max(0.0, candidate_sigs[cid].get("github_activity_score", -1)) for cid,*_ in top150]
    save_p95 = sorted(saves_150)[int(0.95 * len(saves_150))]
    github_p95 = sorted(githubs_150)[int(0.95 * len(githubs_150))]

    composited = []
    for cid, primary, reasoning, parts in top150:
        sig = candidate_sigs[cid]
        comp = composite_score(primary, sig, save_p95, github_p95)
        composited.append((cid, comp, primary, reasoning, parts))

    composited.sort(key=lambda x: (-x[1], x[0]))
    top100 = composited[:100]

    # The output 'score' column uses the composite — it reflects the full
    # evidence picture including recruiter signals, and is non-increasing
    # by construction (we sorted by it).
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (cid, comp, primary, reasoning, parts) in enumerate(top100, start=1):
            w.writerow([cid, i, round(comp, 6), reasoning])

    if args.debug_dump:
        with open(args.debug_dump, "w") as f:
            json.dump(
                [{
                    "candidate_id": cid,
                    "score": comp,
                    "primary_score": primary,
                    "reasoning": r,
                    **p
                } for cid, comp, primary, r, p in top100],
                f, indent=2, default=str
            )

    print(f"Wrote {len(top100)} rows to {args.out}")
    print(f"Total candidates scored: {len(all_scored)}")


if __name__ == "__main__":
    main()
