"""Full-pipeline evaluation: runs every question through the bot, uses the LLM as judge.
Run from project folder:  python eval/llm_eval.py        (add --n 10 for a quick test)
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
config.DB_PATH = config.BASE / "data" / "eval.db"   # keep eval runs out of the real admin dashboard
import db, rag

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="only run first N questions")
ap.add_argument("--delay", type=float, default=12, help="seconds between questions (free-tier rate limit)")
args = ap.parse_args()

db.init_db()
qs = json.load(open(Path(__file__).parent / "questions.json", encoding="utf-8"))["questions"]
if args.n:
    qs = qs[: args.n]

JUDGE = """You are grading a chatbot answer.
Question: {q}
Expected answer: {exp}
Bot answer: {ans}
Does the bot answer contain the expected information and no contradicting facts?
Reply with exactly one word: CORRECT or WRONG."""

correct = ans_total = halluc = unans_total = 0
rows = []
for q in qs:
    res = rag.answer_question(q["question"], [])
    if q["answerable"]:
        ans_total += 1
        verdict = rag._chat([{"role": "user", "content": JUDGE.format(
            q=q["question"], exp=q["expected_answer"], ans=res["answer"])}], 5).upper()
        ok = verdict.startswith("CORRECT")
        correct += ok
    else:
        unans_total += 1
        ok = not res["answered"]          # correct behaviour = refuse
        halluc += not ok
    print(("PASS " if ok else "FAIL ") + f"#{q['id']} {q['question']}")
    if not ok:
        print("      bot said:", res["answer"][:200].replace("\n", " "))
    rows.append({"id": q["id"], "ok": bool(ok), "answer": res["answer"]})
    time.sleep(args.delay)

print(f"\nChunk size {config.CHUNK_SIZE} | top-k {config.TOP_K} | threshold {config.SIM_THRESHOLD}")
print(f"Answer correctness : {correct}/{ans_total} = {correct / max(ans_total, 1):.0%}")
print(f"Hallucination rate : {halluc}/{unans_total} = {halluc / max(unans_total, 1):.0%}  (answered when it should refuse)")
json.dump(rows, open(Path(__file__).parent / "results.json", "w"), indent=2)