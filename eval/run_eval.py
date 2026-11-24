"""Retrieval-only evaluation (no LLM calls, so it's free and fast).

Run from the project folder:   python eval/run_eval.py

  * Hit rate      : for answerable questions, is the correct (file, page) in the top-3 chunks?
  * Refusal rate  : for unanswerable questions, is the best score below the threshold?
Change CHUNK_SIZE / TOP_K in config.py, run  python ingest.py --reset  and re-run to compare.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from rag import retrieve

data = json.load(open(Path(__file__).parent / "questions.json", encoding="utf-8"))["questions"]
hits = total_ans = refused = total_unans = 0

for q in data:
    res = retrieve(q["question"], k=3)
    if q["answerable"]:
        total_ans += 1
        ok = any(r["source"] == q["source_file"] and r["page"] == q["page"] for r in res)
        hits += ok
        if not ok:
            print(f"MISS  #{q['id']}: {q['question']}  (top: {res[0]['source']} p.{res[0]['page']} {res[0]['score']:.2f})")
    else:
        total_unans += 1
        ok = not res or res[0]["score"] < config.SIM_THRESHOLD
        refused += ok
        if not ok:
            print(f"LEAK  #{q['id']}: {q['question']}  (score {res[0]['score']:.2f} >= {config.SIM_THRESHOLD})")

print(f"\nChunk size {config.CHUNK_SIZE}, threshold {config.SIM_THRESHOLD}")
print(f"Retrieval hit rate (top-3): {hits}/{total_ans} = {hits / total_ans:.0%}")
print(f"Correct refusals          : {refused}/{total_unans} = {refused / total_unans:.0%}")
