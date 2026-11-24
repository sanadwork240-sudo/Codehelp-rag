"""Retrieval + prompting + answer logic (with fallback + ticket creation)."""
import os
import time
from dotenv import load_dotenv


from google import genai
from google.genai import types
import config
import db
from ingest import get_collection

load_dotenv()

GREETINGS = {"hi", "hello", "hey", "hii", "thanks", "thank you", "ok", "okay", "bye"}

SYSTEM_PROMPT = f"""You are a helpful college helpdesk assistant.
Rules:
- Answer ONLY using the CONTEXT provided. Never use outside knowledge.
- If the context does not contain the answer, reply exactly: "{config.NOT_FOUND_MSG}"
- Be concise and clear. Use a short list when the answer has several items.
- Never invent fees, dates, percentages or policies.
- If you do calculations (e.g. adding fees), show the numbers briefly.
- The context must explicitly mention the thing asked about. If the question names a program, fee, person or rule that is not stated in the context, reply with the not-found message. Do not assume (for example, B.Tech being offered does not mean M.Tech is offered)."""

def _chat(messages, max_tokens=500):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    client = genai.Client(api_key=key)
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in messages if m["role"] != "system")

    last_err = None
    for model in [config.LLM_MODEL] + config.FALLBACK_MODELS:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=user,
                    config=types.GenerateContentConfig(system_instruction=system or None, temperature=0),
                )
                return (resp.text or "").strip()
            except Exception as e:
                last_err = e
                msg = str(e)
                if any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                    time.sleep(3 * (attempt + 1))   # busy or rate-limited: wait, retry, then next model
                    continue
                if "404" in msg or "NOT_FOUND" in msg:
                    break                            # model unavailable: try the next one
                raise                                # anything else (bad key etc.): show it
    raise last_err

def rewrite_question(question, history):
    """Turn a follow-up ('and for ECE?') into a standalone question using chat history."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        "Given the chat history and the latest user question, rewrite the latest question "
        "as a complete standalone question. Do NOT answer it. If it is already standalone, "
        "return it unchanged. Output only the question.\n\n"
        f"Chat history:\n{convo}\n\nLatest question: {question}\n\nStandalone question:"
    )
    return _chat([{"role": "user", "content": prompt}], max_tokens=100) or question


def retrieve(query, k=config.TOP_K):
    res = get_collection().query(
        query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"]
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "page": meta["page"], "score": 1 - dist})
    return hits


def answer_question(question, history=None):
    history = history or []
    result = {"answer": "", "sources": [], "score": 0.0, "answered": False,
              "ticket_id": None, "log_id": None, "standalone": question}

    if question.strip().lower().strip("!.? ") in GREETINGS:
        result["answer"] = "Hi! Ask me anything about admissions, fees, hostel, exams, attendance or the library."
        return result

    if get_collection().count() == 0:
        result["answer"] = "No documents have been ingested yet. Run `python ingest.py` or upload files in the sidebar."
        return result

    try:
        standalone = rewrite_question(question, history)
        hits = retrieve(standalone)
        good = [h for h in hits if h["score"] >= config.SIM_THRESHOLD]
        result["standalone"] = standalone
        result["score"] = hits[0]["score"] if hits else 0.0

        if not good:
            answered = False
            answer = config.NOT_FOUND_MSG
        else:
            context = "\n\n".join(f"[Source: {h['source']}, page {h['page']}]\n{h['text']}" for h in good)
            answer = _chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {standalone}"},
            ])
            answered = config.NOT_FOUND_MSG.lower() not in answer.lower()
    except Exception as e:  # API down, bad key, etc. -> no ticket, just tell the user
        result["answer"] = f"Sorry, something went wrong: {e}"
        return result

    sources = []
    if answered:
        seen = set()
        for h in good:
            key = (h["source"], h["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"source": h["source"], "page": h["page"]})

    if not answered:
        answer = config.NOT_FOUND_MSG + " I've logged your question so a staff member can follow up."

    src_text = "; ".join(f"{s['source']} p.{s['page']}" for s in sources)
    log_id = db.log_chat(question, result["standalone"], answer, result["score"], answered, src_text)
    ticket_id = None if answered else db.create_ticket(question, log_id)

    result.update(answer=answer, sources=sources, answered=answered, log_id=log_id, ticket_id=ticket_id)
    return result
