"""Streamlit UI: Chat tab + Admin tab.   Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

import config
import db
import ingest
import rag

st.set_page_config(page_title="AI Helpdesk Assistant", page_icon="🎓", layout="wide")
db.init_db()

# ---------------- Sidebar: knowledge base ----------------
with st.sidebar:
    st.header("📚 Knowledge base")
    uploads = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Add to knowledge base", disabled=not uploads):
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            dest = config.RAW_DIR / f.name
            dest.write_bytes(f.getbuffer())
            try:
                n = ingest.ingest_file(dest)
                st.success(f"{f.name}: {n} chunks")
            except Exception as e:
                st.error(f"{f.name}: {e}")
    sources = ingest.list_sources()
    st.caption(f"{len(sources)} documents indexed")
    for s in sources:
        st.write("•", s)

chat_tab, admin_tab = st.tabs(["💬 Chat", "🛠️ Admin"])

# ---------------- Chat tab ----------------
with chat_tab:
    st.title("🎓 AI Helpdesk Assistant")
    st.caption("Ask about admissions, fees, hostel, exams, attendance, library. Answers come only from the uploaded documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["role"] == "assistant":
                if m.get("sources"):
                    st.caption("📄 Sources: " + "; ".join(f"{s['source']} (p.{s['page']})" for s in m["sources"]))
                if m.get("ticket_id"):
                    st.warning(f"Ticket #{m['ticket_id']} created for staff follow-up.")
                if m.get("log_id"):
                    fb = st.feedback("thumbs", key=f"fb_{m['log_id']}")
                    if fb is not None:  # 1 = up, 0 = down
                        db.save_feedback(m["log_id"], 1 if fb == 1 else -1)

    if prompt := st.chat_input("Type your question..."):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching documents..."):
            res = rag.answer_question(prompt, history)
        st.session_state.messages.append({
            "role": "assistant", "content": res["answer"], "sources": res["sources"],
            "ticket_id": res["ticket_id"], "log_id": res["log_id"],
        })
        st.rerun()

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Admin tab ----------------
with admin_tab:
    st.title("Admin dashboard")
    s = db.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total questions", s["total"])
    c2.metric("Answered from docs", f"{s['answer_rate']:.0f}%")
    c3.metric("Open tickets", s["open_tickets"])
    c4.metric("Satisfaction", f"{s['satisfaction']:.0f}% ({s['rated']} votes)" if s["satisfaction"] is not None else "–")

    st.subheader("Most asked questions")
    tq = db.top_questions()
    st.dataframe(pd.DataFrame(tq), use_container_width=True) if tq else st.info("No questions yet.")

    st.subheader("Open tickets (unanswered questions)")
    tickets = db.get_tickets("open")
    if not tickets:
        st.info("No open tickets 🎉")
    for t in tickets:
        a, b = st.columns([6, 1])
        a.write(f"**#{t['id']}** · {t['ts']} · {t['question']}")
        if b.button("Resolve", key=f"res_{t['id']}"):
            db.resolve_ticket(t["id"])
            st.rerun()

    st.subheader("Answers rated 👎")
    bad = db.disliked_answers()
    st.dataframe(pd.DataFrame(bad), use_container_width=True) if bad else st.info("None yet.")
