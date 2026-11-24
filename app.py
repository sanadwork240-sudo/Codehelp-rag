
"""
Streamlit UI: Chat tab + Admin tab.
Run: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import config
import db
import ingest
import rag


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Helpdesk Assistant",
    page_icon="🎓",
    layout="wide"
)

db.init_db()


# =========================================================
# SIDEBAR: KNOWLEDGE BASE
# =========================================================

with st.sidebar:
    st.header("📚 Knowledge base")

    uploads = st.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

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

    for source in sources:
        st.write("•", source)


# =========================================================
# TABS
# =========================================================

chat_tab, admin_tab = st.tabs(
    ["💬 Chat", "🛠️ Admin"]
)


# =========================================================
# CHAT TAB
# =========================================================

with chat_tab:

    st.title("🎓 AI Helpdesk Assistant")

    st.caption(
        "Ask about admissions, fees, hostel, exams, attendance, "
        "and library. Answers come only from uploaded documents."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for m in st.session_state.messages:

        with st.chat_message(m["role"]):

            st.write(m["content"])

            if m["role"] == "assistant":

                if m.get("sources"):
                    source_text = "; ".join(
                        f"{s['source']} (p.{s['page']})"
                        for s in m["sources"]
                    )

                    st.caption("📄 Sources: " + source_text)

                if m.get("ticket_id"):
                    st.warning(
                        f"Ticket #{m['ticket_id']} created "
                        "for staff follow-up."
                    )

                if m.get("log_id"):
                    fb = st.feedback(
                        "thumbs",
                        key=f"fb_{m['log_id']}"
                    )

                    if fb is not None:
                        db.save_feedback(
                            m["log_id"],
                            1 if fb == 1 else -1
                        )

    # Accept a new question
    if prompt := st.chat_input("Type your question..."):

        history = [
            {
                "role": m["role"],
                "content": m["content"]
            }
            for m in st.session_state.messages
        ]

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.spinner("Searching documents..."):
            res = rag.answer_question(prompt, history)

        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "sources": res["sources"],
            "ticket_id": res["ticket_id"],
            "log_id": res["log_id"]
        })

        st.rerun()

    # Clear conversation
    if st.session_state.messages:

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


# =========================================================
# ADMIN TAB
# =========================================================

with admin_tab:

    st.title("🛠️ Admin Dashboard")

    # -----------------------------------------------------
    # ANALYTICS
    # -----------------------------------------------------

    s = db.stats()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total questions",
        s["total"]
    )

    c2.metric(
        "Answered from docs",
        f"{s['answer_rate']:.0f}%"
    )

    c3.metric(
        "Open tickets",
        s["open_tickets"]
    )

    c4.metric(
        "Satisfaction",
        (
            f"{s['satisfaction']:.0f}% ({s['rated']} votes)"
            if s["satisfaction"] is not None
            else "–"
        )
    )

    st.divider()

    # -----------------------------------------------------
    # MOST ASKED QUESTIONS
    # -----------------------------------------------------

    st.subheader("📊 Most asked questions")

    tq = db.top_questions()

    if tq:
        questions_df = pd.DataFrame(tq)

        st.dataframe(
            questions_df[["question", "times"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "question": st.column_config.TextColumn(
                    "Question",
                    width="large"
                ),
                "times": st.column_config.NumberColumn(
                    "Times Asked",
                    format="%d"
                )
            }
        )
    else:
        st.info("No questions yet.")

    st.divider()

    # -----------------------------------------------------
    # AI-MANAGED OPEN TICKETS
    # -----------------------------------------------------

    st.subheader("🎫 AI-managed open tickets")

    tickets = db.get_tickets("open")

    if not tickets:
        st.success("No open tickets 🎉")

    else:

        # -------------------------------------------------
        # TICKET FILTERS
        # -------------------------------------------------

        st.subheader("🔎 Filter and sort tickets")

        col1, col2, col3 = st.columns(3)

        categories = sorted({
            t.get("category") or "Other"
            for t in tickets
        })

        with col1:
            selected_category = st.selectbox(
                "Category",
                ["All"] + categories,
                key="ticket_category_filter"
            )

        with col2:
            selected_priority = st.selectbox(
                "Priority",
                ["All", "Critical", "High", "Medium", "Low"],
                key="ticket_priority_filter"
            )

        with col3:
            sort_option = st.selectbox(
                "Sort by",
                [
                    "Newest First",
                    "Oldest First",
                    "Highest Priority First"
                ],
                key="ticket_sort_option"
            )

        # -------------------------------------------------
        # APPLY FILTERS
        # -------------------------------------------------

        filtered_tickets = tickets.copy()

        if selected_category != "All":
            filtered_tickets = [
                t for t in filtered_tickets
                if (t.get("category") or "Other")
                == selected_category
            ]

        if selected_priority != "All":
            filtered_tickets = [
                t for t in filtered_tickets
                if (t.get("priority") or "Medium")
                == selected_priority
            ]

        # -------------------------------------------------
        # APPLY SORTING
        # -------------------------------------------------

        if sort_option == "Newest First":
            filtered_tickets.sort(
                key=lambda t: t.get("ts") or "",
                reverse=True
            )

        elif sort_option == "Oldest First":
            filtered_tickets.sort(
                key=lambda t: t.get("ts") or ""
            )

        elif sort_option == "Highest Priority First":

            priority_order = {
                "Critical": 0,
                "High": 1,
                "Medium": 2,
                "Low": 3
            }

            filtered_tickets.sort(
                key=lambda t: priority_order.get(
                    t.get("priority") or "Medium",
                    2
                )
            )

        st.caption(
            f"Showing {len(filtered_tickets)} of "
            f"{len(tickets)} open tickets"
        )

        # -------------------------------------------------
        # TICKET DISPLAY
        # -------------------------------------------------

        priority_icons = {
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🟠",
            "Critical": "🔴"
        }

        if not filtered_tickets:
            st.info("No tickets match the selected filters.")

        for t in filtered_tickets:

            priority = t.get("priority") or "Medium"
            category = t.get("category") or "Other"
            summary = t.get("summary") or "Not available"

            department = (
                t.get("department")
                or "General Administration"
            )

            icon = priority_icons.get(priority, "⚪")

            with st.expander(
                f"{icon} Ticket #{t['id']} | "
                f"{category} | {priority}"
            ):

                st.write(f"**Date:** {t['ts']}")

                st.write(
                    f"**Student question:** {t['question']}"
                )

                st.write(
                    f"**AI summary:** {summary}"
                )

                st.write(
                    f"**Category:** {category}"
                )

                st.write(
                    f"**Assigned department:** {department}"
                )

                st.write(
                    f"**Priority:** {icon} {priority}"
                )

                st.write(
                    f"**Status:** {t['status']}"
                )

                if st.button(
                    "Resolve ticket",
                    key=f"resolve_{t['id']}"
                ):

                    db.resolve_ticket(t["id"])

                    st.success(
                        f"Ticket #{t['id']} resolved."
                    )

                    st.rerun()

    st.divider()

    # -----------------------------------------------------
    # NEGATIVE FEEDBACK
    # -----------------------------------------------------

    st.subheader("👎 Answers rated negatively")

    bad = db.disliked_answers()

    if bad:

        st.dataframe(
            pd.DataFrame(bad),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No negative feedback yet.")
