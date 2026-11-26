
"""
Greenfield Institute of Technology
AI Helpdesk Assistant

Run:
    streamlit run app.py
"""

import hmac
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

import config
import db
import ingest
import rag


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("helpdesk.app")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Greenfield AI Helpdesk",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

db.init_db()


# =========================================================
# FIRST-START KNOWLEDGE BASE SETUP
# =========================================================

@st.cache_resource(show_spinner="Preparing knowledge base...")
def bootstrap_kb():
    """Index documents automatically when the vector database is empty."""
    if ingest.get_collection().count() == 0:
        total = ingest.ingest_folder(config.RAW_DIR)
        logger.info("Knowledge base initialized with %s chunks.", total)

    return True


bootstrap_kb()


# =========================================================
# ADMIN ACCESS
# =========================================================

def is_admin():
    return bool(st.session_state.get("is_admin", False))


def admin_unlocked():
    """Require the configured admin password."""
    if is_admin():
        return True

    expected = config.ADMIN_PASSWORD

    if not expected:
        st.warning(
            "Admin access is disabled because ADMIN_PASSWORD "
            "has not been configured."
        )
        return False

    st.info("🔒 Enter the admin password to access the dashboard.")

    password = st.text_input(
        "Admin password",
        type="password",
        key="admin_password_input"
    )

    if password:
        if hmac.compare_digest(
            password.encode("utf-8"),
            expected.encode("utf-8")
        ):
            st.session_state.is_admin = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


# =========================================================
# CUSTOM THEME
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #080b09;
        color: #f1f5f2;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d120f;
        border-right: 1px solid #263b2d;
    }

    section[data-testid="stSidebar"] * {
        color: #e5eee7;
    }

    h1, h2, h3 {
        color: #f1f5f2 !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background-color: #14251a;
        color: #42e58b;
        border: 1px solid #34794c;
        border-radius: 9px;
        padding: 9px 18px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1c3826;
        color: white;
        border-color: #42e58b;
    }

    .stChatInput textarea,
    .stTextInput input {
        background-color: #151b17 !important;
        color: white !important;
        border: 1px solid #34794c !important;
    }

    [data-testid="stFileUploader"] {
        background-color: #111914;
        border: 1px dashed #34794c;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetric"] {
        background-color: #111914;
        border: 1px solid #263b2d;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricValue"] {
        color: #42e58b !important;
    }

    hr {
        border-color: #263b2d;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR: KNOWLEDGE BASE
# =========================================================

with st.sidebar:

    st.markdown("## 🎓 Greenfield AI")
    st.caption("Intelligent Student Support System")
    st.divider()

    st.markdown("### 📚 Knowledge Base")

    if is_admin():

        st.caption("Upload documents for the AI assistant.")

        uploads = st.file_uploader(
            "Upload PDF / DOCX / TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button(
            "➕ Add to Knowledge Base",
            disabled=not uploads,
            use_container_width=True
        ):

            config.RAW_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            for uploaded_file in uploads:

                safe_name = Path(uploaded_file.name).name
                destination = config.RAW_DIR / safe_name

                try:
                    # Do not silently overwrite an existing document.
                    if destination.exists():
                        st.warning(
                            f"{safe_name} already exists. "
                            "Rename the new file before uploading it."
                        )
                        continue

                    destination.write_bytes(
                        uploaded_file.getbuffer()
                    )

                    with st.spinner(f"Indexing {safe_name}..."):
                        chunks = ingest.ingest_file(destination)

                    st.success(
                        f"{safe_name}: {chunks} chunks indexed."
                    )

                except Exception:
                    logger.exception(
                        "Could not process uploaded file: %s",
                        safe_name
                    )
                    st.error(
                        f"Could not process {safe_name}. "
                        "Check the application logs."
                    )

    else:
        st.caption(
            "Admins can add documents after logging in "
            "from the Admin Dashboard."
        )

    try:
        sources = ingest.list_sources()
    except Exception:
        logger.exception("Could not load indexed document list.")
        sources = []

    st.divider()
    st.markdown("### 📈 Knowledge Base Status")

    st.metric("Indexed Documents", len(sources))

    st.markdown("### 📄 Available Documents")

    if sources:
        for source in sources:
            st.caption(f"📘 {source}")
    else:
        st.info("No documents indexed yet.")

    st.divider()
    st.caption("Greenfield Institute of Technology")


# =========================================================
# MAIN NAVIGATION
# =========================================================

chat_tab, admin_tab = st.tabs([
    "💬 Chat Assistant",
    "🛠️ Admin Dashboard"
])


# =========================================================
# CHAT TAB
# =========================================================

with chat_tab:

    st.title("🎓 AI Helpdesk Assistant")

    st.caption(
        "Ask questions about admissions, fees, hostel, "
        "examinations, attendance and library services."
    )

    st.divider()

    if config.get_setting("DEMO_MODE", "0") == "1":
        st.info(
            "🧪 Demo: This assistant answers from the "
            "Greenfield Institute documents."
        )

    st.info(
        "👋 Welcome! Ask me anything about Greenfield Institute. "
        "I'll search the institute documents for your answer."
    )

    # -----------------------------------------------------
    # EXAMPLE QUESTIONS
    # -----------------------------------------------------

    st.markdown("### ✨ Example Questions")

    example_columns = st.columns(3)

    examples = [
        ("🎓 Admissions", "How can I apply?"),
        ("💰 Fees", "What is the CSE tuition fee?"),
        ("🏠 Hostel", "What are the mess timings?")
    ]

    for column, (label, example_question) in zip(
        example_columns,
        examples
    ):

        with column:

            st.markdown(f"**{label}**")

            if st.button(
                example_question,
                key=f"example_{label}",
                use_container_width=True
            ):
                st.session_state.queued_question = example_question
                st.rerun()

    st.divider()

    # -----------------------------------------------------
    # INITIALIZE CHAT STATE
    # -----------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Separate counter so clearing chat does not reset the
    # question limit within the current Streamlit session.
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    # -----------------------------------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])

            if message["role"] == "assistant":

                if message.get("sources"):

                    source_text = "; ".join(
                        f"{source['source']} (p.{source['page']})"
                        for source in message["sources"]
                    )

                    st.caption("📄 Sources: " + source_text)

                if message.get("ticket_id"):

                    st.warning(
                        f"🎫 Ticket #{message['ticket_id']} "
                        "created for staff follow-up."
                    )

                # Admin-only retrieval information.
                if is_admin() and message.get("trace"):

                    with st.expander("🔍 Retrieval trace (admin)"):

                        for item in message["trace"]:

                            used = (
                                "✅ used"
                                if item["used"]
                                else "⛔ below threshold"
                            )

                            st.caption(
                                f"{item['source']} (p.{item['page']}) | "
                                f"similarity {item['score']:.3f} | {used}"
                            )

                            st.text(item["preview"])

                # Feedback buttons.
                if message.get("log_id"):

                    feedback = st.feedback(
                        "thumbs",
                        key=f"feedback_{message['log_id']}"
                    )

                    if feedback is not None:

                        db.save_feedback(
                            message["log_id"],
                            1 if feedback == 1 else -1
                        )

    # -----------------------------------------------------
    # QUESTION INPUT
    # -----------------------------------------------------

    prompt = st.chat_input(
        "Ask your question about the institute..."
    )

    if not prompt and st.session_state.get("queued_question"):
        prompt = st.session_state.pop("queued_question")

    if prompt:

        prompt = prompt.strip()[:config.MAX_QUESTION_CHARS]

        if not prompt:
            st.warning("Please enter a question.")

        elif (
            st.session_state.question_count
            >= config.MAX_QUESTIONS_PER_SESSION
        ):

            st.warning(
                f"Demo limit of {config.MAX_QUESTIONS_PER_SESSION} "
                "questions reached for this session."
            )

        else:

            history = [
                {
                    "role": message["role"],
                    "content": message["content"]
                }
                for message in st.session_state.messages
            ]

            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            st.session_state.question_count += 1

            with st.spinner("🔍 Searching institute documents..."):

                try:

                    response = rag.answer_question(
                        prompt,
                        history
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                        "ticket_id": response["ticket_id"],
                        "log_id": response["log_id"],
                        "trace": response.get("trace", [])
                    })

                except Exception:

                    logger.exception("Chat request failed.")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            "Sorry, something went wrong while "
                            "processing your question. Please try again."
                        ),
                        "sources": [],
                        "ticket_id": None,
                        "log_id": None,
                        "trace": []
                    })

            st.rerun()

    # -----------------------------------------------------
    # CLEAR CONVERSATION
    # -----------------------------------------------------

    if st.session_state.messages:

        st.divider()

        if st.button("🗑️ Clear Conversation"):

            st.session_state.messages = []
            st.rerun()


# =========================================================
# ADMIN DASHBOARD
# =========================================================

def render_admin_dashboard():

    # -----------------------------------------------------
    # ANALYTICS
    # -----------------------------------------------------

    dashboard_stats = db.stats()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Questions",
        dashboard_stats["total"]
    )

    col2.metric(
        "Answered from Docs",
        f"{dashboard_stats['answer_rate']:.0f}%"
    )

    col3.metric(
        "Open Tickets",
        dashboard_stats["open_tickets"]
    )

    col4.metric(
        "Satisfaction",
        (
            f"{dashboard_stats['satisfaction']:.0f}%"
            if dashboard_stats["satisfaction"] is not None
            else "–"
        )
    )

    if dashboard_stats["satisfaction"] is not None:
        st.caption(
            f"Based on {dashboard_stats['rated']} feedback votes."
        )

    st.divider()

    # -----------------------------------------------------
    # MOST ASKED QUESTIONS
    # -----------------------------------------------------

    st.subheader("📊 Most Asked Questions")

    questions = db.top_questions()

    if questions:

        questions_df = pd.DataFrame(questions)

        st.dataframe(
            questions_df[["question", "times"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "question": st.column_config.TextColumn(
                    "Student Question",
                    width="large"
                ),
                "times": st.column_config.NumberColumn(
                    "Times Asked",
                    format="%d"
                )
            }
        )

    else:
        st.info("No questions recorded yet.")

    st.divider()

    # -----------------------------------------------------
    # OPEN TICKETS
    # -----------------------------------------------------

    st.subheader("🎫 AI-Managed Open Tickets")

    tickets = db.get_tickets("open")

    if not tickets:

        st.success("🎉 No open tickets.")

    else:

        priority_icons = {
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🟠",
            "Critical": "🔴"
        }

        for ticket in tickets:

            priority = ticket.get("priority") or "Medium"
            category = ticket.get("category") or "Other"
            summary = ticket.get("summary") or "Not available"
            department = (
                ticket.get("department")
                or "General Administration"
            )

            icon = priority_icons.get(priority, "⚪")

            with st.expander(
                f"{icon} Ticket #{ticket['id']} | "
                f"{category} | {priority}"
            ):

                st.write(f"**Date:** {ticket['ts']}")
                st.write(f"**Student Question:** {ticket['question']}")
                st.write(f"**AI Summary:** {summary}")
                st.write(f"**Category:** {category}")
                st.write(f"**Assigned Department:** {department}")
                st.write(f"**Priority:** {icon} {priority}")
                st.write(f"**Status:** {ticket['status']}")

                if st.button(
                    "✅ Resolve Ticket",
                    key=f"resolve_{ticket['id']}"
                ):

                    db.resolve_ticket(ticket["id"])

                    st.success(
                        f"Ticket #{ticket['id']} resolved."
                    )

                    st.rerun()

    st.divider()

    # -----------------------------------------------------
    # NEGATIVE FEEDBACK
    # -----------------------------------------------------

    st.subheader("👎 Answers Rated Negatively")

    negative_feedback = db.disliked_answers()

    if negative_feedback:

        st.dataframe(
            pd.DataFrame(negative_feedback),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.success("No negative feedback recorded yet.")

    st.divider()

    # -----------------------------------------------------
    # ADMIN LOGOUT
    # -----------------------------------------------------

    if st.button("🔓 Log out of admin"):

        st.session_state.is_admin = False
        st.rerun()

    st.caption(
        "Greenfield AI Helpdesk | Administration Panel"
    )


# =========================================================
# ADMIN TAB
# =========================================================

with admin_tab:

    st.title("🛠️ Admin Dashboard")

    st.caption(
        "Monitor helpdesk activity, student questions "
        "and AI-generated support tickets."
    )

    st.divider()

    if admin_unlocked():
        render_admin_dashboard()
