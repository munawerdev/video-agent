import html
import json
import math
import re
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import ask_question, build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme ──────────────────────────────────────────────────────────────────────
# Every colour comes from these tokens, and every Streamlit widget is restyled
# from them, so dark and light both work (not just the page background).
THEMES = {
    "dark": {
        "bg": "#0f1320",
        "surface": "#171c2c",
        "surface2": "#1e2438",
        "line": "#2b3350",
        "text": "#e8ebf4",
        "muted": "#98a1bb",
        "accent": "#8b9cff",
        "accent-ink": "#0f1320",
        "good": "#5ad1a0",
        "bad": "#ff7b8a",
        "glow": "rgba(139,156,255,.16)",
        "glow2": "rgba(139,156,255,.45)",
    },
    "light": {
        "bg": "#f4f6fb",
        "surface": "#ffffff",
        "surface2": "#eef1f8",
        "line": "#d9deec",
        "text": "#151a2b",
        "muted": "#5d6785",
        "accent": "#3a4fd7",
        "accent-ink": "#ffffff",
        "good": "#12805c",
        "bad": "#c9344b",
        "glow": "rgba(58,79,215,.12)",
        "glow2": "rgba(58,79,215,.4)",
    },
}

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400&display=swap');

html, body, .stApp { font-family: 'DM Sans', system-ui, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.stApp :where(p, li, label, h1, h2, h3, h4, h5, h6, summary) { color: var(--text); }
.stApp :where(h1, h2, h3, h4) { font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.02em; }
.stApp :where(small, [data-testid="stCaptionContainer"]) { color: var(--muted); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 2.2rem; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-testid="stBottom"] > div { background: var(--bg); }

/* ── Inputs ── */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div {
    background: var(--surface2) !important; border-color: var(--line) !important; border-radius: 10px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--glow) !important;
}
input, textarea, [data-baseweb="select"] * { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }
input::placeholder, textarea::placeholder { color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: 1; }
[data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] [data-baseweb="menu"] { background: var(--surface2) !important; }
[data-baseweb="popover"] li { color: var(--text) !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: var(--glow) !important; }
[data-testid="stChatInput"] { background: var(--surface2); border: 1px solid var(--line); border-radius: 14px; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--glow); }

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    background: var(--surface2); color: var(--text); border: 1px solid var(--line);
    border-radius: 10px; font-weight: 700; transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); transform: translateY(-1px); color: var(--text); }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0) scale(.98); }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.stButton > button[kind="primary"]:hover { box-shadow: 0 6px 20px var(--glow2); color: var(--accent-ink); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .65rem 1rem; font-weight: 700; color: var(--muted); background: transparent; }
.stTabs [aria-selected="true"] { color: var(--text); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 3px; border-radius: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }
[data-baseweb="tab-panel"] { animation: fade .3s ease both; padding-top: 1.1rem; }

/* ── Cards (st.container(key="card_…")) ── */
[class*="st-key-card"] { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.1rem 1.3rem; }
.card-h { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; margin: 0 0 .4rem; color: var(--text); }

/* ── Masthead + waveform (the one signature element) ── */
.masthead { display: flex; align-items: flex-end; justify-content: space-between; gap: 2rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.app-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: clamp(2rem, 4.5vw, 3rem); letter-spacing: -0.04em; line-height: 1; color: var(--text); }
.app-sub { color: var(--muted); margin-top: .5rem; max-width: 34rem; }
.wave { display: flex; align-items: center; gap: 3px; height: 56px; }
.wave i { width: 4px; height: var(--h); border-radius: 4px; background: var(--muted); opacity: .55; transform-origin: center;
    animation: bar 2.4s ease-in-out infinite alternate; animation-delay: var(--d); }
.wave.live i { background: var(--accent); opacity: 1; animation-duration: .7s; }
.wave.compact { height: 32px; }

/* ── Pipeline stepper ── */
.stepper { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.1rem 1.3rem; margin-bottom: 1.2rem; }
.stepper .bar { height: 4px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
.stepper .bar i { display: block; height: 100%; background: var(--accent); border-radius: 4px; transition: width .6s ease; }
.stepper ul { list-style: none; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .6rem; margin: 1rem 0 0; padding: 0; }
.stepper li { display: flex; align-items: center; gap: .55rem; font-size: .88rem; color: var(--muted); }
.stepper .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--line); flex: none; }
.stepper li.active { color: var(--text); font-weight: 700; }
.stepper li.active .dot { background: var(--accent); animation: ring 1.2s ease-out infinite; }
.stepper li.done .dot { background: var(--good); }
.stepper li.error { color: var(--bad); font-weight: 700; }
.stepper li.error .dot { background: var(--bad); }
.stepper .meta { margin-top: .8rem; font-size: .8rem; color: var(--muted); }

/* ── Result header + stats ── */
.session-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: clamp(1.5rem, 3vw, 2.1rem); letter-spacing: -0.03em; line-height: 1.15; color: var(--text); }
.chips { display: flex; flex-wrap: wrap; gap: .45rem; margin: .7rem 0 1.1rem; }
.chip { background: var(--surface2); border: 1px solid var(--line); border-radius: 999px; padding: .2rem .7rem; font-size: .8rem; color: var(--muted); }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; margin-bottom: 1.2rem; }
.stat { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: .8rem 1rem; }
.stat b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; letter-spacing: -0.03em; color: var(--text); }
.stat span { font-size: .82rem; color: var(--muted); }
.reveal { animation: rise .55s cubic-bezier(.2,.7,.2,1) both; }

/* ── Transcript ── */
.transcript { font-family: 'JetBrains Mono', monospace; font-size: .84rem; line-height: 1.85; color: var(--muted);
    white-space: pre-wrap; word-break: break-word; max-height: 520px; overflow-y: auto; }
.transcript mark { background: var(--glow2); color: var(--text); border-radius: 3px; padding: 0 2px; }
.progress-line { height: 4px; background: var(--surface2); border-radius: 4px; overflow: hidden; margin: .2rem 0 .8rem; }
.progress-line i { display: block; height: 100%; background: var(--good); transition: width .4s ease; }

/* ── Chat ── */
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; animation: rise .35s ease both; }
.typing { display: inline-flex; gap: 5px; padding: .35rem 0; }
.typing i { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); animation: dots 1.2s infinite; }
.typing i:nth-child(2) { animation-delay: .15s; } .typing i:nth-child(3) { animation-delay: .3s; }

/* ── Empty state ── */
.how { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .9rem; margin-top: 1rem; }
.how div { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; color: var(--muted); font-size: .92rem; }
.how b { display: block; color: var(--text); font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; margin-bottom: .2rem; }

@keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes bar { from { transform: scaleY(.3); } to { transform: scaleY(1); } }
@keyframes ring { 0% { box-shadow: 0 0 0 0 var(--glow2); } 100% { box-shadow: 0 0 0 9px transparent; } }
@keyframes dots { 0%, 80%, 100% { opacity: .25; transform: none; } 40% { opacity: 1; transform: translateY(-3px); } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""

# ─── State ──────────────────────────────────────────────────────────────────────
for key, default in {"history": [], "active": None, "pending_question": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

dark = st.session_state.get("dark_mode", True)
tokens = THEMES["dark" if dark else "light"]
st.markdown(
    "<style>:root{"
    + ";".join(f"--{k}:{v}" for k, v in tokens.items())
    + "}</style>"
    + BASE_CSS,
    unsafe_allow_html=True,
)

STEPS = [
    ("audio", "Preparing audio"),
    ("transcript", "Transcribing"),
    ("title", "Naming the session"),
    ("summary", "Summarising"),
    ("extract", "Finding next steps"),
    ("rag", "Indexing for chat"),
]
SUGGESTIONS = [
    "What were the main decisions?",
    "Who owns which action items?",
    "Give me a 3-bullet recap",
    "What is still unresolved?",
]
STOP = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "have",
    "from",
    "they",
    "them",
    "their",
    "there",
    "were",
    "been",
    "will",
    "would",
    "could",
    "should",
    "about",
    "into",
    "over",
    "also",
    "what",
    "which",
    "when",
    "where",
    "while",
    "your",
    "you",
    "are",
    "was",
    "but",
    "not",
    "can",
    "just",
    "then",
    "than",
    "very",
    "more",
    "most",
    "some",
    "such",
    "like",
    "know",
    "going",
    "think",
    "want",
    "really",
    "right",
    "okay",
    "yeah",
    "well",
}


# ─── Helpers ────────────────────────────────────────────────────────────────────
def validate_source(source: str) -> str | None:
    source = source.strip()
    if not source:
        return "Add a YouTube link or a file path in the sidebar first."
    if re.match(r"^https?://", source):
        return None
    if not Path(source).expanduser().exists():
        return "That file wasn't found. Check the path, or paste a link that starts with https://."
    return None


def wave_html(live: bool = False, compact: bool = False) -> str:
    bars = "".join(
        f'<i style="--h:{20 + 75 * abs(math.sin(i * 0.55) * math.cos(i * 0.21)):.0f}%;--d:-{(i * 0.17) % 2.2:.2f}s"></i>'
        for i in range(30)
    )
    return f'<div class="wave{" live" if live else ""}{" compact" if compact else ""}">{bars}</div>'


def stepper_html(states: dict, started: float) -> str:
    done = sum(v == "done" for v in states.values())
    items = "".join(
        f'<li class="{states.get(k, "pending")}"><span class="dot"></span>{label}</li>'
        for k, label in STEPS
    )
    return (
        f'<div class="stepper"><div class="bar"><i style="width:{done / len(STEPS) * 100:.0f}%"></i></div>'
        f'<ul>{items}</ul><div class="meta">Step {min(done + 1, len(STEPS))} of {len(STEPS)} · '
        f"{time.time() - started:.0f}s elapsed</div></div>"
    )


def run_pipeline(source: str, language: str) -> dict:
    started = time.time()
    states: dict = {}
    box = st.empty()

    @contextmanager
    def step(key):
        states[key] = "active"
        box.markdown(stepper_html(states, started), unsafe_allow_html=True)
        try:
            yield
        except Exception:
            states[key] = "error"
            box.markdown(stepper_html(states, started), unsafe_allow_html=True)
            raise
        states[key] = "done"
        box.markdown(stepper_html(states, started), unsafe_allow_html=True)

    with step("audio"):
        chunks = process_input(source)
    with step("transcript"):
        transcript = transcribe_all(chunks, language)
    with step("title"):
        title = generate_title(transcript)
    with step("summary"):
        summary = summarize(transcript)
    with step("extract"):
        actions = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
    with step("rag"):
        chain = build_rag_chain(transcript)

    return {
        "id": int(time.time() * 1000),
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": actions,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": chain,
        "chat": [],
        "source": source,
        "language": language,
        "elapsed": time.time() - started,
        "created": datetime.now().strftime("%d %b %Y, %H:%M"),
    }


def parse_items(text) -> list[str]:
    items = []
    for line in str(text).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line and not line.startswith("#") and not line.endswith(":"):
            items.append(line)
    return items


def top_keywords(text: str, n: int = 10) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", text.lower())
    return [w for w, _ in Counter(w for w in words if w not in STOP).most_common(n)]


def highlight(text: str, query: str) -> tuple[str, int]:
    safe = html.escape(text)
    if not query.strip():
        return safe, 0
    pattern = re.compile(re.escape(html.escape(query.strip())), re.IGNORECASE)
    return pattern.subn(lambda m: f"<mark>{m.group(0)}</mark>", safe)


def build_report(r: dict) -> str:
    return (
        f"# {r['title']}\n\n_Source: {r['source']} — analysed {r['created']}_\n\n"
        f"## Summary\n\n{r['summary']}\n\n## Action items\n\n{r['action_items']}\n\n"
        f"## Key decisions\n\n{r['key_decisions']}\n\n## Open questions\n\n{r['open_questions']}\n\n"
        f"## Transcript\n\n{r['transcript']}\n"
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 AI Video Assistant")
    source = st.text_input(
        "Video or audio source",
        placeholder="https://youtube.com/watch?v=… or /path/to/file.mp4",
        help="Paste a YouTube link or the full path to a local file.",
    )
    language = st.selectbox(
        "Spoken language", ["english", "hinglish"], format_func=str.capitalize
    )
    run_btn = st.button("Analyse recording", type="primary", use_container_width=True)

    if st.session_state.history:
        st.divider()
        st.caption("Recent sessions")
        for i, h in enumerate(st.session_state.history):
            label = h["title"] if len(h["title"]) <= 34 else h["title"][:33] + "…"
            if st.button(
                label,
                key=f"hist_{h['id']}",
                use_container_width=True,
                type="primary" if i == st.session_state.active else "secondary",
            ):
                st.session_state.active = i
                st.rerun()

    st.divider()
    st.toggle("Dark mode", value=True, key="dark_mode")

# ─── Masthead ───────────────────────────────────────────────────────────────────
has_result = st.session_state.active is not None
st.markdown(
    f'<div class="masthead"><div><div class="app-title">AI Video Assistant</div>'
    f'<div class="app-sub">Turn a recording into a summary, next steps and a transcript you can question.</div></div>'
    f"{wave_html(live=run_btn, compact=has_result)}</div>",
    unsafe_allow_html=True,
)

# ─── Run ────────────────────────────────────────────────────────────────────────
failed = False
if run_btn:
    problem = validate_source(source)
    if problem:
        failed = True
        st.warning(problem)
    else:
        try:
            result = run_pipeline(source.strip(), language)
        except Exception as e:
            failed = True
            st.error(f"Something went wrong: {e}")
            st.caption("Check the source and your API keys, then run it again.")
        else:
            st.session_state.history = ([result] + st.session_state.history)[:6]
            st.session_state.active = 0
            st.rerun()

# ─── Results ────────────────────────────────────────────────────────────────────
r = (
    st.session_state.history[st.session_state.active]
    if st.session_state.active is not None
    else None
)

if r and not failed:
    actions, decisions = parse_items(r["action_items"]), parse_items(r["key_decisions"])
    words = len(r["transcript"].split())

    with st.container(key="results"):
        st.markdown(
            f'<div class="reveal"><div class="session-title">{html.escape(r["title"])}</div>'
            f'<div class="chips"><span class="chip">{r["language"].capitalize()}</span>'
            f'<span class="chip">Analysed {r["created"]}</span>'
            f'<span class="chip">{html.escape(r["source"][:48])}</span></div>'
            f'<div class="stats">'
            f'<div class="stat"><b>{words:,}</b><span>words transcribed</span></div>'
            f'<div class="stat"><b>{max(1, round(words / 200))} min</b><span>to read</span></div>'
            f'<div class="stat"><b>{len(actions)}</b><span>action items</span></div>'
            f'<div class="stat"><b>{len(decisions)}</b><span>decisions</span></div></div></div>',
            unsafe_allow_html=True,
        )

        t_over, t_next, t_text, t_chat = st.tabs(
            ["Overview", "Next steps", "Transcript", "Ask questions"]
        )

        # Overview
        with t_over:
            with st.container(key="card_summary"):
                st.markdown('<div class="card-h">Summary</div>', unsafe_allow_html=True)
                st.markdown(r["summary"])
            kws = top_keywords(r["transcript"])
            if kws:
                st.markdown(
                    '<div class="card-h" style="margin-top:1.2rem">Topics that came up most</div>'
                    '<div class="chips">'
                    + "".join(
                        f'<span class="chip">{html.escape(k)}</span>' for k in kws
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
            d1, d2, d3 = st.columns(3)
            d1.download_button(
                "Report (.md)",
                build_report(r),
                "report.md",
                "text/markdown",
                use_container_width=True,
            )
            d2.download_button(
                "Data (.json)",
                json.dumps(
                    {k: v for k, v in r.items() if k not in ("rag_chain", "chat")},
                    indent=2,
                    default=str,
                ),
                "analysis.json",
                "application/json",
                use_container_width=True,
            )
            d3.download_button(
                "Transcript (.txt)",
                r["transcript"],
                "transcript.txt",
                use_container_width=True,
            )

        # Next steps
        with t_next:
            left, right = st.columns([3, 2], gap="medium")
            with left, st.container(key="card_actions"):
                st.markdown(
                    '<div class="card-h">Action items</div>', unsafe_allow_html=True
                )
                if actions:
                    done = sum(
                        bool(st.session_state.get(f"ai_{r['id']}_{i}"))
                        for i in range(len(actions))
                    )
                    st.markdown(
                        f'<div class="progress-line"><i style="width:{done / len(actions) * 100:.0f}%"></i></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{done} of {len(actions)} done")
                    for i, item in enumerate(actions):
                        st.checkbox(item, key=f"ai_{r['id']}_{i}")
                else:
                    st.caption("No action items were found in this recording.")
            with right:
                with st.container(key="card_decisions"):
                    st.markdown(
                        '<div class="card-h">Key decisions</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(r["key_decisions"] or "_None found._")
                st.write("")
                with st.container(key="card_questions"):
                    st.markdown(
                        '<div class="card-h">Open questions</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(r["open_questions"] or "_None found._")

        # Transcript
        with t_text:
            q_col, k_col = st.columns([2, 3], vertical_alignment="bottom")
            query = q_col.text_input(
                "Search",
                placeholder="Search the transcript",
                label_visibility="collapsed",
            )
            picked = None
            if kws and hasattr(st, "pills"):
                picked = k_col.pills(
                    "Jump to topic",
                    kws,
                    label_visibility="collapsed",
                    key=f"pill_{r['id']}",
                )
            term = query.strip() or (picked or "")
            body, hits = highlight(r["transcript"], term)
            if term:
                st.caption(
                    f"{hits} match{'es' if hits != 1 else ''} for “{term}”"
                    if hits
                    else f"No matches for “{term}”"
                )
            with st.container(key="card_transcript"):
                st.markdown(
                    f'<div class="transcript">{body}</div>', unsafe_allow_html=True
                )

        # Chat
        with t_chat:
            chat = r["chat"]
            if not chat:
                st.caption("Start with a suggestion, or type your own question below.")
                cols = st.columns(2)
                for i, s in enumerate(SUGGESTIONS):
                    if cols[i % 2].button(s, key=f"sugg_{i}", use_container_width=True):
                        st.session_state.pending_question = s
                        st.rerun()

            question = st.chat_input(
                "Ask anything about this recording"
            ) or st.session_state.pop("pending_question", None)

            for msg in chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if question:
                chat.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    slot = st.empty()
                    slot.markdown(
                        '<div class="typing"><i></i><i></i><i></i></div>',
                        unsafe_allow_html=True,
                    )
                    try:
                        answer = ask_question(r["rag_chain"], question)
                    except Exception as e:
                        answer = f"Sorry, I couldn't answer that: {e}"
                    slot.markdown(answer)
                chat.append({"role": "assistant", "content": answer})

            if chat and st.button("Clear conversation"):
                r["chat"] = []
                st.rerun()

elif not failed and not run_btn:
    st.markdown(
        '<div class="how reveal">'
        "<div><b>1. Add a source</b>Paste a YouTube link or a file path in the sidebar.</div>"
        "<div><b>2. Pick the language</b>English and Hinglish are supported.</div>"
        "<div><b>3. Analyse</b>You get a summary, next steps and a chat that knows the transcript.</div></div>",
        unsafe_allow_html=True,
    )
