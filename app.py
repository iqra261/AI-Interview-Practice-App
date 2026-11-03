"""AI Interview Practice App - a Streamlit app.

Pick a job role. The AI asks interview questions one at a time, scores your
answers, and suggests improvements. Powered by the Groq API.
"""
"""AI Customer Review Analyzer - a Streamlit app.

Upload a CSV of product reviews and get sentiment charts, common complaints,
praise themes and (optionally) AI-written insights powered by Groq.
"""
"""AI Interview Practice App - a Streamlit app.

Pick a job role. The AI asks interview questions one at a time, scores your
answers, and suggests improvements. Powered by the Groq API.
"""

import json
import os
import re

import pandas as pd
import streamlit as st

try:
    from groq import Groq
except ImportError:
    Groq = None

GROQ_MODEL = "llama-3.3-70b-versatile"

ROLES = [
    "Software Engineer", "Data Analyst", "Data Scientist", "Product Manager",
    "Marketing Manager", "Sales Representative", "Customer Support Specialist",
    "Project Manager", "UX/UI Designer", "HR Specialist", "Accountant",
    "Teacher", "Nurse", "Other (type your own)",
]
LEVELS = ["Entry level / Intern", "Junior (1-3 years)", "Mid-level (3-6 years)", "Senior (6+ years)"]
TYPES = ["Mixed", "Behavioral", "Technical / Role-specific"]

st.set_page_config(page_title="AI Interview Practice", page_icon="🎤", layout="centered")


# ----------------------------------------------------------------------------
# Groq helpers
# ----------------------------------------------------------------------------
def get_api_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


PREFERRED_MODELS = [
    GROQ_MODEL,
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]
NON_CHAT_WORDS = ("whisper", "guard", "tts", "orpheus", "embed", "playai", "safeguard", "compound")


def pick_model(client):
    """Ask Groq which models exist right now and choose a chat model. Returns a name or None."""
    try:
        ids = {m.id for m in client.models.list().data}
    except Exception:
        return None
    for name in PREFERRED_MODELS:
        if name in ids:
            return name
    for name in sorted(ids):
        if not any(word in name.lower() for word in NON_CHAT_WORDS):
            return name
    return None


def model_is_missing(exc):
    text = str(exc).lower()
    return "NotFound" in type(exc).__name__ or "decommission" in text or "does not exist" in text or (
        "model" in text and "not found" in text
    )


def ask_ai(messages, json_mode=False, temperature=0.4, max_tokens=900):
    """Call Groq. Returns (text, error_message). Switches model automatically if one is retired."""
    if Groq is None:
        return None, "The 'groq' package is not installed. Check requirements.txt."
    key = get_api_key()
    if not key:
        return None, "No Groq API key found. Add GROQ_API_KEY in the app Secrets."

    client = Groq(api_key=key, timeout=40.0, max_retries=1)
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}

    def run(model):
        tokens = max_tokens * 3 if "gpt-oss" in model or "qwen3" in model else max_tokens
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=tokens, **extra
        )
        return (response.choices[0].message.content or "").strip()

    model = st.session_state.get("groq_model", GROQ_MODEL)
    try:
        try:
            return run(model), None
        except Exception as exc:
            if not model_is_missing(exc):
                raise
            replacement = pick_model(client)
            if not replacement or replacement == model:
                return None, "No working Groq chat model was found for your key. Check console.groq.com/docs/models."
            st.session_state["groq_model"] = replacement
            return run(replacement), None
    except Exception as exc:
        name = type(exc).__name__
        if "RateLimit" in name:
            return None, "The AI service is busy (rate limit). Wait a minute and try again."
        if "Timeout" in name or "Connection" in name:
            return None, "The AI service did not respond in time. Please try again."
        if "Authentication" in name or "PermissionDenied" in name:
            return None, "The Groq API key was rejected. Check GROQ_API_KEY in your Secrets."
        return None, "Something went wrong contacting the AI service. Please try again."

def parse_json(text):
    """Parse JSON from the model, tolerating extra text around it."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def as_list(value):
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def interviewer_prompt(setup):
    return (
        f"You are a friendly but professional interviewer hiring for the role of {setup['role']} "
        f"({setup['level']}). Interview style: {setup['type']}."
    )


def generate_question(setup, asked):
    previous = "\n".join(f"- {q}" for q in asked) or "(none yet)"
    messages = [
        {"role": "system", "content": interviewer_prompt(setup)},
        {"role": "user", "content": (
            f"Questions already asked:\n{previous}\n\n"
            f"Ask interview question number {len(asked) + 1} of {setup['count']}. "
            "Ask exactly ONE new question that is different from the ones above. "
            "Return only the question text, with no numbering and no extra commentary."
        )},
    ]
    return ask_ai(messages, temperature=0.8, max_tokens=150)


def evaluate_answer(setup, question, answer):
    system = (
        interviewer_prompt(setup) + " You now evaluate the candidate's answer honestly and constructively. "
        "Be fair but not overly generous: very short, vague or off-topic answers deserve low scores. "
        "Return ONLY valid JSON with these keys: "
        '"score" (integer 1-10), "strengths" (list of 1-3 short strings), '
        '"improvements" (list of 2-4 short, specific, actionable strings), '
        '"better_answer" (a stronger example answer of about 80-150 words, written in first person).'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nCandidate's answer: {answer}"},
    ]
    text, error = ask_ai(messages, json_mode=True, temperature=0.3, max_tokens=900)
    if error:
        return None, error
    data = parse_json(text)
    if not isinstance(data, dict):
        return None, "The AI reply could not be read. Please press the button again."
    try:
        score = max(1, min(10, int(round(float(data.get("score", 5))))))
    except Exception:
        score = 5
    return {
        "score": score,
        "strengths": as_list(data.get("strengths")),
        "improvements": as_list(data.get("improvements")),
        "better_answer": str(data.get("better_answer", "")).strip(),
    }, None


def final_summary(setup, records):
    lines = []
    for i, r in enumerate(records, 1):
        lines.append(f"Q{i}: {r['question']}\nAnswer: {r['answer']}\nScore: {r['feedback']['score']}/10")
    messages = [
        {"role": "system", "content": interviewer_prompt(setup)},
        {"role": "user", "content": (
            "Here is the full mock interview:\n\n" + "\n\n".join(lines) +
            "\n\nWrite a short final report in markdown with these headings: "
            "**Overall impression**, **Top strengths**, **Main areas to improve**, **Next steps to practise**. "
            "Be specific to the answers above and encouraging. Keep it under 250 words."
        )},
    ]
    return ask_ai(messages, temperature=0.4, max_tokens=700)


def build_transcript(setup, records, summary):
    out = [f"Mock interview: {setup['role']} - {setup['level']} - {setup['type']}", ""]
    for i, r in enumerate(records, 1):
        fb = r["feedback"]
        out += [f"Question {i}: {r['question']}", f"Your answer: {r['answer']}", f"Score: {fb['score']}/10"]
        out += ["Strengths: " + "; ".join(fb["strengths"]), "Improvements: " + "; ".join(fb["improvements"])]
        out += ["Stronger answer: " + fb["better_answer"], ""]
    if summary:
        out += ["Final report", summary]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "stage": "setup",      # setup -> interview -> report
        "setup": None,
        "records": [],         # finished questions: question, answer, feedback
        "question": None,      # the current question text
        "feedback": None,      # feedback for the current question (after submitting)
        "answer": "",
        "summary": None,
        "run_id": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def restart():
    run_id = st.session_state.get("run_id", 0) + 1
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    st.session_state["run_id"] = run_id


# ----------------------------------------------------------------------------
# Screens
# ----------------------------------------------------------------------------
def screen_setup():
    st.subheader("Set up your practice interview")
    role_choice = st.selectbox("Job role", ROLES)
    custom = ""
    if role_choice.startswith("Other"):
        custom = st.text_input("Type the job role", placeholder="e.g. Civil Engineer")
    level = st.selectbox("Experience level", LEVELS, index=1)
    interview_type = st.selectbox("Question type", TYPES)
    count = st.slider("Number of questions", 3, 10, 5)

    if st.button("🎤 Start interview", type="primary"):
        role = custom.strip() if role_choice.startswith("Other") else role_choice
        if not role:
            st.warning("Please type a job role first.")
            return
        setup = {"role": role[:80], "level": level, "type": interview_type, "count": count}
        with st.spinner("Preparing your first question..."):
            question, error = generate_question(setup, [])
        if error:
            st.error(error)
            return
        st.session_state.update(setup=setup, question=question, stage="interview",
                                records=[], feedback=None, summary=None)
        st.rerun()


def finish_interview():
    st.session_state["stage"] = "report"
    st.session_state["summary"] = None
    st.rerun()


def screen_interview():
    setup = st.session_state["setup"]
    records = st.session_state["records"]
    number = len(records) + 1
    run_id = st.session_state["run_id"]

    st.progress((number - 1) / setup["count"], text=f"Question {number} of {setup['count']}")
    st.markdown(f"### 🧑‍💼 {st.session_state['question']}")

    feedback = st.session_state["feedback"]
    if feedback is None:
        answer = st.text_area("Your answer", height=180, key=f"answer_{run_id}_{number}",
                              placeholder="Type your answer as you would say it in a real interview...")
        col1, col2 = st.columns(2)
        if col1.button("Submit answer", type="primary"):
            if len(answer.strip()) < 10:
                st.warning("Please write a bit more (at least a sentence) before submitting.")
            else:
                with st.spinner("Evaluating your answer..."):
                    result, error = evaluate_answer(setup, st.session_state["question"], answer.strip())
                if error:
                    st.error(error)
                else:
                    st.session_state["feedback"] = result
                    st.session_state["answer"] = answer.strip()
                    st.rerun()
        if col2.button("End interview now"):
            if records:
                finish_interview()
            else:
                restart()
                st.rerun()
        return

    # Feedback view
    score = feedback["score"]
    st.metric("Your score", f"{score} / 10")
    if feedback["strengths"]:
        st.success("**What went well**\n\n" + "\n".join(f"- {s}" for s in feedback["strengths"]))
    if feedback["improvements"]:
        st.warning("**How to improve**\n\n" + "\n".join(f"- {s}" for s in feedback["improvements"]))
    if feedback["better_answer"]:
        with st.expander("See a stronger example answer"):
            st.write(feedback["better_answer"])

    is_last = number >= setup["count"]
    label = "Finish and see my report" if is_last else "Next question ➡️"
    if st.button(label, type="primary"):
        records.append({"question": st.session_state["question"],
                        "answer": st.session_state["answer"], "feedback": feedback})
        st.session_state["feedback"] = None
        if is_last:
            finish_interview()
        else:
            with st.spinner("Preparing the next question..."):
                question, error = generate_question(setup, [r["question"] for r in records])
            if error:
                records.pop()
                st.session_state["feedback"] = feedback
                st.error(error)
            else:
                st.session_state["question"] = question
                st.rerun()


def screen_report():
    setup = st.session_state["setup"]
    records = st.session_state["records"]
    st.subheader("📊 Your interview report")

    if not records:
        st.info("You did not answer any questions yet.")
    else:
        scores = [r["feedback"]["score"] for r in records]
        c1, c2, c3 = st.columns(3)
        c1.metric("Average score", f"{sum(scores) / len(scores):.1f} / 10")
        c2.metric("Best answer", f"{max(scores)} / 10")
        c3.metric("Questions answered", len(scores))
        st.bar_chart(pd.Series(scores, index=[f"Q{i}" for i in range(1, len(scores) + 1)]))

        if st.session_state["summary"] is None:
            with st.spinner("Writing your final report..."):
                summary, error = final_summary(setup, records)
            if error:
                st.warning(error)
            else:
                st.session_state["summary"] = summary
        if st.session_state["summary"]:
            st.markdown(st.session_state["summary"])

        st.subheader("Question by question")
        for i, r in enumerate(records, 1):
            fb = r["feedback"]
            with st.expander(f"Q{i} ({fb['score']}/10): {r['question']}"):
                st.markdown("**Your answer**")
                st.write(r["answer"])
                if fb["improvements"]:
                    st.markdown("**How to improve**\n" + "\n".join(f"- {s}" for s in fb["improvements"]))
                if fb["better_answer"]:
                    st.markdown("**Stronger example answer**")
                    st.write(fb["better_answer"])

        st.download_button("⬇️ Download transcript", build_transcript(setup, records, st.session_state["summary"]),
                           "interview_practice.txt", "text/plain")

    if st.button("🔄 Practise again"):
        restart()
        st.rerun()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
init_state()
st.title("🎤 AI Interview Practice")
st.write("Choose a job role. The AI asks one question at a time, scores your answers, and shows you how to improve.")
st.caption("🔒 Privacy: your answers are used only in this session and are not stored permanently.")

if Groq is None or not get_api_key():
    st.warning(
        "The AI is not set up yet. Add your Groq key in the app's **Settings → Secrets** as one line:\n\n"
        '`GROQ_API_KEY = "your-key"`\n\nThen reboot the app. See the README for details.'
    )
    st.stop()

if st.session_state["stage"] != "setup":
    setup = st.session_state["setup"]
    st.sidebar.markdown(f"**Role:** {setup['role']}\n\n**Level:** {setup['level']}\n\n**Type:** {setup['type']}")
    if st.sidebar.button("Start over"):
        restart()
        st.rerun()

if st.session_state["stage"] == "setup":
    screen_setup()
elif st.session_state["stage"] == "interview":
    screen_interview()
else:
    screen_report()

import io
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from groq import Groq
except ImportError:  # the app still works without the AI feature
    Groq = None

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
SAMPLE_PATH = APP_DIR / "sample_reviews.csv"
MAX_REVIEWS = 5000
GROQ_MODEL = "llama-3.3-70b-versatile"
NONE_OPTION = "(none)"
SENTIMENTS = ["Positive", "Neutral", "Negative"]

st.set_page_config(page_title="AI Customer Review Analyzer", page_icon="💬", layout="wide")

# ----------------------------------------------------------------------------
# Word lists (written in the code, so nothing needs to be downloaded)
# ----------------------------------------------------------------------------
NEGATORS = {"not", "no", "never", "dont", "doesnt", "didnt", "cant", "wont", "isnt", "wasnt"}

STOPWORDS = set(
    """a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further get
    got had has have having he her here hers him his how i if in into is it its just me more most my
    nor now of off on once only or other our out over own really same she should so some such than
    that the their them then there these they this those through to too under until up us very was
    we were what when where which while who why will with would you your product item bought buy
    im ive thats youre theyre couldnt wouldnt shouldnt one two even still much many lot make made
    """.split()
) | NEGATORS

COMPLAINT_THEMES = {
    "Quality": ["broke", "broken", "cheap", "flimsy", "defect", "faulty", "poor quality", "fell apart",
                "stopped working", "damaged", "scratch", "tear", "tore", "crack", "quality"],
    "Price": ["expensive", "overpriced", "pricey", "price", "costly", "not worth", "waste of money"],
    "Shipping and delivery": ["shipping", "delivery", "arrived late", "late", "delay", "lost",
                              "never arrived", "took weeks", "slow to arrive"],
    "Customer service": ["customer service", "support", "refund", "rude", "no response", "warranty",
                         "ignored", "never answered", "hassle"],
    "Size and fit": ["size", "fit", "too small", "too big", "tight", "loose", "uncomfortable"],
    "Performance": ["battery", "lag", "noisy", "noise", "loud", "doesn't work", "not work", "weak",
                    "overheat", "crash", "disconnect", "leak", "slow to heat", "flat"],
    "Packaging": ["packaging", "package", "box", "wrapped", "crushed", "smell"],
}

PRAISE_THEMES = {
    "Quality and durability": ["quality", "sturdy", "durable", "solid", "well made", "waterproof", "reliable"],
    "Value for money": ["value", "worth", "affordable", "bargain", "great price", "good price"],
    "Fast delivery": ["fast shipping", "quick delivery", "arrived early", "fast delivery", "shipping was fast",
                      "delivery was quick", "quick shipping"],
    "Customer service": ["customer service", "support", "helpful", "friendly", "refund", "responsive"],
    "Comfort and fit": ["comfortable", "comfort", "fits", "perfect fit", "lightweight", "soft"],
    "Performance": ["sound", "battery", "bass", "fast", "works great", "powerful", "noise cancelling",
                    "brews", "taste", "clear"],
    "Ease of use": ["easy", "simple", "intuitive", "setup", "convenient", "handy"],
    "Design and looks": ["design", "sleek", "stylish", "looks", "beautiful", "color", "nice on"],
}


def compile_themes(themes):
    """Turn a theme dictionary into regex patterns that match word beginnings."""
    return {
        name: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + ")")
        for name, keywords in themes.items()
    }


COMPLAINT_PATTERNS = compile_themes(COMPLAINT_THEMES)
PRAISE_PATTERNS = compile_themes(PRAISE_THEMES)


# ----------------------------------------------------------------------------
# Loading and cleaning data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes):
    """Read CSV bytes, trying UTF-8 first and Latin-1 second. Returns (DataFrame, error)."""
    last_error = "The file could not be read as a CSV."
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding=encoding)
            frame.columns = [str(c).strip() for c in frame.columns]
            return frame, None
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The file is empty."
        except Exception:
            last_error = "The file could not be read as a CSV. Please check it is a valid .csv file."
    return None, last_error


def guess_column(columns, keywords):
    """Return the first column whose name contains one of the keywords, else None."""
    for keyword in keywords:
        for col in columns:
            if keyword in col.lower():
                return col
    return None


def prepare(raw, text_col, rating_col, date_col, product_col):
    """Build a clean DataFrame with standard column names. Returns (data, messages)."""
    messages = []
    data = pd.DataFrame(index=raw.index)
    data["review_text"] = (
        raw[text_col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )

    if rating_col:
        ratings = pd.to_numeric(raw[rating_col], errors="coerce")
        bad = int(ratings.isna().sum() - raw[rating_col].isna().sum())
        if ratings.notna().sum() == 0:
            messages.append(("warning", f"The column '{rating_col}' has no numeric ratings, so ratings are ignored."))
        else:
            data["rating"] = ratings
            if bad:
                messages.append(("warning", f"{bad} rating value(s) were not numbers and were left blank."))

    if date_col:
        dates = pd.to_datetime(raw[date_col], errors="coerce", utc=True).dt.tz_convert(None)
        bad = int(dates.isna().sum() - raw[date_col].isna().sum())
        if dates.notna().sum() == 0:
            messages.append(("warning", f"The column '{date_col}' has no valid dates, so dates are ignored."))
        else:
            data["date"] = dates
            if bad:
                messages.append(("warning", f"{bad} date value(s) were invalid and were left blank."))

    if product_col:
        data["product"] = raw[product_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    before = len(data)
    data = data[data["review_text"] != ""]
    empty_removed = before - len(data)

    before = len(data)
    data = data.drop_duplicates()
    dup_removed = before - len(data)

    data = data.reset_index(drop=True)
    if empty_removed or dup_removed:
        messages.append(("info", f"Cleaning removed {empty_removed} empty review(s) and {dup_removed} duplicate row(s)."))

    if len(data) > MAX_REVIEWS:
        messages.append(("warning", f"The file has {len(data):,} reviews. Only the first {MAX_REVIEWS:,} are analysed."))
        data = data.head(MAX_REVIEWS).copy()

    return data, messages


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def find_themes(text, patterns):
    return "; ".join(name for name, pattern in patterns.items() if pattern.search(text))


@st.cache_data(show_spinner=False)
def analyze(data: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment, themes and rating-mismatch flags to the cleaned data."""
    analyzer = SentimentIntensityAnalyzer()
    out = data.copy()
    scores = out["review_text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    out["sentiment_score"] = scores.round(3)
    out["sentiment"] = scores.apply(sentiment_label)

    normal = out["review_text"].str.lower().str.replace("’", "'", regex=False)
    out["complaint_themes"] = normal.apply(lambda t: find_themes(t, COMPLAINT_PATTERNS))
    out["praise_themes"] = normal.apply(lambda t: find_themes(t, PRAISE_PATTERNS))

    out["rating_mismatch"] = ""
    if "rating" in out.columns:
        top = max(5.0, float(out["rating"].max()))
        high = (out["rating"] >= 0.8 * top) & (out["sentiment"] == "Negative")
        low = (out["rating"] <= 0.4 * top) & (out["sentiment"] == "Positive")
        out.loc[high, "rating_mismatch"] = "High rating but negative text"
        out.loc[low, "rating_mismatch"] = "Low rating but positive text"
    return out


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower().replace("’", "").replace("'", ""))


def term_counts(texts):
    """Count how many reviews mention each word and each two-word phrase."""
    words, phrases = Counter(), Counter()
    for text in texts:
        tokens = tokenize(text)
        words.update({w for w in tokens if w not in STOPWORDS and len(w) > 2})
        pairs = set()
        for a, b in zip(tokens, tokens[1:]):
            first_ok = a in NEGATORS or (a not in STOPWORDS and len(a) > 2)
            if first_ok and b not in STOPWORDS and len(b) > 2:
                pairs.add(f"{a} {b}")
        phrases.update(pairs)
    return words, phrases


def count_themes(frame, column):
    series = frame[column].str.split("; ").explode()
    series = series[series.notna() & (series != "")]
    return series.value_counts()


def counter_series(counter, n=15):
    items = counter.most_common(n)
    return pd.Series({k: v for k, v in items}, dtype="int64")


def short(text, limit=160):
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ----------------------------------------------------------------------------
# Page sections
# ----------------------------------------------------------------------------
def render_upload():
    """Upload tab. Returns (results DataFrame, dataset key) or (None, None)."""
    st.header("Upload your reviews")
    st.caption("🔒 Privacy: your file is only used during this browser session and is never saved permanently.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    sample_bytes = SAMPLE_PATH.read_bytes() if SAMPLE_PATH.exists() else None
    col_a, col_b = st.columns(2)
    if sample_bytes:
        col_a.download_button("⬇️ Download sample CSV", sample_bytes, "sample_reviews.csv", "text/csv")
        if col_b.button("▶️ Try the app with the sample data"):
            st.session_state["use_sample"] = True

    source_bytes, source_name = None, None
    if uploaded is not None:
        source_bytes, source_name = uploaded.getvalue(), uploaded.name
    elif sample_bytes and st.session_state.get("use_sample"):
        source_bytes, source_name = sample_bytes, "sample_reviews.csv"

    if source_bytes is None:
        st.info(
            "👋 **Welcome!** Upload a CSV file of product reviews to see what your customers think.\n\n"
            "1. Upload a CSV (or try the sample data).\n"
            "2. Pick which column holds the review text.\n"
            "3. Explore the dashboard, complaints, praise and AI insights."
        )
        return None, None

    raw, error = read_csv_bytes(source_bytes)
    if error:
        st.error(error)
        return None, None
    if raw.empty or len(raw.columns) == 0:
        st.error("The file has no rows to analyse. Please upload a CSV with at least one review.")
        return None, None

    st.success(f"Loaded **{source_name}**: {len(raw):,} rows and {len(raw.columns)} columns.")
    st.dataframe(raw.head(10))

    columns = list(raw.columns)
    sig = abs(hash(tuple(columns)))
    st.subheader("Choose your columns")

    guess_text = guess_column(columns, ["review", "text", "comment", "feedback", "body"])
    text_col = st.selectbox(
        "Review text column (required)", columns,
        index=columns.index(guess_text) if guess_text else 0, key=f"text_{sig}",
    )

    def optional_select(label, keywords, key):
        guess = guess_column([c for c in columns if c != text_col], keywords)
        options = [NONE_OPTION] + columns
        choice = st.selectbox(label, options, index=options.index(guess) if guess else 0, key=f"{key}_{sig}")
        return None if choice == NONE_OPTION else choice

    c1, c2, c3 = st.columns(3)
    with c1:
        rating_col = optional_select("Rating column (optional)", ["rating", "stars", "star", "score"], "rating")
    with c2:
        date_col = optional_select("Date column (optional)", ["date", "time", "created"], "date")
    with c3:
        product_col = optional_select("Product column (optional)", ["product", "item", "name"], "product")

    data, messages = prepare(raw, text_col, rating_col, date_col, product_col)
    for level, text in messages:
        getattr(st, level)(text)

    if data.empty:
        st.error("No usable reviews were found in the selected review column.")
        return None, None

    with st.spinner("Analysing reviews..."):
        results = analyze(data)
    st.success(f"Analysis complete: {len(results):,} reviews ready. Use the other tabs to explore.")
    return results, f"{source_name}_{len(raw)}_{sig}"


def sidebar_filters(results, ds_key):
    """Build sidebar filters and return the filtered DataFrame."""
    st.sidebar.header("Filters")
    selected = st.sidebar.multiselect("Sentiment", SENTIMENTS, default=SENTIMENTS, key=f"f_sent_{ds_key}")
    mask = results["sentiment"].isin(selected)

    if "product" in results.columns:
        products = sorted(results["product"].unique())
        chosen = st.sidebar.multiselect("Product", products, default=products, key=f"f_prod_{ds_key}")
        mask &= results["product"].isin(chosen)

    if "rating" in results.columns:
        lo, hi = float(results["rating"].min()), float(results["rating"].max())
        if lo < hi:
            whole = (results["rating"].dropna() % 1 == 0).all()
            step = 1.0 if whole else 0.1
            picked = st.sidebar.slider("Rating range", lo, hi, (lo, hi), step=step, key=f"f_rate_{ds_key}")
            if picked != (lo, hi):
                mask &= results["rating"].between(picked[0], picked[1])

    if "date" in results.columns:
        dmin, dmax = results["date"].min().date(), results["date"].max().date()
        if dmin < dmax:
            picked = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin,
                                           max_value=dmax, key=f"f_date_{ds_key}")
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1]) + pd.Timedelta(days=1)
                mask &= (results["date"] >= start) & (results["date"] < end)

    st.sidebar.caption(f"Showing {int(mask.sum()):,} of {len(results):,} reviews")
    return results[mask]


def render_dashboard(f):
    st.header("Dashboard")
    pct = f["sentiment"].value_counts(normalize=True) * 100
    cols = st.columns(5)
    cols[0].metric("Total reviews", f"{len(f):,}")
    avg = f["rating"].mean() if "rating" in f.columns else None
    cols[1].metric("Average rating", f"{avg:.2f}" if avg is not None and pd.notna(avg) else "N/A")
    cols[2].metric("😀 Positive", f"{pct.get('Positive', 0):.0f}%")
    cols[3].metric("😐 Neutral", f"{pct.get('Neutral', 0):.0f}%")
    cols[4].metric("😞 Negative", f"{pct.get('Negative', 0):.0f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment distribution")
        st.bar_chart(f["sentiment"].value_counts().reindex(SENTIMENTS, fill_value=0))
    with right:
        st.subheader("Rating distribution")
        if "rating" in f.columns and f["rating"].notna().any():
            st.bar_chart(f["rating"].dropna().round().astype(int).value_counts().sort_index())
        else:
            st.caption("No rating column was selected, so this chart is not available.")

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment over time")
        if "date" in f.columns and f["date"].notna().any():
            series = f.dropna(subset=["date"]).set_index("date")["sentiment_score"].sort_index()
            span = (series.index.max() - series.index.min()).days
            trend = series.resample("MS" if span > 120 else "W").mean().dropna()
            if len(trend) >= 2:
                st.line_chart(trend)
                st.caption("Average sentiment score (-1 to +1) per " + ("month." if span > 120 else "week."))
            else:
                st.caption("Not enough dates in the current selection to draw a trend.")
        else:
            st.caption("No date column was selected, so this chart is not available.")
    with right:
        st.subheader("Average sentiment by product")
        if "product" in f.columns:
            st.bar_chart(f.groupby("product")["sentiment_score"].mean().round(3))
        else:
            st.caption("No product column was selected, so this chart is not available.")

    st.subheader("Most frequent words")
    left, right = st.columns(2)
    for col, label in ((left, "Positive"), (right, "Negative")):
        with col:
            st.markdown(f"**In {label.lower()} reviews**")
            words, _ = term_counts(f.loc[f["sentiment"] == label, "review_text"])
            if words:
                st.bar_chart(counter_series(words))
            else:
                st.caption(f"No {label.lower()} reviews in the current selection.")


def render_sentiment_details(f):
    st.header("Sentiment details")
    st.subheader("Score distribution")
    labels = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
    bands = pd.cut(f["sentiment_score"], bins=[-1.001, -0.6, -0.2, 0.2, 0.6, 1.001], labels=labels)
    counts = bands.value_counts(sort=False)
    counts.index = counts.index.astype(str)
    st.bar_chart(counts)

    if "rating" in f.columns and f["rating"].notna().any():
        st.subheader("Sentiment compared with star rating")
        by_rating = f.dropna(subset=["rating"]).groupby(f["rating"].round().astype(int))["sentiment_score"].mean()
        st.bar_chart(by_rating.round(3))
        st.caption("Average sentiment score for each star rating. Higher stars should mean higher scores.")
        mismatches = f[f["rating_mismatch"] != ""]
        st.subheader(f"Reviews where rating and text disagree ({len(mismatches)})")
        if mismatches.empty:
            st.success("No mismatches found.")
        else:
            st.dataframe(mismatches[["review_text", "rating", "sentiment", "sentiment_score", "rating_mismatch"]])

    st.subheader("Browse reviews")
    query = st.text_input("Search within review text", key="search_reviews")
    table = f[f["review_text"].str.contains(query, case=False, regex=False)] if query else f
    shown = [c for c in ["review_text", "sentiment", "sentiment_score", "rating", "product", "date"] if c in table.columns]
    st.dataframe(table[shown])


def render_topics(f, sentiment, theme_col, title):
    subset = f[f["sentiment"] == sentiment]
    st.subheader(title)
    if subset.empty:
        st.info(f"There are no {sentiment.lower()} reviews in the current selection.")
        return

    themes = count_themes(subset, theme_col)
    if themes.empty:
        st.caption("No clear themes were detected in these reviews.")
    else:
        st.markdown("**Themes (ranked)**")
        st.bar_chart(themes)
        choice = st.selectbox("Show the reviews for a theme", ["(choose a theme)"] + list(themes.index),
                              key=f"theme_{sentiment}")
        if choice != "(choose a theme)":
            matching = subset[subset[theme_col].str.contains(choice, regex=False)]
            st.dataframe(matching[["review_text", "sentiment_score"] + (["rating"] if "rating" in matching.columns else [])])

    words, phrases = term_counts(subset["review_text"])
    lowered = [t.lower().replace("’", "").replace("'", "") for t in subset["review_text"]]
    rows = []
    entries = [("Word", w, c) for w, c in words.most_common(10)]
    entries += [("Phrase", p, c) for p, c in phrases.most_common(10) if c >= 2]
    for kind, term, count in entries:
        example = next((orig for orig, low in zip(subset["review_text"], lowered) if term in low), "")
        rows.append({"Type": kind, "Term": term, "Reviews": count, "Example": short(example)})
    if rows:
        st.markdown("**Most common words and phrases**")
        st.dataframe(pd.DataFrame(rows))


def render_complaints_praise(f):
    st.header("Common complaints and praise")
    render_topics(f, "Negative", "complaint_themes", "😞 What customers complain about")
    st.divider()
    render_topics(f, "Positive", "praise_themes", "😀 What customers praise")


# ----------------------------------------------------------------------------
# AI insights (optional, uses Groq)
# ----------------------------------------------------------------------------
def get_groq_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def build_context(f):
    """Statistics plus a small balanced sample of reviews (kept small on purpose)."""
    pct = f["sentiment"].value_counts(normalize=True) * 100
    lines = [
        f"Total reviews analysed: {len(f)}",
        f"Positive: {pct.get('Positive', 0):.0f}%, Neutral: {pct.get('Neutral', 0):.0f}%, Negative: {pct.get('Negative', 0):.0f}%",
    ]
    if "rating" in f.columns and f["rating"].notna().any():
        lines.append(f"Average rating: {f['rating'].mean():.2f}")
    for label, column in (("Top complaint themes", "complaint_themes"), ("Top praise themes", "praise_themes")):
        sub = f[f["sentiment"] == ("Negative" if "complaint" in column else "Positive")]
        counts = count_themes(sub, column).head(5) if not sub.empty else pd.Series(dtype=int)
        if not counts.empty:
            lines.append(label + ": " + ", ".join(f"{k} ({v})" for k, v in counts.items()))

    sample_parts = []
    for label, size in (("Positive", 25), ("Negative", 25), ("Neutral", 10)):
        sub = f[f["sentiment"] == label]
        if not sub.empty:
            sample_parts.append(sub.sample(min(len(sub), size), random_state=1))
    sample = pd.concat(sample_parts) if sample_parts else f.head(0)
    lines.append("\nSample reviews:")
    for _, row in sample.iterrows():
        extra = f" | {row['rating']:.0f} stars" if "rating" in sample.columns and pd.notna(row.get("rating")) else ""
        lines.append(f"- [{row['sentiment']}{extra}] {short(row['review_text'], 300)}")
    return "\n".join(lines)


PREFERRED_MODELS = [
    GROQ_MODEL,
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]
NON_CHAT_WORDS = ("whisper", "guard", "tts", "orpheus", "embed", "playai", "safeguard", "compound")


def pick_model(client):
    """Ask Groq which models exist right now and choose a chat model. Returns a name or None."""
    try:
        ids = {m.id for m in client.models.list().data}
    except Exception:
        return None
    for name in PREFERRED_MODELS:
        if name in ids:
            return name
    for name in sorted(ids):
        if not any(word in name.lower() for word in NON_CHAT_WORDS):
            return name
    return None


def model_is_missing(exc):
    text = str(exc).lower()
    return "NotFound" in type(exc).__name__ or "decommission" in text or "does not exist" in text or (
        "model" in text and "not found" in text
    )


def call_groq(api_key, messages):
    """Call Groq and return (text, error_message). Switches model automatically if one is retired."""
    try:
        client = Groq(api_key=api_key, timeout=40.0, max_retries=1)

        def run(model):
            tokens = 3600 if "gpt-oss" in model or "qwen3" in model else 1200
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, max_tokens=tokens
            )
            return response.choices[0].message.content

        model = st.session_state.get("groq_model", GROQ_MODEL)
        try:
            return run(model), None
        except Exception as exc:
            if not model_is_missing(exc):
                raise
            replacement = pick_model(client)
            if not replacement or replacement == model:
                return None, "No working Groq chat model was found for your key. Check console.groq.com/docs/models."
            st.session_state["groq_model"] = replacement
            return run(replacement), None
    except Exception as exc:
        name = type(exc).__name__
        if "RateLimit" in name:
            return None, "The AI service is busy (rate limit reached). Please wait a minute and try again."
        if "Timeout" in name or "Connection" in name:
            return None, "The AI service did not respond in time. Please try again."
        if "Authentication" in name or "PermissionDenied" in name:
            return None, "The Groq API key was rejected. Check GROQ_API_KEY in your app Secrets."
        return None, "Something went wrong while contacting the AI service. Please try again."

def render_ai(f, ds_key):
    st.header("AI Insights")
    st.caption("Optional. Only statistics and a small sample of up to 60 reviews are sent to Groq.")

    api_key = get_groq_key()
    if Groq is None:
        st.warning("The 'groq' package is not installed, so AI Insights are unavailable. All other tabs still work.")
        return
    if not api_key:
        st.info(
            "No Groq API key found, so AI Insights are switched off. Everything else works without it.\n\n"
            "To enable it, add `GROQ_API_KEY = \"your-key\"` in your app's Secrets (see the README)."
        )
        return

    context = build_context(f)
    system = (
        "You analyse customer reviews for a small business owner. Use ONLY the statistics and sample "
        "reviews below. Never invent reviews, quotes or numbers. If the data does not answer a question, "
        "say so. Write in plain, simple English.\n\n" + context
    )
    summary_key, chat_key = f"ai_summary_{ds_key}", f"ai_chat_{ds_key}"

    if st.button("✨ Generate AI summary"):
        prompt = (
            "Write: 1) a short plain-English summary of overall customer feedback; 2) the top 5 complaints "
            "with a one-line explanation each; 3) the top 5 positives with a one-line explanation each; "
            "4) 3 to 5 practical actions the business could take. Use clear headings."
        )
        with st.spinner("Asking the AI..."):
            text, error = call_groq(api_key, [{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        if error:
            st.error(error)
        else:
            st.session_state[summary_key] = text
    if st.session_state.get(summary_key):
        st.markdown(st.session_state[summary_key])

    st.subheader("Ask a follow-up question")
    history = st.session_state.setdefault(chat_key, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask something about these reviews...")
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Thinking..."):
            text, error = call_groq(api_key, [{"role": "system", "content": system}] + history[-6:])
        if error:
            history.pop()
            st.error(error)
        else:
            history.append({"role": "assistant", "content": text})
            with st.chat_message("assistant"):
                st.markdown(text)


def render_download(results, filtered):
    st.header("Download results")
    st.caption("The files include sentiment, score, themes and rating-mismatch flags for every review.")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ All results (CSV)", results.to_csv(index=False).encode("utf-8-sig"),
                       "review_analysis_all.csv", "text/csv")
    c2.download_button("⬇️ Filtered results (CSV)", filtered.to_csv(index=False).encode("utf-8-sig"),
                       "review_analysis_filtered.csv", "text/csv")
    st.dataframe(filtered.head(50))


# ----------------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------------
st.title("💬 AI Customer Review Analyzer")
st.write("Understand what your customers think without reading every review.")

tab_upload, tab_dash, tab_sent, tab_comp, tab_ai, tab_dl = st.tabs(
    ["1. Upload and Preview", "2. Dashboard", "3. Sentiment Details",
     "4. Complaints and Praise", "5. AI Insights", "6. Download Results"]
)

with tab_upload:
    results, ds_key = render_upload()

if results is None:
    for tab in (tab_dash, tab_sent, tab_comp, tab_ai, tab_dl):
        with tab:
            st.info("Upload a CSV (or try the sample data) in the first tab to get started.")
    st.stop()

filtered = sidebar_filters(results, ds_key)

if filtered.empty:
    for tab in (tab_dash, tab_sent, tab_comp, tab_ai):
        with tab:
            st.warning("No reviews match the current filters. Try widening them in the sidebar.")
else:
    with tab_dash:
        render_dashboard(filtered)
    with tab_sent:
        render_sentiment_details(filtered)
    with tab_comp:
        render_complaints_praise(filtered)
    with tab_ai:
        render_ai(filtered, ds_key)

with tab_dl:
    render_download(results, filtered)

import json
import os
import re

import pandas as pd
import streamlit as st

try:
    from groq import Groq
except ImportError:
    Groq = None

GROQ_MODEL = "llama-3.3-70b-versatile"

ROLES = [
    "Software Engineer", "Data Analyst", "Data Scientist", "Product Manager",
    "Marketing Manager", "Sales Representative", "Customer Support Specialist",
    "Project Manager", "UX/UI Designer", "HR Specialist", "Accountant",
    "Teacher", "Nurse", "Other (type your own)",
]
LEVELS = ["Entry level / Intern", "Junior (1-3 years)", "Mid-level (3-6 years)", "Senior (6+ years)"]
TYPES = ["Mixed", "Behavioral", "Technical / Role-specific"]

st.set_page_config(page_title="AI Interview Practice", page_icon="🎤", layout="centered")


# ----------------------------------------------------------------------------
# Groq helpers
# ----------------------------------------------------------------------------
def get_api_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def ask_ai(messages, json_mode=False, temperature=0.4, max_tokens=900):
    """Call Groq. Returns (text, error_message)."""
    if Groq is None:
        return None, "The 'groq' package is not installed. Check requirements.txt."
    key = get_api_key()
    if not key:
        return None, "No Groq API key found. Add GROQ_API_KEY in the app Secrets."
    try:
        client = Groq(api_key=key, timeout=40.0, max_retries=1)
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=temperature,
            max_tokens=max_tokens, **extra,
        )
        return response.choices[0].message.content.strip(), None
    except Exception as exc:
        name = type(exc).__name__
        if "RateLimit" in name:
            return None, "The AI service is busy (rate limit). Wait a minute and try again."
        if "Timeout" in name or "Connection" in name:
            return None, "The AI service did not respond in time. Please try again."
        if "Authentication" in name or "PermissionDenied" in name:
            return None, "The Groq API key was rejected. Check GROQ_API_KEY in your Secrets."
        if "NotFound" in name:
            return None, f"The model '{GROQ_MODEL}' is not available. Change GROQ_MODEL in app.py."
        return None, "Something went wrong contacting the AI service. Please try again."


def parse_json(text):
    """Parse JSON from the model, tolerating extra text around it."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def as_list(value):
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def interviewer_prompt(setup):
    return (
        f"You are a friendly but professional interviewer hiring for the role of {setup['role']} "
        f"({setup['level']}). Interview style: {setup['type']}."
    )


def generate_question(setup, asked):
    previous = "\n".join(f"- {q}" for q in asked) or "(none yet)"
    messages = [
        {"role": "system", "content": interviewer_prompt(setup)},
        {"role": "user", "content": (
            f"Questions already asked:\n{previous}\n\n"
            f"Ask interview question number {len(asked) + 1} of {setup['count']}. "
            "Ask exactly ONE new question that is different from the ones above. "
            "Return only the question text, with no numbering and no extra commentary."
        )},
    ]
    return ask_ai(messages, temperature=0.8, max_tokens=150)


def evaluate_answer(setup, question, answer):
    system = (
        interviewer_prompt(setup) + " You now evaluate the candidate's answer honestly and constructively. "
        "Be fair but not overly generous: very short, vague or off-topic answers deserve low scores. "
        "Return ONLY valid JSON with these keys: "
        '"score" (integer 1-10), "strengths" (list of 1-3 short strings), '
        '"improvements" (list of 2-4 short, specific, actionable strings), '
        '"better_answer" (a stronger example answer of about 80-150 words, written in first person).'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nCandidate's answer: {answer}"},
    ]
    text, error = ask_ai(messages, json_mode=True, temperature=0.3, max_tokens=900)
    if error:
        return None, error
    data = parse_json(text)
    if not isinstance(data, dict):
        return None, "The AI reply could not be read. Please press the button again."
    try:
        score = max(1, min(10, int(round(float(data.get("score", 5))))))
    except Exception:
        score = 5
    return {
        "score": score,
        "strengths": as_list(data.get("strengths")),
        "improvements": as_list(data.get("improvements")),
        "better_answer": str(data.get("better_answer", "")).strip(),
    }, None


def final_summary(setup, records):
    lines = []
    for i, r in enumerate(records, 1):
        lines.append(f"Q{i}: {r['question']}\nAnswer: {r['answer']}\nScore: {r['feedback']['score']}/10")
    messages = [
        {"role": "system", "content": interviewer_prompt(setup)},
        {"role": "user", "content": (
            "Here is the full mock interview:\n\n" + "\n\n".join(lines) +
            "\n\nWrite a short final report in markdown with these headings: "
            "**Overall impression**, **Top strengths**, **Main areas to improve**, **Next steps to practise**. "
            "Be specific to the answers above and encouraging. Keep it under 250 words."
        )},
    ]
    return ask_ai(messages, temperature=0.4, max_tokens=700)


def build_transcript(setup, records, summary):
    out = [f"Mock interview: {setup['role']} - {setup['level']} - {setup['type']}", ""]
    for i, r in enumerate(records, 1):
        fb = r["feedback"]
        out += [f"Question {i}: {r['question']}", f"Your answer: {r['answer']}", f"Score: {fb['score']}/10"]
        out += ["Strengths: " + "; ".join(fb["strengths"]), "Improvements: " + "; ".join(fb["improvements"])]
        out += ["Stronger answer: " + fb["better_answer"], ""]
    if summary:
        out += ["Final report", summary]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "stage": "setup",      # setup -> interview -> report
        "setup": None,
        "records": [],         # finished questions: question, answer, feedback
        "question": None,      # the current question text
        "feedback": None,      # feedback for the current question (after submitting)
        "answer": "",
        "summary": None,
        "run_id": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def restart():
    run_id = st.session_state.get("run_id", 0) + 1
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    st.session_state["run_id"] = run_id


# ----------------------------------------------------------------------------
# Screens
# ----------------------------------------------------------------------------
def screen_setup():
    st.subheader("Set up your practice interview")
    role_choice = st.selectbox("Job role", ROLES)
    custom = ""
    if role_choice.startswith("Other"):
        custom = st.text_input("Type the job role", placeholder="e.g. Civil Engineer")
    level = st.selectbox("Experience level", LEVELS, index=1)
    interview_type = st.selectbox("Question type", TYPES)
    count = st.slider("Number of questions", 3, 10, 5)

    if st.button("🎤 Start interview", type="primary"):
        role = custom.strip() if role_choice.startswith("Other") else role_choice
        if not role:
            st.warning("Please type a job role first.")
            return
        setup = {"role": role[:80], "level": level, "type": interview_type, "count": count}
        with st.spinner("Preparing your first question..."):
            question, error = generate_question(setup, [])
        if error:
            st.error(error)
            return
        st.session_state.update(setup=setup, question=question, stage="interview",
                                records=[], feedback=None, summary=None)
        st.rerun()


def finish_interview():
    st.session_state["stage"] = "report"
    st.session_state["summary"] = None
    st.rerun()


def screen_interview():
    setup = st.session_state["setup"]
    records = st.session_state["records"]
    number = len(records) + 1
    run_id = st.session_state["run_id"]

    st.progress((number - 1) / setup["count"], text=f"Question {number} of {setup['count']}")
    st.markdown(f"### 🧑‍💼 {st.session_state['question']}")

    feedback = st.session_state["feedback"]
    if feedback is None:
        answer = st.text_area("Your answer", height=180, key=f"answer_{run_id}_{number}",
                              placeholder="Type your answer as you would say it in a real interview...")
        col1, col2 = st.columns(2)
        if col1.button("Submit answer", type="primary"):
            if len(answer.strip()) < 10:
                st.warning("Please write a bit more (at least a sentence) before submitting.")
            else:
                with st.spinner("Evaluating your answer..."):
                    result, error = evaluate_answer(setup, st.session_state["question"], answer.strip())
                if error:
                    st.error(error)
                else:
                    st.session_state["feedback"] = result
                    st.session_state["answer"] = answer.strip()
                    st.rerun()
        if col2.button("End interview now"):
            if records:
                finish_interview()
            else:
                restart()
                st.rerun()
        return

    # Feedback view
    score = feedback["score"]
    st.metric("Your score", f"{score} / 10")
    if feedback["strengths"]:
        st.success("**What went well**\n\n" + "\n".join(f"- {s}" for s in feedback["strengths"]))
    if feedback["improvements"]:
        st.warning("**How to improve**\n\n" + "\n".join(f"- {s}" for s in feedback["improvements"]))
    if feedback["better_answer"]:
        with st.expander("See a stronger example answer"):
            st.write(feedback["better_answer"])

    is_last = number >= setup["count"]
    label = "Finish and see my report" if is_last else "Next question ➡️"
    if st.button(label, type="primary"):
        records.append({"question": st.session_state["question"],
                        "answer": st.session_state["answer"], "feedback": feedback})
        st.session_state["feedback"] = None
        if is_last:
            finish_interview()
        else:
            with st.spinner("Preparing the next question..."):
                question, error = generate_question(setup, [r["question"] for r in records])
            if error:
                records.pop()
                st.session_state["feedback"] = feedback
                st.error(error)
            else:
                st.session_state["question"] = question
                st.rerun()


def screen_report():
    setup = st.session_state["setup"]
    records = st.session_state["records"]
    st.subheader("📊 Your interview report")

    if not records:
        st.info("You did not answer any questions yet.")
    else:
        scores = [r["feedback"]["score"] for r in records]
        c1, c2, c3 = st.columns(3)
        c1.metric("Average score", f"{sum(scores) / len(scores):.1f} / 10")
        c2.metric("Best answer", f"{max(scores)} / 10")
        c3.metric("Questions answered", len(scores))
        st.bar_chart(pd.Series(scores, index=[f"Q{i}" for i in range(1, len(scores) + 1)]))

        if st.session_state["summary"] is None:
            with st.spinner("Writing your final report..."):
                summary, error = final_summary(setup, records)
            if error:
                st.warning(error)
            else:
                st.session_state["summary"] = summary
        if st.session_state["summary"]:
            st.markdown(st.session_state["summary"])

        st.subheader("Question by question")
        for i, r in enumerate(records, 1):
            fb = r["feedback"]
            with st.expander(f"Q{i} ({fb['score']}/10): {r['question']}"):
                st.markdown("**Your answer**")
                st.write(r["answer"])
                if fb["improvements"]:
                    st.markdown("**How to improve**\n" + "\n".join(f"- {s}" for s in fb["improvements"]))
                if fb["better_answer"]:
                    st.markdown("**Stronger example answer**")
                    st.write(fb["better_answer"])

        st.download_button("⬇️ Download transcript", build_transcript(setup, records, st.session_state["summary"]),
                           "interview_practice.txt", "text/plain")

    if st.button("🔄 Practise again"):
        restart()
        st.rerun()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
init_state()
st.title("🎤 AI Interview Practice")
st.write("Choose a job role. The AI asks one question at a time, scores your answers, and shows you how to improve.")
st.caption("🔒 Privacy: your answers are used only in this session and are not stored permanently.")

if Groq is None or not get_api_key():
    st.warning(
        "The AI is not set up yet. Add your Groq key in the app's **Settings → Secrets** as one line:\n\n"
        '`GROQ_API_KEY = "your-key"`\n\nThen reboot the app. See the README for details.'
    )
    st.stop()

if st.session_state["stage"] != "setup":
    setup = st.session_state["setup"]
    st.sidebar.markdown(f"**Role:** {setup['role']}\n\n**Level:** {setup['level']}\n\n**Type:** {setup['type']}")
    if st.sidebar.button("Start over"):
        restart()
        st.rerun()

if st.session_state["stage"] == "setup":
    screen_setup()
elif st.session_state["stage"] == "interview":
    screen_interview()
else:
    screen_report()
