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
