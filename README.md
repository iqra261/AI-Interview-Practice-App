# 🎤 AI Interview Practice

Choose a job role and practise a mock interview. The AI asks questions one at a time,
scores each answer from 1 to 10, shows strengths and improvements, gives a stronger example answer,
and writes a final report at the end.

## Features
- Pick a role (or type your own), experience level, question type and number of questions.
- One question at a time; the AI avoids repeating questions.
- Feedback after every answer: score, what went well, how to improve, example answer.
- Final report with average score, a score chart and a downloadable transcript.
- Nothing is stored permanently; everything lives in your browser session.

## Files
| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit app |
| `requirements.txt` | Python packages (streamlit, pandas, groq) |
| `.gitignore` | Keeps secrets out of GitHub |

## Groq API key (required)
1. Create a free key at https://console.groq.com (API Keys → Create API Key).
2. On Streamlit Community Cloud: app **Settings → Secrets**, paste one line with straight quotes:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. Save and **Reboot app**.

Locally, put the same line in `.streamlit/secrets.toml` (already in `.gitignore`).
Never put the key inside `app.py`.

## Run on your computer (optional)
```
pip install -r requirements.txt
streamlit run app.py
```

## Storage note
No database is needed. To save practice history later, use a free hosted database such as Neon PostgreSQL;
Streamlit Community Cloud erases local files (including SQLite) when the app restarts.

## Common problems
- **"The AI is not set up yet"**: the secret must be named exactly `GROQ_API_KEY`; reboot after saving.
- **Model not available**: change `GROQ_MODEL` near the top of `app.py`.
- **ModuleNotFoundError**: a package is missing from `requirements.txt`.
