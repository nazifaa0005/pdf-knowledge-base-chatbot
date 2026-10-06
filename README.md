# PDF Knowledge Base Chatbot

Upload a text-based English PDF and ask questions about it. The chatbot searches the document for relevant passages, extracts an answer, and cites the PDF page. If it cannot find a supported answer, it says so. The included particle physics PDF is an example; you can upload another compatible PDF.

## How it works

1. PyMuPDF extracts text from each PDF page.
2. TF-IDF ranks passages relevant to the question.
3. General definition and classification checks handle some direct questions. A pretrained extractive question-answering model, `deepset/roberta-base-squad2`, handles other questions.
4. The app checks the selected answer against its passage and returns a source sentence and page number, or a no-answer message.

The model is downloaded on first use. The project does not train a model or search the web for answers.

## Run on Windows

Install Python 3.12, then open PowerShell in the folder containing `backend`, `frontend`, and this README. Run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload
```

Open http://127.0.0.1:8000. Upload `particle_physics_knowledge_base.pdf` and ask a question such as “What are leptons?” Stop the server with Ctrl+C. On later runs, use only the last command from the same folder; the virtual environment and model download can be reused.

## Project files

| Path | Purpose |
| --- | --- |
| `backend/app.py` | FastAPI endpoints, PDF upload, chat requests, and saved PDF state |
| `backend/knowledge_base.py` | PDF text extraction, passages, and retrieval |
| `backend/answering.py` | Answer selection and validation |
| `frontend/index.html` | Browser upload and chat interface |
| `particle_physics_knowledge_base.pdf` | Example knowledge base |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes local environments, saved uploads, and generated files |

The app accepts one active PDF at a time (up to 10 MB and 200 pages). Scanned image-only PDFs require OCR before upload. The uploaded PDF is saved locally and restored on restart. Chat history is not saved or used as conversational memory. The app is intended for local use and has no user accounts.

## Limitations

Answers depend on the PDF's wording and the retrieval and extraction model. It can miss a supported answer or select imperfect evidence, so check the cited page for important facts. In the project's 46-question evaluation, 42 checks passed, including all 16 tested unrelated questions; four supported-question checks missed their expected answer. Nine API tests passed on Windows, using a fake QA model to check app behavior rather than answer accuracy.
