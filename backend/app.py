"""FastAPI application. Run: python -m uvicorn backend.app:app --reload"""
import base64
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answering import FALLBACK, answer_from_knowledge_base
from .knowledge_base import build_knowledge_base

ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MODEL_NAME = os.getenv('QA_MODEL', 'deepset/roberta-base-squad2')
MODEL_REVISION = os.getenv('QA_MODEL_REVISION',
    'adc3b06f79f797d1c575d5479d6f5efe54a9e3b4' if MODEL_NAME == 'deepset/roberta-base-squad2' else 'main')


def load_qa():
    import torch
    from transformers import pipeline
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return pipeline('question-answering', model=MODEL_NAME, revision=MODEL_REVISION,
                    device=-1, model_kwargs={'use_safetensors': True})


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def create_app(storage_dir=None, qa_factory=None):
    storage = Path(storage_dir) if storage_dir is not None else Path(os.getenv('KB_DATA_DIR', ROOT / 'data'))
    state_file = storage / 'knowledge_base.json'
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.qa = (qa_factory or load_qa)()
        app.state.kb = None
        if state_file.exists():
            try:
                if state_file.stat().st_size > MAX_UPLOAD_BYTES * 2:
                    raise ValueError('Saved file exceeds size limit')
                saved = json.loads(state_file.read_text(encoding='utf-8'))
                pdf = base64.b64decode(saved['pdf'], validate=True)
                app.state.kb = build_knowledge_base(pdf, saved['filename'], app.state.qa.tokenizer)
            except (ValueError, KeyError, OSError, RuntimeError):
                LOGGER.exception('Could not restore the saved PDF; please upload it again.')
        yield

    app = FastAPI(title='Knowledge Base Chatbot', lifespan=lifespan)
    app.state.kb = None

    def status():
        kb = app.state.kb
        if kb is None:
            return {'loaded': False}
        return {'loaded': True, 'filename': kb['filename'], 'pages': kb['page_count'],
                'chunks': len(kb['chunks'])}

    @app.get('/api/knowledge-base')
    def knowledge_base_status():
        return status()

    @app.post('/api/knowledge-base')
    def upload_knowledge_base(file: UploadFile = File(...)):
        try:
            filename = Path((file.filename or '').replace('\\', '/')).name
            if not filename.lower().endswith('.pdf'):
                raise HTTPException(400, 'Please upload a PDF file.')
            pdf = file.file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            file.file.close()
        if len(pdf) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Please upload a PDF smaller than 10 MB.')
        with lock:
            try:
                new_kb = build_knowledge_base(pdf, filename, app.state.qa.tokenizer)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(400, str(exc)) from exc
            try:
                storage.mkdir(parents=True, exist_ok=True)
                temporary = state_file.with_suffix('.tmp')
                temporary.write_text(json.dumps({'filename': filename,
                    'pdf': base64.b64encode(pdf).decode('ascii')}), encoding='utf-8')
                os.replace(temporary, state_file)
            except OSError as exc:
                LOGGER.exception('Could not save uploaded PDF')
                raise HTTPException(503, 'Could not save the PDF. Check the data folder permissions.') from exc
            app.state.kb = new_kb
        return {'message': 'Knowledge base loaded', **status()}

    @app.post('/api/chat')
    def chat(request: ChatRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(400, 'Please enter a question.')
        with lock:
            kb = app.state.kb
            if kb is None:
                raise HTTPException(409, 'Upload a knowledge base first.')
            try:
                result = answer_from_knowledge_base(question, kb, app.state.qa)
            except (RuntimeError, ValueError) as exc:
                LOGGER.exception('QA inference failed')
                raise HTTPException(503, 'The model could not process this question. Please try again.') from exc
            if result is None:
                return {'answer': FALLBACK, 'page': None, 'filename': kb['filename'],
                        'evidence': None, 'method': 'abstain'}
            return {**result, 'filename': kb['filename']}

    @app.get('/')
    def home():
        return FileResponse(ROOT / 'frontend' / 'index.html')

    return app


app = create_app()
