"""PDF extraction and sentence-aware TF-IDF retrieval (no model training)."""
import re

import pymupdf
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STEMMER = SnowballStemmer('english')  # Algorithm only; no NLTK data download.
# General retrieval vocabulary, not answers or particle-physics names.
ALIASES = {'kind': 'type', 'kinds': 'type', 'types': 'type', 'colour': 'color',
           'colours': 'color', 'discovered': 'find', 'discover': 'find',
           'discovery': 'find', 'found': 'find'}
QUERY_WORDS = {'define', 'explain', 'tell', 'describe', 'list', 'name', 'named',
               'number', 'different', 'exact', 'exactly', 'please', 'exist'}
STOP = (set(ENGLISH_STOP_WORDS) - {'top', 'bottom', 'first', 'last'}) | QUERY_WORDS


def terms(text):
    """Normalize lexical forms for retrieval, never the displayed source text."""
    return [STEMMER.stem(ALIASES.get(w, w)) for w in re.findall(r'[a-z]+|\d+(?:\.\d+)?', text.lower())
            if w not in STOP]


def query_text(question):
    # Remove an answer-category word only when the question asks for that category.
    question = re.sub(r"^(?:what|which)\s+(?:particles?|cities|city|countries|country|"
                      r"devices?|organizations?|people|person|years?)\s+", "", question, flags=re.I)
    return question


def question_terms(question):
    return terms(query_text(question))


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text):
    # Protect initials and common abbreviations before splitting. Keep original text.
    text = clean_text(text)
    protected = re.sub(r'\b(?:[A-Z]\.){1,3}(?=\s|$)',
                       lambda m: m.group().replace('.', '\u2024'), text)
    protected = re.sub(r'\b(?:Dr|Mr|Mrs|Prof|e\.g|i\.e)\.',
                       lambda m: m.group().replace('.', '\u2024'), protected)
    return [s.replace('\u2024', '.').strip() for s in
            re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“])', protected) if s.strip()]


def page_sentences(raw):
    # Short standalone headings should not be glued to a sentence subject.
    lines = raw.splitlines()
    blocks = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if (i == 0 and len(line.split()) < 9 and not line.endswith(('.', '!', '?'))):
            continue
        blocks.append(line)
    return [s for s in split_sentences(' '.join(blocks))
            if not s.endswith('?') and len(s.split()) >= 3]


def chunk_text(text, tokenizer, max_tokens=240, overlap_tokens=50):
    """Compatibility helper: preserve original characters, even for long sentences."""
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError('Overlap must be smaller than chunk size.')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    offsets = encoded['offset_mapping']
    output = []
    for start in range(0, len(offsets), max_tokens - overlap_tokens):
        end = min(start + max_tokens, len(offsets))
        if start < end:
            output.append(text[offsets[start][0]:offsets[end - 1][1]])
        if end == len(offsets):
            break
    return output


def build_knowledge_base(pdf_bytes, filename, tokenizer):
    if not pdf_bytes:
        raise ValueError('The PDF is empty.')
    try:
        with pymupdf.open(stream=pdf_bytes, filetype='pdf') as document:
            if document.needs_pass:
                raise ValueError('Password-protected PDFs are not supported.')
            if len(document) > 200:
                raise ValueError('Please use a PDF with at most 200 pages.')
            pages = [page.get_text('text') for page in document]
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError('The file is not a readable PDF.') from exc
    sentences, chunks = [], []
    for page_number, raw in enumerate(pages, 1):
        local = page_sentences(raw)
        for sentence in local:
            sentences.append({'text': sentence, 'page': page_number})
        # Single sentences and adjacent pairs retain antecedents for "They...".
        seen = set()
        for i in range(len(local)):
            for size in (1, 2):
                text = ' '.join(local[i:i + size])
                if text in seen:
                    continue
                seen.add(text)
                for part in chunk_text(text, tokenizer):
                    chunks.append({'text': part, 'page': page_number})
    if not chunks:
        raise ValueError('No readable text found. Scanned PDFs need OCR first.')
    vectorizer = TfidfVectorizer(analyzer=terms, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([c['text'] for c in chunks])
    except ValueError as exc:
        raise ValueError('The PDF has no usable English text.') from exc
    return {'filename': filename, 'page_count': len(pages), 'pages': pages,
            'sentences': sentences, 'chunks': chunks, 'vectorizer': vectorizer,
            'chunk_matrix': matrix, 'vocabulary': set(vectorizer.vocabulary_)}


def retrieve_chunks(question, kb, top_k=12):
    query_terms = set(question_terms(question))
    if not query_terms:
        return []
    scores = cosine_similarity(kb['vectorizer'].transform([query_text(question)]),
                               kb['chunk_matrix']).ravel()
    results = []
    for i in scores.argsort()[::-1]:
        if scores[i] <= 0:
            break
        chunk = kb['chunks'][int(i)]
        coverage = len(query_terms & set(terms(chunk['text']))) / len(query_terms)
        # Coverage makes a passage mentioning both the entity and requested property
        # preferable to one repeating only the entity many times.
        results.append({**chunk, 'similarity': float(scores[i]), 'coverage': coverage,
                        'rank_score': float(scores[i]) * coverage * coverage})
    return sorted(results, key=lambda c: c['rank_score'], reverse=True)[:top_k]
