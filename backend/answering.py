"""Grounded answers: general definition evidence, otherwise extractive QA.

All answers include an unchanged source sentence. No particle names or answers
are embedded in the program. Heuristics are conservative and not a proof of truth.
"""
import re
from .knowledge_base import terms, question_terms, split_sentences, retrieve_chunks

MIN_QA_SCORE = 0.20
MIN_RELEVANCE = 0.08
FALLBACK = "I couldn't find an answer in the provided knowledge base."


def definition_subject(question):
    q = question.strip().rstrip('?.! ')
    match = re.fullmatch(r'(?:what (?:is|are)|define|explain)\s+(?:(?:a|an|the)\s+)?(.+)', q, re.I)
    if not match:
        return None
    subject = match.group(1)
    # Property/list questions need the QA route, not a concept definition.
    if re.search(r'\b(?:of|for|in|on|at|by|with|type|types|kind|kinds)\b', subject, re.I):
        return None
    return set(terms(subject))


def definition_evidence(subject, kb):
    """Recognize explicit class statements and parenthetical class membership.

    Return the entire source sentence; never fabricate a singular/plural rewrite.
    """
    candidates = []
    for entry in kb['sentences']:
        sentence = entry['text']
        if not subject <= set(terms(sentence)):
            continue
        # Explicit enumeration: a topic followed by a colon and a list.
        if ':' in sentence:
            head, tail = sentence.split(':', 1)
            if subject <= set(terms(head)) and ',' in tail and ' and ' in tail:
                candidates.append((1.5, entry))
        # Category (member1 and member2): the category comes from the PDF.
        for match in re.finditer(r'([\w-]+(?:\s+[\w-]+){0,2})\s*\(([^)]+)\)', sentence):
            if subject <= set(terms(match.group(2))) and not re.search(
                    r'\b(?:like|example|include|including|such)\b', match.group(2), re.I):
                candidates.append((3.5 if re.search(r'\b(?:grouped|classified|categories)\b', sentence, re.I) else 2, entry))
        # Match at the start: "whether X is..." is not an asserted definition.
        match = re.match(r'^(?:The |A |An )?(.{1,100}?)\s+(?:is|are)\s+(.+)', sentence, re.I)
        if match and subject <= set(terms(match.group(1))):
            predicate = match.group(2)
            if re.search(r'\b(?:whether|if|not|might|may)\b', match.group(1), re.I):
                continue
            if re.match(r'(?:(?:perhaps|sometimes|often|usually)\s+)?(?:a|an|the)\s+', predicate, re.I) or re.match(r'\w+s\s+(?:that|which)\b', predicate, re.I):
                extra = len(set(terms(match.group(1))) - subject)
                property_penalty = .9 if re.search(r'^(?:(?:perhaps|sometimes|often|usually)\s+)?the (?:only|most|best)', predicate, re.I) else 0
                candidates.append((3 - min(extra, 2) * .3 - property_penalty, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[0])
    return evidence_result(candidates[0][1]['text'], candidates[0][1]['page'], 'definition')


def evidence_result(text, page, method, **extra):
    return {'answer': text, 'page': page, 'evidence': text, 'method': method, **extra}


def answer_from_knowledge_base(question, kb, qa):
    subject = definition_subject(question)
    if subject:
        return definition_evidence(subject, kb)
    query_terms = set(question_terms(question))
    # Unknown requested content (e.g. price) must not disappear from a TF-IDF query.
    if not query_terms or query_terms - kb['vocabulary']:
        return None
    matches = retrieve_chunks(question, kb)
    candidates = []
    for match in matches:
        if match['similarity'] < MIN_RELEVANCE or match['coverage'] < 1.0:
            continue
        prediction = qa(question=question, context=match['text'],
                        handle_impossible_answer=True, max_answer_len=60,
                        max_seq_len=384, max_question_len=64, doc_stride=96)
        span = prediction['answer'].strip()
        if not span or float(prediction['score']) < MIN_QA_SCORE:
            continue
        start, end = prediction['start'], prediction['end']
        if match['text'][start:end].strip() != span:
            continue
        # Keep whole sentences around the span, including a preceding antecedent.
        sentences = split_sentences(match['text'])
        selected = []
        cursor = 0
        for i, sentence in enumerate(sentences):
            offset = match['text'].find(sentence, cursor)
            cursor = offset + len(sentence)
            if offset < end and cursor > start:
                if re.match(r'^(?:They|It|These|This|Those|Their)\b', sentence) and i:
                    selected.append(sentences[i - 1])
                selected.append(sentence)
        evidence = ' '.join(dict.fromkeys(selected))
        # A question's entity/property must be supported in the returned evidence.
        if not evidence or not query_terms <= set(terms(evidence)):
            continue
        count_question = re.match(r'how many\s+(\w+)', question, re.I)
        if count_question:
            number = re.search(r'\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)\b', span, re.I)
            if not number:
                continue
            # The predicted number must quantify the requested noun, not another
            # entity in the same sentence. Allow modifiers and a short apposition.
            unit = set(terms(count_question.group(1)))
            counted = False
            for occurrence in re.finditer(r'\b' + re.escape(number.group()) + r'\b', evidence, re.I):
                tail = re.split(r'[.;!?]', evidence[occurrence.end():], maxsplit=1)[0]
                nearby = ' '.join(tail.split()[:8])
                if unit and unit <= set(terms(nearby)):
                    counted = True
            if not counted:
                continue
        if re.search(r'\b(?:types?|kinds?)\b', question, re.I) and not re.match(r'how many', question, re.I):
            if not (',' in span or ' and ' in span):
                continue
        value = match['rank_score'] * float(prediction['score'])
        candidates.append((value, evidence_result(evidence, match['page'], 'qa',
                           extracted_answer=span, qa_score=float(prediction['score']))))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
