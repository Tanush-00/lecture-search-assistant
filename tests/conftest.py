"""Shared test helpers: a fake embedder, a tiny fake set of transcripts and a fake Ollama server."""
import json
import re
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import rag  # noqa: E402

STOP_WORDS = set("""the a an is are of to and in what how do i does it for on we can you this that with or as be by
at from so if then each every not but than when which who why has have more also one two use used will today video""".split())


DIM = 4096


def fake_embed(texts, progress=None, **_ignored):
    """A stand-in for Ollama: every word is hashed into one of 4096 slots, then the vector gets length 1.
    Texts that share words get similar vectors, so search works - no model download needed.
    Like the real embed_texts(), it reports progress when asked (all at once, since there is only one batch)."""
    vectors = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in STOP_WORDS:
                vectors[row, zlib.crc32(word.encode()) % DIM] += 1.0
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if progress:
        progress(len(texts), len(texts))
    return vectors / np.clip(lengths, 1e-12, None)


SENTENCES = {
    "01_precision_recall": [
        "In this video I explain precision recall and the f1 score.",
        "Imagine one thousand accounts and only fifty of them are fake.",
        "A model that says every account is genuine gets ninety five percent accuracy.",
        "But it catches zero fake accounts so accuracy alone is misleading for imbalanced data.",
        "Precision tells us how many of the accounts we flagged as fake are really fake.",
        "Recall tells us how many of the really fake accounts we managed to catch.",
        "A spam filter needs high precision because good mail must not go to spam.",
        "Fraud detection needs high recall because we cannot miss a fraud case.",
        "The f1 score is the harmonic mean of precision and recall.",
        "If either precision or recall is low then the f1 score becomes low.",
        "The confusion matrix has four parts true positive false positive false negative and true negative.",
        "Thank you for watching this video.",
    ],
    "02_overfitting": [
        "Today I will explain overfitting underfitting and cross validation.",
        "Overfitting means the model memorises the training data like learning answers by heart.",
        "The training score is very high but the score on new data is poor.",
        "Underfitting means the model is too simple and performs badly on both training and test data.",
        "We split the data into a training set and a test set to check new data performance.",
        "The test set must never be seen by the model and we use it only once at the end.",
        "Cross validation splits the data into k folds and trains k times.",
        "Each time one fold is used for testing and the rest for training and then we average the scores.",
        "Stratified folds keep the same class ratio in every fold which matters for imbalanced data.",
        "To reduce overfitting you can collect more data or use a simpler model with smaller depth.",
        "Regularization also helps to reduce overfitting.",
        "If training accuracy is ninety nine percent and test accuracy is seventy percent the model overfits.",
    ],
    "03_sql_joins": [
        "In this video we learn sql joins and group by.",
        "Take two tables customers and orders connected by the customer id.",
        "An inner join returns only the rows that have a match in both tables.",
        "A left join returns every customer even when the customer has no orders and fills nulls.",
        "Group by combines rows into groups so we can use sum count or average.",
        "Where filters rows before grouping while having filters the groups after grouping.",
        "A window function like row number or rank works inside a partition without removing rows.",
        "A common mistake is a one to many join that duplicates rows and doubles the totals.",
        "To count orders per customer join the tables and group by the customer id.",
    ],
}

# (question, video, answer_start, answer_end) - each sentence lasts 4 seconds
QUESTIONS = [
    ("What is the difference between precision and recall?", "01_precision_recall", 16, 24),
    ("Why is accuracy misleading for imbalanced data?", "01_precision_recall", 4, 16),
    ("What is the f1 score?", "01_precision_recall", 32, 40),
    ("What is overfitting?", "02_overfitting", 4, 12),
    ("How does cross validation work?", "02_overfitting", 24, 32),
    ("How can I reduce overfitting?", "02_overfitting", 36, 44),
    ("What is the difference between an inner join and a left join?", "03_sql_joins", 8, 16),
    ("Where versus having in sql?", "03_sql_joins", 20, 24),
    ("What is the capital of France?", "", None, None),
    ("How do I make chicken biryani?", "", None, None),
    ("Who won the cricket world cup?", "", None, None),
]


def make_segments(sentences, seconds_each=4.0):
    return [{"start": i * seconds_each, "end": (i + 1) * seconds_each, "text": s} for i, s in enumerate(sentences)]


@pytest.fixture
def transcripts():
    return {video: make_segments(sentences) for video, sentences in SENTENCES.items()}


@pytest.fixture
def questions_csv(tmp_path):
    """The fake questions as a questions.csv file."""
    lines = ["question,video,answer_start,answer_end"]
    for question, video, start, end in QUESTIONS:
        lines.append(f"{question},{video},{'' if start is None else start},{'' if end is None else end}")
    path = tmp_path / "questions.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def index(transcripts):
    return rag.build_index(transcripts, 30, 5, embed_fn=fake_embed, settings={"embed_model": config.EMBED_MODEL})


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Speaks just enough of Ollama's web API for the tests."""

    def _send(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, request))
        if request.get("model") == "missing-model":
            self._send(404, {"error": "model 'missing-model' not found"})
        elif self.path == "/api/embed":
            self._send(200, {"embeddings": fake_embed(request["input"]).tolist()})
        elif self.path == "/api/generate":
            self._send(200, {"response": f" MOCK ANSWER for a prompt of {len(request['prompt'])} characters. "})
        else:
            self._send(500, {"error": "unknown path"})

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "bge-m3:latest"}, {"name": "llama3.2:latest"}]})
        else:
            self._send(404, {})

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_ollama():
    """Starts the fake server. Gives (its address, the server); server.requests lists what it received."""
    server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", server
    server.shutdown()
    server.server_close()
