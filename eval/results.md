**Test questions:** 65 questions that the 36 lecture videos answer (206 minutes of speech; `eval/questions.csv` says where each answer is said) and 24 questions that they cannot answer (unrelated, or about topics the selected lectures do not cover). Chunks are about 30 s long with 5 s overlap (493 chunks). A result is correct if it comes from the right video and overlaps the time where the answer is said.

#### Keyword search vs search by meaning

| Search method | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| Random guessing (calculated) | 0% | 1% | 2% |
| Keywords (TF-IDF) | 55% (43-67) | 80% (69-88) | 86% (76-93) |
| Meaning (bge-m3 embeddings) | 72% (60-82) | 85% (74-91) | 91% (81-96) |

Recall@k = share of the answerable questions whose right passage is among the top k results. The range in brackets is the 95% confidence interval (Wilson). 'Random guessing' is the exact chance that k chunks picked at random contain the answer, so it shows how much of a result could be luck.
With 65 questions the interval is about +/-9 points around each percentage, so a gap smaller than that between two rows is not a reliable difference.

![Recall of random guessing, keyword search and search by meaning](images/retrieval_methods.png)

#### Chunk length (search by meaning)

| Chunk length | Chunks | Recall@1 | Recall@3 | Recall@5 | Random guessing, Recall@3 |
|---|---|---|---|---|---|
| 15 s (overlap 3 s) | 1057 | 71% | 82% | 88% | 1% |
| 30 s (overlap 5 s) | 493 | 72% | 85% | 91% | 1% |
| 60 s (overlap 10 s) | 257 | 77% | 89% | 92% | 2% |
| 90 s (overlap 15 s) | 178 | 66% | 88% | 94% | 2% |

A result counts as correct when it overlaps the answer, so longer chunks have an easier job: look at how the random column grows. They also point to a coarser spot in the video.

![Recall for different chunk lengths](images/chunk_size.png)

#### The 'not found' cut-off

| Search method | Cut-off | Answerable questions answered | Other questions refused | Balance on unseen questions |
|---|---|---|---|---|
| Meaning (bge-m3 embeddings) | 0.58 | 94% | 100% | 94% (+/- 6) |
| Keywords (TF-IDF) | 0.20 | 78% | 88% | 80% (+/- 9) |

The cut-off is the lowest score at which the assistant still answers. I chose it on these same questions, so the middle two columns are a little optimistic. The last column picks the cut-off on some questions and tests it on others (5-fold, repeated 10 times), which is the fairer number.

![Scores of answerable and not answerable questions](images/not_found_scores.png)
