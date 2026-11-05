# How to write the test questions

`questions.csv` is the exam that the search has to pass. Every row is one question:

| column | what to write |
|---|---|
| `question` | The question, in your own words (like a classmate would ask it). |
| `video` | The lecture it is answered in, e.g. `3.2-V5` (see `catalog.csv`). **Leave empty** if none of the lectures answers it. |
| `answer_start` | Where the answer starts in that video, as `m:ss` (like `0:35`) or seconds (like `35`). |
| `answer_end` | Where the answer ends, same format. |

Example rows:

```
question,video,answer_start,answer_end
How do I choose the threshold for a logistic regression model?,3.2-V5,0:40,1:30
What does the area under the ROC curve tell me?,3.2-V6,2:10,3:05
What is the capital of France?,,,
```

## Rules that keep the test honest

1. **Write the questions after you have read the transcripts** (`transcripts/<id>.txt`, one line per few seconds with
   the time in front), but do not copy sentences from them. A real user would not use the lecturer's exact words.
   Mix short and long questions, and a few in Hinglish.
2. **Aim for 40 questions that the lectures answer** (about one per video, a few across two videos) **and about 20 that
   they cannot answer** (rows with an empty `video`). The 10 that are already in the file are unrelated to the course.
   Add at least 10 *near misses*: questions about data science that these 36 lectures do not cover (for example
   clustering, neural networks, PCA, k-nearest neighbours, time series). They are the hard test of the "not found" cut-off.
3. **Check every "not found" question.** Run `python grep_transcripts.py "k-means|clustering"`. If any line says it,
   a lecture does talk about that topic: delete the question or give it a video and a time range.
4. **Mark a tight time range**: only the part where the answer is actually said (15 to 60 seconds is typical).
   A search result counts as correct only if it overlaps that range, so a wide range makes the test easier.
   Look at the real words with `python grep_transcripts.py --show 3.2-V5 0:40 1:30`.
5. **Check that the answer is not said, just as well, in another lecture.** Run
   `python grep_transcripts.py "area under"`. If two lectures answer the question equally well, a correct result in
   the other lecture would be counted as wrong. Change the question so that only one lecture fits, or leave it out.
6. **Open the file in VS Code or Notepad, not Excel.** Excel turns `0:35` into a clock time and can change the file.
   If you must use Excel, write times in seconds and save as "CSV UTF-8".
7. **Do not change the questions after you have seen the results** just to make the numbers look better.
   If you add questions later, add them as new rows and say so in the README.

## Kinds of questions to include

These are ideas for the *style* only. Write a question only after you have read the passage that answers it.

- **Definition:** What is a threshold in logistic regression? What is multicollinearity?
- **How / why:** How does cross-validation pick the cp value? Why do we split the data into training and test sets?
- **Reading output:** What does a high p-value or a small coefficient mean? What does an AUC of 0.5 tell me?
- **Compare:** What is the difference between sensitivity and specificity? CART or random forest, which is easier to explain?
- **Example from the lecture:** How did Moneyball predict the number of runs a team scores? Why do we remove stop words in text analytics?
- **Vague or short:** "overfitting", "bag of words". Real users type like this.
