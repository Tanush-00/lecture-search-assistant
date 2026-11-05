"""A web page for the Lecture Search Assistant.   Run:  streamlit run app.py"""
import streamlit as st

import config
import rag

CREDITS = ("Lectures: MIT OpenCourseWare 15.071 The Analytics Edge (Spring 2017), Prof. Dimitris Bertsimas, "
           "licensed CC BY-NC-SA 4.0. The videos are not part of this project's code; step 0 downloads them.")

st.set_page_config(page_title="Lecture Search Assistant", layout="centered")
st.title("Lecture Search Assistant")
st.caption("Ask a question. I find the lecture video and the minute where it is explained.")


@st.cache_resource
def get_index(version):
    """Load the index once. `version` changes when step2 rebuilds it, so a rebuilt index is picked up."""
    return rag.load_index(config.INDEX_DIR)


def index_version():
    file = config.INDEX_DIR / "embeddings.npy"
    return file.stat().st_mtime if file.exists() else 0


def show_hit(hit):
    """One passage: video title, where it sits in the course, time range, score, the text and (if the file is here) a player."""
    context = rag.video_context(hit.video)
    st.markdown(f"**{rag.video_title(hit.video)}**  \n"
                + (f"{context}  \n" if context else "")
                + f"{rag.format_time(hit.start)} - {rag.format_time(hit.end)} &nbsp;|&nbsp; score {hit.score:.2f}")
    st.markdown(hit.text.replace("$", "\\$"))
    media = rag.find_media_for(hit.video, config.VIDEO_DIR)
    if media is not None:
        player = st.audio if media.suffix.lower() in rag.AUDIO_EXTENSIONS else st.video
        player(str(media), start_time=int(hit.start))


try:
    index = get_index(index_version())
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

ollama_running, ollama_models = rag.ollama_status()
can_embed = ollama_running and rag.has_model(ollama_models, config.EMBED_MODEL)
can_write = ollama_running and rag.has_model(ollama_models, config.LLM_MODEL)

with st.sidebar:
    st.header("Settings")
    labels = {"Meaning (embeddings)": "embeddings", "Keywords (TF-IDF)": "keywords"}
    choice = st.radio("Search by", list(labels), index=0 if can_embed else 1,
                      help="Meaning needs Ollama. Keywords works everywhere but only matches the same words.")
    method = labels[choice]
    top_k = st.slider("Passages to show", 1, 8, config.TOP_K)
    min_score = st.slider("Minimum score to answer", 0.0, 1.0, float(config.MIN_SCORE[method]), 0.01,
                          key=f"min_score_{method}",
                          help="Below this score the assistant says 'not found'. Find your value with evaluate.py.")
    write_answer = st.checkbox("Write a short answer with the LLM", value=can_write, disabled=not can_write)
    if not ollama_running:
        st.info("Ollama is not running, so only keyword search is available.")
    st.caption(CREDITS)

question = st.text_input("Your question", placeholder="for example: how do I choose the threshold?")

if question.strip():
    problem = None
    try:
        if method == "embeddings":
            rag.check_index_matches_config(index)
        with st.spinner("Searching ..."):
            result = rag.answer_question(question, index, method=method, top_k=top_k,
                                         min_score=min_score, use_llm=write_answer)
    except (rag.OllamaError, ValueError) as error:
        problem = str(error)

    if problem:
        st.error(problem)
    else:
        hits = result["hits"]
        best = float(hits["score"].iloc[0])
        if not result["found"]:
            st.warning(f"Not found in the videos. The best match scored {best:.2f}, "
                       f"below the minimum of {result['cutoff']:.2f}.")
            with st.expander("Closest passage (probably not relevant)"):
                show_hit(next(hits.head(1).itertuples()))
        else:
            if result["answer"]:
                st.subheader("Answer")
                st.write(result["answer"])
            st.subheader("Where it is explained")
            for hit in hits.itertuples():
                show_hit(hit)
                st.divider()
