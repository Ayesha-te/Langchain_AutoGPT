import streamlit as st
from openai import OpenAI
from wikipedia import summary


def get_client() -> OpenAI:
    return OpenAI(api_key=st.secrets["openai"]["apikey"])


def generate_text(system_prompt: str, user_prompt: str) -> str:
    response = get_client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.9,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


st.title("🦜🔗 YouTube GPT Creator")
prompt = st.text_input("Plug in your prompt here")

if "title_history" not in st.session_state:
    st.session_state.title_history = []
if "script_history" not in st.session_state:
    st.session_state.script_history = []

if prompt:
    try:
        wiki_research = summary(prompt, sentences=3, auto_suggest=False)
    except Exception:
        wiki_research = "No Wikipedia summary was found for this topic."

    title = generate_text(
        "You create catchy YouTube video titles.",
        f"Write me one compelling YouTube video title about: {prompt}",
    )
    script = generate_text(
        "You write engaging YouTube scripts.",
        f'Write a YouTube video script based on this title: "{title}" while leveraging this Wikipedia research: {wiki_research}',
    )

    st.session_state.title_history.append(title)
    st.session_state.script_history.append(script)

    st.write("### Generated Title")
    st.success(title)

    st.write("### Video Script")
    st.write(script)

    with st.expander("📝 Title History"):
        st.info("\n\n".join(st.session_state.title_history))

    with st.expander("📜 Script History"):
        st.info("\n\n".join(st.session_state.script_history))

    with st.expander("📚 Wikipedia Research"):
        st.info(wiki_research)
