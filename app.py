# Bring in dependencies
import os
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

# Set OpenAI API key using Streamlit secrets
os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["apikey"]

# App framework
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'], 
    template='Write me a YouTube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'], 
    template='Write me a YouTube video script based on this title: "{title}" while leveraging this Wikipedia research: {wikipedia_research}'
)

# Memory for tracking conversation
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Language models
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

# Wikipedia wrapper
wiki = WikipediaAPIWrapper()

# Run app logic when prompt is entered
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write("### Generated Title")
    st.success(title)

    st.write("### Video Script")
    st.write(script)

    with st.expander('📝 Title History'): 
        st.info(title_memory.buffer)

    with st.expander('📜 Script History'): 
        st.info(script_memory.buffer)

    with st.expander('📚 Wikipedia Research'): 
        st.info(wiki_research)
