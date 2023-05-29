from dotenv import load_dotenv
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent

import streamlit as st
from streamlit_chat import message


load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        user_prompt = prompt.strip().replace(" ", "").lower()

        top = ["top5", "topfive", "1stfive", "1st5", "first5", "firstfive"]
        bottom = ["last5", "lastfive", "bottom5", "bottomfive"]
        if any([x in user_prompt for x in top]):
            response = "Here are the Top 5 transactions"
        elif any([x in user_prompt for x in bottom]):
            response = "Here are the Bottom/Last 5 transactions"
        else:
            response = agent.run(prompt)

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(response)

@st.cache_data
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False).encode('utf-8')

if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

st.title(':blue[Bank Statement Analysis Chatbot] 📊')
uploaded_file = st.file_uploader("Choose a csv file", type='csv')

if "chat_btn_active" not in st.session_state:
    st.session_state.chat_btn_active = False

if uploaded_file is not None:
    csv_data = uploaded_file.read()
    with open(uploaded_file.name, 'wb') as f: 
        f.write(csv_data)

    df = pd.read_csv(uploaded_file.name)
    st.dataframe(df.head(5))
    csv = convert_df(df)

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    agent = create_pandas_dataframe_agent(chat, df, verbose=True)
    # print(agent)

    # place buttons in one line
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    with col2:
        st.download_button(
            label="Generate Overall Report",
            data=csv,
            file_name='large_df.csv',
            mime='text/csv',
        )   
    with col3:
        # st.button("Generate Overall Report", on_click="") 
        chat_btn = st.button("Let's Chat!")

    if st.session_state.prompts:
        for i in range(len(st.session_state.responses)):
            user_prompt = st.session_state.prompts[i].strip().replace(" ", "").lower()
            top = ["top5", "topfive", "1stfive", "1st5", "first5", "firstfive"]
            bottom = ["last5", "lastfive", "bottom5", "bottomfive"]

            message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=83)
            message(st.session_state.responses[i], key=str(i), seed='Milo')

            if any([x in user_prompt for x in top]):
                st.dataframe(df.head())
            elif any([x in user_prompt for x in bottom]):
                st.dataframe(df.tail())
            else:
                pass
            
    if chat_btn or st.session_state.chat_btn_active:
        st.text_input("Ask Something:", key="user")
        send_btn = st.button("Send", on_click=send_click)
        st.session_state.chat_btn_active = True
        
            

