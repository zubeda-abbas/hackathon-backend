from dotenv import load_dotenv
import os
import json
import sys
import pandas as pd
import numpy as np
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_chat import message

from controllers import (validateJSON, get_bankname, parse_sbi, parse_axis, parse_hdfc, bank_classifier_predict)

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        user_prompt = prompt.strip().replace(" ", "").lower()
        response = ""

        top = ["top5row", "topfiverow", "1stfiverow", "1st5row", "first5row", "firstfiverow", "top5ob", "topfiveob", "1stfiveob", "1st5ob", "first5ob", "firstfiveob", "top5tran", "topfiveobtran", "1stfiveobtran", "1st5obtran", "first5obtran", "firstfiveobtran"]
        bottom = ["last5row", "lastfiverow", "bottom5row", "bottomfiverow", "last5ob", "lastfiveob", "bottom5ob", "bottomfiveob", "last5tran", "lastfivetran", "bottom5tran", "bottomfivetran"]
        top_high = ["top5highesttran", "topfivehighesttran", "1stfivehighesttran", "1st5highesttran", "first5highesttran", "firstfivehighesttran"]
        bottom_low = ["lowest5tran", "lowestfivetran"]           
        top_debit = ["top5debit", "topfivedebit", "1stfivedebit", "1st5debit", "first5debit", "firstfivedebit"]
        bottom_debit = ["last5debit", "lastfivedebit", "bottom5debit", "bottomfivedebit"]            
        top_debit_high = ["top5highestdebit", "topfivehighestdebit", "1stfivehighestdebit", "1st5highestdebit", "first5highestdebit", "firstfivehighestdebit"]
        bottom_debit_low = ["lowest5debit", "lowestfivedebit"]            
        top_credit = ["top5credit", "topfivecredit", "1stfivecredit", "1st5credit", "first5credit", "firstfivecredit"]
        bottom_credit = ["last5credit", "lastfivecredit", "bottom5credit", "bottomfivecredit"]            
        top_credit_high = ["top5highestcredit", "topfivehighestcredit", "1stfivehighestcredit", "1st5highestcredit", "first5highestcredit", "firstfivehighestcredit"]
        bottom_credit_low = ["lowest5credit", "lowestfivecredit"]

        top_condn = any([x in user_prompt for x in top])
        bottom_condn = any([x in user_prompt for x in bottom])
        top_high_condn = any([x in user_prompt for x in top_high])
        low_condn = any([x in user_prompt for x in bottom_low])
        top_debit_condn = any([x in user_prompt for x in top_debit])
        bottom_debit_condn = any([x in user_prompt for x in bottom_debit])
        top_debit_high_condn = any([x in user_prompt for x in top_debit_high])
        low_debit_condn = any([x in user_prompt for x in bottom_debit_low])
        top_credit_condn = any([x in user_prompt for x in top_credit])
        bottom_credit_condn = any([x in user_prompt for x in bottom_credit])
        top_credit_high_condn = any([x in user_prompt for x in top_credit_high])
        low_credit_condn = any([x in user_prompt for x in bottom_credit_low])

        if top_debit_high_condn:
            response += "The Top 5 highest debit transactions, "
        if top_credit_high_condn:
            response += "The Top 5 highest credit transactions, "
        if top_high_condn:
            response += "The Top 5 highest transactions, "
        if top_debit_condn:
            response += "The Top 5 debit transactions, "
        if top_credit_condn:
            response += "The Top 5 credit transactions, "
        if top_condn:
            response += "The Top 5 transactions, "

        if low_debit_condn:
            response += "The Lowest 5 debit transactions, "
        if low_credit_condn:
            response += "The Lowest 5 credit transactions, "
        if low_condn:
            response += "The Lowest 5 transactions, "
        if bottom_debit_condn:
            response += "The Bottom/Last 5 debit transactions, "
        if bottom_credit_condn:
            response += "The Bottom/Last 5 credit transactions, "
        if bottom_condn:
            response += "The Bottom/Last 5 transactions, "

        if not top_condn and not bottom_condn and not top_debit_condn and not top_credit_condn and not bottom_debit_condn and not bottom_credit_condn and not top_high_condn and not top_debit_high_condn and not top_credit_high_condn and not low_condn and not low_debit_condn and not low_credit_condn:
            try:
                response = agent.run("""
                For the following query:
                If it is just asking a question that do not require to display dataframe or multiple rows or plot a graph or chart return the response as it is.
                If it requires to plot a graph or chart return the data in a JSON format. If it is a bar graph or chart assign the data to the key "bar", if it is a line graph or chart assign the data to the key "line", if it is a pie chart or graph assign the data to the key "pie". For graphs and charts, assign the X-axis data to JSON key "x" and Y-axis data to JSON key "y". Convert all the list to string. Return all the output as a string. 
                If it requires to display a dataframe or table or multiple rows instead of plotting a graph or chart assign the data to the JSON key "multiple". Convert all the list to string. Return all the output as a string.
                Here is the query: {}
                """.format(prompt))
                print("Response: ", response)
            except Exception as e:
                print("Error: ", e)
                response = "Sorry, unable to answer, try asking in a simpler way!"

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(response)
        st.session_state.user = ""

@st.cache_data
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False).encode('utf-8')


if 'prompts' not in st.session_state:
    st.session_state.prompts = [""]
if 'responses' not in st.session_state:
    st.session_state.responses = ["Hello there, this is a Bank Statement Analysis chatbot! You can ask me any queries related to the uploaded bank statement, I will try my best to answer all your queries. How can i help you?"]

st.title(':blue[Bank Statement Analysis Chatbot] 📊')
uploaded_file = st.file_uploader("Choose a pdf file", type='pdf')

if "chat_btn_active" not in st.session_state:
    st.session_state.chat_btn_active = False

if "parsed" not in st.session_state:
    st.session_state.parsed = ""

if uploaded_file is not None:

    pdf_data = uploaded_file.read()
    with open(uploaded_file.name, 'wb') as f: 
        f.write(pdf_data)

    if st.session_state.parsed == "":
        parser_bank_name = get_bankname(uploaded_file.name)
        print("Bank name: ", parser_bank_name)
        parse_df = pd.DataFrame()
        if parser_bank_name == "SBI Bank":
            parse_df = parse_sbi(uploaded_file.name)
            print("SBI: ", parse_df.shape)
        elif parser_bank_name == "Axis Bank":
            parse_df = parse_axis(uploaded_file.name)
            print("AXIS: ", parse_df.shape)
        elif parser_bank_name == "HDFC Bank":
            parse_df = parse_hdfc(uploaded_file.name)
            print("HDFC: ", parse_df.shape)
        else:
            pass

        print("DF: ", parse_df.shape)
        st.session_state.parsed = bank_classifier_predict(parse_df)

    if st.session_state.parsed != "": 
        df = pd.read_csv(st.session_state.parsed)
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
            
        if st.session_state.prompts and chat_btn or st.session_state.chat_btn_active:
            for i in range(len(st.session_state.responses)):
                if i == 0:
                    message(st.session_state.responses[i], key=str(i), seed='Milo')
                if st.session_state.prompts[i] != "":
                    user_prompt = st.session_state.prompts[i].strip().replace(" ", "").lower()
                    top = ["top5row", "topfiverow", "1stfiverow", "1st5row", "first5row", "firstfiverow", "top5ob", "topfiveob", "1stfiveob", "1st5ob", "first5ob", "firstfiveob", "top5tran", "topfiveobtran", "1stfiveobtran", "1st5obtran", "first5obtran", "firstfiveobtran"]
                    bottom = ["last5row", "lastfiverow", "bottom5row", "bottomfiverow", "last5ob", "lastfiveob", "bottom5ob", "bottomfiveob", "last5tran", "lastfivetran", "bottom5tran", "bottomfivetran"]
                    top_high = ["top5highesttran", "topfivehighesttran", "1stfivehighesttran", "1st5highesttran", "first5highesttran", "firstfivehighesttran"]
                    bottom_low = ["lowest5tran", "lowestfivetran"]           
                    top_debit = ["top5debit", "topfivedebit", "1stfivedebit", "1st5debit", "first5debit", "firstfivedebit"]
                    bottom_debit = ["last5debit", "lastfivedebit", "bottom5debit", "bottomfivedebit"]            
                    top_debit_high = ["top5highestdebit", "topfivehighestdebit", "1stfivehighestdebit", "1st5highestdebit", "first5highestdebit", "firstfivehighestdebit"]
                    bottom_debit_low = ["lowest5debit", "lowestfivedebit"]            
                    top_credit = ["top5credit", "topfivecredit", "1stfivecredit", "1st5credit", "first5credit", "firstfivecredit"]
                    bottom_credit = ["last5credit", "lastfivecredit", "bottom5credit", "bottomfivecredit"]            
                    top_credit_high = ["top5highestcredit", "topfivehighestcredit", "1stfivehighestcredit", "1st5highestcredit", "first5highestcredit", "firstfivehighestcredit"]
                    bottom_credit_low = ["lowest5credit", "lowestfivecredit"]

                    top_condn = any([x in user_prompt for x in top])
                    bottom_condn = any([x in user_prompt for x in bottom])
                    top_high_condn = any([x in user_prompt for x in top_high])
                    low_condn = any([x in user_prompt for x in bottom_low])
                    top_debit_condn = any([x in user_prompt for x in top_debit])
                    bottom_debit_condn = any([x in user_prompt for x in bottom_debit])
                    top_debit_high_condn = any([x in user_prompt for x in top_debit_high])
                    low_debit_condn = any([x in user_prompt for x in bottom_debit_low])
                    top_credit_condn = any([x in user_prompt for x in top_credit])
                    bottom_credit_condn = any([x in user_prompt for x in bottom_credit])
                    top_credit_high_condn = any([x in user_prompt for x in top_credit_high])
                    low_credit_condn = any([x in user_prompt for x in bottom_credit_low])

                    message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=83)

                    if validateJSON(st.session_state.responses[i]) == False:
                        message(st.session_state.responses[i], key=str(i), seed='Milo')
                    elif validateJSON(st.session_state.responses[i]) == True:
                        result = json.loads(st.session_state.responses[i])
                        if result.get("bar"):
                            try:
                                plot_df = pd.DataFrame(result["bar"])
                                message("The bar chart is plotted,", key=str(i), seed='Milo')
                                st.bar_chart(plot_df, x="x", y="y")
                            except Exception as e:
                                message("Sorry, unable to answer, try asking in a simpler way!", key=str(i), seed='Milo')
                        if result.get("line"):
                            try:
                                plot_df = pd.DataFrame(result["line"])
                                message("The line graph is plotted,", key=str(i), seed='Milo')
                                st.line_chart(plot_df, x="x", y="y")
                            except Exception as e:
                                message("Sorry, unable to answer, try asking in a simpler way!", key=str(i), seed='Milo')
                        if result.get("pie"):
                            try:
                                plot_df = pd.DataFrame(result["pie"])
                                message("The pie chart is plotted,", key=str(i), seed='Milo')
                                fig, ax = plt.subplots(figsize=(5, 5))
                                ax.pie(plot_df["y"], labels=plot_df["x"], wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white'})
                                #display a white circle in the middle of the pie chart
                                p = plt.gcf()
                                p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
                                st.pyplot(fig)
                            except Exception as e:
                                message("Sorry, unable to answer, try asking in a simpler way!", key=str(i), seed='Milo')
                        if result.get("multiple"):
                            try:
                                plot_df = pd.DataFrame(result["multiple"])
                                message("The result transactions are,", key=str(i), seed='Milo')
                                st.dataframe(plot_df)
                            except Exception as e:
                                message("Sorry, unable to answer, try asking in a simpler way!", key=str(i), seed='Milo')
                        if not result.get("bar") and not result.get("line") and not result.get("pie") and not result.get("multiple"):
                            try:
                                message(st.session_state.responses[i], key=str(i), seed='Milo')
                            except Exception as e:
                                message("Sorry, unable to answer, try asking in a simpler way!", key=str(i), seed='Milo')
                    else:
                        try:
                            message(st.session_state.responses[i], key=str(i), seed='Milo')
                        except Exception as e:
                            message("Sorry, unable to answer, try asking in a simpler way!", key=str(i), seed='Milo')

                    if top_debit_high_condn:
                        top_df = df.loc[df["transactionValue"] < 0]
                        top_df["transactionValue"] = abs(top_df["transactionValue"])
                        st.dataframe(top_df.nlargest(5, ['transactionValue']))
                    if top_credit_high_condn:
                        st.dataframe(df.loc[df["transactionValue"] > 0].nlargest(5, ['transactionValue']))
                    if top_high_condn:
                        top_df = df.copy()
                        top_df["transactionValue"] = abs(top_df["transactionValue"])
                        st.dataframe(top_df.nlargest(5, ['transactionValue']))
                    if top_credit_condn:
                        st.dataframe(df.loc[df["transactionValue"] > 0].head())
                    if top_debit_condn:
                        st.dataframe(df.loc[df["transactionValue"] < 0].head())
                    if top_condn:
                        st.dataframe(df.head())

                    if low_debit_condn:
                        top_df = df.loc[df["transactionValue"] < 0]
                        top_df["transactionValue"] = abs(top_df["transactionValue"])
                        st.dataframe(top_df.nsmallest(5, ['transactionValue']))
                    if low_credit_condn:
                        st.dataframe(df.loc[df["transactionValue"] > 0].nsmallest(5, ['transactionValue']))
                    if low_condn:
                        top_df = df.copy()
                        top_df["transactionValue"] = abs(top_df["transactionValue"])
                        st.dataframe(top_df.nsmallest(5, ['transactionValue']))
                    if bottom_debit_condn:
                        st.dataframe(df.loc[df["transactionValue"] < 0].tail())
                    if bottom_credit_condn:
                        st.dataframe(df.loc[df["transactionValue"] > 0].tail())
                    if bottom_condn:
                        st.dataframe(df.tail())     

        if chat_btn or st.session_state.chat_btn_active:
            st.text_input("Ask Something:", key="user")
            send_btn = st.button("Send", on_click=send_click)
            st.session_state.chat_btn_active = True
        
            

