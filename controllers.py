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

def validateJSON(jsonData):
    try:
        result = json.loads(jsonData) and type(json.loads(jsonData)) is dict
        if result:
            return True
        else:
            return False
    except ValueError as err:
        return False