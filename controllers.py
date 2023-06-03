import os
import json
import sys
import pandas as pd
import numpy as np
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_pandas_dataframe_agent
import csv
import datetime
from tokenize import Number
import pandas
import numpy as np
import requests
import time
import math
import io
import re
import boto3
import json
import tensorflow
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
# nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def validateJSON(jsonData):
    try:
        result = json.loads(jsonData) and type(json.loads(jsonData)) is dict
        if result:
            return True
        else:
            return False
    except ValueError as err:
        return False
    
def preprocess_text(sentence):
    sentence = str(sentence)     # Convert sentence to into string
    sentence = sentence.upper()  # Uppercase
    sentence = sentence.replace('{html}',"")  # Replace html to blank
    cleanr = re.compile('<.*?>')   # Special characters
    cleantext = re.sub(cleanr, '', sentence) 
    tokens = word_tokenize(cleantext)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]  # Words that are not in stop words & occured     more than 2 times
    return " ".join(filtered_words)  # Join the filtered tokens

def postprocess(df_new):
    df_new.loc[(df_new["category_new"] == "Direct Expense") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Transfer-in/Revenue-other") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Adjustment/reversal") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Agency/Vendor Expense") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Bank charges") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Cash/cheque deposit") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Investment-New") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Investment/FD deposit") & (df_new["transactionValue"] > 0), "category_new"] = "Investment/FD redeem"
    df_new.loc[(df_new["category_new"] == "Investment/FD redeem") & (df_new["transactionValue"] < 0), "category_new"] = "Investment/FD deposit"
    df_new.loc[(df_new["category_new"] == "Loan repayment") & (df_new["transactionValue"] > 0), "category_new"] = "Loan-in"
    df_new.loc[(df_new["category_new"] == "Loan-in") & (df_new["transactionValue"] < 0), "category_new"] = "Loan repayment"
    df_new.loc[(df_new["category_new"] == "Outward bounce") & (df_new["transactionValue"] > 0), "category_new"] = "Inward bounce"
    df_new.loc[(df_new["category_new"] == "Inward bounce") & (df_new["transactionValue"] < 0), "category_new"] = "Outward bounce"
    df_new.loc[(df_new["category_new"] == "Rental expense") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Revenue-PG-Lender-Escrow") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Revenue-PG-Non-split") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Revenue-UPI") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Salary/Emp/Consultant") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Tax") & (df_new["transactionValue"] > 0), "category_new"] = "Tax/other-credit"
    df_new.loc[(df_new["category_new"] == "Tax/other-credit") & (df_new["transactionValue"] < 0), "category_new"] = "Tax"
    df_new.loc[(df_new["category_new"] == "Utilities/Bill") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Cash Expense") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "OD/CC Repayment") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Interest income") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Deposit-credit") & (df_new["transactionValue"] < 0), "category_new"] = "Deposit-debit"
    df_new.loc[(df_new["category_new"] == "Deposit-debit") & (df_new["transactionValue"] > 0), "category_new"] = "Deposit-credit"
    df_new.loc[(df_new["category_new"] == "Revenue-POS") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Revenue-PG-split") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Revenue-COD") & (df_new["transactionValue"] < 0) & (df_new["description"].str.strip().str.upper().str.contains("DELHIV", case=False)), "category_new"] = "Saas/Tech"
    df_new.loc[(df_new["category_new"] == "Revenue-COD") & (df_new["transactionValue"] < 0) & (df_new["description"].str.strip().str.upper().str.contains("DELHIV", case=False) == False), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Marketing") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category_new"] = "Revenue-Mktplace"
    df_new.loc[(df_new["category_new"] == "Marketing") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False) == False), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Saas/Tech") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category_new"] = "Revenue-Mktplace"
    df_new.loc[(df_new["category_new"] == "Saas/Tech") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("DELHIV", case=False)), "category_new"] = "Revenue-COD"
    df_new.loc[(df_new["category_new"] == "Saas/Tech") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("DELHIV", case=False) == False) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ") == False), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Revenue-Mktplace") & (df_new["transactionValue"] < 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False) == False), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Revenue-Mktplace") & (df_new["transactionValue"] < 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category_new"] = "Marketing"
    df_new.loc[(df_new["category_new"] == "Revenue-Mktplace") & (df_new["transactionValue"] < 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZONSE", case=False)), "category_new"] = "Saas/Tech"
    df_new.loc[(df_new["category_new"] == "Revenue-Mktplace") & (df_new["transactionValue"] < 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category_new"] = "Marketing"
    df_new.loc[(df_new["category_new"] == "Refund") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category_new"] = "Revenue-Mktplace"
    df_new.loc[(df_new["category_new"] == "Refund") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False) == False), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Nach payment") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("RAZ", case=False)), "category_new"] = "Revenue-PG-split"
    df_new.loc[(df_new["category_new"] == "Nach payment") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("LOAN|INCRED|BAJAJ", regex=True, case=False)), "category_new"] = "Loan-in"
    df_new.loc[(df_new["category_new"] == "Nach payment") & (df_new["transactionValue"] > 0) & (df_new["description"].str.strip().str.upper().str.contains("LOAN|INCRED|BAJAJ|RAZ", regex=True, case=False) == False), "category_new"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category_new"] == "Revenue-Escrow") & (df_new["transactionValue"] < 0), "category_new"] = "Direct Expense"
    df_new.loc[(df_new["category_new"] == "Trading/Investment") & (df_new["transactionValue"] > 0), "category_new"] = "Transfer-in/Revenue-other"
    
    return df_new

def bank_classifier_predict(file_name):
    # grab environment variables
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    
    start = time.time()
    runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1', aws_access_key_id=os.environ["SAGEMAKER_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["SAGEMAKER_SECRET_ACCESS_KEY"])
    print("EndpointName: ", ENDPOINT_NAME)

    print("Running classifier predict...")

    cat_dict = {0: 'Cash/cheque deposit', 1: 'Outward bounce', 2: 'Transfer-in/Revenue-other', 3: 'Revenue-UPI', 4: 
        'Direct Expense', 5: 'Bank charges', 6: 'Tax', 7: 'Revenue-COD', 8: 'Trading/Investment', 
        9: 'Revenue-PG-Non-split', 10: 'Marketing', 11: 'Utilities/Bill', 12: 'Revenue-PG-split', 
        13: 'Salary/Emp/Consultant', 14: 'Rental expense', 15: 'Saas/Tech', 16: 'Cash Expense', 
        17: 'Revenue-Mktplace', 18: 'Refund', 19: 'Investment-New', 20: 'Adjustment/reversal', 
        21: 'Inward bounce', 22: 'Loan-in', 23: 'Agency/Vendor Expense', 24: 'Investment/FD redeem', 
        25: 'Nach payment', 26: 'OD/CC Repayment', 27: 'Interest income', 28: 'Investment/FD deposit', 
        29: 'Revenue-Escrow', 30: 'Revenue-PG-Lender-Escrow', 31: 'Loan repayment', 32: 'Tax/other-credit', 
        33: 'Deposit-debit', 34: 'Deposit-credit', 35: 'Revenue-POS'}
    
    payload_df = pd.read_csv(file_name)

    if payload_df.shape[0] > 0:
        df_new = payload_df[["_id", "description", "transactionValue"]]
        df_new["description"] = df_new["description"].astype('string')
        print("DF: ", df_new)
        print("Info: ", df_new.info())

        df_clean = df_new.copy()
        df_clean["description"] = df_new["description"].map(lambda s: preprocess_text(s))

        max_len = 200
        X = tokenizer(
            text=df_clean['description'].tolist(),
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=True
        )
        print("X: ", X)

        input_id = X["input_ids"].numpy()
        input_id = json.dumps(input_id.tolist())
        attention_mask = X["attention_mask"].numpy()
        attention_mask = json.dumps(attention_mask.tolist())

        inp_data = []
        for inpt, attn in zip(json.loads(input_id), json.loads(attention_mask)):
            if len(inpt) < max_len:
                diff = max_len - len(inpt)
                n = [0] * diff
                inpt.extend(n)
            if len(attn) < max_len:
                diff = max_len - len(attn)
                n = [0] * diff
                attn.extend(n)
            dic = {'input_ids': inpt, 'attention_mask': attn}
            inp_data.append(dic)
    

        if df_clean.shape[0] > 360:
            predictions = []
            
            num = math.ceil(len(inp_data) / 360)
            print("No. of Batches: ", num)
            json_list = np.array_split(inp_data, num)
            
            for idx, res in enumerate(json_list):
                print("Batch: ", idx + 1)
                json_df = json.dumps(res.tolist())
                # df_parq = res[["description"]].to_parquet()
                # res["description"] = res["description"].astype('string')
                start1 = time.time()
                response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, Body=json_df, ContentType='application/json')
                # print("Response: ", response)
                end1 = time.time()
                print("Endpoint time taken: {}".format(end1 - start1))

                result = json.loads(response['Body'].read().decode())
                # print("Result: ", result)
                        
                final_predicted = [cat_dict[val] for val in result["Predictions"]]
                predictions.extend(final_predicted)
                
            df_new["category"] = predictions
            df_new = postprocess(df_new)

            end = time.time()
            print("Successfully predicted...Time taken: {} secs".format(end - start))
            return df_new
        else:
            print("Batches: 1")
            # df_new["description"] = df_new["description"].astype('string')
            # df_parq = df_new.to_parquet()
            json_df = json.dumps(inp_data)
            start1 = time.time()
            response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, Body=json_df, ContentType='application/json')
            # print("Response: ", response)
            end1 = time.time()
            print("Endpoint time taken: {}".format(end1 - start1))

            result = json.loads(response['Body'].read().decode())
            # print("Result: ", result)
                    
            final_predicted = [cat_dict[val] for val in result["Predictions"]]
            # ids = payload_df["_id"].values.tolist()
            
            df_new["category"] = final_predicted
            df_new = postprocess(df_new)
            
            end = time.time()
            print("Successfully predicted...Time taken: {} secs".format(end - start))
            return df_new
    else:
        return "No data found..."