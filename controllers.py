import os
import json
import sys
import pandas as pd
import numpy as np
from dateutil import parser as date_parser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfdevice import PDFDevice
import pdfminer
import PyPDF2
import tabula
from tabula import read_pdf
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_pandas_dataframe_agent
import csv
import datetime
from datetime import datetime
import PyPDF2
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
    
# get bank name
def get_bankname(file_path):
    pdfFileObj = open(file_path, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    pageObj = pdfReader.pages[0]
    pagelen = len(pdfReader.pages[0])
    pageData = pageObj.extract_text()
    bank_name = {'ICIC':'ICICI Bank', 'SBIN':'SBI Bank','HDFC':'HDFC Bank','UTIB':'Axis Bank','IDFB':'IDFC Bank'}
    listbyline = pageData.split('\n')
    name = ''
    for x in listbyline:
        if re.search(r"IFS Code|IFSC|IFSC Code", x):
            if re.search(r"([A-Z]{4})0[0-9]{6}", x):# checkfor ifsc code
                name = bank_name[re.search(r"([A-Z]{4})0[0-9]{6}", x).group()[0:4]] # bank name extraction
                break
    return name

def sonata(y):
        x=date_parser.parse(y,dayfirst=True)
        return x


def getnumber(text):
    isNegative = False

    if isinstance(text, int)==True or isinstance(text, float)==True:
        result=float(text)
    else:
        getVals=[]
        for val in text:
            if "-" in text:
                isNegative = True
            if val.isnumeric()==True or val==".":
                getVals.append(val)
        result = "".join(getVals)
        result=float(result)
    if isNegative:
        return result * -1
    return result

# get accountNumber for any bank out 5
def get_account_number(text):
    listbyline=text.split('\n')
    account_no=''
    for x in listbyline:
        if re.search(r"ACCOUNT NO :|A/C No:|Account No : ", x):
            if " " not in x.split(":")[1]:
                account_no = x.split(":")[1]
            else:
                account_no = x.split(":")[1].split(" ")[1]
        elif re.search(r"Account Number", x):
            account_no = x.split(":")[-1].replace("\t", '')
        elif re.search(r"Account No :", x):
            account_no = x.split(":")[1].split(" ")[0]
    return int(account_no)

# HDFC
def getcoord(path,tol_diff):
    fp = open(path, 'rb')
    # Create a PDF parser object associated with the file object.
    parser = PDFParser(fp)
    # Create a PDF document object that stores the document structure.
    # Password for initialization as 2nd parameter
    # password = ''
    document = PDFDocument(parser)
    # Check if the document allows text extraction. If not, abort.
    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()
    # Create a PDF device object.
    device = PDFDevice(rsrcmgr)
    # BEGIN LAYOUT ANALYSIS
    # Set parameters for analysis.
    laparams = LAParams()
    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    output=[]
    def parse_obj(lt_objs):
        # loop over the object list
        for obj in lt_objs:
            # if it's a textbox, print text and location
            if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
                # print(obj.bbox[0], obj.bbox[1], obj.get_text().replace('\n', '_'))
                output.append([obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3],obj.get_text().replace('\n', '_')])
            # if it's a container, recurse
            elif isinstance(obj, pdfminer.layout.LTFigure):
                parse_obj(obj._objs)
                output.append(parse_obj(obj._objs))
    # loop over all pages in the document
    for page in PDFPage.create_pages(document):
        # read the page into a layout object
        interpreter.process_page(page)
        layout = device.get_result()
        # extract text from this object
        parse_obj(layout._objs)

    x0=0
    x1=0
    x2=0
    x3=0
    x4=0
    x5=0
    x6=0
    x7=0
    # print(output)

    for i in output:
        if i!=None:
            if 'Date_' == i[4]:
                x0=i[0] + tol_diff + 10
            if 'Narration_' == i[4]:
                x1=i[0] + tol_diff + 110
            if 'Chq./Ref.No._' == i[4]:
                x2=i[2] + tol_diff + 50
            if "Value Dt Withdrawal Amt._" == i[4]:
                x3=i[2] + tol_diff
            if "Deposit Amt._" == i[4]:
                x4=i[2] + tol_diff
            if "Closing Balance_" == i[4]:
                x5=i[2] + tol_diff

    print([x0,x1,x2,x3,x4,x5])

    return [x0,x1,x2,x3,x4,x5]

def parse_hdfc(path):
    print("File name: ", path)
    tolerance=10
    columns=getcoord(path,tolerance)

    
    pdf = PyPDF2.PdfReader(open(path,'rb'))
    # pdf.decrypt(b'CONS866823038')
    pages=len(pdf.pages)

    text = pdf.pages[0].extract_text()


    a_num=get_account_number(text)
    arr= []
    for p in range(1,pages+1):

        x=tabula.read_pdf(path,guess=False,stream=False,columns=columns,pages=p,multiple_tables=True,pandas_options={'header':None})

        for i in x:
            i=i.fillna(0)

            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            #     print(i)
            # print(i)
            for j in range(0,len(i)):
                    flag=0

                    try:
                        sonata(i[0][j])
                        total=i[5][j]

                        flag=1

                        if total==0:
                            flag=0

                    except Exception: pass


                    if flag!=0:

                        date=sonata(i[0][j])
                        # print(date)

                        transaction_value=0

                        transaction_type=""

                        # print(i[4][j])
                        # print(type (i[4][j]))

                        if getnumber(i[3][j])>0 and getnumber(i[4][j]) == 0:
                            transaction_value=getnumber(i[3][j])
                            transaction_type="debit"

                        if getnumber(i[4][j])>0 and getnumber(i[3][j]) == 0:
                            transaction_value=getnumber(i[4][j])
                            transaction_type="credit"


                        desc = i[1][j]
                        for k in range(1,4):
                            try:
                                val_1=float(i[3][j+k])+float(i[4][j+k])+float(i[5][j+k])
                                if val_1==0:
                                    desc=desc + i[1][j+k]
                            except Exception: break

                        balance=getnumber(i[5][j])

                        final_res={
                            "accountNumber": a_num,
                            "bankName": "HDFC Bank",
                            "balance": float(balance),
                            "date": date,
                            "transactionValue":float(transaction_value),
                            "transactionType": transaction_type,
                            "description": desc.upper()
                          }
                        
                        arr.append(final_res)

    # print(arr)
    df = pd.DataFrame(arr)
    print(df.info())
    return df

# ICICI
def getcoord1(path,tol_diff):
    fp = open(path, 'rb')
    # Create a PDF parser object associated with the file object.
    parser = PDFParser(fp)
    # Create a PDF document object that stores the document structure.
    # Password for initialization as 2nd parameter
    # password = ''
    document = PDFDocument(parser)
    # Check if the document allows text extraction. If not, abort.
    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()
    # Create a PDF device object.
    device = PDFDevice(rsrcmgr)
    # BEGIN LAYOUT ANALYSIS
    # Set parameters for analysis.
    laparams = LAParams()
    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    output=[]
    def parse_obj(lt_objs):
        # loop over the object list
        for obj in lt_objs:
            # if it's a textbox, print text and location
            if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
                # print(obj.bbox[0], obj.bbox[1], obj.get_text().replace('\n', '_'))
                output.append([obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3],obj.get_text().replace('\n', '_')])
            # if it's a container, recurse
            elif isinstance(obj, pdfminer.layout.LTFigure):
                parse_obj(obj._objs)
                output.append(parse_obj(obj._objs))
    # loop over all pages in the document
    for page in PDFPage.create_pages(document):
        # read the page into a layout object
        interpreter.process_page(page)
        layout = device.get_result()
        # extract text from this object
        parse_obj(layout._objs)

    x0=0
    x1=0
    x2=0
    x3=0
    x4=0
    x5=0
    x6=0
    x7=0
    # print(output)

    for i in output:
        if i!=None:
            if 'Sl_No_1_' in i[4]:
                x0=i[0] + tol_diff + 70
            if 'Transaction_' == i[4]:
                x1=i[0] + tol_diff
            if 'Transaction_Posted Date_' in i[4]:
                x2=i[0] - tol_diff
                x3=i[2] + tol_diff + 40
            # if 'Cheque no /_' == i[4]:
            if 'Remarks_SI/' in i[4]:
                x4=i[0] + tol_diff + 50
            if "Withdra_wal (Dr)_" == i[4]:
                x5=i[0] - tol_diff
            if "Deposit_" == i[4]:
                x6=i[0] - tol_diff
            if "Balance_" == i[4]:
                x7=i[0] - tol_diff

    print([x0,x1,x2,x3,x4,x5,x6,x7])

    return [x0,x1,x2,x3,x4,x5,x6,x7]

def parse_icici(path):
    print("File name: ", path)
    tolerance=10
    columns=getcoord1(path,tolerance)

    
    pdf = PyPDF2.PdfReader(open(path,'rb'))
    # pdf.decrypt(b'CONS866823038')
    pages=len(pdf.pages)

    text = pdf.pages[0].extract_text()


    a_num=get_account_number(text)
    arr= []
    for p in range(1,pages+1):

        x=tabula.read_pdf(path,guess=False,stream=False,columns=columns,pages=p,multiple_tables=True,pandas_options={'header':None})

        for i in x:
            i=i.fillna(0)

            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            #     print(i)
    #         # print(i)
            for j in range(0,len(i)):
                    flag=0

                    try:
                        sonata(i[1][j])
                        total=i[8][j]

                        flag=1

                        if total==0:
                            flag=0

                    except Exception: pass


                    if flag!=0:

                        date=sonata(i[1][j])
                        # print(date)

                        transaction_value=0

                        transaction_type=""

                        # print(i[4][j])
                        # print(type (i[4][j]))

                        if getnumber(i[6][j])>0 and getnumber(i[7][j])==0:
                            transaction_value=getnumber(i[6][j])
                            transaction_type="debit"

                        if getnumber(i[7][j])>0 and getnumber(i[6][j])==0:
                            transaction_value=getnumber(i[7][j])
                            transaction_type="credit"


                        desc = str(i[3][j]) + str(i[4][j])
                        for k in range(1,7):
                            try:
                                val_1=float(i[6][j+k])+float(i[7][j+k])
                                if val_1==0:
                                    desc=desc + str(i[3][j+k]) + str(i[4][j+k])
                            except Exception: break

                        balance=getnumber(i[8][j]) + getnumber(i[8][j+1])

                        final_res={
                            "accountNumber": a_num,
                            "bankName": "ICICI Bank",
                            "balance": float(balance),
                            "date": date,
                            "transactionValue":float(transaction_value),
                            "transactionType": transaction_type,
                            "description": desc.upper()
                          }
                        
                        print(final_res)
                        arr.append(final_res)

    df = pd.DataFrame(arr)
    df.dropna(inplace=True)
    print(df.info())
    return df

# SBI 
def parse_sbi(filename):
    print("File name: ", filename)
    df = read_pdf(filename,pages="all") #address of pdf file
    for page in range(len(df)):
        descrip=''
        loc_to_update=0; #inital location to update;
        for row in range(len(df[page])):
            if(str(df[page].loc[row][0])=='nan'):
                if(str(df[page].loc[row][2])!='nan'):
                    descrip+=str(df[page].loc[row][2])
            else:
        #         this block is for updating value
        #         loc_to_update
                df[page].loc[loc_to_update][2]=descrip
                loc_to_update=row
                descrip=str(df[page].loc[row][2])
            if(row+1 == len(df[page])):
                df[page].loc[loc_to_update][2]=descrip
          
    #    
    for pg in range(len(df)):
        for x in df[pg].index:
            if(str(df[pg].loc[x][0]) == 'nan'):
                df[pg].drop(x, inplace = True)

    for p in range(len(df)):
        df[p].fillna(0, inplace = True)    

    df_big = pd.concat(df)  #make single df from list of df 
    remo_column=[]
    for col in df_big.columns:
        if 'Unnamed' in col:
            remo_column.append(col)

    new_ind=[]
    inc=0
    for x in range(len(df)):
        for i in df[x].index:
            new_ind.append(inc)
            inc+=1

    df_big.set_index(pd.Index(new_ind),inplace=True) #set new index for df
    df_big.drop(remo_column, axis=1, inplace=True) #unwanted column removed
    # df_big.drop(['Txn Date','Ref No./Cheque','Branch'], axis=1, inplace=True) #unwanted column removed
    df_big['transactionValue']=0
    df_big['transactionType']=''
    # add two column tansaction type and value
    for x in df_big.index:
        if(df_big.loc[x,'Debit']==0):
            df_big.loc[x, "transactionValue"] =df_big.loc[x,'Credit']
            df_big.loc[x, "transactionType"] ='credit'
        else:    
            df_big.loc[x, "transactionValue"] =df_big.loc[x,'Debit']
            df_big.loc[x, "transactionType"] ='debit'

    df_big.drop(['Debit','Credit'], axis=1, inplace=True) #unwanted column removed
    df_big = df_big.set_axis(['date','description', 'balance', 'transactionValue', 'transactionType'], axis=1)
    df_big['bankName'] = 'SBI Bank'
    df_big['date'] = pd.to_datetime(df_big['date'],dayfirst=True, format='%d/%m/%Y')
    df_big['transactionValue'] = df_big['transactionValue'].str.replace(',','').astype(float)
    df_big['balance'] = df_big['balance'].str.replace(',','').astype(float)

    print(df_big.info())
    return df_big

# IDFC
def getcoord2(path,tol_diff):
    fp = open(path, 'rb')
    # Create a PDF parser object associated with the file object.
    parser = PDFParser(fp)
    # Create a PDF document object that stores the document structure.
    # Password for initialization as 2nd parameter
    # password = ''
    document = PDFDocument(parser)
    # Check if the document allows text extraction. If not, abort.
    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()
    # Create a PDF device object.
    device = PDFDevice(rsrcmgr)
    # BEGIN LAYOUT ANALYSIS
    # Set parameters for analysis.
    laparams = LAParams()
    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    output=[]
    def parse_obj(lt_objs):
        # loop over the object list
        for obj in lt_objs:
            # if it's a textbox, print text and location
            if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
                # print(obj.bbox[0], obj.bbox[1], obj.get_text().replace('\n', '_'))
                output.append([obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3],obj.get_text().replace('\n', '_')])
            # if it's a container, recurse
            elif isinstance(obj, pdfminer.layout.LTFigure):
                parse_obj(obj._objs)
                output.append(parse_obj(obj._objs))
    # loop over all pages in the document
    for page in PDFPage.create_pages(document):
        # read the page into a layout object
        interpreter.process_page(page)
        layout = device.get_result()
        # extract text from this object
        parse_obj(layout._objs)

    x0=0
    x1=0
    x2=0
    x3=0
    x4=0
    x5=0
    x6=0
    x7=0
    # print(output)

    for i in output:
        if i!=None:
            if 'Transaction Date Value Date_' == i[4]:
                x0=i[0] + tol_diff + 70
                x1=i[2] + tol_diff
            if 'Particulars_' == i[4]:
                x2=i[2] + tol_diff + 50
            if 'Cheque _No_' == i[4]:
                x3=i[2] + tol_diff + 10
            if "Debit_" == i[4]:
                x4=i[2] + tol_diff + 10
            if "Credit_" == i[4]:
                x5=i[2] + tol_diff + 5
            # if "Balance_" == i[4]:
            #     x6=i[2] + tol_diff

    print([x0,x1,x2,x3,x4,x5])

    return [x0,x1,x2,x3,x4,x5]

def parse_idfc(path):
    print("File name: ", path)
    tolerance=10
    columns=getcoord2(path,tolerance)

    
    pdf = PyPDF2.PdfReader(open(path,'rb'))
    # pdf.decrypt(b'CONS866823038')
    pages=len(pdf.pages)

    text = pdf.pages[0].extract_text()


    a_num=get_account_number(text)
    arr= []
    for p in range(1,pages+1):

        x=tabula.read_pdf(path,guess=False,stream=False,columns=columns,pages=p,multiple_tables=True,pandas_options={'header':None},encoding= 'unicode_escape')

        for i in x:
            i=i.fillna(0)

            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            #     print(i)
            # print(i)
            for j in range(0,len(i)):
                    flag=0

                    try:
                        sonata(i[0][j])
                        total=i[6][j]

                        flag=1

                        if total==0:
                            flag=0

                    except Exception: pass


                    if flag!=0:

                        date=sonata(i[0][j])
                        # print(date)

                        transaction_value=0

                        transaction_type=""

                        # print(i[4][j])
                        # print(type (i[4][j]))

                        if getnumber(i[4][j])>0 and getnumber(i[5][j]) == 0:
                            transaction_value=getnumber(i[4][j])
                            transaction_type="debit"

                        if getnumber(i[5][j])>0 and getnumber(i[4][j])==0:
                            transaction_value=getnumber(i[5][j])
                            transaction_type="credit"


                        desc = i[2][j]
                        for k in range(1,4):
                            try:
                                val_1=float(i[4][j+k])+float(i[5][j+k])
                                if val_1==0:
                                    desc=desc + i[2][j+k]
                            except Exception: break

                        balance=getnumber(i[6][j])

                        final_res={
                            "accountNumber": a_num,
                            "bankName":"IDFC Bank",
                            "balance": float(balance),
                            "date": date,
                            "transactionValue":float(transaction_value),
                            "transactionType": transaction_type,
                            "description": desc.upper()
                          }
                        
                        # print(final_res)
                        arr.append(final_res)

    # print(arr)
    df = pd.DataFrame(arr)
    print(df.info())
    return df

# AXIS
def parse_axis(filename):
    
    print("file name",filename)
    df = read_pdf(filename,pages="all") #address of pdf file
    # set columns to all df  till second last page
    for page in range(len(df)-2):
        df[page] = df[page].set_axis(df[0].columns, axis=1)
        
    for pg in range(len(df)):
        for x in df[pg].index:
            if str(df[pg].loc[x][0]) == 'nan':
                df[pg].drop(x, inplace = True)
            
    for p in range(len(df)-2):
        df[p].fillna(0, inplace = True)
    
    # import pandas as pd

    df_big = pd.concat(df[0:-2])  #make single df from list of df
    
    new_ind=[]
    inc=0
    for x in range(len(df)-2):

        for i in df[x].index:
            new_ind.append(inc)
            inc+=1
            
    df_big = df_big.set_index(pd.Index(new_ind)) #set new index for df        
    
    df_big.drop(['Chq No','Branch Name','Chq No','Value Date'], axis=1, inplace=True) #unwanted column removed
    df_big["Transaction Type"] = ""
    df_big.loc[df_big["DR/CR"] == 'DR', "Transaction Type"] = "debit"
    df_big.loc[df_big["DR/CR"] == 'CR', "Transaction Type"] = "credit"

    df_big.drop('DR/CR',axis=1, inplace=True)
        
    df_big = df_big.set_axis(['date','description', 'balance', 'transactionValue', 'transactionType'], axis=1)
    df_big['bankName']='AXIS Bank'
    
    removepoint=0
    for x in df_big.index:
#         print(df_big.loc[x][0])
        if(df_big.loc[x][0]=='Sr. No.'):
            removepoint=x
            break
    
    df_big=df_big.loc[0:removepoint-1]
    
    df_big['date']=pd.to_datetime(df_big['date'],dayfirst=True, format='%d-%m-%Y')
    df_big['transactionValue'] = df_big['transactionValue'].astype(float)
    df_big['balance'] = df_big['balance'].astype(float)
#     df_big['date']=pd.to_datetime(df_big['date'],dayfirst=True)
    print(df_big.info())
    
    return df_big
    
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
    df_new.loc[(df_new["category"] == "Direct Expense") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Transfer-in/Revenue-other") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Adjustment/reversal") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Agency/Vendor Expense") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Bank charges") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Cash/cheque deposit") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Investment New") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Investment/FD deposit") & (df_new["transactionType"] == "credit"), "category"] = "Investment/FD redeem"
    df_new.loc[(df_new["category"] == "Investment/FD redeem") & (df_new["transactionType"] == "debit"), "category"] = "Investment/FD deposit"
    df_new.loc[(df_new["category"] == "Loan repayment") & (df_new["transactionType"] == "credit"), "category"] = "Loan-in"
    df_new.loc[(df_new["category"] == "Loan-in") & (df_new["transactionType"] == "debit"), "category"] = "Loan repayment"
    df_new.loc[(df_new["category"] == "Outward bounce") & (df_new["transactionType"] == "credit"), "category"] = "Inward bounce"
    df_new.loc[(df_new["category"] == "Inward bounce") & (df_new["transactionType"] == "debit"), "category"] = "Outward bounce"
    df_new.loc[(df_new["category"] == "Rental expense") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Revenue PG Lender Escrow") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Revenue PG Non split") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Revenue UPI") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Salary/Emp/Consultant") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Tax") & (df_new["transactionType"] == "credit"), "category"] = "Tax/other-credit"
    df_new.loc[(df_new["category"] == "Tax/other-credit") & (df_new["transactionType"] == "debit"), "category"] = "Tax"
    df_new.loc[(df_new["category"] == "Utilities/Bill") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Cash Expense") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "OD/CC Repayment") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Interest income") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Deposit credit") & (df_new["transactionType"] == "debit"), "category"] = "Deposit debit"
    df_new.loc[(df_new["category"] == "Deposit debit") & (df_new["transactionType"] == "credit"), "category"] = "Deposit credit"
    df_new.loc[(df_new["category"] == "Revenue POS") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Revenue PG split") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Revenue COD") & (df_new["transactionType"] == "debit") & (df_new["description"].str.strip().str.upper().str.contains("DELHIV", case=False)), "category"] = "Saas/Tech"
    df_new.loc[(df_new["category"] == "Revenue COD") & (df_new["transactionType"] == "debit") & (df_new["description"].str.strip().str.upper().str.contains("DELHIV", case=False) == False), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Marketing") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category"] = "Revenue Marketplace"
    df_new.loc[(df_new["category"] == "Marketing") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False) == False), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Saas/Tech") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category"] = "Revenue Marketplace"
    df_new.loc[(df_new["category"] == "Saas/Tech") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("DELHIV", case=False)), "category"] = "Revenue COD"
    df_new.loc[(df_new["category"] == "Saas/Tech") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("DELHIV", case=False) == False) & (df_new["description"].str.strip().str.upper().str.contains("AMAZ") == False), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Revenue Marketplace") & (df_new["transactionType"] == "debit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False) == False), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Revenue Marketplace") & (df_new["transactionType"] == "debit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category"] = "Marketing"
    df_new.loc[(df_new["category"] == "Revenue Marketplace") & (df_new["transactionType"] == "debit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZONSE", case=False)), "category"] = "Saas/Tech"
    df_new.loc[(df_new["category"] == "Revenue Marketplace") & (df_new["transactionType"] == "debit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category"] = "Marketing"
    df_new.loc[(df_new["category"] == "Refund") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False)), "category"] = "Revenue Marketplace"
    df_new.loc[(df_new["category"] == "Refund") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("AMAZ", case=False) == False), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Nach payment") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("RAZ", case=False)), "category"] = "Revenue PG split"
    df_new.loc[(df_new["category"] == "Nach payment") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("LOAN|INCRED|BAJAJ", regex=True, case=False)), "category"] = "Loan-in"
    df_new.loc[(df_new["category"] == "Nach payment") & (df_new["transactionType"] == "credit") & (df_new["description"].str.strip().str.upper().str.contains("LOAN|INCRED|BAJAJ|RAZ", regex=True, case=False) == False), "category"] = "Transfer-in/Revenue-other"
    df_new.loc[(df_new["category"] == "Revenue Escrow") & (df_new["transactionType"] == "debit"), "category"] = "Direct Expense"
    df_new.loc[(df_new["category"] == "Trading/Investment") & (df_new["transactionType"] == "credit"), "category"] = "Transfer-in/Revenue-other"
    
    return df_new

def bank_classifier_predict(df):
    # grab environment variables
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    
    start = time.time()
    runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1', aws_access_key_id=os.environ["SAGEMAKER_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["SAGEMAKER_SECRET_ACCESS_KEY"])
    print("EndpointName: ", ENDPOINT_NAME)

    print("Running classifier predict...")

    cat_dict = {0: 'Cash/cheque deposit', 1: 'Outward bounce', 2: 'Transfer-in/Revenue-other', 3: 'Revenue UPI', 4: 
        'Direct Expense', 5: 'Bank charges', 6: 'Tax', 7: 'Revenue COD', 8: 'Trading/Investment', 
        9: 'Revenue PG Non split', 10: 'Marketing', 11: 'Utilities/Bill', 12: 'Revenue PG split', 
        13: 'Salary/Emp/Consultant', 14: 'Rental expense', 15: 'Saas/Tech', 16: 'Cash Expense', 
        17: 'Revenue Marketplace', 18: 'Refund', 19: 'Investment New', 20: 'Adjustment/reversal', 
        21: 'Inward bounce', 22: 'Loan-in', 23: 'Agency/Vendor Expense', 24: 'Investment/FD redeem', 
        25: 'Nach payment', 26: 'OD/CC Repayment', 27: 'Interest income', 28: 'Investment/FD deposit', 
        29: 'Revenue Escrow', 30: 'Revenue PG Lender Escrow', 31: 'Loan repayment', 32: 'Tax/other-credit', 
        33: 'Deposit debit', 34: 'Deposit credit', 35: 'Revenue POS'}

    if df.shape[0] > 0:
        df_new = df.dropna()
        df_new["description"] = df_new["description"].astype('string')
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
            df_new.to_csv("bank_data.csv", index=False)

            end = time.time()
            print("Successfully predicted...Time taken: {} secs".format(end - start))
            return "bank_data.csv"
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
            df_new.to_csv("bank_data.csv", index=False)

            end = time.time()
            print("Successfully predicted...Time taken: {} secs".format(end - start))
            return "bank_data.csv"
    else:
        return "No data found..."