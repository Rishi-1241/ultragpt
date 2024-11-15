from functools import wraps
import os
import uuid
import openai
import weaviate
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from collections import Counter
from tempfile import TemporaryDirectory
import pytesseract
from pathlib import Path
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import logging
import requests
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
import re
import io
from io import StringIO
from googleapiclient.discovery import build
import json
from flask import send_from_directory, send_file
import time, datetime
import threading
from flask_mysqldb import MySQL
from flask import Response
import gpt3_tokenizer
import google
import google.oauth2.credentials
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64
import urllib
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from functools import wraps
import jwt
import logging
from sqlalchemy import Column, Integer, String, Boolean
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from shutil import move
from werkzeug.utils import secure_filename
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import jwt
import time
import datetime
import json
import urllib.parse
import openai
import mysql.connector as mysql
import mysql.connector
from mysql.connector import pooling, OperationalError
# Initialize FastAPI app
app = FastAPI()

# environment setup
open_api_key = ""  # add your openai api key here
os.environ["OPENAI_API_KEY"] =  "sk-qZA5vCHZR0_tgTBYmMiOxDMwSqM-tzv5oNIs313-nHT3BlbkFJw-dAjV_QA9Qa8eYkc4mWTHTnF3vLuIBqx-SaQz6dcA"
openai.api_key = "sk-qZA5vCHZR0_tgTBYmMiOxDMwSqM-tzv5oNIs313-nHT3BlbkFJw-dAjV_QA9Qa8eYkc4mWTHTnF3vLuIBqx-SaQz6dcA"
YTapi_key = "AIzaSyAHlhEhjECdTBDLs_JeGygh9J5T3tBwDd4"
Gapi_key = "AIzaSyAHlhEhjECdTBDLs_JeGygh9J5T3tBwDd4"
cx = "f6102f35bce1e44ed"
num_results = 4  # main to dekh bhi nhi rha tha 200 se isi lie start kia tha Xd  GUD GUD tu hi idhar le aaya xD

# Logging configuration
logging.basicConfig(format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

# Database setup (SQLAlchemy)
SQLALCHEMY_DATABASE_URL = "mysql://root:abc123@localhost/prakhar"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# User model
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email_id = Column(String, unique=True, index=True)
    password = Column(String)
    bots = Column(String, default="[]")  # Storing bots as JSON string

Base.metadata.create_all(bind=engine)

class Bot(Base):
    __tablename__ = 'bots'

    id = Column(Integer, primary_key=True, index=True)
    botid = Column(String, unique=True, index=True)
    username = Column(String, index=True)
    personal = Column(Boolean, default=False)  # To track if the bot is primary or not

# Pydantic models for request validation
class UserRegisterRequest(BaseModel):
    name: str
    email_id: str
    password: str

class UserResponse(BaseModel):
    success: bool
    message: str
    token: str
    data: dict
    bots: list = []

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth verification decorator
from fastapi import Request

# db_config = {
#     "user": "root",
#     "password": "abc123",
#     "host": "127.0.0.1",
#     "port": "3306",
#     "database": "prakhar",
# }
# pool = pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **db_config)

# def get_connection():
#     try:
#         conn = pool.get_connection()
#         conn.ping(reconnect=True, attempts=3, delay=5)
#         return conn
#     except OperationalError as e:
#         print("Reconnecting to MySQL:", e)
#         conn = pool.get_connection()
#         return conn

# sending gmail without the below functions, as they don't work
def sendMail(sender, to, subject, msg="", msgHtml=None):
    print("Parmas", sender, to, subject, msg, msgHtml)
    # getting refresh token from sql record of sender email
    conn = mysql.connect()
    cur = conn.cursor()
    query = "SELECT refresh_token FROM users WHERE email_id=%s"
    cur.execute(query, (sender,))
    result = cur.fetchall()
    cur.close()
    conn.close()
    refresh_token = result[0][0]
    if (refresh_token == "" or refresh_token == None):
        return "Need Google Sign In for this feature"
    print("Got refresh token", refresh_token, "for", sender, "from sql")

    # getting new access_token from refresh token
    url = "https://www.googleapis.com/oauth2/v4/token"
    payload = {
        "grant_type": "refresh_token",
        "client_id": "",
        "client_secret": "",
        "refresh_token": refresh_token
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    print("RESponse", response)
    try:
        acctok422 = response.json()["access_token"]
    except:
        return "Need to sign in again"
    print("Access token", acctok422)

    credentials = google.oauth2.credentials.Credentials(
        token=acctok422,
        refresh_token=refresh_token,
        token_uri='https://oauth2.googleapis.com/token',
        client_id="",
        client_secret='',
        scopes=["https://www.googleapis.com/auth/gmail.send"]
    )
    # http = credentials.authorize(httplib2.Http())
    service = build('gmail', 'v1', credentials=credentials)
# API functions for ChatCompletion Functions (right now only weather & YT search are used)
def get_weather(city):
    print("Getting weather of", city)
    api_key = "bbdce49abdbc412d9457fb27eaef8a5c"
    base_url = "https://api.weatherbit.io/v2.0/current"
    params = {
        "key": api_key,
        "city": city
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code == 200:
        weather = data["data"][0]
        city_name = weather["city_name"]
        temperature = weather["temp"]
        humidity = weather["rh"]
        pressure = weather["pres"]
        wind_speed = weather["wind_spd"]
        cloud_cover = weather["clouds"]

        output = f"City: {city_name}\nTemperature: {temperature}°C\nHumidity: {humidity}%\nPressure: {pressure} mb\nWind Speed: {wind_speed} m/s\nCloud Cover: {cloud_cover}%"
        return output
    else:
        return "Failed to retrieve weather information."


def search_videos(query, max_results=3):
    """
    Search for videos on YouTube based on a query and return the top results.
    """
    api_key = "AIzaSyAHlhEhjECdTBDLs_JeGygh9J5T3tBwDd4"
    # Perform a search request
    youtube = build('youtube', 'v3', developerKey=api_key)
    search_request = youtube.search().list(
        q=query,
        part='id',
        maxResults=max_results,
        type='video'
    )
    search_response = search_request.execute()

    videos = []

    # Iterate through the search response and fetch video details
    for item in search_response['items']:
        video_id = item['id']['videoId']

        # Fetch video details using the video ID
        video_request = youtube.videos().list(
            part='snippet',
            id=video_id
        )
        video_response = video_request.execute()

        # Extract the video details
        video_title = video_response['items'][0]['snippet']['title']
        channel_title = video_response['items'][0]['snippet']['channelTitle']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        channel_url = f"https://www.youtube.com/channel/{video_response['items'][0]['snippet']['channelId']}"

        videos.append({
            'title': video_title,
            'channel': channel_title,
            'video_url': video_url,
            'channel_url': channel_url,
        })

    return videos


import mysql.connector
from fastapi.responses import StreamingResponse
import openai

# Database connection configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "abc123",
    "database": "prakhar"
}


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


@app.get('/general-bot/{token}/{message}')
async def general_bot(token: str, message: str):
    message = urllib.parse.unquote(message)
    if not token:
        raise HTTPException(status_code=401, detail="Token is missing!")
    try:
        # decoding the payload to fetch the stored details
        data = jwt.decode(token, "h1u2m3a4n5i6z7e8", algorithms=["HS256"])
        print("decr data", data)
        username = data.get("username")
        print("decr username", username)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=401, detail="Token is invalid!")

    inpt = message
    time1 = time.time()
    query2 = "SELECT * FROM messages WHERE (username=%s AND botid=%s) ORDER BY timestamp DESC LIMIT 5"

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query2, (username, "prakhar"))
        messages = cur.fetchall()
        time2 = time.time()
        print("Took time", time2 - time1, "seconds to fetch statements")
    finally:
        cur.close()
        conn.close()

    # arranging the chat data into a format of list of dictionaries
    chats = []
    for message in messages:
        words = message[4].split()
        first_100_words = ' '.join(words[:100])
        chats.append({"role": message[3], "content": first_100_words})
    chats.reverse()
    print("Previous chats", chats)

    # Your existing prompt
    given_prompt = """
You are a helpful assistant. Strictly Remember You are capable of sending emails but the Email feature is under Google Verification so user need to wait for few days. You are one of the 'Humanized AI Bot', that helps users with their general query, as well as queries related to HumanizeAI Platform. You only use the provided functions when found necessary. If you are required to send email, verify the details like mail id & the content before sending through the function provided AND REMEMBER you can send Emails, as a function is provided to you for sending email, for other functions, give a well formatted response.
Try not to stretch messages too long.
Morsesl is a platform where people can create AI Bots that can replicate them, or a hypothetical character to help communicate with masses, embed the bot in their website to work as assistant for their users, and similar for discord and telegram as well.
Creating a bot is very simple for users here,
1. Just choose a username
2. Fill the information required like how will bot act, what strict rules to follow, or user's company information if the user is a business.
3. And boom, the bot is ready to play by all the users & to get embedded in the user's website or discord or telegram.
4. This is just the beginning, many more features are up on the line. The user should stay tuned.
Some features are about to released by month end, like Lead Generation (Lead generation option collects user's name, phone, email & other details with their consent & stores it for you in your database).
"""

    async def stream_response():
        print("Prompt", given_prompt)
        messages = [
            {"role": "system", "content": given_prompt},
            *chats,
            {"role": "user", "content": inpt},
        ]
        generated_text = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=[
                {
                    "name": "get_weather",
                    "description": "A function to get weather of any city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city you want to get weather of"
                            }
                        },
                        "required": ["city"]
                    }
                },
                {
                    "name": "search_videos",
                    "description": "A function to search videos on youtube and get their links based on user's query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query you want to search videos for"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of video links you want to get"
                            }
                        }
                    },
                    "required": ["query"]
                }
            ],
            temperature=0.7,
            max_tokens=512,
            stream=True,
        )

        response = ""
        function_to_call = None
        function_arguments = ""
        function_name = None

        try:
            for chunk in generated_text:
                delta = chunk.choices[0].delta

                # Handle function calls
                if delta.function_call:
                    if delta.function_call.name:
                        functions = {
                            "get_weather": get_weather,
                            "search_videos": search_videos,
                            "send_mail": sendMail
                        }
                        function_name = delta.function_call.name
                        function_to_call = functions[function_name]
                    else:
                        function_arguments += delta.function_call.arguments or ""

                # Handle function execution
                elif chunk.choices[0].finish_reason and chunk.choices[0].finish_reason != "stop" and chunk.choices[
                    0].finish_reason != "timeout":
                    if chunk.choices[0].finish_reason == "function_call":
                        print("Function to call", function_to_call)
                        print("Function args", function_arguments)
                        jsonified_args = json.loads(function_arguments)
                        print("Jsonified args", jsonified_args)

                        if function_name == "send_mail":
                            funcresponse = function_to_call(username, **jsonified_args)
                        else:
                            funcresponse = function_to_call(**jsonified_args)

                        print("Response", funcresponse)
                        messages.extend([
                            {
                                "role": "user",
                                "content": function_arguments,
                            },
                            {
                                "role": "function",
                                "name": function_name,
                                "content": str(funcresponse),
                            }
                        ])

                        print("respo", messages)
                        generated_text2 = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.7,
                            max_tokens=512,
                            stream=True,
                        )

                        for chunk2 in generated_text2:
                            if chunk2.choices[0].delta.content:
                                yield f'data: {chunk2.choices[0].delta.content}\n\n'
                                response += chunk2.choices[0].delta.content
                            else:
                                # Save chat to database when stream ends
                                try:
                                    conn = get_db_connection()
                                    cur = conn.cursor()
                                    query = "INSERT INTO messages (username, botid, sender, message, timestamp) VALUES (%s, %s, %s, %s, %s)"
                                    cur.execute(query, (username, "humanize", "user", inpt, datetime.datetime.now()))
                                    cur.execute(query,
                                                (username, "humanize", "assistant", response, datetime.datetime.now()))
                                    conn.commit()
                                finally:
                                    cur.close()
                                    conn.close()

                # Handle regular content
                elif delta.content:
                    yield f'data: {delta.content}\n\n'
                    response += delta.content
                else:
                    # Save chat to database when stream ends
                    print("Stream ended successfully")
                    print("Resp total", response)
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        query = "INSERT INTO messages (username, botid, sender, message, timestamp) VALUES (%s, %s, %s, %s, %s)"
                        cur.execute(query, (username, "humanize", "user", inpt, datetime.datetime.now()))
                        cur.execute(query, (username, "humanize", "assistant", str(response), datetime.datetime.now()))
                        conn.commit()
                    finally:
                        cur.close()
                        conn.close()

        except Exception as e:
            print(f"Error during streaming: {e}")
            raise

    return StreamingResponse(stream_response(), media_type='text/event-stream')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
