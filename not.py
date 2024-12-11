from functools import wraps
import os
import uuid
import openai
import weaviate
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
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
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import logging
import requests
import re
import io
from io import StringIO
from googleapiclient.discovery import build
import json
from flask import send_from_directory, send_file
import time, datetime
import threading

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
from sqlalchemy import create_engine, Column, Integer, String, JSON, func
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
from typing import List, Optional
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
from fastapi import Request
from fastapi import Request, HTTPException, Depends
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import mysql.connector
from fastapi.responses import StreamingResponse
import openai
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
#////////////////////////////////////////////////////////////////////////////////

# Initialize FastAPI app
app = FastAPI()

# environment setup
open_api_key = ""  # add your openai api key here
os.environ[
    "OPENAI_API_KEY"] = "sk-qZA5vCHZR0_tgTBYmMiOxDMwSqM-tzv5oNIs313-nHT3BlbkFJw-dAjV_QA9Qa8eYkc4mWTHTnF3vLuIBqx-SaQz6dcA"
openai.api_key = "sk-qZA5vCHZR0_tgTBYmMiOxDMwSqM-tzv5oNIs313-nHT3BlbkFJw-dAjV_QA9Qa8eYkc4mWTHTnF3vLuIBqx-SaQz6dcA"
YTapi_key = "AIzaSyAHlhEhjECdTBDLs_JeGygh9J5T3tBwDd4"
Gapi_key = "AIzaSyAHlhEhjECdTBDLs_JeGygh9J5T3tBwDd4"
cx = "f6102f35bce1e44ed"
num_results = 4  # main to dekh bhi nhi rha tha 200 se isi lie start kia tha Xd  GUD GUD tu hi idhar le aaya xD

# Logging configuration
logging.basicConfig(format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

# Database setup (SQLAlchemy)
SQLALCHEMY_DATABASE_URL = "mysql://root:abc123@localhost/ayush"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
from sqlalchemy.orm import relationship
import json

Base = declarative_base()

'''///////////////////////////////////////////////////////////////////////////////////////'''

# User model with optional fields
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email_id = Column(String, unique=True, index=True)
    password = Column(String)
    bots = Column(JSON, default=[])  # Storing bots as JSON list (can be more structured than string)
    whatsapp = Column(String, nullable=True)
    telegram = Column(String, nullable=True)
    discord = Column(String, nullable=True)
    instagram = Column(String, nullable=True)
    twitter = Column(String, nullable=True)
    youtube = Column(String, nullable=True)
    pic = Column(String, nullable=True)  # Optional picture field
    created_at = Column(Integer, default=func.now())  # Optional: store creation timestamp

    # Establishing relationship with bots (optional, if you need this)
    bots_relation = relationship("Bot", back_populates="owner")

'''/////////////////////////////////////////////////////////////////////////////////////////'''

# Bot model with optional fields
class Bot(Base):
    __tablename__ = 'bot_profiles'

    id = Column(Integer, primary_key=True, index=True)
    botid = Column(String, unique=True, index=True)
    username = Column(String, index=True)
    personal = Column(Boolean, default=False)  # To track if the bot is personal (primary) or not
    name = Column(String, nullable=True)  # Optional field for bot's name
    description = Column(String, nullable=True)  # Optional field for bot description
    pic = Column(String, nullable=True)  # Optional picture field
    interactions = Column(Integer, default=0)  # Optional: Store number of interactions with the bot
    likes = Column(Integer, default=0)  # Optional: Store number of likes for the bot
    created_at = Column(Integer, default=func.now())  # Optional: Store creation timestamp

    # Relationship to the User model
    owner = relationship("User", back_populates="bots_relation")

'''/////////////////////////////////////////////////////////////////////////////////////////////'''
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
async def token_required(request: Request):
    token = request.headers.get("x-access-token")
    if not token:
        raise HTTPException(status_code=401, detail="Token is missing!")
    try:
        data = jwt.decode(token, "h1u2m3a4n5i6z7e8", algorithms=["HS256"])
        username = data.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token data")
        return username
    except Exception as e:
        print("Token Error:", e)
        raise HTTPException(status_code=401, detail="Token is invalid!")
'''////////////////////////////////////////////////////////////////////////////////'''

# Database connection configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "abc123",
    "database": "ayush"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# Utility to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

'''///////////////////////////////////////////////////////////////////////////////////'''


@app.post("/store-bot-data")
async def store_bot_data(
        username: str = Depends(token_required),  # Retrieve username from the token
        botid: str = Form(...),
        name: str = Form(...),
        description: str = Form(...),
        public: bool = Form(...),
        pic: Optional[UploadFile] = File(None),
        company_info: Optional[str] = Form(None),
        whatsapp: Optional[str] = Form(None),
        telegram: Optional[str] = Form(None),
        discord: Optional[str] = Form(None),
        youtube: Optional[str] = Form(None),
        instagram: Optional[str] = Form(None),
        twitter: Optional[str] = Form(None),
        linkedin: Optional[str] = Form(None),
        website: Optional[str] = Form(None),
):
    try:
        # Save the uploaded profile image if provided
        pic_filename = None
        if pic and allowed_file(pic.filename):
            pic_filename = secure_filename(pic.filename)
            save_path = os.path.join(os.getcwd(), "assets", pic_filename)
            with open(save_path, "wb") as f:
                f.write(pic.file.read())

        # Database connection
        conn = get_db_connection()
        cur = conn.cursor()

        # Prepare and execute the query for the bots table
        bot_update_query = """
            UPDATE bots 
            SET 
                name=%s, description=%s, pic=%s, 
                whatsapp=%s, telegram=%s, discord=%s, youtube=%s, 
                instagram=%s, twitter=%s, linkedin=%s, website=%s, 
                company_info=%s, public=%s 
            WHERE botid=%s
        """

        # Handle updates to the users table if the bot is primary
        if username == botid:
            cur.execute("UPDATE users SET setup=%s WHERE email_id=%s", (1, username))

            user_update_query = """
                UPDATE users 
                SET 
                     whatsapp=%s, telegram=%s, discord=%s, youtube=%s, 
                    instagram=%s, twitter=%s, linkedin=%s, website=%s 
                WHERE email_id=%s OR username=%s
            """
            cur.execute(
                user_update_query,
                (
                    whatsapp, telegram, discord, youtube, instagram,
                    twitter, linkedin, website, username, username
                ),
            )

        cur.execute(
            bot_update_query,
            (
                name, description, pic_filename,
                whatsapp, telegram, discord, youtube, instagram, twitter,
                linkedin, website, company_info, public, botid,
            ),
        )

        # Commit the transaction and close the cursor
        conn.commit()
        cur.close()

        return JSONResponse(content={"success": True, "message": "Bot data stored successfully."})

    except Exception as e:
        print("MYSQL ERR", e)
        raise HTTPException(status_code=500, detail="Error in writing bot data to Database")

'''/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": "localhost",   # Ensure this is correct
    "user": "root",        # Ensure this is correct
    "password": "abc123",  # Ensure this is correct
    "database": "ayush"    # Ensure this is correct
}

def get_db_connection():
    try:
        # Attempt to connect to the database
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            print("Successfully connected to the database")
            return conn
        else:
            print("Connection failed: Database not available")
            return None
    except Error as e:
        print(f"Error connecting to the database: {e}")
        return None


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import os
from werkzeug.utils import secure_filename

# Add CORS Middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# Database connection (example, replace with your implementation)

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)
# Token dependency (placeholder for authentication)
async def token_required(request: Request):
    token = request.headers.get("x-access-token")
    if not token:
        raise HTTPException(status_code=401, detail="Token is missing!")
    try:
        data = jwt.decode(token, "h1u2m3a4n5i6z7e8", algorithms=["HS256"])
        username = data.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token data")
        return username
    except Exception as e:
        print("Token Error:", e)
        raise HTTPException(status_code=401, detail="Token is invalid!")
@app.put("/update-bot-data")
async def update_bot_data(
        username: str = Depends(token_required),  # Retrieve username from the token
        botid: str = Form(...),
        name: str = Form(...),
        description: str = Form(...),
        botrole: str = Form(...),
        steps: str = Form(...),
        purpose: str = Form(...),
        public: bool = Form(...),
        pic_filename: Optional[UploadFile] = File(None),
        company_info: Optional[str] = Form(None),
        whatsapp: Optional[str] = Form(None),
        telegram: Optional[str] = Form(None),
        discord: Optional[str] = Form(None),
        youtube: Optional[str] = Form(None),
        instagram: Optional[str] = Form(None),
        twitter: Optional[str] = Form(None),
        linkedin: Optional[str] = Form(None),
        website: Optional[str] = Form(None),
):
    try:
        # Save the uploaded profile image if provided
        pic_filename = None
        if pic_filename and allowed_file(pic_filename.filename):
            pic_filename = secure_filename(pic_filename.filename)
            save_path = os.path.join(os.getcwd(), "assets", pic_filename)
            with open(save_path, "wb") as f:
                f.write(pic_filename.file.read())

        # Database connection
        conn = get_db_connection()
        cur = conn.cursor()

        # Update the bot_profiles table
        bot_update_query = """
            UPDATE bot_profiles 
            SET 
                name=%s, description=%s, pic_filename=%s, botrole=%s, 
                steps=%s, purpose=%s, whatsapp=%s, telegram=%s, 
                discord=%s, youtube=%s, instagram=%s, twitter=%s, 
                linkedin=%s, website=%s, company_info=%s, public=%s 
            WHERE botid=%s
        """
        cur.execute(
            bot_update_query,
            (
                name, description, pic_filename, botrole, steps, purpose,
                whatsapp, telegram, discord, youtube, instagram, twitter,
                linkedin, website, company_info, public, botid,
            ),
        )

        # Check if the bot is primary, and update the users table accordingly
        primary_query = "SELECT username FROM bot_profiles WHERE botid=%s"
        cur.execute(primary_query, (botid,))
        data = cur.fetchone()
        if data and data[0]:
            user_update_query = """
                UPDATE users 
                SET 
                    purpose=%s, whatsapp=%s, telegram=%s, discord=%s, 
                    youtube=%s, instagram=%s, twitter=%s, linkedin=%s, 
                    website=%s 
                WHERE email_id=%s OR username=%s
            """
            cur.execute(
                user_update_query,
                (
                    purpose, whatsapp, telegram, discord, youtube, instagram,
                    twitter, linkedin, website, data[1], data[1]
                ),
            )

        # Commit the transaction and close the cursor
        conn.commit()
        cur.close()

        return JSONResponse(content={"success": True, "message": "Bot profile data updated successfully."})

    except Exception as e:
        print("MYSQL ERR", e)
        raise HTTPException(status_code=500, detail="Error updating bot profile data in the database")

'''///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''









if __name__ == "__main__":
    import uvicorn
    connection = get_db_connection()
    if connection:
        print("Connection successful!")
        connection.close()
    else:
        print("Connection failed.")
    uvicorn.run(app, host="0.0.0.0", port=3000)