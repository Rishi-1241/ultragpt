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

# Initialize FastAPI app
app = FastAPI()

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

def token_required(request: Request):
    token = request.headers.get('x-access-token')
    if not token:
        raise HTTPException(status_code=401, detail="Token is missing !!")
    
    try:
        # Decode the token to get the username
        data = jwt.decode(token, "h1u2m3a4n5i6z7e8", algorithms=["HS256"])
        username = data.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Token is invalid !!")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token is invalid !!")
    
    return username  # Returns the username to be injected into the route handler


import os
from pathlib import Path
from tempfile import TemporaryDirectory
from pdf2image import convert_from_path
import easyocr
from PIL import Image

def OCRFINAL(pdf_name, output_file, out_directory=Path("~").expanduser(), dpi=200):
    # Initialize EasyOCR reader (English language assumed)
    reader = easyocr.Reader(['en'])
    
    PDF_file = Path(pdf_name)
    image_file_list = []
    text_file = out_directory / Path(output_file)

    with TemporaryDirectory() as tempdir:
        pdf_pages = convert_from_path(PDF_file, dpi=dpi, poppler_path="C:\\Users\\Prakhar Agrawal\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin")
        print("pdf_pages", pdf_pages)
        for page_enumeration, page in enumerate(pdf_pages, start=1):
            filename = f"{tempdir}/page_{page_enumeration:03}.jpg"  # Corrected path separator for cross-platform compatibility
            page.save(filename, "JPEG")
            image_file_list.append(filename)

        # Extract text using EasyOCR
        with open(text_file, "a") as output_file:
            for image_file in image_file_list:
                # Read text from the image using EasyOCR
                text = " ".join(reader.readtext(image_file, detail=0))
                text = text.replace("-\n", "")
                output_file.write(text)

        # Read the whole extracted text
        with open(text_file, "r") as f:
            textFinal = f.read()

        # Split text into paragraphs of 150 words each
        paragraphs = []
        words = textFinal.split()
        for i in range(0, len(words), 150):
            paragraphs.append(' '.join(words[i:i + 150]))

        # Delete the text file after processing
        if os.path.exists(text_file):
            os.remove(text_file)

    return paragraphs
# Register user endpoint
@app.post("/register", response_model=UserResponse)
def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    name, email_id, password = request.name, request.email_id, request.password

    if not name or not email_id or not password:
        raise HTTPException(status_code=400, detail="Please fill all the fields.")

    try:
        # Check if user already exists
        user = db.query(User).filter(User.email_id == email_id).first()
        if user:
            raise HTTPException(status_code=400, detail="User already exists.")

        # Add user to the database (hashing password should be done here)
        hashed_password = password  # Placeholder, ideally use something like bcrypt to hash the password
        new_user = User(name=name, email_id=email_id, password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Create JWT token
        token = jwt.encode({'username': email_id}, "h1u2m3a4n5i6z7e8", algorithm="HS256")

        return JSONResponse(content={
            "success": True,
            "message": "Account created successfully",
            "token": token,
            "data": {"name": name, "email_id": email_id},
            "bots": []
        }, status_code=201)

    except Exception as e:
        logging.error(f"MYSQL ERR: {e}")
        raise HTTPException(status_code=500, detail="Error in writing user data to Database")

# Bot creation request model
class BotRequest(BaseModel):
    botid: str
    primary: bool = False
from sqlalchemy import text

@app.post("/create-bot")
async def create_bot(bot_request: BotRequest, username: str = Depends(token_required), db: Session = Depends(get_db)):
    print("USER:", username)
    print("BODY:", bot_request.dict())

    try:
        botid = bot_request.botid
        primary = bot_request.primary

        # Check if botid already exists
        bot = db.query(Bot).filter(Bot.botid == botid).first()
        if bot:
            raise HTTPException(status_code=400, detail="Bot ID already exists.")

        # Simulating bot creation with print statements
        print("Class created", time.time())
        
        # Add the new bot
        if primary:
            db.execute(
                text("INSERT INTO bots (botid, username, personal) VALUES (:botid, :username, 1)"),
                {"botid": botid, "username": botid}
            )
            # Update the user's bot list
            user = db.query(User).filter(User.email_id == username).first()
            bots = json.loads(user.bots)
            bots.append(botid)
            db.execute(
                text("UPDATE users SET bots=:bots WHERE email_id=:email_id"),
                {"bots": json.dumps(bots), "email_id": username}
            )
        else:
            user = db.query(User).filter(User.email_id == username).first()
            db.execute(
                text("INSERT INTO bots (botid, username) VALUES (:botid, :username)"),
                {"botid": botid, "username": user.email_id}
            )

        db.commit()
        return JSONResponse(status_code=201, content={"success": True, "message": "Bot created successfully."})

    except Exception as e:
        print("MYSQL ERR:", e)
        raise HTTPException(status_code=500, detail="Error in writing bot data to Database")
    
UPLOAD_FOLDER = "./assets"  # Folder to save the uploaded files
ALLOWED_EXTENSIONS = {'pdf'}  # Allowed file types

# Helper function to check allowed file extensions
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    try:
        # Check if allowed file
        if 'pdf' not in pdf.filename:
            raise HTTPException(status_code=400, detail="No file part")
        
        if pdf.filename == '':
            raise HTTPException(status_code=400, detail="No selected file")
        
        if allowed_file(pdf.filename):
            # Save file
            filename = secure_filename(pdf.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await pdf.read())
            
            txt = filename.replace(".pdf", ".txt")
            inpt = OCRFINAL(file_path, txt)  # Assuming OCRFINAL is implemented elsewhere
            
            string = "".join(inpt)
            
            # Clean up files after processing
            try:
                os.remove(file_path)
                os.remove(os.path.join(UPLOAD_FOLDER, txt))
            except Exception as e:
                print(f"Error deleting files: {e}")
            
            return JSONResponse(
                content={"success": True, "message": "File uploaded successfully.", "data": string},
                status_code=200
            )
        else:
            raise HTTPException(status_code=400, detail="File type not allowed")
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error in uploading file")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
