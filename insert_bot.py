from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import json
import mysql.connector
import time
import jwt

app = FastAPI()

# Define the model for the request body
class BotRequest(BaseModel):
    botid: str
    primary: bool

# MySQL connection settings
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "abc123"
MYSQL_DB = "prakhar"

# JWT Secret key
JWT_SECRET_KEY = "h1u2m3a4n5i6z7e8"

from fastapi import Header, HTTPException
import jwt

# Function to extract the token from the Authorization header
def token_required(authorization: str = Header(...)):
    try:
        # Expecting "Bearer <token>"
        token = authorization.split(" ")[1]  # Extract token after "Bearer"
        data = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        username = data.get("username")
        if username is None:
            raise HTTPException(status_code=401, detail="Token is invalid.")
        return username
    except IndexError:
        raise HTTPException(status_code=401, detail="Token is missing or malformed.")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token is invalid.")

# Function to connect to the MySQL database
def get_db():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    return conn

# Create bot endpoint
@app.post("/create-bot")
async def create_bot(bot_request: BotRequest, token: str = Depends(token_required), db: Session = Depends(get_db)):
    print("USER:", token)  # Username from token
    print("BODY:", bot_request.dict())

    try:
        botid = bot_request.botid
        primary = bot_request.primary

        # Check if botid already exists
        cursor = db.cursor()
        cursor.execute("SELECT * FROM bots WHERE botid=%s", (botid,))
        bot = cursor.fetchone()
        if bot:
            raise HTTPException(status_code=400, detail="Bot ID already exists.")

        # Simulating class creation
        print("Class created", time.time())

        # Create classes in your external client
        class_obj = {
            "class": botid + "_ltm",
            "vectorizer": "text2vec-openai"
        }
        client.schema.create_class(class_obj)  # Assuming `client` is defined somewhere for schema operations
        print("Saved 3", time.time())
        class_obj = {
            "class": botid + "_images",
            "vectorizer": "text2vec-openai"
        }
        client.schema.create_class(class_obj)  # Assuming `client` is defined somewhere
        print("Saved 4", time.time())

        if primary:
            # Insert into bots table
            query1 = "INSERT INTO bots (botid, username, personal) VALUES (%s, %s, %s)"
            cursor.execute(query1, (botid, botid, 1))

            # Update the user's bot list if it's primary
            cursor.execute("SELECT bots FROM users WHERE email_id=%s", (token,))
            user_bots = json.loads(cursor.fetchone()[0])
            user_bots.append(botid)
            cursor.execute("UPDATE users SET bots=%s WHERE email_id=%s", (json.dumps(user_bots), token))

            db.commit()
            cursor.close()

            return JSONResponse(status_code=201, content={"success": True, "message": "Bot created successfully."})
        else:
            # Get username from database (username can be email)
            cursor.execute("SELECT username FROM users WHERE username=%s OR email_id=%s", (token, token))
            username = cursor.fetchone()[0]

            # Insert bot for non-primary user
            query = "INSERT INTO bots (botid, username) VALUES (%s, %s)"
            cursor.execute(query, (botid, username))
            db.commit()
            cursor.close()

            return JSONResponse(status_code=201, content={"success": True, "message": "Bot created successfully."})

    except Exception as e:
        print("MYSQL ERR:", e)
        raise HTTPException(status_code=500, detail="Error in writing bot data to Database")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
