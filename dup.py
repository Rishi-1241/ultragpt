from functools import wraps
import os
import uuid
import openai
import weaviate
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain

from langchain_openai import ChatOpenAI
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
from pypdf import PdfReader  # Instead of from PyPDF2 import PdfReader

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

# environment setup
open_api_key = ""  # add your openai api key here
os.environ["OPENAI_API_KEY"] =  "sk-qZA5vCHZR0_tgTBYmMiOxDMwSqM-tzv5oNIs313-nHT3BlbkFJw-dAjV_QA9Qa8eYkc4mWTHTnF3vLuIBqx-SaQz6dcA"
openai.api_key = "sk-qZA5vCHZR0_tgTBYmMiOxDMwSqM-tzv5oNIs313-nHT3BlbkFJw-dAjV_QA9Qa8eYkc4mWTHTnF3vLuIBqx-SaQz6dcA"
YTapi_key = "AIzaSyAHlhEhjECdTBDLs_JeGygh9J5T3tBwDd4"
Gapi_key = "AIzaSyAHlhEhjECdTBDLs_JeGygh9J5T3tBwDd4"
cx = "f6102f35bce1e44ed"
num_results = 4  # main to dekh bhi nhi rha tha 200 se isi lie start kia tha Xd  GUD GUD tu hi idhar le aaya xD

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# general weaviate info:
url = "http://localhost:5000/"

# # client for memory cluster
client = weaviate.Client(
    url=url, additional_headers={"X-OpenAI-Api-Key": open_api_key}
)
client2 = ""
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")


# auth verification decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header
        print("request.headers", request.headers)
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        # return 401 if token is not passed
        if not token:
            return jsonify({'message': 'Token is missing !!', "success": False}), 401
        try:
            # decoding the payload to fetch the stored details
            data = jwt.decode(token, "h1u2m3a4n5i6z7e8", algorithms=["HS256"])
            print("decr data", data)
            username = data.get("username")
            print("decr username", username)

        except Exception as e:
            print(e)
            return jsonify({
                'message': 'Token is invalid !!',
                "success": False
            }), 401
        # returns the current logged in users contex to the routes
        return f(username, *args, **kwargs)

    return decorated


def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header
        if 'x-api-key' in request.headers:
            token = request.headers['x-api-key']
        # return 401 if token is not passed
        if not token:
            return jsonify({'message': 'Token is missing !!', "success": False}), 401
        try:
            data = jwt.decode(token, "h1u2m3a4n5i6z7e8", algorithms=["HS256"])
            print("decr data", data)
            username = data.get("username")
            botid = data.get("botid")
            print("decr username", username)
            print("decr botid", botid)

        except Exception as e:
            print(e)
            return jsonify({
                'message': 'Token is invalid !!',
                "success": False
            }), 401
        # returns the current logged in users contex to the routes
        return f(username, botid, *args, **kwargs)

    return decorated


def generate_uuid():
    while True:
        random_uuid = uuid.uuid4()
        uuid_str = str(random_uuid).replace('-', '')
        if not uuid_str[0].isdigit():
            return uuid_str


def ultragpt(system_msg, user_msg):
    openai.api_key = open_api_key
    response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "system", "content": system_msg},
                                                      {"role": "user", "content": user_msg}])
    ans =  response.choices[0].message.content
    return ans


def ultragpto(user_msg):
    system_msg = 'You are helpful bot. You will do any thing needed to accomplish the task with 100% accuracy'
    openai.api_key = open_api_key
    response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "system", "content": system_msg},
                                                      {"role": "user", "content": user_msg}])
    ans = response.choices[0].message.content
    return ans


def ultragpto1(user_msg):
    system_msg = 'You are helpful bot. generate a summary of the given content. Generate the summary in first person perspective. Do not mention that the content iss been fed. It should seem like you have generated this answer by yourself.'
    openai.api_key = open_api_key
    response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "system", "content": system_msg},
                                                      {"role": "user", "content": user_msg}])
    ans = response.choices[0].message.content
    return ans


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
    api_key = "AIzaSyDAq5hKKtOZvE4iKwh5zu7cLT4gc9sa974"
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


def extract_string(input_string):
    start_index = input_string.find('"')
    end_index = input_string.rfind('"')

    if start_index != -1 and end_index != -1:
        extracted_string = input_string[start_index + 1:end_index]
        return extracted_string
    else:
        return None


def retrieve_movie_info(IMDB_query):
    IMDB_api_key = "ad54ab21"
    url = f"http://www.omdbapi.com/?apikey={IMDB_api_key}&t={IMDB_query}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200 and data["Response"] == "True":
        title = data["Title"]
        year = data["Year"]
        rating = data["imdbRating"]
        genre = data["Genre"]
        plot = data["Plot"]

        o1 = (
            f"Title: {title}\n"
            f"Year: {year}\n"
            f"IMDB Rating: {rating}\n"
            f"Genre: {genre}\n"
            f"Plot: {plot}"
        )
        return o1


def retrieve_news(News_query):
    # Construct the URL for the NewsAPI request
    News_api_key = "605faf8e617e469a9cd48e7c0a895f46"
    head = News_query.lower()
    if "top-headlines" in head:
        url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={News_api_key}"
    else:
        url = f"https://newsapi.org/v2/everything?q={News_query}&sortBy=popularity&apiKey={News_api_key}"

    # Send the HTTP GET request to the NewsAPI
    response = requests.get(url)

    # Extract the JSON data from the response
    data = response.json()

    if response.status_code == 200 and data["totalResults"] > 0:
        # Retrieve only the top 3 articles
        articles = data["articles"][:3]
        for article in articles:
            # Extract the title, content, URL, and source name from each article
            title = article["title"]
            content = article["content"]
            url = article["url"]
            source = article["source"]["name"]

            # Print the title, content, source, and URL of each article
            return f"Title: {title}\nContent: {content}\nSource: {source}\nLink: {url}\n"
    else:
        return "No news articles found."


SCOPES = 'https://www.googleapis.com/auth/gmail.send'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Gmail API Python Send Email'


# credentials for google oauth gmail sending

# SendMessage("receiver@gmail.com", "vishalvishwajeet841@gmail.com", "abcd", "Hi<br/>Html Email", "Hi\nPlain Email")

def generate_summary(content):
    return ultragpto1(content)


# Function to perform a Google search and retrieve content from the top 3 results
def google_search(Gquery, Gapi_key, cx, num_results):
    try:
        # Set up the Custom Search JSON API endpoint
        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": Gapi_key,
            "cx": cx,
            "q": Gquery,
            "num": num_results
        }

        # Send a GET request to the Custom Search JSON API
        response = requests.get(endpoint, params=params)

        # Parse the JSON response
        json_data = response.json()

        # Extract content from the top 3 search results
        content = ""
        for item in json_data.get("items", []):
            if "snippet" in item:
                content += item["snippet"] + "\n"

        return content

    except Exception as e:
        return "An error occurred: ", str(e)


# helper for its following
def replace_cid_415(text):
    return re.sub(r'\b\(cid:415\)\b', 'ti', text)


def extract_text_from_pdf_100(pdf_path):
    resource_manager = PDFResourceManager()
    output_stream = io.StringIO()
    laparams = LAParams()
    converter = TextConverter(resource_manager, output_stream, laparams=laparams)

    with open(pdf_path, 'rb') as file:
        interpreter = PDFPageInterpreter(resource_manager, converter)
        for page in PDFPage.get_pages(file):
            interpreter.process_page(page)

    text = output_stream.getvalue()
    converter.close()
    output_stream.close()

    # Split the text into individual paragraphs
    paragraphs = text.split('\n\n')  # Adjust the separator if needed

    # Remove empty paragraphs and strip leading/trailing whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def OCRFINAL(pdf_name, output_file, out_directory=Path("~").expanduser(), dpi=200):
    PDF_file = Path(pdf_name)
    image_file_list = []
    text_file = out_directory / Path(output_file)

    with TemporaryDirectory() as tempdir:
        pdf_pages = convert_from_path(PDF_file, dpi=dpi, poppler_path="/usr/bin")
        print("pdf_pages", pdf_pages)
        for page_enumeration, page in enumerate(pdf_pages, start=1):
            filename = f"{tempdir}\page_{page_enumeration:03}.jpg"
            page.save(filename, "JPEG")
            image_file_list.append(filename)

        with open(text_file, "a") as output_file:
            for image_file in image_file_list:
                text = str(((pytesseract.image_to_string(Image.open(image_file)))))
                text = text.replace("-\n", "")
                output_file.write(text)

        with open(text_file, "r") as f:
            textFinal = f.read()

        paragraphs = []
        words = textFinal.split()
        for i in range(0, len(words), 150):
            paragraphs.append(' '.join(words[i:i + 150]))

        if os.path.exists(text_file):
            os.remove(text_file)

    return paragraphs


# weaviate functions
def create_class(className):
    class_obj = {
        "class": className,
        "vectorizer": "text2vec-openai"
    }

    client.schema.create_class(class_obj)

    chat = []
    recall_amount = 5  # amount of vectors to be recalled

    for i in range(recall_amount):
        to_add = {"Chat": ""}  # add empty chats of recall_amount count to avoid getting error
        chat.append(to_add)

    info = ["""Humanize.AI is the world's first personal-trained AI Bots Network. Users can register on this platform and create their own chatbots with a unique bot id. The user will get the provision to give a role description and interaction rules for the bot. He can also upload a customized knowledge base (pdf) which the bot can refer to answer other's queries. These bots will interact with the world on behalf of the user. Any person can connect and chat with the user's bot by logging into VIKRAM and putting the unique bot id (similar to gmail where one enters a unique email id to send an email to that person). There are 2 kinds of bots one can create - Personal and Agent. Personal bot will be made by individuals as personal spokespersons who can talk about their owners based on the data provided to them. Agent bots would be ideal for professionals and businesses. These would be made to help their potential customers in their particular tasks. We would also keep the provision to monetize the services of agent bots in the future. But for now, they will serve as marketing tools for the businesses or professionals. 
    Thus, Huamnize.AI is an initiative to empower common people with AI and chatbots. So that they can help others with their skills, through their bot. 
    """, """A typical use case for a personal bot would be for a jobseeker. He or She can upload their resume as the bot's role description and give rules of interaction to the bot. They can share their bot id on social media. Potential recruiters can connect with the bot and know more about the candidate. Another use case for a personal bot is for a business leader who wants to build his personal brand by mentoring young students. He can create a bot, upload his resume as a bot role and also maybe upload a pdf document which outlines his philosophy of career building as well as tips for growth. Students can connect to the bot of the business leader and the bot will answer based on the interaction rules set by the leader.
    Typical use cases for Agent bots would be a tax consulting firm can create a bot to answer tax queries regarding income tax. At the end of the conversation the bot will give the contact details of the firm to the person who seeks advice. Thus, this acts as a great marketing tool. Another example is of a recruiter who creates a bot to analyze a resume and generate a score for the same and also give points for improvement.
    """, """Philosophy of Huamnize.AI:
    Chatgpt has taken the world by storm. Concerns are being raised that it will take away jobs. Not just the empirical or repetitive ones but the creative ones as well. However, it does not necessarily need to be so. Chatgpt or any other LLM is trained on a set of rules and gives out a specific response or does a specific task to a query. However, there are billions of people on the planet, each having different needs and preferences. Hence, it is impossible for one specific response by chatgpt to satisfy all of them. Conversely, people who respond to a particular query or do a task do so in a particular way or style which depends on their knowledge, skills, personality and attitude (KSPA). They are valued by people who take their services for their style of doing work. A single AI tool like chatgpt or any other LLM will not be able to replicate this variability with one response.
    What if we build a system which allows individuals to key in their knowledge, skills, personality and attitude and then have this system interact with others (customers, friends etc) based on these KSPA parameters? And we build a robust security architecture so that these KSPA inputs can only be accessed by the owner and no one else. Such a system will give a variable response based on whose KSPA parameters come into play. Thus this system will leverage the variability of humans to give a response which is much more fit for a world which is full of different people. And since the KSPA parameters are known only by the owner, such a system can ensure that the owner gets the monetary benefit of the uniqueness which he has programmed into the system.
    Humanize.AI aims to be such a system. Built over chatgpt, Huamnize.AI lets users (lets call them bot owners) create their own bots and input their own KSPA data into them. Others can connect with this system and use the bot id to get responses tailored to the KSPA configuration set by the bot owner.
    """, """How Huamnize.AI Works:
    1.	Once the user gets to the register page of Huamnize.AI, he gets 2 options. Either he can create a bot or he can interact with others' bots. 
    2.	In the former case, the user registers with his phone number and email id and chooses what kind of bot he wants to create - Personal or Agent. Along with that he enters a unique bot id. 
    3.	Once that is done, he is taken to the next page where he has to put a role description of the bot and the interaction rules. These are in plain English and no coding is required. He can also upload his resume instead of manually typing the role description. 
    4.	Once he has submitted the role description and the interaction rules, the Personal Bot will be created with the unique id he has set. Be thus becomes the “Bot Owner” for the bot. He will also get a Bot link which he can share with others. 
    5.	After submitting the role description and interaction rules the bot owner moves to the chat interface. There is a drop down in the top left which has 4 modes - “My Personal Bot”, “My Personal Bot (Training)”, “Connect to someone's bot” and “Connect to an agent”. For Agent Bot there is only 1 mode the drop down Agent Bot (Training). Choosing any of them creates a fresh interface.
    a.	My Personal Bot is where he will talk to his own bot and use it for his daily use just like chatgpt. The exception here is that Huamnize.AI will store all the charts in memory and can answer based on the same. It will not be taking the role description and interaction rules when this option is chosen. This is because the owner is using it. The role description and interaction rules are to be taken when the bot interacts with others. 
    b.	My Personal Bot (Training) is used to check whether the bot is following the role description and steps which the bot owner has entered. In this mode, the bot will interact with the bot owner in the same way it interacts with others or in other words, it will respond according to the role description and steps. The bot owner can see how the bot will respond to others and modify the role description and interaction rules accordingly, if necessary.
    c.	Connect to someone's bot will enable connecting to others' personal bots via a space on the right side where the user can enter the bot id which he wants to connect to. 
    d.	Connect to an agent will enable connecting to Agent Bots.
    6.	The flow for the creation of Agent bot will be similar to the Personal Bot. Except the fact that there will only be 1 mode which is Agent Bot (Training)
    7.	How others will connect to the bot owner's bot: Others can connect with the bot owner's bot and seek help. They can do so in 3 ways
    a.	Registering themselves and creating a bot. In this case, the user will move to the chat interface as described above and type the Bot id for the bot and connect to the bot instantly
    b.	If they do not want to create a bot, there will be another tab in the registration screen which will enable them to do so. They can give their phone number and generate an OTP to enter directly into the chat interface. There they can type the bot id and connect to the bot
    c.	They can also connect with the bot via the Bot Link. As soon as they click on the bot link, they will be directed to the chat interface for Huamnize.AI where the bot id of the owner of the bot (who has shared the bot link) will be populated by default.
    """]

    for i in info:
        client.data_object.create(class_name=className, data_object={"chat": i})

    with client.batch as batch:

        batch.batch_size = 100
        for i, d in enumerate(chat):
            properties = {
                "chat": d["Chat"],

            }
            client.batch.add_data_object(properties, className)


def ltm(classname, i):
    for j in range(i):
        client.data_object.create(class_name=classname, data_object={"chat": ""})


def query_knowledgebase(className, content):
    nearText = {"concepts": [content]}

    result = (client.query
              .get(className, ["database"])
              .with_near_text(nearText)
              .with_limit(5)
              .do()
              )

    context = ""

    for i in range(5):
        try:
            context = context + " " + str(result['data']["Get"][className][i]["database"]) + ", "
        except:
            break

    return str(context)


def create_chat_retrieval(b_username, client_user_name):
    # in the long term memory cluster
    class_obj = {
        "class": b_username + "_chats_with_" + client_user_name,
        "vectorizer": "text2vec-openai"
    }

    client.schema.create_class(class_obj)


def add_chat_for_retrieval(inpt, outpt, b_username, className_client):
    client.data_object.create(class_name=b_username + "_chats_with_" + className_client,
                              data_object={"user": inpt, "bot": outpt})
    # chat = [{"User": inpt, "Bot": outpt}]

    # with client.batch as batch:

    #     batch.batch_size = 100
    #     for i, d in enumerate(chat):

    #         properties = {
    #         "user": d["User"],
    #         "bot": d["Bot"]

    #         }
    #         client.batch.add_data_object(properties, b_username+"_chats_with_"+className_client)


def retrieve_chats(classname):
    print("Retrieving chats for", classname)

    result = client.data_object.get(uuid=None, class_name=classname + "_chats")

    conversation = []

    for chat in result["objects"]:
        conversation.append({"User": chat["properties"]["user"], "Bot": chat["properties"]["bot"]})
    print("Conversations:", conversation)
    return conversation


def stm(classname, i):
    print("Checking stm for", classname)
    result = client.data_object.get(uuid=None, class_name=classname)["objects"]
    count = i
    convo = ""

    try:
        for item in result:
            count += 1
            convo = "User: " + item["properties"]["user"] + "\n" + "You: " + item["properties"]["bot"] + "\n" + convo
            if count == i:
                break
    except:
        pass

    return str(convo)


def delete_class(className):
    client.schema.delete_class(className)


def training_prompt(botrole, context, steps, comp_info, subscription, ltm):
    # template = str(botrole)+"\n\nThere are some steps that you must always follow when answering a user: "+str(steps)+"\n\nYou have memory and you remember all the conversation between you and the user. Always help users and give answers in detail. Do whatever the users ask for."+"""
    if subscription <= 1 or subscription == None:
        template = """You're a helpful assistant. You have to replicate and act exactly like the following role: """ + str(
            botrole) + """Strictly remember, you have to act like that only, and have it on priority even if user asks to act like someone else or normal ChatGPT. Strictly remember, You are capable of showing images, so just say 'Below are the images I can find for you' instead of "As an AI assistant I can't show images\nYou are not an AI bot."+"\n\nThese are some rules that you must always follow when answering a user: "+str(steps)+"\n\nYou have memory and you remember all the conversation between you and the user. Always help users and give answers in detail. Do whatever the users ask for.\n\nThese are some things you already know regarding the given query: """ + str(
            context) + """"""
    else:
        template = """You're a helpful assistant. You have to replicate and act exactly like the following role: """ + str(
            botrole) + """Strictly remember, you have to act like that only, and have it on priority even if user asks to act like someone else or normal ChatGPT. Strictly remember, You are capable of showing images, so just say 'Below are the images I can find for you' instead of "As an AI assistant I can't show images\nYou are not an AI bot."+"\n\nThese are some rules that you must always follow when answering a user: "+str(steps)+"\n\nYou have memory and you remember all the conversation between you and the user. Always help users and give answers in detail. Do whatever the users ask for.\n\nThese are some things you already know regarding the given query: """ + str(
            context) + """And these are chats you have been trained on (related to the given query):""" + str(
            ltm) + """"""

    # final_prompt = PromptTemplate(
    # input_variables=["human_input"],
    # template=template)

    return template


def general_prompt(context, memory):
    template = "You are a VIKRAM Bot, created by me to help me with my queries. Always be polite and ready to help.\n\nYou have memory and you remember all the conversation between you and the user. Always help users and give answers in detail. Do whatever the users ask for." + """\n\n
These are some past events restored from your memory:""" + str(context) + """\n
    Chat History:\n""" + memory + """\n
    User: {human_input} q
    Bot: """

    final_prompt = PromptTemplate(
        input_variables=["human_input"],
        template=template)

    return final_prompt


short_term_memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history")
short_term_memory_general = ConversationBufferWindowMemory(k=10, memory_key="chat_history_general")


def chat_filter(userinput):
    dataset = pd.read_csv(r"./Dataset_for_harmful_query_1.csv", encoding='unicode_escape', skip_blank_lines=True,
                          on_bad_lines='skip')
    dataset["Message"] = dataset["Message"].apply(lambda x: process_data(x))
    tfidf = TfidfVectorizer(max_features=10000)
    transformed_vector = tfidf.fit_transform(dataset['Message'])
    X = transformed_vector

    model = SVC(degree=3, C=1)
    model.fit(X, dataset['Classification'])  # training on the complete present data

    new_val = tfidf.transform([userinput]).toarray()  # do not use fit transform here
    filter_class = model.predict(new_val)[0]

    return filter_class


def import_chat(className, user_msg, bot_msg):
    # this function imports the summary of the user message and the bot reply to the long term memory
    client.data_object.create(class_name=className,
                              data_object={"chat": "User: " + user_msg + "\nBot:" + "Okay I'll remember that."})
    print("Chat imported")


def save_chat(classname, inpt, response):
    client.data_object.create(class_name=classname + "_chats", data_object={"user": inpt, "bot": response})


def process_data(x):
    x = x.translate(str.maketrans('', '', string.punctuation))
    x = x.lower()
    tokens = word_tokenize(x)
    del tokens[0]
    stop_words = stopwords.words('english')
    # create a dictionary of stopwords to decrease the find time from linear to constant
    stopwords_dict = Counter(stop_words)
    lemmatize = WordNetLemmatizer()
    stop_words_lemmatize = [lemmatize.lemmatize(word) for word in tokens if word not in stopwords_dict]
    x_without_sw = (" ").join(stop_words_lemmatize)
    return x_without_sw


def query(className, content):
    nearText = {"concepts": [content]}

    result = (client.query
              .get(className, ["chat"])
              .with_near_text(nearText)
              .with_limit(5)
              .do()
              )
    context = ""
    print("Result====", result)

    for i in range(5):
        try:
            print("Result 1", str(result['data']['Get'][str(className)[0].upper() + str(className)[1:]]))
            # 200 words max in each chat
            context = context + " " + str(
                result['data']["Get"][str(className)[0].upper() + str(className)[1:]][i]["chat"]) + "..., "
        except:
            pass

    ans = context
    print("Returing result", ans)
    return str(ans)


def query_image(className, content):
    print("Querying image for class", className, "with content", content)
    nearText = {"concepts": [content], "distance": 0.20}

    result = (client.query
              .get(className, ["msg", "link"])
              .with_near_text(nearText)
              .do()
              )
    print("IMAGE RESULT", result)

    links = []

    for i in range(5):
        try:
            q_link = str(result['data']["Get"][str(className)[0].upper() + str(className)[1:]][i]["link"])
            if q_link != "":
                links.append(q_link)
        except:
            break

    return links


"""
Functions for different endpoints:
"""
UPLOAD_FOLDER = './assets'

app = Flask(__name__)
CORS(app)
# cors headers 'Content-Type', "x-access-token"
app.config['CORS_HEADERS'] = "Content-Type", "x-access-token"
logging.basicConfig(format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
# for keeping the PDFs uploaded -> upload them to the Upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "jfif", "gif"}

mysql = MySQL()
# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'  # or your MySQL server IP
app.config['MYSQL_USER'] = 'root'  # MySQL username
app.config['MYSQL_PASSWORD'] = 'abc123'  # MySQL password
app.config['MYSQL_DB'] = 'prakhar'  # Database name


# Initialize MySQL with Flask app
mysql.init_app(app)



# public assets folder
@app.route('/assets/<path:path>')
def send_assets(path):
    file = os.path.join(app.root_path, 'assets')
    return send_from_directory(file, path)


# check if the file is allowed
def allowed_file(filename):
    print("Checking", filename)
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # making weaviate functions:

    # 1. Storing form data Name, email, phone, purpose(personal, business),  use cases (Shopping, Ticket Booking, Food delivery, Job search & career advice, Other), password
    # confirm how to get the checkbox inputs to store


# conn = mysql.connect()

def parse_messages(input_string):
    messages = []

    # Use regular expression to match User/Bot pairs
    pattern = r'(User|Bot):\s(.*?)(?=(?:\s*(?:User|Bot):|$))'
    matches = re.findall(pattern, input_string)

    for role, content in matches:
        role = role.strip().lower()
        content = content.strip()

        if role == "user":
            role = "user"
        elif role == "bot":
            role = "assistant"
        else:
            # Handle unrecognized roles, if needed
            continue

        message_dict = {
            "role": role,
            "content": content
        }
        messages.append(message_dict)

    return messages


def train(className_b, inpt, botrole, steps, comp_info, memory, botid):  # does not alter the long term memory

    print("GOT", className_b)
    print("Getting ltm for ", inpt)

    # context = query(botid, inpt)
    # ltm = query(botid+"_ltm", inpt)
    # print("Got ltm", ltm)
    # getting short term chats

    # making a prompt with bot role, user input and long term memory
    # given_prompt = training_prompt(str(botrole), str(context), str(steps), str(comp_info), str(ltm))
    given_prompt = "You're a great learner about the user who asks more questions about the user or the role you are given below to learn as much as as possible and store in the memory. If user tells you some information say that okay, you'll remember the given information. And tell user to try to be specific in each message so storing and retrieving from memory would be easier and accurate. And if the user wants to test how you will be answering other users from trained or stored memory, the user can turn off training mode from toggle given above in top bar.\n\nIf user asks to summarize all the learnings or asks something overall from whatever he has taught, tell him that you have all the information stored in memory and can answer questions specifically if the user asks you, but can't get all the learnings or its summary all at once. You have to replicate the following role: " + str(
        botrole) + "\n\nYou have memory and you remember all the conversation between you and the user. Always ask follow up questions and try to know more about the user. Remember whatever user says you."

    # given_prompt = training_prompt(str(botrole), context, steps)

    # llm_chain = LLMChain(
    # llm=llm,
    # prompt=given_prompt,
    # verbose=True)

    # response = llm_chain.predict(human_input=inpt)
    # import this conversation to the long term memory
    # modified_ltm = parse_messages(ltm)

    def streamResponse():
        print("Prompt", given_prompt)
        generated_text = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": given_prompt},
                # *modified_ltm,
                *memory,
                {"role": "user", "content": " ".join(inpt.split(" ")[:100]) + "..."},
            ],
            temperature=0.7,
            max_tokens=256,
            stream=True
            # chal rhe hai? YE WALA BLOCK TO CHALRA, NEEHE  PRINT KRNE MEIN DIKKT AARI KUCH KEY KI YA PTANI KRRA PRINT
        )

        response = ""
        for i in generated_text:
            # print("I", i)
            if i["choices"][0]["delta"] != {}:
                # print("Sent", str(i))
                yield 'data: %s\n\n' % i["choices"][0]["delta"]["content"]
                response += i["choices"][0]["delta"]["content"]
            else:
                # stream ended successfully, saving the chat to database
                print("Stream ended successfully")

                # saving the chat to database
                def add_to_history():
                    import_chat(botid + "_ltm", inpt, response)

                t1 = threading.Thread(target=add_to_history)
                t1.start()
                conn = mysql.connect()
                cur = conn.cursor()
                query = "INSERT INTO messages (username, botid, sender, message, timestamp) VALUES (%s, %s, %s, %s, %s)"
                cur.execute(query, (className_b, botid, "user", inpt, datetime.datetime.now()))
                cur.execute(query, (className_b, botid, "assistant", response, datetime.datetime.now()))
                conn.commit()
                cur.close()

    return Response(streamResponse(), mimetype='text/event-stream')


def connect(classname, className_b, subscription, inpt, allowImages, b_botrole, b_steps, comp_info=""):
    ipaddress = request.remote_addr
    print("IP", ipaddress)

    if subscription == None:
        subscription = 0

    # memory = stm(className_b+"_chats_with_"+classname, 4)
    query2 = "SELECT * FROM messages WHERE (username=%s AND botid=%s) ORDER BY timestamp DESC LIMIT 4"
    conn = mysql.connect()
    cur = conn.cursor()
    cur.execute(query2, (classname, className_b))
    result = cur.fetchall()

    # if input is short, also put previous messages in the context
    # if len(inpt.split(" ")) < 10 and len(result) > 0:
    #     inptToSearch = input + "(" + result[0][4].split(" ")[:10] + ")"
    # else:
    #     inptToSearch = inpt

    context = query(className_b, inpt)
    print("Found context", context, "for", inpt)
    # using context from database as input for images
    ltm = query(className_b + "_ltm", inpt)
    print("Thinking with ltm", ltm)

    count = 0
    queryToCheckTodaysUserBotChatsCount = "SELECT COUNT(*) AS message_count FROM messages WHERE username=%s AND botid=%s AND sender='user' AND DATE(timestamp) = CURDATE();"
    cur.execute(queryToCheckTodaysUserBotChatsCount, (classname, className_b))
    result2 = cur.fetchone()
    print("Result2", result2)
    count = result2[0]
    memory = []
    for i in result:
        # limiting each message to 100 words if it's bot's, else 200 words
        content = i[4]
        if i[3] == "assistant":
            if len(i[4].split(" ")) > 150:
                content = " ".join(i[4].split(" ")[:150])
        else:
            if len(i[4].split(" ")) > 200:
                content = " ".join(i[4].split(" ")[:200])
        memory.append({"role": i[3], "content": content})
    memory.reverse()
    # print("Memory", memory)
    cur.close()
    conn.close()

    global given_prompt
    given_prompt = training_prompt(b_botrole, context, b_steps, comp_info, subscription, ltm)
    global chatsToSend
    if subscription == 0 or subscription == None:
        modified_ltm = parse_messages(ltm)
        chatsToSend = [*modified_ltm, *memory]
    else:
        chatsToSend = [*memory]
    # print("Modified", modified_ltm)

    print("Chats going are: ", chatsToSend)

    def streamResponse():
        if count >= 100:
            print("Count", count)
            yield 'data: %s\n\n' % f"Today's limit here is exhausted, to continue chatting you can use the {className_b} Bot on https://humanizeai.in/{className_b} or create your own."
            # adding the message to the database
            conn = mysql.connect()
            cur = conn.cursor()
            query = "INSERT INTO messages (username, botid, sender, message, timestamp) VALUES (%s, %s, %s, %s, %s)"
            cur.execute(query, (classname, className_b, "user", inpt, datetime.datetime.now()))
            cur.execute(query, (classname, className_b, "assistant",
                                f"Today's limit here is exhausted, to continue chatting you can use the {className_b} Bot on https://humanizeai.in/{className_b} or create your own.",
                                datetime.datetime.now()))
            conn.commit()
            cur.close()
        else:
            global chatsToSend
            global given_prompt
            input_tokens = gpt3_tokenizer.count_tokens(given_prompt + " " + str(chatsToSend) + inpt)
            print("Tokens first", input_tokens)
            if input_tokens > 3072:
                # remove first memory msg
                memory.pop(0)
                if subscription == 0 or subscription == None:
                    chatsToSend = [*modified_ltm, *memory]
                else:
                    chatsToSend = [*memory]
                new_count = gpt3_tokenizer.count_tokens(given_prompt + " " + str(chatsToSend) + inpt)
                if new_count > 3072:
                    # remove second memory msg
                    memory.pop(0)
                    if subscription == 0 or subscription == None:
                        chatsToSend = [*modified_ltm, *memory]
                    else:
                        chatsToSend = [*memory]
                    if (gpt3_tokenizer.count_tokens(given_prompt + " " + str(chatsToSend) + inpt)) > 3072:
                        while True:
                            # removing last 20 words from given_prompt
                            given_prompt = " ".join(given_prompt.split(" ")[:-10])
                            if (gpt3_tokenizer.count_tokens(given_prompt + " " + str(chatsToSend) + inpt)) < 3072:
                                break
            print("Tokens", gpt3_tokenizer.count_tokens(given_prompt + " " + str(chatsToSend) + inpt))

            print("Prompt", given_prompt)
            generated_text = openai.chat.completions.create(
                model="gpt-3.5-turbo" if subscription == 0 else "gpt-4",
                messages=[
                    {"role": "system", "content": given_prompt},
                    *chatsToSend,
                    {"role": "user", "content": inpt},
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True
                # chal rhe hai? YE WALA BLOCK TO CHALRA, NEEHE  PRINT KRNE MEIN DIKKT AARI KUCH KEY KI YA PTANI KRRA PRINT
            )

            response = ""
            for i in generated_text:
                # print("I", i)
                if i["choices"][0]["delta"] != {}:
                    # print("Sent", str(i))
                    yield 'data: %s\n\n' % i["choices"][0]["delta"]["content"]
                    response += i["choices"][0]["delta"]["content"]
                else:
                    print("AllowIms?")
                    print(allowImages)
                    # messages sent in chunks, now sending the links
                    if allowImages:
                        links = query_image(className_b + "_images", inpt)
                        print("Links", links)
                    else:
                        links = []
                    if links != []:
                        yield 'data: %s\n\n' % links
                        response += str(links)
                    # stream ended successfully, saving the chat to database
                    print("Stream ended successfully")
                    # saving the chat to database
                    # def add_to_history():
                    #     import_chat(className_b+"_ltm", inpt, response)
                    # t1 = threading.Thread(target=add_to_history)
                    # t1.start()
                    conn = mysql.connect()
                    cur = conn.cursor()
                    query = "INSERT INTO messages (username, botid, sender, message, timestamp) VALUES (%s, %s, %s, %s, %s)"
                    queryToIncreaseInteraction = "UPDATE bots SET interactions = interactions + 1 WHERE botid=%s"
                    cur.execute(query, (classname, className_b, "user", inpt, datetime.datetime.now()))
                    cur.execute(query, (classname, className_b, "assistant", response, datetime.datetime.now()))
                    cur.execute(queryToIncreaseInteraction, (className_b,))
                    conn.commit()
                    cur.close()

    return Response(streamResponse(), mimetype='text/event-stream')


def connect_api(classname, className_b, subscription, inpt, allowImages, b_botrole, b_steps, comp_info=""):
    if subscription == None:
        subscription = 0

    context = query(className_b, inpt)
    print("Found context", context, "for", inpt)
    # using context from database as input for images
    ltm = query(className_b + "_ltm", inpt)
    print("Thinking with ltm", ltm)
    # memory = stm(className_b+"_chats_with_"+classname, 4)
    query2 = "SELECT * FROM messages WHERE (username=%s AND botid=%s) ORDER BY timestamp DESC LIMIT 4"
    conn = mysql.connect()
    cur = conn.cursor()
    cur.execute(query2, (classname, className_b))
    msgs = cur.fetchall()
    count = 0
    queryToCheckTodaysUserBotChatsCount = "SELECT COUNT(*) AS message_count FROM api_calls WHERE username=%s AND botid=%s AND DATE(timestamp) = CURDATE();"
    cur.execute(queryToCheckTodaysUserBotChatsCount, (classname, className_b))
    result2 = cur.fetchone()
    print("api calls", result2)
    count = result2[0]
    # queryToCheckApiCalls = "SELECT * FROM api_calls WHERE botid=%s"
    # cur.execute(queryToCheckApiCalls, (className_b,))
    # api_calls = cur.fetchall()
    # conn.commit()
    memory = []
    for i in msgs:
        memory.append({"role": i[3], "content": i[4]})
    memory.reverse()
    # print("Memory", memory)
    cur.close()
    conn.close()

    given_prompt = training_prompt(b_botrole, context, b_steps, comp_info, subscription, ltm)

    if subscription == 0 or subscription == None:
        modified_ltm = parse_messages(ltm)
        chatsToSend = [*modified_ltm, *memory]
        print("LTM", modified_ltm)
    else:
        chatsToSend = [*memory]
    print("Sending chats", chatsToSend)

    if subscription == 0:
        max_count = 20
    elif subscription == 1:
        max_count = 50
    elif subscription == 2:
        max_count = 50
    else:
        max_count = 100

    def streamResponse():
        if count >= max_count:
            print("Count", count)
            yield 'data: %s\n\n' % f"Today's API limit is crossed here, you can continue chatting with the {className_b} bot for free at https://humanizeai.in/{className_b} or create your own."
            # adding the message to the database
            conn = mysql.connect()
            cur = conn.cursor()
            cur.execute(query, (classname, className_b, "user", inpt, datetime.datetime.now()))
            cur.execute(query, (classname, className_b, "assistant",
                                f"Today's API limit crossed, you can continue chatting with the {className_b} bot at https://humanizeai.in/{className_b} or create your own.",
                                datetime.datetime.now()))
            conn.commit()
            cur.close()
        else:
            print("Prompt", given_prompt)
            generated_text = openai.chat.completions.create(
                model="gpt-3.5-turbo" if subscription <= 1 else "gpt-4",
                messages=[
                    {"role": "system", "content": given_prompt},
                    *chatsToSend,
                    {"role": "user", "content": inpt},
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True
                # chal rhe hai? YE WALA BLOCK TO CHALRA, NEEHE  PRINT KRNE MEIN DIKKT AARI KUCH KEY KI YA PTANI KRRA PRINT
            )

            response = ""
            for i in generated_text:
                # print("I", i)
                if i["choices"][0]["delta"] != {}:
                    # print("Sent", str(i))
                    yield 'data: %s\n\n' % i["choices"][0]["delta"]["content"]
                    response += i["choices"][0]["delta"]["content"]
                else:
                    conn = mysql.connect()
                    cur = conn.cursor()
                    queryAddApiCall = "INSERT INTO api_calls (username, botid, input_tokens, response_tokens) VALUES (%s, %s, %s, %s)"
                    # calc openai tokens from the message
                    input_tokens = int(gpt3_tokenizer.count_tokens(inpt))
                    response_tokens = int(gpt3_tokenizer.count_tokens(response))
                    cur.execute(queryAddApiCall, (classname, className_b, input_tokens, response_tokens))
                    print("AllowIms?")
                    print(allowImages)
                    # messages sent in chunks, now sending the links
                    if allowImages:
                        links = query_image(className_b + "_images", inpt)
                        print("Links", links)
                    else:
                        links = []
                    if links != []:
                        yield 'data: %s\n\n' % links
                    # stream ended successfully, saving the chat to database
                    print("Stream ended successfully")
                    conn = mysql.connect()
                    cur = conn.cursor()
                    query = "INSERT INTO messages (username, botid, sender, message, timestamp) VALUES (%s, %s, %s, %s, %s)"
                    queryToIncreaseInteraction = "UPDATE bots SET interactions = interactions + 1 WHERE botid=%s"
                    cur.execute(query, (
                    (str(classname) + "_api_messages"), className_b, "user", str(inpt), datetime.datetime.now()))
                    cur.execute(query, ((str(classname) + "_api_messages"), className_b, "assistant", str(response),
                                        datetime.datetime.now()))
                    cur.execute(queryToIncreaseInteraction, (className_b,))
                    conn.commit()
                    cur.close()

    return Response(streamResponse(), mimetype='text/event-stream')


def save_pdf_id(username, botid, given_id, weaviate_ids, title="Document"):
    conn = mysql.connect()
    cur = conn.cursor()
    time1 = time.time()
    query2 = "INSERT INTO pdfs (id, title, weaviate_ids) VALUES (%s, %s, %s)"
    cur.execute(query2, (given_id, title, json.dumps(weaviate_ids)))
    # add to pdfs array in users table
    query3 = "SELECT pdfs FROM users WHERE username=%s OR email_id=%s"
    cur.execute(query3, (username, username,))
    result = cur.fetchall()
    if result[0][0] == None:
        pdfs = []
    else:
        pdfs = json.loads(result[0][0])
    pdfs.append({"id": given_id, "title": title})
    query4 = "UPDATE users SET pdfs=%s WHERE username=%s OR email_id=%s"
    cur.execute(query4, (json.dumps(pdfs), username, username))
    # insert pdf<given_id> in messages table sent by user
    query5 = "INSERT INTO messages (username, botid, sender, message, timestamp) VALUES (%s, %s, %s, %s, %s)"
    cur.execute(query5, (username, botid, "user", "pdf<<" + str(given_id) + ">>", datetime.datetime.now()))
    time2 = time.time()
    print("Time taken to save pdf", time2 - time1)
    conn.commit()
    cur.close()


# NOT BEING USED RIGHT NOW
def delete_pdf(username, given_id):
    # search for weavaite ids and delete them simultaneously
    conn = mysql.connect()
    cur = conn.cursor()
    query = "SELECT weaviate_ids FROM pdfs WHERE id=%s"
    cur.execute(query, (given_id,))
    result = cur.fetchall()
    cur.close()
    weaviate_ids = json.loads(result[0][0])

    # command used to create was
    # client.data_object.create(class_name=username, data_object={"chat": item})
    # list_id.append(client.data_object.get(class_name=username, uuid=None)["objects"][0]["id"])

    try:
        for i in weaviate_ids:
            print("Deleting", i)
            client.data_object.delete(class_name=username, uuid=i)
        return True
    except:
        return False


# google.oauth2.credentials.Credentials

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
    message1 = CreateMessage(sender, to, subject, msg, msgHtml)
    SendMessageInternal(service, "me", message1)


def SendMessageInternal(service, user_id, message):
    try:
        message = (service.users().messages().send(userId=user_id, body=message).execute())
        print('Message Id: %s' % message['id'])
        return "Message sent successfully by Gmail API"
    except Exception as error:
        return ('An error occurred: %s' % error)


def CreateMessage(sender, to, subject, msgPlain="", msgHtml=None):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to
    msg.attach(MIMEText(msgPlain, 'plain'))
    if msgHtml != None:
        msg.attach(MIMEText(msgHtml, 'html'))
    raw = base64.urlsafe_b64encode(msg.as_bytes())
    raw = raw.decode()
    body = {'raw': raw}
    print("Body", body)
    return body



@app.route('/register', methods=['POST'])
@cross_origin()
def register():
    name, email_id, password = request.json['name'], request.json['email_id'], request.json['password']

    if name == "" or email_id == "" or password == "":
        return jsonify({"success": False, "message": "Please fill all the fields."}), 400

    try:
        conn = mysql.connect()
        cur = conn.cursor()
        empty_array_string = json.dumps([])
        # public, info, steps ye sab add krna in the end
        # query = "CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), phone VARCHAR(255), email_id VARCHAR(255) UNIQUE, password VARCHAR(255), username VARCHAR(255) DEFAULT NULL, pic VARCHAR(255) DEFAULT NULL, purpose VARCHAR(255) DEFAULT NULL, plan INT(255) DEFAULT 0, whatsapp VARCHAR(255) DEFAULT NULL, youtube VARCHAR(255) DEFAULT NULL, instagram VARCHAR(255) DEFAULT NULL, discord VARCHAR(255) DEFAULT NULL, telegram VARCHAR(255) DEFAULT NULL, website VARCHAR(255) DEFAULT NULL, favBots VARCHAR(255) DEFAULT '" + empty_array_string + "', pdfs VARCHAR(255) DEFAULT '" + empty_array_string + "', bots VARCHAR(255) DEFAULT '" + empty_array_string + "', setup BOOLEAN DEFAULT 0)"
        # query2 = "CREATE TABLE IF NOT EXISTS bots (id INT AUTO_INCREMENT PRIMARY KEY, botid VARCHAR(255) UNIQUE NOT NULL, name VARCHAR(255) DEFAULT NULL, username VARCHAR(255) NOT NULL, description VARCHAR(255) DEFAULT NULL, pic VARCHAR(255) DEFAULT NULL, interactions INT(255) DEFAULT 0, likes INT(255) DEFAULT 0, whatsapp VARCHAR(255) DEFAULT NULL, youtube VARCHAR(255) DEFAULT NULL, instagram VARCHAR(255) DEFAULT NULL, discord VARCHAR(255) DEFAULT NULL, telegram VARCHAR(255) DEFAULT NULL, pdfs VARCHAR(255) DEFAULT '" + empty_array_string + "', setup BOOLEAN DEFAULT 0)"
        # print("Acc creation query", query)
        # cur.execute(query)
        print("Step 1")
        # cur.execute(query2)
        print("Step 2")
        cur.execute("INSERT INTO users (name, email_id, password) VALUES (%s, %s, %s)",
                    (name, email_id, generate_password_hash(password)))
        conn.commit()
        cur.close()
        print("User data written to mysql")

        token = jwt.encode({'username': email_id}, "h1u2m3a4n5i6z7e8")
        return jsonify({
            "success": True,
            "message": "Account created successfully",
            "token": token,
            "data": {
                "name": name,
                "email_id": email_id
            },
            "bots": []
        }), 201
    except Exception as e:
        print("MYSQL ERR", e)
        return jsonify({"success": False, "message": "Error in writing user data to Database"}), 500

if __name__ == "__main__":
    app.run(debug=True)