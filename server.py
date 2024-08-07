import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
import openai
import os

api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print("API Key fetched successfully")
else:
    print("API Key not found")

# Initialize FastAPI
app = FastAPI()

# Enable CORSMiddleware
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],  # Adjust this as needed for your production environment
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

# Load OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

context_text = pd.read_csv('context.csv').to_string(index=False)

SYSTEM_PROMPT = "Your role is to welcome and assist visitors on this website, ensuring they find what they are " \
                "looking for. Provide concise answers (20 words or less) focused on the website’s content. If unsure " \
                "about an answer, admit it. Recommend scheduling a call with our experts for a turn-key solution. " \
                "Share URLs only when asked, referring to them as "'this link'". Remind users occasionally that this " \
                "interaction is powered by ChatGPT and that accuracy is aimed for, but mistakes can occur. Tune " \
                "responses to be friendly and engaging, and always end with a question to keep the conversation going."
SUGGESTIONS_PROMPT = "Based on the context provide three replies as the client to the last question presented by the " \
                     "assistant. the replies should be concise, 2-3 word (questions, answers, or schedule a call). " \
                     "Separate each reply with '|'"

DEFAULT_CHAT = "Hello and welcome to Tensor Technologies! How can I assist you today?"
DEFAULT_SUGGESTIONS = ["מה הם השירותים שאתם מספקים?", "Tell me more about your services.",
                       "What technologies do you use?"]

# In-memory store for user messages (stateful conversation)
user_conversations = {}


class ChatRequest(BaseModel):
    # Pydantic model for chat request
    user_id: str
    message: str


@app.post("/chatbot")
async def chatbot(chat_request: ChatRequest):
    user_id = chat_request.user_id
    user_message = chat_request.message

    # Initialize user conversation if not already present
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    # Construct prompt with context, system prompt, and conversation history
    conversation_history = user_conversations[user_id]

    prompt = f"{SYSTEM_PROMPT}\n\n{context_text}\n\n{format_conversation(conversation_history)}User: {user_message}\nAssistant:"

    try:
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        # Call OpenAI API for response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[

                {"role": "system", "content": SYSTEM_PROMPT},

                {"role": "system", "content": context_text},

                *conversation_history,

                {"role": "user", "content": user_message},

            ],

            max_tokens=150,
            n=1,
            stop=["User:", "Assistant:"]
        )

        # Get the assistant's response
        assistant_response = response.choices[0].message.content.strip()

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": assistant_response})

    return {"response": assistant_response}


@app.post("/suggestions")
async def generate_suggestions(chat_request: ChatRequest):
    user_id = chat_request.user_id
    user_message = chat_request.message

    # Initialize user conversation if not already present
    if user_id not in user_conversations:
        user_conversations[user_id] = []
    else:
        last_message = user_conversations[user_id]
        last_message = last_message[-1]['content']

    # Construct prompt with context, system prompt, and conversation history

    try:
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        # Call OpenAI API for response
        suggestions_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SUGGESTIONS_PROMPT},
                {"role": "system", "content": context_text},
                {"role": "assistant", "content": last_message},
            ],
            max_tokens=20,
            temperature=0.7,
            n=1,
        )
        suggestions = suggestions_response.choices[0].message.content.split('|')
        print(suggestions)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"suggestions": suggestions}


@app.get("/default-suggestions")
async def default_suggestions():
    return {"suggestions": DEFAULT_SUGGESTIONS}

@app.get("/default-chat")
async def default_chat():
    return {"chat": [DEFAULT_CHAT]}


def format_conversation(conversation):
    formatted = ""
    for msg in conversation:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"

    return formatted


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
