import imaplib
import email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import lancedb
import os
import json
import requests
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- SECURE CONFIGURATION (from Environment Variables) ---
IMAP_SERVER = os.getenv('IMAP_SERVER', 'imap.gmail.com')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
EMAIL_ACCOUNT = os.getenv('EMAIL_ACCOUNT')
EMAIL_PASSWORD = os.getenv('EMAIL_APP_PASSWORD')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- STATIC CONFIGURATION ---
DB_PATH = "/tmp/lancedb"
TABLE_NAME = "pnb_faqs_filtered"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Knowledge Base Object ---
# This avoids reloading models on every single request.
knowledge_base = None

def initialize_knowledge_base():
    """Connects to the LanceDB table. This is called once before the first request."""
    global knowledge_base
    if knowledge_base is None:
        print("Initializing knowledge base connection...")
        if not os.path.exists(DB_PATH):
             raise FileNotFoundError(f"LanceDB database not found at {DB_PATH}. Please run the scraper script first.")

        db = lancedb.connect(DB_PATH)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        knowledge_base = LanceDB(connection=db, embedding=embeddings, table_name=TABLE_NAME)
        print("Knowledge base connection successful.")

def generate_gemini_reply(context, question):
    """
    Generates a reply using the Gemini API with an improved prompt for better persona and response quality.
    """
    print("Generating response with Gemini...")

    # --- Step 1: Handle simple greetings and small talk ---
    normalized_question = question.lower().strip()
    greetings = ["hi", "hello", "hey", "how are you", "how are you?"]

    # If the user's entire message is just a greeting, provide a canned response.
    if normalized_question in greetings:
        print("Detected simple greeting. Replying with a standard greeting.")
        return "Hello! I'm Arya, your PNB Housing assistant. How can I help you with our Home Loan or Fixed Deposit products today?"
    
    # Also handle cases where the message is very short and likely just a greeting
    if len(normalized_question.split()) <= 2 and any(g in normalized_question for g in greetings):
        print("Detected short greeting. Replying with a standard greeting.")
        return "Hello! I'm Arya, your PNB Housing assistant. How can I help you with our Home Loan or Fixed Deposit products today?"

    # --- Step 2: Build the enhanced prompt for the LLM ---
    prompt_template = f"""
    You are "Arya", a professional and friendly customer service assistant for PNB Housing.

    **Your Core Directives:**
    1.  **Persona**: You are always polite, helpful, and clear. Your goal is to provide detailed and useful information, not just short answers.
    2.  **Greeting**: Every single response MUST begin with a friendly greeting. Examples: "Hi there! I'm Arya.", "Hello! My name is Arya.", "Hello! I'm Arya from PNB Housing."
    3.  **Grounding**: You MUST base your answers strictly on the information provided in the "CONTEXT" section. Do not use any outside knowledge.
    4.  **Handling Missing Information**: If the CONTEXT does not contain the answer, you MUST reply with: "Hello! I'm Arya. I'm sorry, but I couldn't find specific information about your query in our knowledge base. I can assist with questions about PNB Housing's Home Loans and Fixed Deposits."

    **Example of a Perfect Response:**
    ---
    USER'S QUESTION: Can I open multiple accounts?
    CONTEXT: Question: Can a depositor open multiple accounts? Answer: Yes, you can open multiple accounts, but for the purpose of computation of tax liability all the accounts will be clubbed.
    YOUR ANSWER:
    Hello! I'm Arya.

    Yes, you can certainly open multiple Fixed Deposit accounts with PNB Housing. Please keep in mind that for the purpose of computing tax liability, all of your accounts will be clubbed together.
    ---

    Now, answer the following user's question based on the provided context.

    ---
    CONTEXT:
    {context}
    ---

    USER'S QUESTION:
    {question}

    YOUR ANSWER:
    """

    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt_template.strip()}]}]
    })

    # --- Step 3: Call the Gemini API ---
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if "candidates" in result and result["candidates"][0]["content"]["parts"][0]["text"]:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            print("Gemini API returned an unexpected response structure.")
            return "I seem to be having a technical issue. Please try again in a moment."

    except requests.exceptions.HTTPError as http_err:
        print(f"Gemini API HTTP error: {http_err} - {response.text}")
        return "I am currently facing a technical issue and cannot reply at the moment. Please try again later."
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return "I am sorry, but I encountered an error while processing your request."

def send_reply(to_address, subject, body):
    """Sends an email reply."""
    print(f"Sending reply to {to_address}...")
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ACCOUNT
        msg['To'] = to_address
        msg['Subject'] = f"Re: {subject}"
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Reply sent successfully.")
        return True
    except Exception as e:
        print(f"Failed to send reply: {e}")
        return False

def check_and_process_emails():
    """The core logic to fetch and process unread emails."""
    print("\nChecking for new emails...")
    processed_count = 0
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
        mail.select('inbox')
        status, messages = mail.search(None, 'UNSEEN')
        if status != 'OK' or not messages[0]:
            print("No new unread emails.")
            mail.logout()
            return "No new emails to process."

        email_ids = messages[0].split()
        print(f"Found {len(email_ids)} new email(s).")

        for email_id in email_ids:
            _, msg_data = mail.fetch(email_id, '(RFC822)')
            msg = email.message_from_bytes(msg_data[0][1])
            
            from_header = email.utils.getaddresses([msg['From']])
            if not from_header: continue
            sender_name, sender_address = from_header[0]
            if '@' not in sender_address: continue

            subject_header = email.header.decode_header(msg['Subject'])
            subject = subject_header[0][0]
            if isinstance(subject, bytes): subject = subject.decode(subject_header[0][1] or 'utf-8')

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = msg.get_payload(decode=True).decode()

            print(f"\n--- Processing email from: {sender_address} ---")
            user_query = f"Subject: {subject}\n\n{body}"

            retrieved_docs = knowledge_base.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            generated_answer = generate_gemini_reply(context, user_query)
            
            if send_reply(sender_address, subject, generated_answer):
                mail.store(email_id, '+FLAGS', '\\Seen')
                processed_count += 1

        mail.logout()
        return f"Successfully processed {processed_count} of {len(email_ids)} email(s)."

    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        return f"An error occurred: {e}"

# --- API Endpoint Definition ---
@app.route('/trigger-email-check', methods=['POST'])
def trigger_email_check():
    """
    An API endpoint that, when called, runs the email checking process.
    """
    # This ensures the knowledge base is loaded before the first request
    if knowledge_base is None:
        initialize_knowledge_base()
        
    result = check_and_process_emails()
    return jsonify({"status": "completed", "message": result})

# This block allows you to run the app directly for local testing
if __name__ == '__main__':
    # Check for required environment variables before starting
    if not all([EMAIL_ACCOUNT, EMAIL_PASSWORD, GEMINI_API_KEY]):
         print("="*50)
         print("ERROR: Missing required environment variables.")
         print("       Please ensure EMAIL_ACCOUNT, EMAIL_APP_PASSWORD, and")
         print("       GEMINI_API_KEY are set in your .env file.")
         print("="*50)
    else:
        port = int(os.getenv('FLASK_RUN_PORT', 6004))
        # Initialize the knowledge base once on startup
        initialize_knowledge_base()
        app.run(host='0.0.0.0', port=port, debug=True)
