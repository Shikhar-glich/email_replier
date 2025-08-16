# 📧 RAG Email Replier – PNB Housing Assistant

This project implements an **automated email responder** powered by **Retrieval-Augmented Generation (RAG)**.  
It connects to Gmail, reads unread emails, searches through **PNB Housing FAQs** stored in **LanceDB**, and generates contextual replies using the **Google Gemini API**.  

The assistant persona is **Arya**, a friendly customer service agent for PNB Housing’s Home Loan and Fixed Deposit products.

---

## ✨ Features
- 🔍 **FAQ Scraping** – Extracts Home Loan & Fixed Deposit FAQs from PNB Housing’s website.  
- 📚 **Knowledge Base** – Stores FAQs in **LanceDB** with HuggingFace sentence embeddings.  
- 🤖 **Smart Replies** – Uses Google Gemini API to generate contextual, human-like responses.  
- 📩 **Email Automation** – Reads Gmail inbox, processes new emails, and replies automatically.  
- 🌐 **Flask API** – Provides `/trigger-email-check` endpoint to trigger processing on demand.  

---

## 📂 Project Structure
├── app.py # Flask API + Email handling + Gemini integration
├── scraper.py # Scrapes FAQs & builds LanceDB knowledge base
├── requirements.txt # Python dependencies
├── .env.example # Example environment config (no secrets)
├── LICENSE # MIT License
└── README.md


---

## ⚙️ Setup & Run Instructions

1️⃣ Clone the Repository
git clone https://github.com/Shikhar-glich/email_replier.git
cd rag-email-replier

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Configure Environment Variables

Copy .env.example → .env

Edit .env and add your credentials:

EMAIL_ACCOUNT="your_email@gmail.com"
EMAIL_APP_PASSWORD="your_gmail_app_password"
GEMINI_API_KEY="your_gemini_api_key"
FLASK_APP=app.py
FLASK_RUN_PORT=6004


🔐 Important: Use a Gmail App Password, not your real login password.

5️⃣ Build the Knowledge Base

Scrape FAQs and store them in LanceDB:

python scraper.py

6️⃣ Run the Flask Server
python app.py


Server will start at:

http://localhost:6004

📡 API Endpoint
Trigger Email Processing

POST /trigger-email-check

Example with curl:

curl -X POST http://localhost:6004/trigger-email-check


Example response:

{
  "status": "completed",
  "message": "Successfully processed 3 of 3 email(s)."
}


## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Maintainer
- **Name:** Shikhar Jaglan  
- **GitHub:** [@Shikhar-glich](https://github.com/Shikhar-glich)
