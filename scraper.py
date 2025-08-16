import requests
from bs4 import BeautifulSoup
import lancedb
import os
from langchain_community.vectorstores import LanceDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def scrape_pnb_faqs(url):
    """
    Scrapes ONLY the Home Loan and Fixed Deposit FAQs from the PNB Housing website,
    based on the specific HTML structure of those sections.

    Args:
        url (str): The URL of the FAQ page.

    Returns:
        list: A list of strings, where each string is a "Question: [question] Answer: [answer]" pair.
    """
    print(f"Scraping targeted FAQs from: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    faq_data = []
    
    # Define the keywords for the sections we want to scrape
    target_keywords = ['home loan', 'fixed deposit']

    # Find all potential FAQ sections on the page
    all_sections = soup.find_all('div', class_='tabReapeate')
    
    if not all_sections:
        print("Error: Could not find any FAQ section containers with class 'tabReapeate'.")
        return []

    print(f"Found {len(all_sections)} potential sections. Filtering for Home Loan and Fixed Deposit...")

    for section in all_sections:
        heading_tag = section.find('h3')
        if not heading_tag:
            continue

        heading_text = heading_tag.get_text(strip=True).lower()

        # Check if the section heading matches our target keywords
        if any(keyword in heading_text for keyword in target_keywords):
            print(f"\n--- Processing Section: {heading_tag.get_text(strip=True)} ---")
            
            # Find all question containers within this specific section
            question_containers = section.find_all('div', class_='question')
            
            for q_container in question_containers:
                question_tag = q_container.find('div', class_='QuesLists')
                answer_container = q_container.find_next_sibling('div', class_='answer')

                if question_tag and answer_container:
                    answer_tag = answer_container.find('div', class_='AnsLists')
                    if answer_tag:
                        question = ' '.join(question_tag.get_text(strip=True).split())
                        answer = ' '.join(answer_tag.get_text(strip=True).split())
                        
                        if question and answer:
                            full_text = f"Question: {question} Answer: {answer}"
                            faq_data.append(full_text)
                            print(f"  - Scraped Q: {question[:60]}...")
    
    print(f"\nSuccessfully scraped {len(faq_data)} Q&A pairs from the target sections.")
    return faq_data

def create_lancedb_knowledge_base(documents):
    """
    Creates a searchable RAG knowledge base using LanceDB.

    Args:
        documents (list): A list of text documents.

    Returns:
        LanceDB: A LanceDB vector store object that can be queried.
    """
    if not documents:
        print("No documents provided to create the knowledge base.")
        return None

    print("\n--- Creating Knowledge Base with LanceDB ---")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text('\n\n'.join(documents))
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    print("Loading embedding model (this may take a moment on first run)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embedding model loaded.")

    # Setup LanceDB
    db_path = "/tmp/lancedb"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    db = lancedb.connect(db_path)
    # Using a more specific table name for the filtered data
    table_name = "pnb_faqs_filtered"
    
    if table_name in db.table_names():
        db.drop_table(table_name)
        print(f"Dropped existing LanceDB table: {table_name}")

    print("Creating LanceDB vector store...")
    vector_store = LanceDB.from_texts(chunks, embeddings, connection=db, table_name=table_name)
    print("Knowledge base created successfully in LanceDB!")

    return vector_store

def main():
    """
    Main function to run the scraping and knowledge base creation process.
    """
    faq_url = 'https://www.pnbhousing.com/faqs'
    
    scraped_documents = scrape_pnb_faqs(faq_url)
    
    if not scraped_documents:
        print("\nScraping failed or no data was found. Exiting.")
        return

    knowledge_base = create_lancedb_knowledge_base(scraped_documents)
    
    if not knowledge_base:
        print("\nFailed to create the knowledge base. Exiting.")
        return

    print("\n--- Testing the Knowledge Base with a Sample Query ---")
    
    query = "what are the interest rates for fixed deposit?"
    print(f"Sample Query: '{query}'")
    
    results = knowledge_base.similarity_search(query, k=3)
    
    print("\nTop 3 most relevant results from the knowledge base:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:350] + "...")
        print("-" * 20)

    print("\nProcess complete. The filtered LanceDB knowledge base is ready.")


if __name__ == '__main__':
    main()
