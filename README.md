# FAQ Chatbot with Web Scraping, Semantic Search, and BM25

## This project enhances a Rasa chatbot with the ability to:

- Scrape FAQs from webpages
  
- Save them to CSV
  
- Perform smart FAQ lookup using:
  
- Semantic Search (FAISS + Sentence Transformers)

- BM25 Ranking
  
- Fuzzy Matching (RapidFuzz)


# Features

## 1. Action: action_faq_generate

Scrapes FAQs from a list of webpages and saves them to a CSV.

# Functionality:

- Uses requests and BeautifulSoup to parse HTML.

- Extracts headings  as questions.

- Extracts paragraph elements as answers.

- Saves questions and answers into a CSV (faqs.csv).


# Source URLs:

[
    "https://prpvoice.com/About.aspx",
    "https://prpvoice.com/LMS.aspx",
    "https://prpvoice.com/Whatsapp.aspx",
    "https://prpvoice.com/WhatsApp-Marketing.aspx",
    "https://prpvoice.com/IVR.aspx",
    "https://prpvoice.com/SMS.aspx",
    "https://prpvoice.com/Email.aspx"
]


---

## 2. Action: action_faq_lookup

Looks up the best answer from saved FAQs based on user queries using three methods.

# Techniques Used:

FAISS + SentenceTransformers (all-MiniLM-L6-v2): Semantic search.

BM25Okapi (from rank_bm25): Token-based ranking.

Fuzzy Matching (RapidFuzz): Text similarity.

Combined scoring: Returns the best match based on score.


# Lookup Flow:

## 1. User query is received.


## 2. Three types of matches are computed:

FAISS semantic search (top 3).

Fuzzy match (if score > 60).

BM25 top score.



## 3. All results are ranked and the best match is returned.




---

# Requirements

Install dependencies with:

pip install rasa requests beautifulsoup4 lxml sentence-transformers faiss-cpu rank-bm25 rapidfuzz nltk scikit-learn

Also, run once to download tokenizer (for NLTK):

import nltk
nltk.download('punkt')


---

# File Structure

actions/

│

├── actions.py   

├── faqs.csv          


---

# How to Use

## Step 1: Generate FAQ CSV

Run the following Rasa custom action to scrape and save FAQs:

- intent: generate_faq
  action: action_faq_generate

## Step 2: Ask a Question

Once faqs.csv is populated, you can ask a question:

- intent: ask_question
  examples: |
    - What is WhatsApp marketing?
    - Tell me about IVR service
  action: action_faq_lookup


---

# Future Enhancements

Add support for multilingual FAQ lookup.

Schedule automatic scraping and updating of FAQ data.

Allow user-specific FAQ databases.



---

## Author

Shani | prpvoice.com


---
