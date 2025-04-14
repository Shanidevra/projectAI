from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
from bs4 import BeautifulSoup
import csv
from rapidfuzz import fuzz, process
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from rasa_sdk.events import SlotSet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lxml import html

class ActionFAQGenerate(Action):

    def name(self) -> Text:
        return "action_faq_generate"

    def extract_faqs_from_url(self, url: str) -> List[Dict[str, str]]:
        faqs = []
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            question_elements = soup.find_all(['h1', 'h2', 'h3'])  # Treat all headings as potential questions
            answer_elements = soup.find_all('p')

            for i in range(min(len(question_elements), len(answer_elements))):
                question = question_elements[i].get_text(strip=True)
                answer = answer_elements[i].get_text(strip=True)
                if question and answer:
                    faqs.append({"question": question, "answer": answer})
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
        return faqs

    def save_faqs_to_csv(self, faqs: List[Dict[str, str]], filename: str = "faqs.csv"):
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['question', 'answer']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(faqs)
            print(f"FAQs saved to {filename}")
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        urls = [
            "https://prpvoice.com/About.aspx",
            "https://prpvoice.com/LMS.aspx",
            "https://prpvoice.com/Whatsapp.aspx",
            "https://prpvoice.com/WhatsApp-Marketing.aspx",
            "https://prpvoice.com/IVR.aspx",
            "https://prpvoice.com/SMS.aspx",
            "https://prpvoice.com/Email.aspx"

        ]

        all_faqs = []
        for url in urls:
            faqs = self.extract_faqs_from_url(url)
            all_faqs.extend(faqs)

        self.save_faqs_to_csv(all_faqs)

        dispatcher.utter_message(text="FAQs have been generated and saved.")
        return []
    
class ActionFAQLookup(Action):

    def name(self) -> Text:
        return "action_faq_lookup"

    def __init__(self):
        self.faqs = self.load_faqs()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.faiss_index, self.embeddings = self.build_faiss_index(self.faqs)
        self.tokenized_questions = [self.tokenize(faq["question"]) for faq in self.faqs]
        self.bm25 = BM25Okapi(self.tokenized_questions)

    def load_faqs(self, csv_path: str = "faqs.csv") -> List[Dict[str, str]]:
        faqs = []
        try:
            with open(csv_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    question = row.get("question", "").strip()
                    answer = row.get("answer", "").strip()
                    if question and answer:
                        faqs.append({"question": question, "answer": answer})
        except Exception as e:
            print(f"[FAQ ERROR] Couldn't load CSV: {e}")
        return faqs

    def build_faiss_index(self, faqs: List[Dict[str, str]]):
        questions = [faq["question"] for faq in faqs]
        embeddings = self.model.encode(questions, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, embeddings

    def tokenize(self, text: str):
        return re.findall(r'\w+', text.lower())

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get("text")

        if not self.faqs:
            dispatcher.utter_message(text="FAQ database is currently unavailable.")
            return []

        # FAISS Semantic Match
        query_embedding = self.model.encode([user_query], convert_to_numpy=True)
        _, faiss_indices = self.faiss_index.search(query_embedding, k=3)
        faiss_candidates = [self.faqs[i] for i in faiss_indices[0]]

        # Fuzzy Match
        fuzzy_match, fuzzy_score, fuzzy_index = process.extractOne(user_query, [f["question"] for f in self.faqs], scorer=fuzz.token_sort_ratio)
        fuzzy_candidate = self.faqs[fuzzy_index] if fuzzy_score > 60 else None

        # BM25 Match
        tokenized_query = self.tokenize(user_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        best_bm25_index = int(np.argmax(bm25_scores))
        bm25_candidate = self.faqs[best_bm25_index] if bm25_scores[best_bm25_index] > 0 else None

        # Combine candidates and select best match
        candidates = []
        if fuzzy_candidate:
            candidates.append(("fuzzy", fuzzy_candidate, fuzzy_score))
        if bm25_candidate:
            candidates.append(("bm25", bm25_candidate, bm25_scores[best_bm25_index]))
        for candidate in faiss_candidates:
            score = fuzz.token_sort_ratio(user_query, candidate["question"])
            candidates.append(("semantic", candidate, score))

        best_match = sorted(candidates, key=lambda x: x[2], reverse=True)[0][1]

        dispatcher.utter_message(text=f"{best_match['answer']}")
        return []
    
