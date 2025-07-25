import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLanguageModel


load_dotenv()
GORQ_API_KEY = os.getenv("GORQ_API_KEY")
GORQ_MODEL = os.getenv("GORQ_MODEL")


def scrape_site(url):
    print(f"Scraping {url}")
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    texts = soup.find_all(['p', 'li', 'h1', 'h2', 'h3'])
    content = "\n".join([t.get_text(strip=True) for t in texts])
    return content

data_changi = scrape_site("https://www.changiairport.com")
data_jewel = scrape_site("https://www.jewelchangiairport.com")
full_data = data_changi + "\n" + data_jewel


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(full_data)


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True)


dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings))


faiss.write_index(faiss_index, "faiss_index.index")
with open("texts.pkl", "wb") as f:
    pickle.dump(chunks, f)


class GorqLLM(LLM):
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    @property
    def _llm_type(self):
        return "gorq"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.5,
        }
        res = requests.post("https://api.groq.com/openai/v1/completions", json=body, headers=headers)
        return res.json()['choices'][0]['text'].strip()


embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_func, allow_dangerous_deserialization=True)


retriever = db.as_retriever(search_kwargs={"k": 4})
llm = GorqLLM(api_key=GORQ_API_KEY, model=GORQ_MODEL)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


print("📦 Changi Airport Chatbot is ready! Ask anything (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        print("Goodbye! 👋")
        break
    answer = qa_chain.run(query)
    print("Bot:", answer)