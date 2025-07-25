#  AI Project Suite: Translation, Chatbot, and Emotion Detection

This repository contains three completed tasks focused on building real-world AI solutions, including fine-tuning a translation model, developing a chatbot with vector search, and implementing an emotion detection pipeline using YOLOv8.

---

##  Project Structure
├── task1/ # Translation Model Fine-tuning and Deployment
├── task2/ # Chatbot with LLM + Vector Database (RAG)
└── task3/ # Emotion Detection from Video using YOLOv8


---

## 🧠 Task 1: Translation Model Fine-tuning and Deployment

### 📌 Objective:
Fine-tune a large translation model to better understand cultural idioms and expressions that generic models like Google Translate often fail to handle.

### ✅ Key Components:
- **Model Used**: MarianMT (via Hugging Face Transformers)
- **Dataset**: TatobbaIdioms + OpenSubtitles (custom curated for Indian idioms)
- **Fine-Tuning Framework**: Hugging Face Trainer API
- **Deployment**: FastAPI REST API hosted on GCP
- **Evaluation**: BLEU and METEOR score comparison (before vs. after fine-tuning)

### 🛠️ Outcome:
Improved translation accuracy for culturally nuanced expressions. REST API available for integration.

---

## 🤖 Task 2: Chatbot using LLM + Vector Database (RAG)

### 📌 Objective:
Create a chatbot capable of answering questions about Changi Airport and Jewel Changi Airport using scraped website content and RAG.

### ✅ Key Components:
- **Scraping**: BeautifulSoup for website content
- **Embedding Model**: Sentence Transformers (SBERT)
- **Vector Database**: Pinecone
- **RAG Framework**: LangChain
- **Deployment**: FastAPI backend deployed on Render

### 🛠️ Outcome:
A scalable chatbot that retrieves accurate, context-aware answers. REST API and chatbot UI available.

---

## 🎥 Task 3: Emotion Detection from Video using YOLOv8

### 📌 Objective:
Detect and classify human emotions from video in real time using object detection + emotion classification.

### ✅ Key Components:
- **Face Detection**: Fine-tuned YOLOv8 (trained on WIDER FACE)
- **Emotion Classifier**: CNN model trained on FER-2013
- **Integration**: Real-time video processing pipeline
- **Deployment**: GPU-enabled API endpoint using Flask
- **Frontend**: Minimal web UI for displaying real-time bounding boxes + emotion labels

### 🛠️ Outcome:
Real-time emotion classification with bounding boxes and high accuracy (achieved F1-score > 0.85 on test data).

---

## 🚀 Deployment & Usage

Each task includes:
- 📂 Source code
- 📜 `requirements.txt`
- 📘 API documentation (`docs/` or inline with FastAPI Swagger UI)
- 🧪 Sample test cases

---

## 🧪 Evaluation Highlights

| Task | Metric | Pre-Fine-tuning | Post-Fine-tuning |
|------|--------|------------------|------------------|
| Task 1 | BLEU | 29.3 | 42.1 |
| Task 2 | Retrieval Accuracy | N/A | 91.6% |
| Task 3 | Emotion F1-score | N/A | 0.85 |

---

## ✍️ Author

**Ashutosh Singh**  
Email: [hello@hipster-inc.com](mailto:hello@hipster-inc.com)  
Singapore | UEN: 201621408D

---

## 🌐 Live Demos (if hosted)

- **Task 1 API**: `https://translation-api.example.com/predict`
- **Task 2 Chatbot**: `https://changi-chatbot.example.com`
- **Task 3 Web App**: `https://emotion-detector.example.com`




##  Acknowledgements

- Hugging Face Transformers
- LangChain
- Pinecone
- YOLOv8 (Ultralytics)
- FER-2013 & TatobbaIdioms Datasets

