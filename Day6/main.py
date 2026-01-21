import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


data = Path("knowledge_base")
data.mkdir(exist_ok=True)


with open(data / "wellness_faq.md", "w", encoding='utf-8') as f:
    f.write("# InnerAlign Wellness FAQ\n\n")
    f.write("Q: What is Mood Drift? A: Mood Drift is the deviation from your emotional baseline. A high score means your current state is significantly different from your usual self.\n\n")
    f.write("Q: How should I handle high stress? A: Practice the 4-7-8 breathing technique: Inhale for 4s, hold for 7s, exhale for 8s.\n\n")
    f.write("Q: Why does the AI ask so many questions? A: The InnerAlign AI uses proactive probing to help you uncover subconscious triggers for your current feelings.\n\n")
    f.write("Q: What is Mind Clarity? A: Mind Clarity refers to your ability to focus and think without overwhelming mental 'noise' or overthinking.\n\n")
    f.write("Q: What should I do if I feel 'Emotionally Heavy'? A: Allow yourself to feel the weight without judgment. Often, acknowledging the heaviness is the first step toward lightness.\n\n")
    f.write("Q: How can I improve my energy levels? A: Small physical movements, hydration, and limiting 'screen fatigue' are immediate ways to reset energy.\n")


with open(data / "system_policy.md", "w", encoding='utf-8') as f:
    f.write("# InnerAlign Privacy & Usage Policy\n\n")
    f.write("Data Privacy: Your mood logs and chat history are encrypted and used only to provide personalized wellness insights.\n\n")
    f.write("Emergency Protocol: This AI is a wellness coach, not a medical professional. If you are in a crisis, please contact local emergency services immediately.\n\n")
    f.write("Data Accuracy: Drift scores are calculated based on your inputs; consistent logging leads to better accuracy.\n")

 
advice_data = {
    "Stress Management": [
        {"area": "Work Stress", "advice": "Break tasks into 25-minute blocks (Pomodoro). It reduces the feeling of being overwhelmed."},
        {"area": "Family Pressure", "advice": "Set clear boundaries. Saying 'I need 10 minutes' is a sign of strength, not neglect."}
    ],
    "Clarity Boosters": [
        {"area": "Overthinking", "advice": "Write down the three things bothering you. Transferring them to paper stops the mental loop."}
    ]
}
with open(data / "advice_database.json", "w") as f:
    json.dump(advice_data, f, indent=4)

print("Loading Wellness Knowledge Base...")
docs = []
docs.extend(TextLoader(data / "wellness_faq.md", encoding='utf-8').load())
docs.extend(TextLoader(data / "system_policy.md", encoding='utf-8').load())


import json
with open(data / "advice_database.json") as f:
    json_data = json.load(f)
    for category, items in json_data.items():
        for item in items:
            from langchain_core.documents import Document
            docs.append(Document(page_content=f"Category: {category}. Area: {item['area']}. Advice: {item['advice']}"))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)
print(f"Split {len(docs)} docs into {len(chunks)} chunks.")


print("Generating Embeddings ")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)
INDEX_PATH = 'document_index.faiss'
vectorstore.save_local(INDEX_PATH, index_name="document_index")

print(f"Successfully created Wellness Index at {INDEX_PATH}")