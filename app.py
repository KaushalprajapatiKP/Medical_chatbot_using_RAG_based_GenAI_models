from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat_temp.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        # Get the input message
        msg = request.form.get("msg", "").strip().lower()

        # Handle simple greetings
        if msg in ["hey", "hello", "hi"]:
            return jsonify({"response": "Hello! How can I assist you today?"})

        # Invoke the RAG chain
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "I'm sorry, I couldn't find an answer to your question.")
        
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 5000, debug= True)