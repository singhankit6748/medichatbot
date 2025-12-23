# app.py
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os, traceback

# light imports only at top-level
from src.helper import download_hugging_face_embeddings  # keep but we will call it lazily
from src.prompt import *  # system_prompt etc.

app = Flask(__name__)
load_dotenv()

# env keys (may be None)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# set envs if present (safe but not required)
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Try to import optional Groq integration (won't crash if missing)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    ChatGroq = None
    GROQ_AVAILABLE = False
    # log import error for debugging on Render if needed
    with open("/tmp/groq_import_error.txt", "w") as f:
        f.write("langchain_groq import failed:\n")
        f.write(traceback.format_exc())

# We'll lazy-initialize these globals on first request
_rag_chain = None
_retriever = None
_chat_model = None

def get_rag_chain():
    """
    Initialize embeddings, pinecone index, retriever, chatModel and chain lazily.
    Returns a rag_chain-like object with .invoke({"input": ...}).
    """
    global _rag_chain, _retriever, _chat_model

    if _rag_chain is not None:
        return _rag_chain

    # Basic env checks
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set in environment.")
    # If you want Groq to be optional, we handle it below.

    # Import heavy libraries only when needed
    try:
        # embeddings (this is heavy)
        embeddings = download_hugging_face_embeddings()
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings: {e}")

    try:
        # Import Pinecone store lazily
        from langchain_pinecone import PineconeVectorStore
    except Exception as e:
        raise RuntimeError(f"Failed to import PineconeVectorStore: {e}")

    # Pinecone index usage
    index_name = "medical-chatbot"
    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Pinecone index '{index_name}': {e}")

    _retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Import chain builders lazily
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as e:
        raise RuntimeError(f"Failed to import LangChain chain builders: {e}")

    # Create or fallback for chat model
    if GROQ_AVAILABLE and ChatGroq is not None:
        # If GROQ available, use it
        try:
            _chat_model = ChatGroq(
                model="llama-3.3-70b-versatile",  # change if needed
                api_key=GROQ_API_KEY,
                temperature=0
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatGroq: {e}")
    else:
        # Groq is not available — fall back or raise informative error.
        raise RuntimeError("Groq model not available in this deployment (langchain_groq missing).")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(_chat_model, prompt)
    _rag_chain = create_retrieval_chain(_retriever, question_answer_chain)

    return _rag_chain


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify(status="ok"), 200


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    """
    Main endpoint: ensure the RAG chain is initialized lazily.
    If initialization fails, return a 500 with useful message.
    """
    msg = request.form.get("msg") or request.args.get("msg") or ""
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    try:
        rag_chain = get_rag_chain()
    except RuntimeError as e:
        # Return helpful message for missing deps / env vars
        return jsonify({"error": "Initialization failed", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Unexpected initialization error", "details": str(e)}), 500

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer") if isinstance(response, dict) else str(response)
    except Exception as e:
        # Log to /tmp for debugging if needed
        with open("/tmp/rag_runtime_error.txt", "w") as f:
            f.write(traceback.format_exc())
        return jsonify({"error": "Chain execution failed", "details": str(e)}), 500

    return str(answer)


if __name__ == '__main__':
    # Use $PORT if provided (Render sets this env var)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
