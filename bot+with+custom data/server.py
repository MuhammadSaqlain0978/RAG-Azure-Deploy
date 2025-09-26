# server.py
import os
import threading
import shutil
import tempfile
import zipfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from rag_bot import UniversityRAGBot  # put your class in rag_bot.py

# Config from env
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
AZ_CONN_STR = os.environ.get("DefaultEndpointsProtocol=https;AccountName=botstorageaccountname1;AccountKey=0uszWr82Az7C+K5dAmkfRCSd3qi0/UziFuuUx8gU3v7OJ84B4ejflJwiLIU3BDhl2VshBVvzd5a++AStcq6Puw==;EndpointSuffix=core.windows.net")  # or use SAS
BLOB_CONTAINER = os.environ.get("AZURE_BLOB_CONTAINER", "vectorstores")
BLOB_NAME = os.environ.get("AZURE_BLOB_NAME", "vector_store.zip")
DATABASE_FOLDER = os.environ.get("DATABASE_FOLDER", "database")
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "./database_vector_store")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "changeme")  # simple auth for reindex/upload

app = Flask(__name__)
CORS(app)
bot = None
blob_service = None

def init_blob_client():
    global blob_service
    if AZ_CONN_STR:
        blob_service = BlobServiceClient.from_connection_string(AZ_CONN_STR)
        # create container if not exists
        try:
            blob_service.create_container(BLOB_CONTAINER)
        except Exception:
            pass

def upload_vectorstore_to_blob(local_folder, container=BLOB_CONTAINER, blob_name=BLOB_NAME):
    # zip folder and upload
    base = tempfile.mkdtemp()
    zip_path = os.path.join(base, "vector_store.zip")
    shutil.make_archive(zip_path.replace(".zip",""), 'zip', local_folder)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
    with open(zip_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

def download_and_extract_blob(dest_folder, container=BLOB_CONTAINER, blob_name=BLOB_NAME):
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
    try:
        stream = blob_client.download_blob().readall()
    except Exception:
        return False
    base = tempfile.mkdtemp()
    zip_path = os.path.join(base, "vector_store.zip")
    with open(zip_path, "wb") as f:
        f.write(stream)
    # extract
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_folder)
    return True

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/query", methods=["POST"])
def query():
    global bot
    if not bot or not bot.vector_store:
        return jsonify({"error":"vector store not loaded"}), 400
    data = request.get_json()
    question = data.get("question") if data else None
    if not question:
        return jsonify({"error":"no question provided"}), 400
    result = bot.answer_query(question)
    return jsonify(result)

@app.route("/reindex", methods=["POST"])
def reindex():
    key = request.headers.get("x-admin-key")
    if key != ADMIN_KEY:
        return jsonify({"error":"unauthorized"}), 401
    try:
        bot.load_university_data(database_folder=DATABASE_FOLDER, vector_store_path=VECTOR_STORE_PATH)
        # after building, upload to blob for persistence
        if blob_service:
            upload_vectorstore_to_blob(VECTOR_STORE_PATH)
        return jsonify({"status":"ok", "message":"reindex complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    key = request.headers.get("x-admin-key")
    if key != ADMIN_KEY:
        return jsonify({"error":"unauthorized"}), 401
    if 'file' not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    f = request.files['file']
    save_path = os.path.join(DATABASE_FOLDER, f.filename)
    os.makedirs(DATABASE_FOLDER, exist_ok=True)
    f.save(save_path)
    # optionally reindex synchronously (or return and let user call /reindex)
    try:
        bot.load_university_data(database_folder=DATABASE_FOLDER, vector_store_path=VECTOR_STORE_PATH)
        if blob_service:
            upload_vectorstore_to_blob(VECTOR_STORE_PATH)
    except Exception as e:
        return jsonify({"warning":"file uploaded but reindex failed", "error": str(e)}), 202
    return jsonify({"status":"uploaded and reindexed"})

def create_bot_and_load():
    global bot
    init_blob_client()
    bot = UniversityRAGBot(api_key=MISTRAL_API_KEY)
    # Try restore vector store from blob
    if AZ_CONN_STR:
        try:
            ok = download_and_extract_blob(VECTOR_STORE_PATH)
            if ok:
                # load local vector store
                bot.load_university_data(database_folder=DATABASE_FOLDER, vector_store_path=VECTOR_STORE_PATH)
                print("Loaded vector store from blob.")
                return
        except Exception as e:
            print("No vector store in blob or failed to download:", e)
    # fallback: build locally (will upload later)
    try:
        bot.load_university_data(database_folder=DATABASE_FOLDER, vector_store_path=VECTOR_STORE_PATH)
        if AZ_CONN_STR:
            upload_vectorstore_to_blob(VECTOR_STORE_PATH)
    except Exception as e:
        print("Warning: initial load failed:", e)

if __name__ == "__main__":
    create_bot_and_load()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
