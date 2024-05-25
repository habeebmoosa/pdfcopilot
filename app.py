from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import time

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

index = None
query_engine = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global index, query_engine

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], "temp")
        os.makedirs(temp_dir, exist_ok=True)
        os.rename(filepath, os.path.join(temp_dir, filename))

        document = SimpleDirectoryReader(temp_dir).load_data()
        index = VectorStoreIndex.from_documents(document)
        query_engine = index.as_query_engine()

        os.remove(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)

        flash('File successfully uploaded and processed')
        return redirect(url_for('query'))
    else:
        flash('Allowed file types are pdf')
        return redirect(request.url)

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        query = request.form['query']
        start_time = time.time()
        response = query_engine.query(query)
        end_time = time.time()
        total_time = end_time - start_time

        return render_template('query.html', query=query, response=response, time_taken=total_time)

    return render_template('query.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
