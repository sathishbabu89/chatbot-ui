from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
import spacy
from pypdf import PdfReader
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to extract text from a PDF file
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    pdf_file = request.files['pdf_file']
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
    pdf_file.save(pdf_path)
    return redirect(url_for('chatbot_page', pdf_filename=pdf_file.filename))

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_answer():
    question = request.form['question']
    pdf_filename = request.args.get('pdf_filename')
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    pdf_text = extract_pdf_text(pdf_path)
    
    # Process the text
    doc = nlp(pdf_text)
    answer = qa_pipeline(question=question, context=pdf_text)['answer']
    
    return render_template('chatbot.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
