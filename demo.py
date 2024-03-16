import cohere
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from annoy import AnnoyIndex
import numpy as np
import pandas as pd

from data import question
from data import text

#integration

from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config["TEMPLATES_AUTO_RELOAD"] = True

#->
# Split into a list of paragraphs
texts = text.split('\n')
# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts if t])
co = cohere.Client('3RBqyUGXWqQ6ClruJ9VibxRUtVtopwzQ1tzfvLmf')

# Get the embeddings
response = co.embed( texts=texts.tolist()).embeddings

# Check the dimensions of the embeddings
embeds = np.array(response)

# Create the search index, pass the size of embedding
search_index = AnnoyIndex(embeds.shape[1], 'angular')

# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10) # 10 trees
search_index.save('test.ann')

def search_text(query):
    # Get the query's embedding
    query_embed = co.embed(texts=[query]).embeddings
    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 10, include_distances=True)
    search_results = texts[similar_item_ids[0]]
    return search_results

#creates prompt on the basis of context and question 
def ask_llm(question, num_generations=1):
    # Search the text archive
    results = search_text(question)

    # Get the top result
    context = results[0]

    # Prepare the prompt
    prompt = f"""
    Information regarding tenants rights: 
    {context}
    
    Question: {question}

    Please extract the relevant information from the provided data. Properly address the question first and then if needed add more related responses.
    Ensure the response is contained within a single paragraph. If there's irrelevant question or if the data doesn't contain the answer, reply that the information is not available within the provided context."""

    prediction = co.generate(
        prompt=prompt,
        max_tokens=120,
        model="command-nightly",
        temperature=0.5,
        num_generations=num_generations
    )
    return prediction.generations

# question = "can my landlord come and tell me to leave?"
# results = ask_llm(question,)
# print(results[0])

# question = input("Enter your question....")
# results = ask_llm(question,)
# print(results[0])

#
 
IMG_FOLDER = os.path.join("./static", "img")
app.config["UPLOAD_FOLDER"] = IMG_FOLDER

@app.route('/')
def index():
    Main_ = os.path.join(app.config["UPLOAD_FOLDER"], "Default.svg")
    Query_ = os.path.join(app.config["UPLOAD_FOLDER"], "Vector.svg")
    Microphone_ = os.path.join(app.config["UPLOAD_FOLDER"], "microphone.svg")
    Flash_ = os.path.join(app.config["UPLOAD_FOLDER"], "flash.svg")
    return render_template("main.html", answer = None, headingImg=Main_, shiningStar=Query_, microPhone=Microphone_, flash = Flash_)

@app.route('/lawyer', methods=['POST'])
def process_text():
    text_input = request.form.get('text')
    print("User input:", text_input)
    answer = ask_llm(text_input)
    Main_ = os.path.join(app.config["UPLOAD_FOLDER"], "Default.svg")
    Query_ = os.path.join(app.config["UPLOAD_FOLDER"], "Vector.svg")
    Microphone_ = os.path.join(app.config["UPLOAD_FOLDER"], "microphone.svg")
    Flash_ = os.path.join(app.config["UPLOAD_FOLDER"], "flash.svg")
    return render_template("main.html", answer = answer[0], question = text_input, headingImg=Main_, shiningStar=Query_, microPhone=Microphone_, flash = Flash_)

if __name__ == '__main__':
    app.run(host='localhost', port=5172)  # Running Flask app on port 5172