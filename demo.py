import cohere
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from annoy import AnnoyIndex
import numpy as np
import pandas as pd

from data import question
from data import text

from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enabling CORS(allowing sharing of data)) for all routes
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Spliting into a list of paragraphs
texts = text.split('\n')

# Cleaning up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts if t])

# Creating an instance of Cohere API client
co = cohere.Client('GRAB_YOUR_COHERE_API_KEY') # get your key API key after signing up for Cohere

# Get the embeddings
response = co.embed( texts=texts.tolist()).embeddings

# Checking the dimensions of embeddings
embeds = np.array(response)

# Creating the search index, passing the size of embedding
search_index = AnnoyIndex(embeds.shape[1], 'angular')

# Adding all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10) # Number of tress: 10
search_index.save('test.ann') #saving all the vector embeddings of our data inside a test.ann file


def search_text(query):
    # Getting the query's embedding
    query_embed = co.embed(texts=[query]).embeddings
    # Retrieving the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 10, include_distances=True)
    search_results = texts[similar_item_ids[0]] 
    return search_results


#creating prompt on the basis of context and question 
def ask_llm(question, num_generations=1):
    
    # Searching the text archive
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
 

#flask for backend of webapp

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