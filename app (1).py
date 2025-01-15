# @title website run

from flask import Flask, request, render_template
from pyngrok import ngrok
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

NGROK_AUTHTOKEN = "2e5HfihdTLSWD2nmcoXXDzjxib9_7TVpQakJDni6zHKDv4ag7"
ngrok.set_auth_token(NGROK_AUTHTOKEN)

word_embeddings = {}
with open('/content/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs

def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def sentence_vector_func(sentences_cleaned):
    sentence_vector = []
    for i in sentences_cleaned:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vector.append(v)
    return sentence_vector

def summary_text_with_tfidf(test_text):
    sentences = sent_tokenize(test_text)

    # Calculate target length (50% of original)
    total_chars = len(test_text)
    target_chars = total_chars // 2

    # Clean sentences and remove stopwords
    clean_sentences = pd.Series(sentences).str.replace("[^a-z A-Z 0-9]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_sentences)

    sim_mat = cosine_similarity(tfidf_matrix, tfidf_matrix)
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    # Rank sentences by their PageRank scores
    ranked_sentences = sorted(((scores[i], s, len(s)) for i, s in enumerate(sentences)), reverse=True)

    # Select sentences until we reach target length
    summarised_string = ""
    current_length = 0

    for _, sentence, sent_len in ranked_sentences:
        if current_length + sent_len <= target_chars:
            summarised_string += sentence + " "
            current_length += sent_len
        else:
            break

    return summarised_string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        summary = summary_text_with_tfidf(text)
        return render_template('index.html', summary=summary, original_text=text)

if __name__ == '__main__':
    try:
        public_url = ngrok.connect(5000)
        print(f"Public URL: {public_url}")
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error: {e}")
        ngrok.kill()