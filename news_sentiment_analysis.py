from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import textwrap
import streamlit as st

def fetech_articles(query, APIkey, num_days=30):
    articles = []
    for i in range(0,num_days,2):
        from_date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
        to_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')

        url = f"https://newsapi.org/v2/everything?q={query}&language=en&from={from_date}&to={to_date}&apiKey={APIkey}"
        res = requests.get(url)
        data = res.json()

        if "articles" in data:
            articles.extend(data['articles'])
        else:
            print(f"Error fetcing news from {from_date} to {to_date}")
            print(res)



    return articles


def preprocess_articles(articles):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    for article in articles:
        text = article['title'] + ". " + article['description'] if article['description'] else '' + article['content'][:-17] if article['content'] else ''
        article['text'] = text
        article['inputs'] = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    del tokenizer

    return articles


def sentiment_analysis(articles):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment").to(device)
    sentiment_map_reverse = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    with torch.inference_mode():
        for article in tqdm(articles):
            outputs = model(**article['inputs'].to(device), output_hidden_states=True)
            probs = torch.softmax(outputs.logits, dim=1)
            sentiment = probs.argmax().item()
            embedding = outputs.hidden_states[-1][0,0].to('cpu')
            article['sentiment'] = sentiment_map_reverse[sentiment]
            article['embedding'] = embedding
    return articles


def reduce_dim(articles):
    vectors_list = []
    for article in articles:
        vectors_list.append(article['embedding'])

    vectors = torch.stack(vectors_list)

    vectors_2d = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=3).fit_transform(vectors)

    for i, article in enumerate(articles):
        article['embedding_2d'] = vectors_2d[i]
    return articles


def plot_articles(articles, color='sentiment'):
    df = pd.DataFrame(columns=['title', 'publisher', 'x', 'y', 'description', 'sentiment', 'date'])
    for article in articles:
        df.loc[len(df)] = [
            '<br>'.join(textwrap.wrap(article['title'], width=80)),
            article['source']['name'],
            article['embedding_2d'][0].item(),
            article['embedding_2d'][1].item(),
            '<br>'.join(textwrap.wrap(article['description'], width=80)) if article['description'] else '',
            article['sentiment'],
            article['publishedAt'].split("T")[0]
        ]

    fig = px.scatter(
        df, x='x', y='y', color=color,
        hover_data=['title', 'publisher', 'description', 'sentiment'],
        width=1200,
        height=700,
    )
    fig.update_layout(
        yaxis=dict(scaleanchor='x', scaleratio=1) 
    )
    fig.show()


st.set_page_config(page_title="News Sentiment Visualizer", layout="wide")
st.title("News Sentiment Visualizer with Embeddings")

query = st.text_input("Enter a topic to search news for", "Iran Israel conflict")
api_key = st.text_input("Enter your NewsAPI key", type="password")
num_days = st.slider("Number of past days to search", 2, 30, 30, step=2)
color_option = st.selectbox("Colour points by", ["sentiment", "date", "publisher"])

st.markdown("### Optional: Upload your own JSON file of news articles")
st.markdown("""
            **Expected JSON format:**
            
            A list of article objects, each contraining at least:

            ```json
            {
                "title": "Headline...",
                "description": "Short summary...",
                "content": "Full text...",
                "source": { "name": "Publisher name" },
                "publishedAt": "YYYY-MM-DDTHH:MM:SSZ"
            }
""")

uploaded_file = st.file_uploader("Upload a JSON file", type="json")


if st.button("Fetch and Analyze News"):
    if uploaded_file:
        try:
            articles = pd.read_json(uploaded_file).to_dict(orient='records')
            st.success(f"Loaded {len(articles)} articles from uploaded JSON")
        except Exception as e:
            st.error(f"Failed to read uploaded JSON file: {e}")
            st.stop()


    else:
        if not api_key:
            st.error("Please provide a valid NewsAPI key.")
            st.stop()

        with st.spinner("Fatching articles..."):
            articles = fetech_articles(query, api_key, num_days)
        
        if not articles:
            st.warnings("No articles found. Try a different query or key.")
            st.stop()
        else:
            with st.spinner("Preprocessing.."):
                articles = preprocess_articles(articles)
            
            with st.spinner("Running sentiment analysis and embedding..."):
                articles = sentiment_analysis(articles)
            
            with st.spinner("Reducing embedding to 2D..."):
                articles = reduce_dim(articles)

            st.success(f"Fetched and processed {len(articles)} articles!")

            st.subheader("Embedding Visualiztion")
            df = pd.DataFrame(columns=['title', 'publisher', 'x', 'y', 'description', 'sentiment', 'date'])

            for article in articles:
                df.loc[len(df)] = [
                    '<br>'.join(textwrap.wrap(article['title'], width=80)),
                    article['source']['name'],
                    article['embedding_2d'][0].item(),
                    article['embedding_2d'][1].item(),
                    '<br>'.join(textwrap.wrap(article['description'], width=80)) if article['description'] else '',
                    article['sentiment'],
                    article['publishedAt'].split("T")[0]
                ]

            fig = px.scatter(
                df, x='x', y='y', color=color_option,
                hover_data=['title', 'publisher', 'description', 'sentiment', 'date'],
                width=1200,
                height=700,
            )
            fig.update_layout(yaxis=dict(scaleanchor='x', scaleratio=1))

            st.plotly_chart(fig, use_container_width=True)

            st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="news_articles.csv")