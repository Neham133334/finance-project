import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import base64

# Set page config
st.set_page_config(
    page_title="ESG Analytics Dashboard",
    page_icon="🌱",
    layout="wide"
)

# Constants
DEFAULT_TICKER = "AAPL"
DEFAULT_DAYS = 365
MAX_NEWS_ARTICLES = 15  # Reduced for CPU efficiency

@st.cache_resource
def load_sentiment_model():
    """Load a lightweight sentiment analysis model"""
    try:
        model_name = "finiteautomata/bertweet-base-sentiment-analysis"  # Smaller model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return (model, tokenizer)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def get_esg_data(ticker):
    """Fetch ESG scores with error handling"""
    try:
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability
        return {
            'environment': esg_data.loc.get('environmentScore', {}).get('Value', 0) / 100,
            'social': esg_data.loc.get('socialScore', {}).get('Value', 0) / 100,
            'governance': esg_data.loc.get('governanceScore', {}).get('Value', 0) / 100
        } if esg_data is not None else None
    except:
        return None

def get_stock_data(ticker, days):
    """Fetch stock data with simplified error handling"""
    try:
        data = yf.download(
            ticker,
            start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            end=datetime.now().strftime('%Y-%m-%d')
        return data[['Close']].reset_index()
    except:
        return None

def analyze_sentiment(headlines, model_info):
    """Lightweight sentiment analysis"""
    if not model_info:
        return [{'label': 'NEUTRAL', 'score': 0}] * len(headlines)
    
    model, tokenizer = model_info
    inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return [
        {'label': model.config.id2label[pred.item()], 'score': prob.item()}
        for pred, prob in zip(torch.argmax(outputs.logits, dim=1), torch.nn.functional.softmax(outputs.logits, dim=1).max(dim=1).values)
    ]

def main():
    st.title("🌱 ESG Dashboard")
    
    # Sidebar controls
    ticker = st.sidebar.text_input("Stock Ticker", DEFAULT_TICKER).upper()
    days = st.sidebar.slider("Analysis Period (Days)", 30, 730, DEFAULT_DAYS)
    
    # Load model first
    with st.spinner("Loading AI model (first time may take ~60s)..."):
        model_info = load_sentiment_model()
    
    # Fetch data
    esg_scores = get_esg_data(ticker)
    stock_data = get_stock_data(ticker, days)
    
    # Display metrics
    if esg_scores:
        cols = st.columns(3)
        cols[0].metric("Environmental", f"{esg_scores['environment']:.2f}/1.0")
        cols[1].metric("Social", f"{esg_scores['social']:.2f}/1.0")
        cols[2].metric("Governance", f"{esg_scores['governance']:.2f}/1.0")
    
    # Stock chart
    if stock_data is not None:
        st.plotly_chart(px.line(stock_data, x='Date', y='Close'))
    
    # News analysis
    if st.sidebar.checkbox("Show News Analysis"):
        newsapi = NewsApiClient(api_key=st.secrets["NEWS_API_KEY"])
        news = newsapi.get_everything(
            q=ticker,
            page_size=MAX_NEWS_ARTICLES,
            language='en'
        )
        
        if news['articles']:
            headlines = [a['title'] for a in news['articles']]
            sentiments = analyze_sentiment(headlines, model_info)
            
            st.subheader("Recent News Sentiment")
            for headline, sentiment in zip(headlines, sentiments):
                st.write(f"{sentiment['label']}: {headline}")

if __name__ == "__main__":
    main()
