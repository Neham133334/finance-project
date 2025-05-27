import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Tuple
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="ESG Analytics Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_TICKER = "AAPL"
DEFAULT_DAYS = 365
SENTIMENT_WEIGHTS = {
    'positive': 1,
    'neutral': 0,
    'negative': -1
}
ESG_WEIGHTS = {
    'environment': 0.3,
    'social': 0.3,
    'governance': 0.2,
    'sentiment': 0.2
}

@st.cache_resource
def load_sentiment_model():
    """Load a lighter sentiment analysis model for CPU"""
    try:
        # Using a smaller model for CPU efficiency
        model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Ensure model is in CPU mode
        model = model.to('cpu')
        model.eval()
        
        return (model, tokenizer)
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

def analyze_sentiment(text: str, model_info) -> Dict:
    """Perform sentiment analysis on text"""
    if not model_info:
        return {'label': 'neutral', 'score': 0.0}
    
    model, tokenizer = model_info
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}  # Ensure on CPU
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_label = torch.max(probs, dim=1)
        
        label = model.config.id2label[top_label.item()]
        score = top_prob.item()
        
        return {'label': label, 'score': score}
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return {'label': 'neutral', 'score': 0.0}

def get_esg_data(ticker: str) -> Dict[str, float]:
    """Fetch ESG scores from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability
        
        if esg_data is None or esg_data.empty:
            return None
            
        esg_scores = {
            'environment': esg_data.loc['environmentScore', 'Value'] / 100 if 'environmentScore' in esg_data.index else 0,
            'social': esg_data.loc['socialScore', 'Value'] / 100 if 'socialScore' in esg_data.index else 0,
            'governance': esg_data.loc['governanceScore', 'Value'] / 100 if 'governanceScore' in esg_data.index else 0,
            'total': esg_data.loc['totalEsg', 'Value'] / 100 if 'totalEsg' in esg_data.index else 0
        }
        return esg_scores
    except Exception as e:
        st.error(f"Error fetching ESG data: {e}")
        return None

def get_stock_data(ticker: str, days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """Fetch historical stock price data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock_data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if stock_data.empty:
            return None
            
        stock_data = stock_data[['Close']].reset_index()
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data['Pct Change'] = stock_data['Close'].pct_change() * 100
        stock_data['Cumulative Return'] = (1 + stock_data['Pct Change'] / 100).cumprod() - 1
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def get_company_name(ticker: str) -> str:
    """Get company name from ticker"""
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('longName', ticker)
    except:
        return ticker

def get_news_sentiment(ticker: str, company_name: str, model_info, days: int = 30) -> Tuple[pd.DataFrame, float]:
    """Fetch news and perform sentiment analysis"""
    try:
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=st.secrets["NEWS_API_KEY"])
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch news (limit to 20 articles to reduce processing load)
        query = f"{company_name} OR {ticker}"
        news = newsapi.get_everything(
            q=query,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=20  # Reduced number of articles for CPU efficiency
        )
        
        if not news or not news.get('articles'):
            return pd.DataFrame(), 0.0
            
        # Process articles
        articles = news['articles']
        headlines = [article['title'] for article in articles]
        sources = [article['source']['name'] for article in articles]
        dates = [article['publishedAt'] for article in articles]
        urls = [article['url'] for article in articles]
        
        # Perform sentiment analysis
        sentiments = []
        scores = []
        
        for headline in headlines:
            result = analyze_sentiment(headline, model_info)
            sentiments.append(result['label'].lower())
            scores.append(result['score'])
        
        # Create DataFrame
        news_df = pd.DataFrame({
            'Date': dates,
            'Source': sources,
            'Headline': headlines,
            'Sentiment': sentiments,
            'Score': scores,
            'URL': urls
        })
        
        # Convert dates
        news_df['Date'] = pd.to_datetime(news_df['Date'])
        
        # Calculate average sentiment score
        news_df['Weighted_Sentiment'] = news_df.apply(
            lambda x: SENTIMENT_WEIGHTS.get(x['Sentiment'], 0) * x['Score'], axis=1
        )
        avg_sentiment = news_df['Weighted_Sentiment'].mean()
        
        return news_df, avg_sentiment
    except Exception as e:
        st.error(f"Error fetching or analyzing news: {e}")
        return pd.DataFrame(), 0.0

def calculate_composite_score(esg_scores: Dict[str, float], sentiment_score: float) -> float:
    """Calculate weighted composite ESG + Sentiment score"""
    if not esg_scores:
        return 0.0
    
    # Normalize sentiment score from [-1, 1] to [0, 1]
    normalized_sentiment = (sentiment_score + 1) / 2
    
    composite_score = (
        esg_scores['environment'] * ESG_WEIGHTS['environment'] +
        esg_scores['social'] * ESG_WEIGHTS['social'] +
        esg_scores['governance'] * ESG_WEIGHTS['governance'] +
        normalized_sentiment * ESG_WEIGHTS['sentiment']
    )
    
    return composite_score

def display_metrics(esg_scores: Dict[str, float], sentiment_score: float, composite_score: float):
    """Display key metrics in columns"""
    if not esg_scores:
        st.warning("No ESG data available for this company")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Environmental Score", f"{esg_scores['environment']:.2f}")
    with col2:
        st.metric("Social Score", f"{esg_scores['social']:.2f}")
    with col3:
        st.metric("Governance Score", f"{esg_scores['governance']:.2f}")
    with col4:
        # Display sentiment with color
        sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "gray"
        st.metric("News Sentiment", f"{sentiment_score:.2f}", delta=None, delta_color="off")
    with col5:
        st.metric("Composite ESG+ Score", f"{composite_score:.2f}")

def plot_stock_performance(stock_data: pd.DataFrame):
    """Plot stock performance charts"""
    if stock_data is None or stock_data.empty:
        st.warning("No stock data available")
        return
    
    fig1 = px.line(
        stock_data,
        x='Date',
        y='Close',
        title='Stock Price Trend',
        labels={'Close': 'Price (USD)'}
    )
    
    fig2 = px.line(
        stock_data,
        x='Date',
        y='Cumulative Return',
        title='Cumulative Returns',
        labels={'Cumulative Return': 'Return (%)'}
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

def display_news_sentiment(news_df: pd.DataFrame):
    """Display news sentiment analysis results"""
    if news_df.empty:
        st.warning("No news articles found")
        return
    
    # Add color mapping for sentiment
    sentiment_color_map = {
        'positive': 'green',
        'neutral': 'gray',
        'negative': 'red'
    }
    news_df['Sentiment_Color'] = news_df['Sentiment'].map(sentiment_color_map)
    
    # Display sentiment distribution
    fig = px.pie(
        news_df,
        names='Sentiment',
        title='Sentiment Distribution',
        color='Sentiment',
        color_discrete_map=sentiment_color_map
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display news table with filtering options
    st.subheader("Recent News Headlines")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        date_filter = st.selectbox(
            "Filter by date range:",
            options=["Last 7 days", "Last 30 days", "All"],
            index=1
        )
    with col2:
        sentiment_filter = st.multiselect(
            "Filter by sentiment:",
            options=["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"]
        )
    
    # Apply filters
    filtered_df = news_df.copy()
    if date_filter == "Last 7 days":
        cutoff_date = datetime.now() - timedelta(days=7)
        filtered_df = filtered_df[filtered_df['Date'] >= cutoff_date]
    elif date_filter == "Last 30 days":
        cutoff_date = datetime.now() - timedelta(days=30)
        filtered_df = filtered_df[filtered_df['Date'] >= cutoff_date]
    
    if sentiment_filter:
        filtered_df = filtered_df[filtered_df['Sentiment'].isin(sentiment_filter)]
    
    # Display table
    st.dataframe(
        filtered_df[['Date', 'Source', 'Headline', 'Sentiment', 'Score']].sort_values('Date', ascending=False),
        column_config={
            "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
            "Score": st.column_config.ProgressColumn(
                "Confidence",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Sentiment": st.column_config.TextColumn(
                "Sentiment",
                help="Sentiment of the news headline",
            ),
            "Headline": st.column_config.TextColumn(
                "Headline",
                help="News headline",
                width="large"
            ),
        },
        hide_index=True,
        use_container_width=True
    )

def get_table_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

def main():
    """Main Streamlit application"""
    st.title("🌱 ESG Analytics Dashboard")
    st.markdown("""
        Analyze Environmental, Social, and Governance (ESG) scores combined with financial 
        performance and news sentiment for public companies.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        ticker = st.text_input("Enter Stock Ticker", DEFAULT_TICKER).upper()
        days = st.slider("Historical Days for Stock Data", 30, 730, DEFAULT_DAYS)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
            This dashboard provides:
            - ESG scores from Yahoo Finance
            - Stock performance metrics
            - News sentiment analysis using financial NLP
            - Composite ESG+ score combining all factors
        """)
        st.markdown("""
            **Optimized for CPU deployment** with lighter models.
        """)
    
    # Load data with spinners
    with st.spinner("Loading sentiment model (this may take a minute)..."):
        model_info = load_sentiment_model()
    
    with st.spinner("Fetching company data..."):
        company_name = get_company_name(ticker)
    
    st.header(f"{company_name} ({ticker}) Analysis")
    
    with st.spinner("Fetching ESG scores..."):
        esg_scores = get_esg_data(ticker)
    
    with st.spinner("Fetching stock data..."):
        stock_data = get_stock_data(ticker, days)
    
    with st.spinner("Analyzing news sentiment..."):
        news_df, avg_sentiment = get_news_sentiment(ticker, company_name, model_info)
    
    # Calculate composite score
    composite_score = 0.0
    if esg_scores:
        composite_score = calculate_composite_score(esg_scores, avg_sentiment)
    
    # Display metrics
    display_metrics(esg_scores, avg_sentiment, composite_score)
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Financial Performance", "News Sentiment", "Export Data"])
    
    with tab1:
        st.subheader("Financial Performance")
        plot_stock_performance(stock_data)
    
    with tab2:
        st.subheader("News Sentiment Analysis")
        display_news_sentiment(news_df)
    
    with tab3:
        st.subheader("Export Data")
        
        if esg_scores and stock_data is not None and not news_df.empty:
            # Create combined ESG data
            esg_df = pd.DataFrame.from_dict(esg_scores, orient='index', columns=['Value'])
            esg_df.reset_index(inplace=True)
            esg_df.columns = ['Metric', 'Value']
            
            # Create sentiment summary
            sentiment_summary = news_df['Sentiment'].value_counts().reset_index()
            sentiment_summary.columns = ['Sentiment', 'Count']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ESG Data")
                st.dataframe(esg_df)
                st.markdown(get_table_download_link(esg_df, f"{ticker}_esg_data.csv"), unsafe_allow_html=True)
                
                st.markdown("### Stock Data")
                st.dataframe(stock_data)
                st.markdown(get_table_download_link(stock_data, f"{ticker}_stock_data.csv"), unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Sentiment Summary")
                st.dataframe(sentiment_summary)
                st.markdown(get_table_download_link(sentiment_summary, f"{ticker}_sentiment_summary.csv"), unsafe_allow_html=True)
                
                st.markdown("### News Data")
                st.dataframe(news_df)
                st.markdown(get_table_download_link(news_df, f"{ticker}_news_data.csv"), unsafe_allow_html=True)
        else:
            st.warning("No data available to export")

if __name__ == "__main__":
    main() 
