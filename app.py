import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# 初始化VADER情感分析器
analyzer = SentimentIntensityAnalyzer()

# 预加载分类模型和向量化器
model = MultinomialNB()
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# 简单的UI界面
st.title("NLP Text Analysis Tool")

uploaded_file = st.file_uploader("Upload a text file", type="txt")

if uploaded_file is not None:
    # 读取文件内容
    text = uploaded_file.read().decode('utf-8')
    
    # 显示原始文本
    st.subheader("Original Text")
    st.write(text)
    
    # 文本预处理（假设之前的模型和向量器已训练好）
    # 在这里我们应该使用已经训练好的tfidf_vectorizer和model，但为了简化例子，我们使用新向量化器。
    processed_text = ' '.join(text.split())  # 简单预处理，去除多余空格等
    X_input = tfidf_vectorizer.fit_transform([processed_text]).toarray()
    
    # 文本分类
    prediction = model.predict(X_input)
    st.subheader("Text Classification")
    st.write(f"Predicted Category: {prediction[0]}")
    
    # 情感分析
    sentiment_score = analyzer.polarity_scores(text)['compound']
    st.subheader("Sentiment Analysis")
    st.write(f"Sentiment Score: {sentiment_score} ({'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'})")
    
    # 关键词提取
    tfidf = TfidfVectorizer(max_features=10)
    keywords = tfidf.fit([text]).get_feature_names_out()
    st.subheader("Keywords")
    st.write(', '.join(keywords))
