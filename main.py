# Import necessary libraries
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, precision_score, recall_score
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset from the provided URL
url = 'https://raw.githubusercontent.com/Heydeaddad/dataset/main/AB_NYC_2019.csv'
data = pd.read_csv(url)
print(data.columns)
print(data.head())

# Select the text column for analysis
text_column = 'name'

# Data Preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Apply preprocessing
data[f'cleaned_{text_column}'] = data[text_column].apply(preprocess_text)

print(data[[text_column, f'cleaned_{text_column}']].head())

# Descriptive Statistics
columns_to_describe = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
print(data[columns_to_describe].describe())

# Word Cloud for 'cleaned_name'
all_words_name = ' '.join([text for text in data[f'cleaned_{text_column}']])
wordcloud_name = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_name)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_name, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Listing Names')
plt.show()

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data[f'cleaned_{text_column}'].dropna()).toarray()

print(X.shape)

# Sentiment Analysis
sid = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if text == "":
        return 0
    sentiment = sid.polarity_scores(text)
    return sentiment['compound']

data['sentiment_score'] = data[f'cleaned_{text_column}'].apply(get_sentiment)
data['sentiment'] = data['sentiment_score'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

print(data[[f'cleaned_{text_column}', 'sentiment_score', 'sentiment']].head())

# Collaborative Filtering
# Use a subset of the dataset for faster processing
small_data = data[['host_id', 'id', 'price']].dropna().head(10000)
small_data['price'] = small_data['price'].astype(float)

# Load data into Surprise format
reader = Reader(rating_scale=(small_data['price'].min(), small_data['price'].max()))
surprise_data = Dataset.load_from_df(small_data, reader)
trainset, testset = train_test_split(surprise_data, test_size=0.2)

# User-based collaborative filtering
algo = KNNBasic()
algo.fit(trainset)

# Predictions
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)
print(f'User-based CF RMSE: {rmse}')

# Extract true and predicted labels for precision and recall calculation
testset_df = pd.DataFrame(testset, columns=['host_id', 'id', 'price'])
true_labels = testset_df['price'].apply(lambda x: 1 if x > 100 else 0).tolist()
pred_labels = [1 if pred.est >= 100 else 0 for pred in predictions]

precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)

print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Content-Based Filtering with Sampling
sampled_data = data.sample(1000, random_state=1)  # Sample 1000 listings for memory efficiency
sampled_data = sampled_data.reset_index(drop=True)  # Reset index to ensure proper alignment
tfidf_matrix_sampled = tfidf.fit_transform(sampled_data[f'cleaned_{text_column}'].dropna())
cosine_sim = cosine_similarity(tfidf_matrix_sampled, tfidf_matrix_sampled)

def recommend_listings_sampled(listing_id, cosine_sim=cosine_sim, data=sampled_data):
    idx = data[data['id'] == listing_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar listings
    listing_indices = [i[0] for i in sim_scores]
    return data['id'].iloc[listing_indices]

print(recommend_listings_sampled(sampled_data['id'].iloc[0]))  # Example listing_id

# Evaluation for Sentiment Analysis
# For this example, let's assume we have true sentiments
# Note: The dataset doesn't actually have sentiment labels, so this is a placeholder
true_sentiments = data['price'].apply(lambda x: 'positive' if x > 100 else 'negative')
print(classification_report(true_sentiments, data['sentiment']))

