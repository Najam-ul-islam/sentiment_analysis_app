import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
# from collections import Counter

class TweetSentimentApp:
    def __init__(self):
        self.model = load_model('lstm_sentiment_model.h5')
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(['example text'])
        self.sent_analyzer = SentimentIntensityAnalyzer()
        self.positive_words = self.read_word_list('positive_words.txt')
        self.negative_words = self.read_word_list('negative_words.txt')
        self.neutral_words = self.read_word_list('neutral_words.txt')
        self.agitative_words = set(open('agitative_words.txt').read().splitlines())
        self.max_sequence_length = 100

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove mentions
        text = re.sub(r'#', '', text)  # remove hashtags
        text = re.sub(r'RT[\s]+', '', text)  # remove retweets
        text = re.sub(r'https?:\/\/\S+', '', text)  # remove links
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # remove special characters
       
        return " ".join(nltk.word_tokenize(text.lower().strip()))
        

    def analyze_sentiment(self, text):
        
        sid = SentimentIntensityAnalyzer()
        sentiment_score = sid.polarity_scores(text)['compound']
        sentiment_tag = 'positive' if sentiment_score > 0 else ('negative' if sentiment_score < 0 else 'neutral')
        
        return sentiment_score, sentiment_tag

    def predict_sentiment(self, text):
        
        if isinstance(text, str):
            text = str(text)
            sequences = self.tokenizer.texts_to_sequences([text])
            if sequences:
                padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
                predicted_sentiment = self.model.predict(padded_sequences)[0][0]
                return predicted_sentiment
      
        return "N/A", None

    def sentiment_label(self, sentiment_score):

        if sentiment_score > 0.2:
            return 'positive'
        elif sentiment_score < -0.2:
            return 'negative'
        else:
            return 'neutral'

    def get_sentiment_score(self, tweet):
        
        if isinstance(tweet, str):
            return self.sent_analyzer.polarity_scores(tweet)['compound']
        return 0.0  # Default score for non-text values
    def calculate_positive_sentiment_percentage(self, dataframe):
        
        # Apply sentiment analysis to each tweet
        dataframe['sentiment_score'] = dataframe['tweets'].apply(lambda tweet: self.get_sentiment_score(tweet))

        # Categorize sentiment into tags
        dataframe['sentiment_tag'] = dataframe['sentiment_score'].apply(self.sentiment_label)

        # Group by user name and sentiment, and count the number of tweets for each combination
        sentiment_counts = dataframe.groupby(['username', 'sentiment_tag']).size().reset_index(name='count')

        # Pivot the sentiment counts dataframe to have sentiments as columns
        sentiment_pivot = sentiment_counts.pivot(index='username', columns='sentiment_tag', values='count').fillna(0)

        # Calculate the percentage of positive sentiment for each user
        sentiment_pivot['total_tweets'] = sentiment_pivot.sum(axis=1)
        sentiment_pivot['positive_percentage'] = (sentiment_pivot['positive'] / sentiment_pivot['total_tweets']) * 100
        
        return sentiment_pivot[['positive_percentage']]
    
    def plot_sentiments_by_politician(self, df):
        
        politician_sentiments = df.groupby('username')['sentiment_tag'].value_counts(normalize=True).unstack().fillna(0)
        # Plot bar chart for sentiments
        politician_sentiments.plot(kind='bar', figsize=(10, 6))
        plt.xlabel('Politician')
        plt.ylabel('Percentage of Sentiments')
        plt.title('Sentiments by Politician')
        plt.xticks(rotation=45)
        st.pyplot()
        
    def find_most_active_politician(self, df):

        politician_tweet_counts = df['username'].value_counts()
        most_active_politician = politician_tweet_counts.idxmax()
        num_tweets = politician_tweet_counts.max()
        return most_active_politician, num_tweets
    # =============================================================================================
    def read_word_list(self, filename):
        
        with open(filename, 'r') as file:
            
            return [line.lower().strip() for line in file]

    # =============================================================================================
    def analyze_tweets_sentiment(self, dataframe):

        mentions = []

        for index, row in dataframe.iterrows():
            username = row['username']
            tweet = str(row['tweets'])  # Convert to string to handle potential non-string values

            blob = TextBlob(tweet)
            for word in blob.words:
                if word in self.positive_words:
                    mentions.append((username, word, 'positive'))
                elif word in self.negative_words:
                    mentions.append((username, word, 'negative'))
                elif word in self.neutral_words:
                    mentions.append((username, word, 'neutral'))
                # elif word in self.agitative_words:
                #     mentions.append((username, word, 'agitative'))
        
        return mentions
    
    def sentiment_count(self, dataframe):
        
        # Group the dataframe by user name and count the number of tweets
        tweet_count_df = dataframe.groupby('sentiment_tag')['tweets'].count().reset_index()
        
        # Find the user with the highest tweet count
        user_with_highest_tweet_count = tweet_count_df.loc[tweet_count_df['tweets'].idxmax()]
        
        # Print the user name and highest tweet count
        user_name = user_with_highest_tweet_count['sentiment_tag']
        tweet_count = user_with_highest_tweet_count['tweets']
        print(f"Overall Sentiment Count: {user_name}, Tweet Count: {tweet_count}")
        
        # Plot tweet count of each leader
        plt.bar(tweet_count_df['sentiment_tag'], tweet_count_df['tweets'], color=['red', 'green', 'blue'], width=0.5)
        plt.xlabel('Sentiment')
        plt.ylabel('Sentiment Count')
        plt.title('Overall Sentiment Count')
        plt.xticks(rotation=45)
        # Show the tweet count on top of each bar
        for i, count in enumerate(tweet_count_df['tweets']):
            plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot()
    # =============================================================================================

    def plot_word_frequency_by_user(self, df):
        user_word_frequency = {}

        for index, row in df.iterrows():
            username = row['username']
            tweet = str(row['tweets'])  # Convert to string to handle potential non-string values

            blob = TextBlob(tweet)
            for word in blob.words:
                if word in self.positive_words or word in self.negative_words or word in self.neutral_words:
                    sentiment = ''
                    if word in self.positive_words:
                        sentiment = 'positive'
                    elif word in self.negative_words:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                        
                    if username not in user_word_frequency:
                        user_word_frequency[username] = {
                            'positive': {},
                            'negative': {},
                            'neutral': {}
                        }
                    if word not in user_word_frequency[username][sentiment]:
                        user_word_frequency[username][sentiment][word] = 0
                    user_word_frequency[username][sentiment][word] += 1

        for user, sentiment_freq in user_word_frequency.items():
            for sentiment, word_freq in sentiment_freq.items():
                self.plot_word_frequency(user, word_freq, sentiment)
    
    def plot_word_frequency(self, user, word_freq, sentiment):
        word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
        plt.figure(figsize=(10, 6))
        plt.bar(word_freq_df['Word'], word_freq_df['Frequency'], color='green')
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.title(f'{sentiment.title()} Word Frequency for User: {user}')
        plt.xticks(rotation=45)
        st.pyplot()
    # =============================================================================================
    def calculate_sentiment_percentages(self, df):
        sentiment_percentages = {}

        for index, row in df.iterrows():
            username = row['username']
            tweet = str(row['tweets'])  # Convert to string to handle potential non-string values

            blob = TextBlob(tweet)
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for word in blob.words:
                if word in self.positive_words:
                    positive_count += 1
                elif word in self.negative_words:
                    negative_count += 1
                elif word in self.neutral_words:
                    neutral_count += 1

            if username not in sentiment_percentages:
                sentiment_percentages[username] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }

            total_words = positive_count + negative_count + neutral_count
            if total_words > 0:
                sentiment_percentages[username]['positive'] += (positive_count / total_words)
                sentiment_percentages[username]['negative'] += (negative_count / total_words)
                sentiment_percentages[username]['neutral'] += (neutral_count / total_words)

        return sentiment_percentages
# =============================================================================================
    def find_most_patriotic_politician(self, df):
        politician_sentiment_count = {}

        for index, row in df.iterrows():
            username = row['username']
            sentiment = row['sentiment_tag']

            if sentiment == 'positive':
                if username not in politician_sentiment_count:
                    politician_sentiment_count[username] = 0
                politician_sentiment_count[username] += 1

        if politician_sentiment_count:
            most_patriotic_politician = max(politician_sentiment_count, key=politician_sentiment_count.get)
            return most_patriotic_politician, politician_sentiment_count[most_patriotic_politician]
        else:
            return None, None
# =============================================================================================
    def find_user_with_highest_tweet_count(self, dataframe):
        # Group the dataframe by user name and count the number of tweets
        tweet_count_df = dataframe.groupby('username')['tweets'].count().reset_index()
        
        # Find the user with the highest tweet count
        user_with_highest_tweet_count = tweet_count_df.loc[tweet_count_df['tweets'].idxmax()]
        
        # Print the user name and highest tweet count
        user_name = user_with_highest_tweet_count['username']
        tweet_count = user_with_highest_tweet_count['tweets']
        st.write(f"#### Highest Tweet Count: {user_name}, Tweet Count: {tweet_count}")
        
        # Plot tweet count of each leader
        plt.bar(tweet_count_df['username'], tweet_count_df['tweets'], color=['red', 'green', 'blue'], width=0.3)
        plt.xlabel('Politician')
        plt.ylabel('Tweet Count')
        plt.title('Tweet Count of Each Leader')
        plt.xticks(rotation=45)
        # Show the tweet count on top of each bar
        for i, count in enumerate(tweet_count_df['tweets']):
            plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot()
# =============================================================================================
    def detect_agitation(self, tweet):
        # Search for agitation keywords in the tweet text
        for word in self.agitative_words:
            if re.search(r'\b' + word + r'\b', tweet, re.IGNORECASE):
                return True
        return False
    #     with open(filename, 'r') as file:
    #         return [line.lower().strip() for line in file]
        

    def analyze_tweets_for_agitation(self, dataframe):
        agitating_politicians = {}

        for index, row in dataframe.iterrows():
            username = row['username']
            tweet = row['tweets']

            # Handle non-string values in tweet column
            if not isinstance(tweet, str):
                continue  # Skip this row

            # Apply sentiment analysis
            blob = TextBlob(tweet)
            sentiment_score = blob.sentiment.polarity

            # Check for agitation keywords
            agitation_keywords = self.get_agitation_keywords(tweet)
            has_agitation = any(keyword in tweet.lower() for keyword in agitation_keywords)

            # Determine if politician is inciting agitation
            if sentiment_score < 0 and has_agitation:
                if username in agitating_politicians:
                    agitating_politicians[username]['count'] += len(agitation_keywords)
                    agitating_politicians[username]['keywords'].update(agitation_keywords)
                else:
                    agitating_politicians[username] = {
                        'count': len(agitation_keywords),
                        'keywords': agitation_keywords
                    }

        return agitating_politicians
    def find_most_agitative_politician(self, agitating_politicians):
        most_agitative_politician = max(agitating_politicians, key=lambda x: agitating_politicians[x]['count'])
        return most_agitative_politician
    
    def get_agitation_keywords(self, tweet):
        agitation_keywords = set()

        for word in self.agitative_words:
            if re.search(r'\b' + word + r'\b', tweet, re.IGNORECASE):
                agitation_keywords.add(word)

        return agitation_keywords

    def run(self):
        st.title('Tweet Sentiment Analysis')

        uploaded_file = st.file_uploader('Upload a file', type=['csv', 'xlsx'])

        if uploaded_file is not None:
            if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':  # XLSX
                df = pd.read_excel(uploaded_file)
            else:  # CSV
                df = pd.read_csv(uploaded_file)
        

            df['cleaned_text'] = df['tweets'].apply(self.clean_text)

 
            df['sentiment_score'], df['sentiment_tag'] = zip(*df['cleaned_text'].apply(self.analyze_sentiment))
            
            
            df['predicted_sentiment'] = df['cleaned_text'].apply(self.predict_sentiment)
            
            
            df['predicted_sentiment_label'] = df['predicted_sentiment'].apply(self.sentiment_label)
            
            
            
            # positive_percentage = (df['sentiment_tag'] == 'positive').mean() * 100

            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize=(8, 6))
            plt.hist(df['sentiment_score'], bins=20, color='green', alpha=0.7)
            plt.xlabel('Sentiment Score')
            plt.ylabel('Frequency')
            plt.title('Sentiment Distribution')
            st.pyplot()
            st.write("### Data Content")
            st.dataframe(df)
            most_active_politician, num_tweets = self.find_most_active_politician(df)

            mentions_df = pd.DataFrame(self.analyze_tweets_sentiment(df),columns=['User', 'Mentioned Word', 'Sentiment'])

            # Display the combined DataFrame using st.dataframe()
            st.write("### Positive, Negative, Neutral Mentions by each Politician")
            st.dataframe(mentions_df)
            # Plot word frequency for each user
            self.plot_word_frequency_by_user(df)
            st.write('### Sentiment Analysis Results')
            st.write(df)
            # Display positive, negative, neutral sentiment graph
            self.plot_sentiments_by_politician(df)
            self.calculate_positive_sentiment_percentage(df)
            
            # Calculate sentiment percentages
            sentiment_percentages = self.calculate_sentiment_percentages(df)
            # Convert sentiment percentages to dataframe
            sentiment_percentages_df = pd.DataFrame(sentiment_percentages).transpose()
            # Plot sentiment percentages
            sentiment_percentages_df.plot(kind='bar', figsize=(10, 6))
            plt.xlabel('Politician')
            plt.ylabel('Percentage of Sentiments')
            plt.title('Sentiments')
            plt.xticks(rotation=45)
            st.pyplot()

            # Find the highest tweet count and plot it
            self.find_user_with_highest_tweet_count(df)
            # Display sentiment percentages dataframe
            st.write('### Sentiment Percentages')
            st.dataframe(sentiment_percentages_df)

            # Find most active politician
            st.write(f"#### Most Active Politician: {most_active_politician} with {num_tweets} tweets")
            # Find most patriotic politician
            most_patriotic_politician, patriotic_count = self.find_most_patriotic_politician(df)

            if most_patriotic_politician:
                st.write(f"#### Most Patriotic Politician: {most_patriotic_politician}")
                # st.write(f"### Positive Sentiment Mentions: {patriotic_count}")
            else:
                st.write("No positive sentiment mentions found.")

            # Detect politicians inciting agitation
            agitating_politicians = self.analyze_tweets_for_agitation(df)

            # Find the most agitative politician
            # most_agitative_politician = self.find_most_agitative_politician(agitating_politicians)
            # Detect politicians inciting agitation
            agitating_politicians = self.analyze_tweets_for_agitation(df)

            # Display politicians inciting agitation in a table
            st.write("### Politicians Inciting Agitation:")
            agitating_table_data = []
            for politician, data in agitating_politicians.items():
                keywords = ", ".join(data['keywords'])
                frequency = data['count']
                agitating_table_data.append((politician, keywords, frequency))

            st.table(pd.DataFrame(agitating_table_data, columns=['Politician', 'Agitation Keywords', 'Keyword Frequency']))


if __name__ == '__main__':
    app = TweetSentimentApp()
    app.run()
