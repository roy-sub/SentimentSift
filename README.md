# Tripadvisor Data Analysis and DL Model Development

![banner](https://github.com/roy-sub/Tripadvisor/blob/main/Images/banner%20I.jpg)

## Project Overview
In this project, we will be exploring the hotel reviews and the rating base on customer hotel experience. We will be also looking at feature engineering and designing a deep learning model to predict ratings based on reviews.We also be using NLP tools for feature extractions and preparing the data for deep learning models.

## About Tripadvisor
Tripadvisor, Inc. is an American online travel company that operates a website and mobile app with user-generated content and a comparison shopping website. It also offers online hotel reservations and bookings for transportation, lodging, travel experiences, and restaurants. Its headquarters are in Needham, Massachusetts. Wikipedia

## Project Structure

### 1. Data Extraction

Hotels play a crucial role in traveling and with the increased access to information new pathways of selecting the best ones emerged. With this dataset, consisting of 20k reviews crawled from Tripadvisor, you can explore what makes a great hotel and maybe even use this model in your travels! Dataset Link : [CLick Here](https://zenodo.org/records/1219899#.YHwt1J_ivIU)

### 2. Data Preprocessing

**a. Compound Score & Sentiment -** Utilized `VADER (Valence Aware Dictionary and Sentiment Reasoner)` which is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media for sentimental scoring and then converting those scores into 3 categorical Sentiments, Positive Negative, and Neutral. [Click Here](https://github.com/cjhutto/vaderSentiment#introduction) to learm more ! 

**b. Applying Functions -** Applied the above functions to the original database to create additional columns for Sentiment Score and Sentiment.And finally saved the preprocessed file for future use.

### 3. Data Analysis and Visualization 

**a. Countplot of Sentiments -** Most of the comments are Positive, as shown in seaborn countplot.

![img1](https://github.com/roy-sub/Tripadvisor/blob/main/Images/Countplot%20of%20Sentiments.png)

**b. Plotting the Bar Graph -** In the bar plot, we can see the distribution of sentiment and rating, people with 5-star ratings have the highest positive sentiment. whereas at lower ratings its mixed emotions showed by customers review, this can be related to sarcasm.

![img2](https://github.com/roy-sub/Tripadvisor/blob/main/Images/Plotting%20the%20Bar%20Graph.png)

**c. Plotting a pie chart of ratings -** A simple pie chart using Plotly library can give you an idea of the distribution of different ratings. The majority of people are giving a positive and 4-5star rating.

![img3](https://github.com/roy-sub/Tripadvisor/blob/main/Images/Plotting%20a%20pie%20chart%20of%20ratings.png)

**d. Violion plot -** Violion plot gives us a better picture of the relationship between Ratings and Sentiments. From 3 to 5 rating most of the review sentiments are positive.

![img4](https://github.com/roy-sub/Tripadvisor/blob/main/Images/Violin%20plot.png)

**e. Wordcloud of Different Sentiments -** The most common word used in all three Sentiments was a hotel room. Which is quite obvious ,hotel managers can to focus on if they want a better rating from customers.

![img5](https://github.com/roy-sub/Tripadvisor/blob/main/Images/Wordcloud%20of%20Different%20Sentiments.png)

**f. Applying Keywords to the Dataframe -** 
Extracted keywords using Gensim's `summarization.keywords module` which contains functions to find keywords of the text and building graph on tokens from text. More can be found at [Gensim](https://radimrehurek.com/gensim_3.8.3/summarization/keywords.html). And then used Python Counter to identify the top ten keywords and created a barplot of the top 20 keywords.

![img6](https://github.com/roy-sub/Tripadvisor/blob/main/Images/Barplot%20of%20Top%2020%20Keywords.png)

### 4. Review Text Processing using NLTK

**a. Downloaded NLTK for natural language processing.**

```
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

**b. Removed common words and stopwords -** To enhance model performance.

**c. Stopwords -** The stop words are words which are filtered out before or after processing of natural language data (text). Though "stop words" usually refers to the most common words in a language, there is no single universal list of stop words used by all natural language processing tools, and indeed not all tools even use such a list. Some tools specifically avoid removing these stop words to support phrase search.

**d. Employed lemmatization to convert words to their base form -** Lemmatization is the process of converting a word to its base form. The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors. For more information visit [Lemmatization Approaches with Examples in Python](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/)

**e. Text Joining -** Making all the comma seperated lemmatized words back into a string. Also used PoerterStemmer to improve the performance and you can also use other yet processing to improve the performance metric.

**f. Utilized the Keras Tokenizer class to vectorize the text corpus -** The Tokenizer class of Keras is used for vectorizing a text corpus. For this either, each text input is converted into integer sequence or a vector that has a coefficient for each token in the form of binary values. Also employed the texts_to_sequences method helps in converting tokens of text corpus into a sequence of integers.

**g. Remapped ratings -** To reduce the model output size from 6 to 5.

### 5. Building the Deep Learning Model (LSTM)

**a. Implemented a Long Short Term Memory (LSTM) architecture for sentiment prediction. -** Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition,speech recognition and anomaly detection in network traffic or IDSs (intrusion detection systems).

![img7](https://github.com/roy-sub/Tripadvisor/blob/main/model_architecture.png)

### 6. Visualized Model Performance

**a. Plotted accuracy and sparse categorical cross-entropy for both training and validation sets**

![img8](https://github.com/roy-sub/Tripadvisor/blob/main/Images/metrics%20i.png)

**b. Visualized model performance using a confusion matrix heatmap**

![img9](https://github.com/roy-sub/Tripadvisor/blob/main/Images/confusion%20matrix.png)

**c. Visualized model performance using a classification report**

![img9](https://github.com/roy-sub/Tripadvisor/blob/main/Images/Classification%20Report.png)

### 7. Testing Saved Model
Saved the trained model has been saved as [BiLSTM.h5](https://github.com/roy-sub/Tripadvisor/blob/main/BiLSTM.h5) for easy replication and testing.

### 8. Conclusion

**a. Tripadvisor Prediction model -** Overall my deep learning model performed well with limited resources and memory restrain. I think using the BERT model can increase accuracy by +20 percent. I haven't experimented with other ML models, but in my experience with gradient booster and logistic models do not perform well in text classification.

**b. Final Thoughts -** Sentiments of reviews were all over the place and they did not have any effect on ratings. In reviews sometimes people are being sarcastic which is hard to pick by machine without context. Overall by analyzing keywords I have realized people were mostly writing reviews about the Hotel room, service, staff, and breakfast. Which is a good indicator for a hotel management team so they can focus on it, to get better reviews and 5 stars ratings.

![conclusion](https://github.com/roy-sub/Tripadvisor/blob/main/Images/banner%20II.jpg)
