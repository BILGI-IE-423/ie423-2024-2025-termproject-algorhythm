[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/7dVfiuFW)
# Mood Prediction of Songs

122203088 Belinay Keleş  
121203034 Sude Şintürk  
121203079 Yeliz Avcı  
120203034 Sevgi Gündoğdu  

## 1.Scope of the Project

This project aims to predict the emotional states conveyed by songs using their multidimensional features. In addition to the Spotify dataset, song lyrics, album cover images, and data collected from other music platforms are used. The main goal of the project is to classify the emotion a song conveys by holistically analyzing audio-based (valence and energy), visual (album cover), and textual (lyrics) features using machine learning techniques. The project is based on the two-dimensional energy-valence matrix, commonly used in psychology models such as Thayer's Mood Model, to define emotions. Songs are categorized into emotional clusters such as joyful, sad, energetic, and anxious. The core focuses of the project are:
Performing textual analysis (NLP) on song lyrics to identify emotional expressions and integrate them into the model,

![](Desktop/Thayer Model.jpeg)

Using visual processing techniques to analyze album cover features and utilize them in emotion prediction,


Determining the emotional positioning of songs on the mood map using audio-based features (valence and energy) retrieved from Spotify.
Ultimately, this project aims to build an innovative, machine learning-enhanced emotion prediction system based on a multimodal approach and use it to generate emotion-based playlists for users.

## 2.Research Questions

How accurately can a song's emotion be predicted based on its energy and valence values, and how reliably can playlists be generated based on these predictions?


To what extent do the visual features of album covers (e.g., color, contrast, visual density) and song lyrics align with the predicted emotional state?


Among various machine learning models (e.g., XGBoost, Random Forest, SVM), which algorithm performs best in predicting a song’s emotion?




## 3.Preprocessing Steps

### 3.1.Libraries
In this project, the libraries OpenCV,  pandas, matploblib, seaborn, Pillow, io,  tqdm, NumPy, selenium and scikit-learn were used.
In the parts where data is generated, the additional libraries selenium, time, requests, and beautifulsoup4 were used.






### 3.3.Process
Data Loading and Initial Cleaning  

The primary dataset, spotify_songs.csv, was loaded and all rows containing missing values were removed using the dropna() function. The describe() function was then used to examine the feature distributions and value ranges. It was observed that the loudness feature included values outside the expected range of -60 to 0, which were identified as outliers and subsequently removed. Additionally, duplicate entries based on the track_id column were detected and eliminated to ensure data integrity.  

Mood Label Assignment Using a 3x3 Grid  

According to Lata (2024), Spotify songs were classified based on emotions using Robert Thayer’s traditional two-dimensional mood model, which is built on the dimensions of energy and valence. While studies typically employ a 2x2 (four-cluster) structure, our project divided the valence (emotional positivity) and energy axes into three equal intervals, resulting in a 3x3 grid that enables a more detailed representation of emotions. As a result, nine distinct mood clusters were obtained. This approach preserves the simplicity and clarity of Thayer’s model while allowing for a more nuanced emotional mapping. Cluster labeling was inspired by the core mood states defined in Thayer’s model, such as Exuberance, Anxiety, Contentment, and Depression. This method provides a strong foundation for the development of mood prediction systems and mood-based music recommendation engines. Then, a function named assign_mood was created to assign these moods to the dataset.  

![](images/3x3 Mood Table.jpeg)
 
One-Hot Encoding  

The categorical feature playlist_subgenre was transformed using one-hot encoding to create binary columns for each genre, enabling it to be used in machine learning models.  

