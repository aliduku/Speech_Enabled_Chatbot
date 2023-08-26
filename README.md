# Speech-Enabled Chatbot

Welcome to the Speech-Enabled Chatbot project! This project combines speech recognition and natural language processing to create a chatbot that accepts both text and speech input from users. The chatbot transcribes speech into text, processes user input, and generates responses using cosine similarity with Bag of Words (BOW) and TF-IDF techniques.

## Overview

This project aims to create a chatbot that can interact with users using both text and speech. The chatbot processes user input and provides responses based on the similarity of input to predefined responses from a dataset. It utilizes speech recognition for transcribing user speech into text and then leverages NLP techniques to generate relevant responses.

## Features

- Accepts both text and speech input from users.
- Utilizes cosine similarity with BOW and TF-IDF to generate responses.
- Provides a seamless user experience through a Streamlit web app.

## How to Use

1. Install the required packages using:
   ```
   pip install streamlit speech_recognition audio_recorder_streamlit nltk pandas
   ```

2. Run the Streamlit app:
   ```
   streamlit run Speech_Enabled_Chatbot.py
   ```

3. In the app, type your message in the text area or click the "Record" button to provide speech input. You can also save your text input or recorded audio.

4. Click the "Submit" button to see the chatbot's responses generated using cosine similarity.

## Try It Out

[Click here](https://speechenabledchatbot-yaj2rsyrwjqcvxksh6o5bf.streamlit.app/) to try out the Speech-Enabled Chatbot web app!
