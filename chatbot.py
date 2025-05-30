import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import random

# Load training data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Preprocess data
def preprocess_data(data):
    training_sentences = []
    training_labels = []
    labels = []
    responses = {}

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses[intent['tag']] = intent['responses']
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    label_encoder = LabelEncoder()
    training_labels = label_encoder.fit_transform(training_labels)

    return training_sentences, training_labels, labels, responses, label_encoder

# Create and train the model
def train_model(training_sentences, training_labels, vocab_size=1000, epochs=200):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(training_sentences)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)

    model = Sequential([
        Dense(128, input_shape=(padded_sequences.shape[1],), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(set(training_labels)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=1)

    return model, tokenizer

# Chat with the bot
def chat(model, tokenizer, label_encoder, responses):
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=model.input_shape[1])
        prediction = model.predict(padded_sequence)
        tag = label_encoder.inverse_transform([np.argmax(prediction)])
        print("Bot:", random.choice(responses[tag[0]]))

# Main function
if __name__ == "__main__":
    data_file = "intents.json"  # Replace with your intents file path
    data = load_data(data_file)
    training_sentences, training_labels, labels, responses, label_encoder = preprocess_data(data)
    model, tokenizer = train_model(training_sentences, training_labels)
    chat(model, tokenizer, label_encoder, responses)