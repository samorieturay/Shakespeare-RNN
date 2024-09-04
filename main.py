import sys
import io
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from datetime import datetime

# Set UTF-8 encoding for standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the text data
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]  # selected part of the text

# Create a sorted list of unique characters in the text
characters = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40

# For training the data 

# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i: i + SEQ_LENGTH])
#     next_characters.append(text[i + SEQ_LENGTH])

# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
# y = np.zeros((len(sentences), len(characters)), dtype=bool)

# for i, sentence in enumerate(sentences):
#     for t, character in enumerate(sentence):
#         x[i, t, char_to_index[character]] = 1
#     y[i, char_to_index[next_characters[i]]] = 1


# For saving the data to the  model
# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# model.fit(x, y, batch_size=256, epochs=4)

# # Save the model with the appropriate extension
# model.save('textgenerator.keras')




# Load the pre-trained model
model = tf.keras.models.load_model('textgenerator.keras')

# Sample function to predict the next character based on the probability distribution
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function that generates text of a given length and temperature
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        # Create an empty array of shape (1, SEQ_LENGTH, len(characters))
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        # Predict the next character
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    
    return generated

# Open the file to append the output
with open('output.txt', 'a', encoding='utf-8') as f:
    # Write the current date and time at the top
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"Generated on: {current_time}\n")
    
    # Generate and write text with different temperature values
    f.write('......0.2.....\n')
    f.write(generate_text(300, 0.2) + '\n')

    f.write('......0.4.....\n')
    f.write(generate_text(300, 0.4) + '\n')

    f.write('......0.6.....\n')
    f.write(generate_text(300, 0.6) + '\n')

    f.write('......0.8.....\n')
    f.write(generate_text(300, 0.8) + '\n')

    f.write('......1.0.....\n')
    f.write(generate_text(300, 1.0) + '\n')

    # Add a separator for readability between runs
    f.write('\n' + '-'*50 + '\n\n')
