import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# read text file with nursery rhymes
with open("nursery_rhymes-2.txt", "r") as file:
    text = file.read().lower()

# split nursery rhymes into separate strings
rhymes = re.split("\n\n\n\n", text)
# each rhyme goes to a dictionary
rhyme_dict = {}
for rhyme in rhymes:
    lines = rhyme.split("\n\n") # splits the title from rhyme as there is two lines following each
    if len(lines) < 2: # accounting for the rhymes that have non text
        continue # skip this rhyme if it doesn't have a title and at least one line of text
    title, text = lines[0], "\n".join(lines[1:])
    rhyme_dict[title.strip()] = text.strip()
# create dictionary of tokenized nursery rhymes
tokenizer = Tokenizer()
rhyme_sequences = {}
for title, text in rhyme_dict.items():
    tokenizer.fit_on_texts([text])
    rhyme_sequences[title] = tokenizer.texts_to_sequences([text])[0]
    tokenizer.work_index = {}



# calculate max sequence length
max_sequence_len = max([len(seq) for seq in rhyme_sequences.values()])

# create predictors and target
sequences = []
for seq in rhyme_sequences.values():
    for i in range(1, len(seq)):
        seq_slice = seq[:i+1]
        sequences.append(seq_slice)
sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)

# define LSTM model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 50, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# train the model
model.fit(X, y, epochs=500, verbose=1)

# function to generate new nursery rhymes
def generate_rhyme(seed_text, num_lines=30, words_per_line=20):
    for line_num in range(num_lines):
        # tokenize the seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # pad the sequences
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        # get the predicted class probabilities for the next word
        predicted_probs = model.predict(token_list, verbose=0)
        # get the index of the word with the highest probability
        predicted = np.argmax(predicted_probs)
        # convert the index to the corresponding word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        # add the predicted word to the output line
        seed_text += " " + output_word
        # start a new line after every "words_per_line" words
        if (line_num+1) % words_per_line == 0:
            seed_text += "\n"
    return seed_text.strip()
# example usage
seed_text = "Harry and Greg drank from a keg"
generated_rhyme = generate_rhyme(seed_text, num_lines=30, words_per_line=20)
# to a text file
with open("generated_rhyme(500epoch).txt", "w") as file:
    file.write(generated_rhyme)
