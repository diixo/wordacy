from pathlib import  Path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Правильные слова для обучения
words = []

test_txts = "db-full.txt"

file_path = Path(test_txts)
if file_path.exists():
    words = file_path.read_text(encoding="utf-8").splitlines()

# Токенизация символов
tokenizer = Tokenizer(char_level=True)  # Символьная токенизация
tokenizer.fit_on_texts(words)

# Преобразование слов в последовательности
sequences = tokenizer.texts_to_sequences(words)

# Подготовка входных и целевых данных
X = []
y = []
for seq in sequences:
    for i in range(1, len(seq)):
        X.append(seq[:i])  # Входная последовательность
        y.append(seq[i])   # Следующий символ

# Приведение к одинаковой длине
max_length = max(len(seq) for seq in X)
X = pad_sequences(X, maxlen=max_length, padding='post')
y = pad_sequences([[target] for target in y], maxlen=1)


from keras.utils import to_categorical

vocab_size = len(tokenizer.word_index) + 1  # Размер словаря
X = to_categorical(X, num_classes=vocab_size)
y = to_categorical(y, num_classes=vocab_size)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Параметры
embedding_dim = 250
lstm_units = 64

# Модель
model = Sequential([
    LSTM(units=lstm_units, input_shape=(X.shape[1], X.shape[2])),
    Dense(units=vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, batch_size=256, epochs=50)

import numpy as np

def correct_word(input_word):
    seq = tokenizer.texts_to_sequences([input_word])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    padded_seq = to_categorical(padded_seq, num_classes=vocab_size)
    
    pred_seq = []
    for i in range(len(input_word)):
        pred = model.predict(padded_seq)
        next_char = np.argmax(pred, axis=-1)[0]
        pred_seq.append(next_char)
        padded_seq = np.roll(padded_seq, -1, axis=1)  # Сдвиг окна
        padded_seq[0, -1] = next_char
    
    corrected_word = ''.join([tokenizer.index_word[idx] for idx in pred_seq if idx in tokenizer.index_word])
    return corrected_word

print(correct_word("mchine"))  # Ожидается: "машина"
