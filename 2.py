import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, Input, TimeDistributed, Bidirectional, Attention

# Пример данных: ошибки и исправленные версии
texts_with_errors = ["mnw nravitsa chitat", "on igraet v fytbol", "ya ychys v shkole"]
correct_texts = ["mne nravitsya chitat", "on igraet v futbol", "ya uchus v shkole"]

# Создаем список всех символов
chars = sorted(set("".join(correct_texts + texts_with_errors)))
char2idx = {c: i+1 for i, c in enumerate(chars)}  # +1 для padding
idx2char = {i: c for c, i in char2idx.items()}

# Параметры
max_len = max(len(t) for t in correct_texts)  # Максимальная длина текста
vocab_size = len(chars) + 1  # Размер алфавита

# Преобразуем текст в последовательности индексов
def text_to_seq(texts):
    return pad_sequences([[char2idx[c] for c in text] for text in texts], maxlen=max_len, padding='post')

X = text_to_seq(texts_with_errors)
y = text_to_seq(correct_texts)

# One-hot encoding для y
y = np.array([to_categorical(seq, num_classes=vocab_size) for seq in y])

# Входной слой
inputs = Input(shape=(max_len,))

# Embedding
embedding = Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len)(inputs)

# Bidirectional LSTM
lstm_out, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(128, return_sequences=True, return_state=True))(embedding)

# Attention слой
attention = Attention()([lstm_out, lstm_out])

# Выходной слой
outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(attention)

# Создаем модель
model = Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучаем модель
model.fit(X, y, epochs=50, batch_size=5)

# Функция для исправления текста
def correct_text(text):
    seq = text_to_seq([text])  # Преобразуем текст в индексы
    pred = model.predict(seq)[0]  # Предсказания модели
    
    corrected_chars = []
    for vec in pred:
        char_idx = np.argmax(vec)
        if char_idx != 0:  # Игнорируем padding
            corrected_chars.append(idx2char[char_idx])
    
    return "".join(corrected_chars).strip()

# Тестирование
test_text = "ychys"
print("Исправленный текст:", correct_text(test_text))
