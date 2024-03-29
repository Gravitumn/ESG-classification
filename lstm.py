import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout , SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import spacy
import re
import string
import seaborn as sns
from keras import Model
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Layer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(int(input_shape[-1]),),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class TextAttBiRNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num=4,
#                  class_num=1, #old
#                  last_activation='sigmoid'): #old
                 last_activation='softmax'):
        super(TextAttBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        # 64,16 or 32,8
        self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))  # LSTM 
        self.bi_rnn = Bidirectional(LSTM(64, return_sequences=True))  # LSTM 
        self.bi_rnn = Bidirectional(LSTM(8, return_sequences=True)) 
        # self.bi_rnn = Bidirectional(SimpleRNN(128, return_sequences=True))
        self.attention = Attention(self.maxlen)
        self.dropout = Dropout(0.2)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextAttBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextAttBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.bi_rnn(embedding)
        x = self.attention(x)
        output = self.classifier(x)
        return output
    


train_data = pd.read_csv("cleaned_train.csv", encoding= 'unicode_escape')
val_data = pd.read_csv("cleaned_val.csv", encoding= 'unicode_escape')
# test_data = pd.read_csv("test2.csv", encoding= 'unicode_escape')

train_ori_text = train_data['ESGN'].values
val_ori_text = val_data['ESGN'].values
# test_text = test_data['ESGN'].values
train_labels = train_data['class'].values
val_labels = val_data['class'].values
# test_labels = test_data['class'].values
train_text = train_data['Cleaned_ESGN'].values
val_text = val_data['Cleaned_ESGN'].values

# Tokenize and pad sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_text)

max_length = 100  # Adjust as needed
train_sequences = tokenizer.texts_to_sequences(train_text)
val_sequences = tokenizer.texts_to_sequences(val_text)
# test_sequences = tokenizer.texts_to_sequences(test_text)

train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post')
val_padded = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=max_length, padding='post')
# test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post')

# One-hot encode labels
num_classes = len(set(train_labels))
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
val_labels_one_hot = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)
# test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# Define the LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 30
hidden_dim = 64

# Assuming your input tensor has shape (batch_size, sequence_length)
# Adjust parameters accordingly based on your actual data
model = TextAttBiRNN(max_features=150000, embedding_dims=50, maxlen=100, class_num=4, last_activation='softmax')

# Define optimizer and compile the model
learning_rate = 1e-3  
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping based on validation accuracy
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Model checkpoint to save the model when early stopping occurs
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('results\\lstm2', monitor='val_accuracy', save_best_only=True)

# Train the model
num_epochs = 100
batch_size = 8
history = model.fit(train_padded, train_labels_one_hot, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(val_padded, val_labels_one_hot), callbacks=[early_stopping, model_checkpoint])

# Plot accuracy graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load the best model
best_model = tf.keras.models.load_model('results\\lstm2')

# Evaluate on the test set
test_loss, test_accuracy = best_model.evaluate(val_padded, val_labels_one_hot)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions on the test set
test_predictions = best_model.predict(val_padded)
predicted_labels = tf.argmax(test_predictions, axis=1).numpy()

# Calculate additional metrics if needed
precision = precision_score(val_labels, predicted_labels, average='weighted')
recall = recall_score(val_labels, predicted_labels, average='weighted')
f1 = f1_score(val_labels, predicted_labels, average='weighted')

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Print confusion matrix
conf_mat = confusion_matrix(val_labels, predicted_labels)

# Display confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(train_labels)), yticklabels=sorted(set(train_labels)))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Create a DataFrame with the original text, cleaned text, predicted label, and true label
results_df = pd.DataFrame({
    'Original Text': val_ori_text,
    'Predicted Label': predicted_labels,
    'True Label': val_labels
})

# Save the DataFrame to a CSV file
results_df.to_csv('results\\lstm.csv', index=False)
