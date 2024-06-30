import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.layers import Embedding, LSTM, Dense

from FastaManager import FastaManager
from MiRNA2Vec import MiRNA2Vec
from Screening import Screening
from Visualizer import Visualizer

"""
def prepare_lstm_data(tokenized_sequences, max_len):
    # Convert tokens to integer indices
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_sequences)
    sequences = tokenizer.texts_to_sequences(tokenized_sequences)
    word_index = tokenizer.word_index

    # Pad sequences
    data = pad_sequences(sequences, maxlen=max_len, padding='post')

    return data, word_index
"""

# Define the input directory
input_directory = "./data/sequences"

# Create miRNA2Vec instance
miRNA2Vec = MiRNA2Vec(k_mers=3, vector_size=16, epochs=4)

# Get sequences
X_positive_sequences = FastaManager.get_all_sequences(input_directory + "/positive_samples")
X_artificial_positive_sequences = FastaManager.get_all_sequences(input_directory + "/artificial_positive_samples")
X_negative_sequences = FastaManager.get_all_sequences(input_directory + "/negative_samples")
X = X_positive_sequences + X_artificial_positive_sequences + X_negative_sequences

print(f"positive sequences: {len(X_positive_sequences)}")
print(f"artificial positive sequences: {len(X_artificial_positive_sequences)}")
print(f"negative sequences: {len(X_negative_sequences)}")


# Create target list
y_positive = np.ones(len(X_positive_sequences))
y_positive_with_data_augmentation = np.ones(len(X_artificial_positive_sequences))
y_negative = np.zeros(len(X_negative_sequences))
y = np.append(y_positive, y_positive_with_data_augmentation)
y = np.append(y, y_negative)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create the dataset for Word2Vec
tokenized_sequences = FastaManager.create_dataset(X_train, miRNA2Vec)
# print(tokenized_sequences[0])

# Train the Word2Vec model
miRNA2Vec.train_word2vec(tokenized_sequences)
# miRNA2Vec.load_model('./pretrained/dna2vec.w2v', tokenized_sequences)

# Get embeddings for sequences
X_train = miRNA2Vec.get_average_embeddings(X_train)
X_test = miRNA2Vec.get_average_embeddings(X_test)
print(f"X_train for SVM model: {X_train[0:2]}")

# Train SVM
svm_model = SVC(kernel='rbf', class_weight="balanced")
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1: {f1 * 100:.2f}%")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example: Getting embeddings for a new sequence
new_sequence = "CCUGGGAGGUGUGAUAUCAUGGUUCCUGGGAGGUGUGAUCCUGUGCUUCCUGGGAGGUGUGAUAUCGUGGUUCCUGGG"
new_embedding = miRNA2Vec.get_average_embeddings([new_sequence])
new_prediction = svm_model.predict(new_embedding)
print(f"Prediction for new sequence: {new_prediction[0]}")

# Visualizer.show_variance(X_test)

# Visualizer.show_2d_scatter(X_test, y_test, title="PCA visualization of sequence embeddings")

# Visualizer.show_3d_scatter(X_test, y_test, title="PCA visualization of sequence embeddings")

screening = Screening(X_positive_sequences, X_negative_sequences)
sequence_for_screening = screening.get_sequence_for_screening()
screen_size = screening.nt_with_most_data
results = {0: 0, 1: 1}
predicted_positive_positions = []
start = time.time()
for idx in range(0, len(sequence_for_screening[:3400])-screen_size):
    new_sequence = sequence_for_screening[idx:idx+screen_size]
    new_embedding = miRNA2Vec.get_average_embeddings([new_sequence])
    new_prediction = svm_model.predict(new_embedding)
    results[int(new_prediction[0])] += 1
    if new_prediction:
        predicted_positive_positions.append(idx)
end = time.time()
print(f"negative values: {results[0]}")
print(f"positive values: {results[1]}")
print(f"positions of positive values: {predicted_positive_positions}")
print(f"time: {end-start} seconds")
# Save the model
output_model_path = "./pretrained/miRNAFromWord2Vec.w2v"
miRNA2Vec.model.wv.save(output_model_path)

# Example: Getting embeddings for k-mers
# kmer = "AUG"
# embedding = miRNA2Vec.model.wv[kmer]
# print(f"Embedding for {kmer} with {np.size(embedding)} elements: {embedding}")


"""comparar FS4"""
"""comparar rapidez con screening de secuencia larga"""
"""armar secuencia aleatoriamente con secuencias del dataset, <SEQ1><nucleotidos_aleatorios><SEQ2><nt_aleatorios>..."""


"""probar con naive bayes para obtener probabilidades"""
"""conseguir probabilidades del SVM"""
"""mientras se aproxima, la probabilidad debe ir creciendo"""
"""investigar sobre metrica para señal de probabilidades"""
"""comparar con trabajo reciente (últimos 5 años)"""