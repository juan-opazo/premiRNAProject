import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

from FastaManager import FastaManager
from MiRNA2Vec import MiRNA2Vec
from Visualizer import Visualizer

# Define the input directory
input_directory = "./data/sequences"

# Create miRNA2Vec instance
miRNA2Vec = MiRNA2Vec(k_mers=3, vector_size=16, epochs=100)

# Create the dataset
positive_tokenized_sequences = FastaManager.create_fasta_dataset(input_directory + "/positive_samples", miRNA2Vec)
negative_tokenized_sequences = FastaManager.create_fasta_dataset(input_directory + "/negative_samples", miRNA2Vec)
tokenized_sequences = positive_tokenized_sequences + negative_tokenized_sequences

# Train the Word2Vec model
miRNA2Vec.train_word2vec(tokenized_sequences)

# Get embeddings for sequences
X_positive = miRNA2Vec.get_average_embeddings(FastaManager.get_all_sequences(input_directory + "/positive_samples"))
X_negative = miRNA2Vec.get_average_embeddings(FastaManager.get_all_sequences(input_directory + "/negative_samples"))
X = X_positive + X_negative

# Create target list
y_positive = np.ones(len(X_positive))
y_negative = np.zeros(len(X_negative))
y = np.append(y_positive, y_negative)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf')
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

Visualizer.show_2d_scatter(X, y, title="PCA visualization of sequence embeddings")

Visualizer.show_3d_scatter(X, y, title="PCA visualization of sequence embeddings")

# Save the model
# output_model_path = "./data/processed/word2vec_model.model"
# save_embeddings(w2v_model, output_model_path)

# Example: Getting embeddings for k-mers
# kmer = "AUG"
# embedding = miRNA2Vec.model.wv[kmer]
# print(f"Embedding for {kmer} with {np.size(embedding)} elements: {embedding}")