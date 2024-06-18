from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pickle


class MiRNA2Vec:
    def __init__(self, k_mers: int = 3, vector_size: int = 100, epochs: int = 5):
        self.__model = None
        self.k_mers = k_mers
        self.vector_size = vector_size
        self.epochs = epochs
        self.window = 5
        self.min_count = 1
        self.workers = 4
        self.skip_gram = 1

    @property
    def model(self):
        return self.__model

    def __get_tokens(self, sequence):
        return [sequence[i:i + self.k_mers] for i in range(len(sequence) - self.k_mers + 1)]

    def tokenize_sequences(self, sequences):
        """Tokenize sequences into k-mers."""
        tokenized_sequences = []
        for seq in sequences:
            tokens = self.__get_tokens(seq)
            tokenized_sequences.append(tokens)
        return tokenized_sequences

    def train_word2vec(self, tokenized_sequences):
        """Train Word2Vec model on tokenized sequences."""
        self.__model = Word2Vec(sentences=tokenized_sequences,
                                vector_size=self.vector_size,
                                window=self.window,
                                min_count=self.min_count,
                                workers=self.workers,
                                epochs=self.epochs,
                                sg=self.skip_gram)
        return self.__model

    def get_average_embeddings(self, sequences):
        """Get average embedding for each sequence."""
        embeddings = []
        for seq in sequences:
            tokens = self.__get_tokens(seq)
            token_embeddings = [self.model.wv[token] for token in tokens if token in self.model.wv]
            if token_embeddings:
                avg_embedding = np.mean(token_embeddings, axis=0)
            else:
                avg_embedding = np.zeros(self.model.vector_size)
            embeddings.append(avg_embedding.tolist())
        return np.array(embeddings)

    def load_model(self, path, tokenized_sequences):
        self.__model = Word2Vec(sentences=tokenized_sequences,
                                vector_size=self.vector_size,
                                window=self.window,
                                min_count=self.min_count,
                                workers=self.workers,
                                epochs=self.epochs)
        self.__model.wv = KeyedVectors.load_word2vec_format(path,
                                          binary=False)  # Change binary to False if the model is in text format
        # Continue training the Word2Vec model with your new sequences
        self.__model.build_vocab(tokenized_sequences, update=True)  # Update the vocabulary with new sentences
        self.__model.train(tokenized_sequences, total_examples=self.__model.corpus_count, epochs=self.__model.epochs)
        return self.__model
