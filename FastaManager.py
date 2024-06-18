import os
from Bio import SeqIO


class FastaManager:
    FASTA = "fasta"
    FA = "fa"

    @staticmethod
    def read_fasta(file_path):
        """Read sequences from a FASTA file."""
        sequences = []
        with open(file_path, "r") as file:
            for record in SeqIO.parse(file, FastaManager.FASTA):
                sequences.append(str(record.seq))
        return sequences

    @staticmethod
    def get_all_sequences(input_dir):
        """Read FASTA files from input_dir and return sequences."""
        all_sequences = []
        for file_name in os.listdir(input_dir):
            if file_name.endswith(f".{FastaManager.FASTA}") or file_name.endswith(f".{FastaManager.FA}"):
                file_path = os.path.join(input_dir, file_name)
                sequences = FastaManager.read_fasta(file_path)
                all_sequences.extend(sequences)
        return all_sequences

    @staticmethod
    def create_fasta_dataset(input_dir, embedding):
        """Read FASTA files from input_dir and return tokenized sequences."""
        return embedding.tokenize_sequences(FastaManager.get_all_sequences(input_dir))

    @staticmethod
    def create_dataset(sequences, embedding):
        """Read list of sequences and return tokenized sequences."""
        return embedding.tokenize_sequences(sequences)