from FastaManager import FastaManager
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO


def write_fasta(sequences, output_file):
    seq_records = []
    for seq_id, description, sequence in sequences:
        seq_record = SeqRecord(Seq(sequence), id=seq_id, description=description)
        seq_records.append(seq_record)

    with open(output_file, "w") as output_handle:
        SeqIO.write(seq_records, output_handle, "fasta")


def generate_positive_samples_with_sequence_reversed():
    input_directory = "./data/sequences"
    new_sequences = []
    for sequence in FastaManager.get_all_sequences(input_directory + "/positive_samples"):
        new_sequences.append(('-', '-', sequence[::-1]))

    print(f"{len(new_sequences)} new sequences generated")
    # Specify the output file path
    output_fasta_file = "./data/sequences/artificial_positive_samples/sequences_reversed.fasta"

    # Write the sequences to the FASTA file
    write_fasta(new_sequences, output_fasta_file)


generate_positive_samples_with_sequence_reversed()
