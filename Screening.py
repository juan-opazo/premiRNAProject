import matplotlib.pyplot as plt
from FastaManager import FastaManager


class Screening:

    def __init__(self, positive_sequences=None, negative_sequences=None):
        self.list_of_positions_of_positive_samples = []
        self.positive_sequences = positive_sequences
        self.negative_sequences = negative_sequences
        self.positive_map = {0: []}
        self.negative_map = {}
        for seq in positive_sequences:
            if self.positive_map.get(len(seq)):
                self.positive_map[len(seq)].append(seq)
            else:
                self.positive_map[len(seq)] = []
                self.positive_map[len(seq)].append(seq)
        for seq in negative_sequences:
            if self.negative_map.get(len(seq)):
                self.negative_map[len(seq)].append(seq)
            else:
                self.negative_map[len(seq)] = []
                self.negative_map[len(seq)].append(seq)
        self.nt_with_most_data = 0
        for nt in self.positive_map.keys():
            if len(self.positive_map[nt]) > len(self.positive_map[self.nt_with_most_data]):
                self.nt_with_most_data = nt

    def get_sequence_for_screening(self, nt=None, k=None):
        self.list_of_positions_of_positive_samples = []
        if not nt:
            nt = self.nt_with_most_data
        sequence = ""
        positive_samples = self.positive_map.get(nt)
        if not positive_samples:
            return sequence
        k = int(len(self.negative_sequences) / len(positive_samples)) if not k else k
        for idx, pos_sample in enumerate(positive_samples):
            sequence += ''.join(self.negative_sequences[idx * k: idx * k + k])
            self.list_of_positions_of_positive_samples.append(len(sequence))
            sequence += pos_sample
        # check if there is missing negative sample to extend to the end of total sequence
        if (len(positive_samples) - 1) * k < len(self.negative_sequences):
            sequence + ''.join(self.negative_sequences[(len(positive_samples) - 1) * k:-1])

        return ''.join(sequence)

    def get_map_nt_number_of_sequence(self):
        map_to_return = {}
        for nt in self.positive_map.keys():
            map_to_return[nt] = len(self.positive_map.get(nt))
        return map_to_return


input_directory = "./data/sequences"

# Get sequences
X_positive_sequences = FastaManager.get_all_sequences(input_directory + "/positive_samples")
X_artificial_positive_sequences = FastaManager.get_all_sequences(input_directory + "/artificial_positive_samples")
X_negative_sequences = FastaManager.get_all_sequences(input_directory + "/negative_samples")
X = X_positive_sequences + X_artificial_positive_sequences + X_negative_sequences
"""
print(f"positive sequences: {len(X_positive_sequences)}")
print(f"artificial positive sequences: {len(X_artificial_positive_sequences)}")
print(f"negative sequences: {len(X_negative_sequences)}")
print("------------------------------------------")
screening = Screening(X_positive_sequences, X_negative_sequences)
print(f"positive map: {screening.get_map_nt_number_of_sequence()}")
print(f"nt with most positive sequence: {screening.nt_with_most_data}")
print(f"sequence for screening len: {len(screening.get_sequence_for_screening())}")
print(f"{len(screening.list_of_positions_of_positive_samples)} positions of positive samples in sequence: {screening.list_of_positions_of_positive_samples[0:5]}...")
print(f"sequence 60 nt for screening len: {len(screening.get_sequence_for_screening(60))}")
print(f"{len(screening.list_of_positions_of_positive_samples)} positions of positive samples in sequence: {screening.list_of_positions_of_positive_samples[0:5]}...")
# print(f"sequence for screening: {screening.get_sequence_for_screening()}")

D = dict(sorted(screening.get_map_nt_number_of_sequence().items()))

plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()))
plt.show()
"""