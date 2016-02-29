"""Load training data and perform preprocessing"""


def load_training_data():
    """Load training data and perform preprocessing"""
    with open('data/poker-hand-training-true.data', 'r') as training_file:
        training_lines = [line.split(',') for line in training_file.readlines()]

    # Separate target column (poker hand classes) from training columns
    training_data = []
    raw_target_data = []
    for line in training_lines:
        training_data.append(line[:-1])
        raw_target_data.append(line[-1])

    # Transform target column values from 0-9 to a bit vector
    def bit_vec_transform(num):
        vec = [0]*10
        vec[int(num)] = 1
        return vec
    target_data = [bic_vec_transform(num) for num in raw_target_data]

    return training_data, target_data

if __name__ == '__main__':
    tr, ta = load_training_data()
