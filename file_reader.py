from csv import reader

class WineFileReader():
    def __init__(self, filename):
        self.raw_dataset = load_csv(filename)
        for row in self.raw_dataset:
            row.append(row.pop(0))
        self.dataset = self.get_processed_dataset(self.raw_dataset)

    def get_processed_dataset(self, raw_dataset):
        dataset = convert_values_to_float(raw_dataset)
        return dataset


##################################################################################################
class SonarFileReader():
    def __init__(self, filename):
        raw_dataset = load_csv(filename)
        self.dataset = self.get_processed_dataset(raw_dataset)

    def get_processed_dataset(self, raw_dataset):
        dataset_with_float_values = convert_values_to_float(raw_dataset)
        dataset = convert_class_name_to_int(dataset_with_float_values, len(dataset_with_float_values[0]) - 1)
        return dataset

# Helper methods#################################################################################

def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_file = reader(file)
        for row in csv_file:
            if row:
                dataset.append(row)
    return dataset


def convert_values_to_float(raw_dataset):
    row_length = len(raw_dataset[0]) - 1
    for row in raw_dataset:
        for col in range(0, row_length):
            row[col] = float(row[col].strip())
    return raw_dataset


def convert_class_name_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return dataset