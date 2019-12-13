# Random forest algorithm
from csv import reader


####################################################################################################
class SonarFileReader():
    def __init__(self, filename):
        raw_dataset = self.load_csv(filename)
        self.dataset = self.get_processed_dataset(raw_dataset)

    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_file = reader(file)
            for row in csv_file:
                if row:
                    dataset.append(row)
        return dataset

    def get_processed_dataset(self, raw_dataset):
        dataset_with_float_values = self.convert_values_to_float(raw_dataset)
        dataset = self.convert_class_name_to_int(dataset_with_float_values, len(dataset_with_float_values[0]) - 1)
        return dataset

    def convert_values_to_float(self, raw_dataset):
        row_length = len(raw_dataset[0]) - 1
        for row in raw_dataset:
            for col in range(0, row_length):
                row[col] = float(row[col].strip())
        return raw_dataset

    def convert_class_name_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return dataset