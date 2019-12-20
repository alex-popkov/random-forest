from math import sqrt
from random import randrange

import random_forest
import file_reader

###################################################################

sonar_filename = 'data/sonar/sonar.all-data.csv'
wine_filename = 'data/wine/wine.data.csv'
max_depth = 10
min_size = 1
sonar_file_reader = file_reader.SonarFileReader(sonar_filename)
wine_file_reader = file_reader.WineFileReader(wine_filename)
sonar_dataset = sonar_file_reader.dataset
wine_dataset = wine_file_reader.dataset
trees_count = 50


########################################################################

def get_train_and_test_datasets(dataset):
    copy_dataset = list(dataset)
    test_length = int(len(dataset) / 5)
    test_dataset = []

    while len(test_dataset) < test_length:
        random_id = randrange(len(copy_dataset))
        test_dataset.append(copy_dataset.pop(random_id))

    return copy_dataset, test_dataset


########################################################################


def demo():
    train_dataset, test_dataset = get_train_and_test_datasets(wine_dataset)
    wins = 0
    rfc = random_forest.CustomRandomForestClassifier(train_dataset, max_depth, min_size,
                                                     int(sqrt(len(train_dataset[0]) - 1)),
                                                     trees_count, 'gini')

    for sample in test_dataset:
        prediction = rfc.predict(sample[:-1])
        if prediction == sample[-1]:
            wins += 1

    print(int(100 * wins / len(test_dataset)))
