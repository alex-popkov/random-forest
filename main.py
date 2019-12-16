import math
from random import seed
from math import sqrt
from random import randrange
import time
from sklearn.ensemble import RandomForestClassifier

import random_forest
import file_reader

seed(42)


###############################################################################################################
def test_custom(dataset, test_dataset, max_depth, min_size, features_count_for_splitting, trees_count, criterion='gini'):
    wins = 0
    train_start = time.time()
    rfc = random_forest.CustomRandomForestClassifier(dataset, max_depth, min_size, features_count_for_splitting,
                                                     trees_count, criterion)

    train_end = time.time()
    for sample in test_dataset:
        prediction = rfc.predict(sample[0: -1])
        if prediction == sample[-1]:
            wins += 1

    predict_end = time.time()
    return int(100 * wins / len(test_dataset)), train_end - train_start, predict_end - train_end


###############################################################################################################
def test_sk(dataset, test_dataset, max_depth, min_size, features_count_for_splitting, trees_count, criterion='gini'):
    wins = 0

    sk_dataset = [row[0: -1] for row in dataset]
    sk_classes = [row[-1] for row in dataset]

    rfc = RandomForestClassifier(n_estimators=trees_count, max_depth=max_depth,
                                 max_features=features_count_for_splitting, criterion=criterion)
    train_start = time.time()

    rfc.fit(sk_dataset, sk_classes)
    train_end = time.time()

    for sample in test_dataset:
        k_prediction = rfc.predict([sample[0: -1]])
        if k_prediction == sample[-1]:
            wins += 1
    predict_end = time.time()

    return int(100 * wins / len(test_dataset)), train_end - train_start, predict_end - train_end


###############################################################################################################
def get_train_and_test_datasets(dataset):
    copy_dataset = list(dataset)
    test_length = int(len(dataset) / 5)
    test_dataset = []

    while len(test_dataset) < test_length:
        random_id = randrange(len(copy_dataset))
        test_dataset.append(copy_dataset.pop(random_id))

    return copy_dataset, test_dataset


###############################################################################################################
def test_sk_vs_custom(dataset, max_depth, min_size, trees_count, criterion):
    count = 1
    float_count = float(count)

    average_custom_accuracy = 0
    average_sk_accuracy = 0
    average_custom_train_time = 0
    average_sk_train_time = 0
    average_custom_predict_time = 0
    average_sk_predict_time = 0

    train_sonar_dataset, test_sonar_dataset = get_train_and_test_datasets(dataset)

    for _ in range(count):
        custom_accuracy, custom_train_time, custom_predict_time = test_custom(train_sonar_dataset, test_sonar_dataset,
                                                                              max_depth, min_size,
                                                                              int(sqrt(len(sonar_dataset[0]) - 1)),
                                                                              trees_count,
                                                                              criterion)
        sk_accuracy, sk_train_time, sk_predict_time = test_sk(train_sonar_dataset, test_sonar_dataset, max_depth,
                                                              min_size, int(sqrt(len(sonar_dataset[0]) - 1)),
                                                              trees_count,
                                                              criterion)
        average_custom_accuracy += custom_accuracy
        average_custom_train_time += custom_train_time
        average_custom_predict_time += custom_predict_time

        average_sk_accuracy += sk_accuracy
        average_sk_train_time += sk_train_time
        average_sk_predict_time += sk_predict_time

    average_custom_accuracy /= float_count
    average_custom_train_time /= float_count
    average_custom_predict_time /= float_count

    average_sk_accuracy /= float_count
    average_sk_train_time /= float_count
    average_sk_predict_time /= float_count

    print('Custom RFC. accuracy:', average_custom_accuracy, '%,', 'training time:', average_custom_train_time,
          ', prediction time:', average_custom_predict_time)
    print('Sklearn RFC. accuracy:', average_sk_accuracy, '%,', 'training time:', average_sk_train_time,
          ', prediction time:', average_sk_predict_time)


###############################################################################################################
def test_time_complexity(dataset, max_depth, min_size, trees_count, multi_n_index=1, multi_p_index=1):
    train_dataset, test_dataset = get_train_and_test_datasets(dataset)

    train_multi_dataset = []
    test_multi_dataset = []
    copy_train_dataset = list(train_dataset)
    copy_test_dataset = list(test_dataset)

    if multi_p_index > 1:
        for row in copy_train_dataset:
            multi_row = list(row)
            for _ in range(multi_p_index - 1):
                multi_row += row
            train_multi_dataset.append(multi_row)

        for row in copy_test_dataset:
            multi_row = list(row)
            for _ in range(multi_p_index - 1):
                multi_row += row
            test_multi_dataset.append(multi_row)
    else:
        train_multi_dataset = list(copy_train_dataset)
        test_multi_dataset = list(copy_test_dataset)

    if multi_n_index > 1:
        for _ in range(multi_n_index - 1):
            train_multi_dataset += train_multi_dataset
            test_multi_dataset += test_multi_dataset

    multi_acc, multi_train_time, multi_predict_time = test_custom(
        train_multi_dataset,
        test_multi_dataset,
        max_depth,
        min_size,
        int(sqrt(len(train_multi_dataset[0]) - 1)),
        trees_count
    )

    acc, train_time, predict_time = test_custom(
        dataset,
        test_dataset,
        max_depth,
        min_size,
        int(sqrt(len(train_dataset[0]) - 1)),
        trees_count
    )

    print('Multi N index:', multi_n_index, ', multi p index: ', multi_p_index, ', deceleration index: train:',
          multi_train_time / train_time,  ', predict:', multi_predict_time / predict_time)


###############################################################################################################
sonar_filename = 'data/sonar/sonar.all-data.csv'
wine_filename = 'data/wine/wine.data.csv'
max_depth = 10
min_size = 1
sonar_file_reader = file_reader.SonarFileReader(sonar_filename)
wine_file_reader = file_reader.WineFileReader(wine_filename)
sonar_dataset = sonar_file_reader.dataset
wine_dataset = wine_file_reader.dataset
trees_count = 50

test_sk_vs_custom(sonar_dataset, max_depth, min_size, trees_count, 'entropy')
test_sk_vs_custom(wine_dataset, max_depth, min_size, trees_count, 'entropy')
# test_time_complexity(sonar_dataset, max_depth, min_size, trees_count, 3, 1)
# test_time_complexity(sonar_dataset, max_depth, min_size, trees_count, 3, 3)
# test_time_complexity(sonar_dataset, max_depth, min_size, trees_count, 1, 3)
