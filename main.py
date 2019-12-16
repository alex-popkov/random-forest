from random import seed
from math import sqrt
import time
from sklearn.ensemble import RandomForestClassifier

import random_forest
import file_reader

#######################################################################################################
sonar_filename = 'data/sonar/sonar.all-data.csv'
wine_filename = 'data/wine/wine.data.csv'
max_depth = 10
min_size = 1
sample_size = 1.0
seed(42)
sonar_file_reader = file_reader.SonarFileReader(sonar_filename)
wine_file_reader = file_reader.WineFileReader(wine_filename)
sonar_dataset = sonar_file_reader.dataset
wine_dataset = wine_file_reader.dataset
features_count_for_splitting = int(sqrt(len(wine_dataset[0]) - 1))
trees_count = 5

# calculation accuracy ###############################################################################
wins = 0
sk_wins = 0
count = len(wine_dataset)
speed_times = 0

for i in range(count):
    copy_dataset = list(wine_dataset)
    predicted_data = copy_dataset.pop(i)
    start = time.time()
    rf = random_forest.CustomRandomForestClassifier(copy_dataset, max_depth, min_size, features_count_for_splitting, trees_count)
    prediction = rf.predict(predicted_data)
    end = time.time()
    if prediction == predicted_data[-1]:
        wins += 1

    sk_dataset = [row[0:-1] for row in copy_dataset]
    sk_classes = [row[-1] for row in copy_dataset]

    sk_start = time.time()
    clf = RandomForestClassifier(n_estimators=trees_count, max_depth=max_depth,
                                 max_features=features_count_for_splitting)
    clf.fit(sk_dataset, sk_classes)
    sk_prediction = clf.predict([predicted_data[0:-1]])
    sk_end = time.time()

    if sk_prediction == predicted_data[-1]:
        sk_wins += 1

    speed_times += (end - start) / (sk_end - sk_start)

print(speed_times / count)
print('sk', 100 * sk_wins / count)
print('my', 100 * wins / count)
