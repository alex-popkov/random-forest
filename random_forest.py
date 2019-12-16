# Random forest algorithm
from csv import reader
from random import randrange
from random import seed
from math import sqrt

####################################################################################################
# class WineFileReader():
#     def __init__(self, filename):
#         raw_dataset = self.load_csv(filename)
#         self.dataset = self.get_processed_dataset(raw_dataset)
#         for row in self.dataset:
#             class_id = row.pop()
#
#
#     def load_csv(self, filename):
#         dataset = list()
#         with open(filename, 'r') as file:
#             csv_file = reader(file)
#             for row in csv_file:
#                 if row:
#                     dataset.append(row)
#         return dataset
#
#     def get_processed_dataset(self, raw_dataset):
#         dataset_with_float_values = self.convert_values_to_float(raw_dataset)
#         dataset = self.convert_class_name_to_int(dataset_with_float_values, len(dataset_with_float_values[0]) - 1)
#         return dataset
#
#     def convert_values_to_float(self, raw_dataset):
#         row_length = len(raw_dataset[0]) - 1
#         for row in raw_dataset:
#             for col in range(0, row_length):
#                 row[col] = float(row[col].strip())
#         return raw_dataset
#
#     def convert_class_name_to_int(self, dataset, column):
#         class_values = [row[column] for row in dataset]
#         unique = set(class_values)
#         lookup = dict()
#         for i, value in enumerate(unique):
#             lookup[value] = i
#         for row in dataset:
#             row[column] = lookup[row[column]]
#         return dataset


##################################################################################################

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


##################################################################################################

class DecisionTree():
    def __init__(self, dataset, max_depth, min_size, features_count):
        self.dataset = dataset
        self.max_depth = max_depth
        self.min_size = min_size
        self.features_count = features_count
        self.root = DecisionTreeNode(dataset, features_count, max_depth, min_size, 0)

    # Print a decision tree
    def print_tree(self, node, depth=0):
        if isinstance(node, DecisionTreeNode):
            print('%s[X%d < %.3f]' % (depth * ' ', (node.split_index + 1), node.split_value))
            self.print_tree(node.left_child, depth + 1)
            self.print_tree(node.right_child, depth + 1)
        else:
            print('%s[%s]' % (depth * ' ', node.y))

    def predict(self, predicted_data, node=None):
        node = node or self.root
        if predicted_data[node.split_index] < node.split_value:
            if isinstance(node.left_child, DecisionTreeNode):
                return self.predict(predicted_data, node.left_child)
            else:
                return node.left_child.y
        else:
            if isinstance(node.right_child, DecisionTreeNode):
                return self.predict(predicted_data, node.right_child)
            else:
                return node.right_child.y


#################################################################################################

class DecisionTreeNode():
    def __init__(self, dataset, features_count, max_depth, min_size, current_depth):
        self.left_child = None
        self.right_child = None
        self.splitted_groups = None
        self.split_index = 1
        self.split_value = 1
        self.split_score = float('inf')
        self.dataset = dataset
        self.features_count = features_count
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.min_size = min_size

        self.init_node(self.dataset, self.features_count)
        self.create_child_nodes()

    def init_node(self, dataset, features_count):
        class_values = list(set(row[-1] for row in dataset))
        features = list()
        range_end = len(dataset[0]) - 1
        while len(features) < features_count:
            random_id = randrange(range_end)
            if random_id not in features:
                features.append(random_id)
        for index in features:
            for row in dataset:
                tested_groups = get_potential_splitted_groups(index, row[index], dataset)
                gini_index = get_gini_index(tested_groups, class_values)
                if gini_index < self.split_score:
                    self.split_index = index
                    self.split_value = row[index]
                    self.split_score = gini_index
                    self.splitted_groups = tested_groups

    def create_child_nodes(self):
        left, right = self.splitted_groups
        self.splitted_groups = None
        if not left or not right:
            self.left_child = self.right_child = TerminalTreeNode(left + right)
            return
        if self.current_depth >= self.max_depth:
            self.left_child = TerminalTreeNode(left)
            self.right_child = TerminalTreeNode(right)
            return
        if len(left) <= self.min_size:
            self.left_child = TerminalTreeNode(left)
        else:
            self.left_child = DecisionTreeNode(
                left, self.features_count, self.max_depth, self.min_size, self.current_depth + 1
            )
        if len(right) <= self.min_size:
            self.right_child = TerminalTreeNode(right)
        else:
            self.right_child = DecisionTreeNode(
                right, self.features_count, self.max_depth, self.min_size, self.current_depth + 1
            )


##################################################################################################

class TerminalTreeNode():
    def __init__(self, group):
        self.y = self.get_y(group)

    def get_y(self, group):
        class_list = [row[-1] for row in group]
        counter = 0
        most_frequent_class = class_list[0]

        for class_id in class_list:
            curr_frequency = class_list.count(class_id)
            if (curr_frequency > counter):
                counter = curr_frequency
                most_frequent_class = class_id

        return most_frequent_class


##################################################################################################


class RandomForest():
    def __init__(self, dataset, max_depth, min_size, features_count, trees_count):
        self.treeList = []
        self.init_forest(dataset, max_depth, min_size, features_count, trees_count)

    def predict(self, predicted_data):
        prediction_list = []
        for tree in self.treeList:
            prediction_list.append(tree.predict(predicted_data))

        # print(prediction_list)
        class_list = [p for p in prediction_list]
        counter = 0
        most_frequent_class = class_list[0]

        for class_id in class_list:
            curr_frequency = class_list.count(class_id)
            if (curr_frequency > counter):
                counter = curr_frequency
                most_frequent_class = class_id

        return most_frequent_class

    def init_forest(self, dataset, max_depth, min_size, features_count, dataset_subsample_count):
        subsample_list = self.get_subsample_list(dataset, dataset_subsample_count)
        for subsample in subsample_list:
            self.treeList.append(DecisionTree(subsample, max_depth, min_size, features_count))

    def get_subsample_list(self, source_dataset, sample_count):
        sample_list = []
        dataset_length = len(source_dataset)
        for index in range(sample_count):
            sample = []
            while len(sample) < dataset_length:
                random_id = randrange(dataset_length - 1)
                sample.append(source_dataset[random_id])
            sample_list.append(sample)
        return sample_list

        # dataset_split = list()
        # dataset_copy = list(source_dataset)
        # fold_size = int(len(source_dataset) / sample_count)
        # for i in range(sample_count):
        #     fold = list()
        #     while len(fold) < fold_size:
        #         index = randrange(len(dataset_copy))
        #         fold.append(dataset_copy.pop(index))
        #     dataset_split.append(fold)
        # return dataset_split

# Helper methods#################################################################################


# Calculate the Gini index for a split dataset
def get_gini_index(groups, classes):
    total_samples = float(sum([len(group) for group in groups]))
    gini_index = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        proportion_sum = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            proportion_sum += proportion * proportion
        gini_index += (1.0 - proportion_sum) * (size / total_samples)
    return gini_index


# Split a dataset based on an attribute and an attribute value
def get_potential_splitted_groups(index, value, dataset):
    left = list()
    right = list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


#######################################################################################################
sonar_filename = 'data/sonar/sonar.all-data.csv'
wine_filename = 'data/wine/wine.data.csv'
max_depth = 10
min_size = 1
sample_size = 1.0
# seed(42)
sonar_file_reader = SonarFileReader(sonar_filename)
# wine_file_reader = WineFileReader(sonar_filename)
sonar_dataset = sonar_file_reader.dataset
# wine_dataset =wine_file_reader.dataset
features_count_for_splitting = int(sqrt(len(sonar_dataset[0]) - 1))
trees_count = 5

# calculation accuracy ###############################################################################
wins = 0
tries = int(len(sonar_dataset) / 2)
for i in range(tries):
    copy_dataset = list(sonar_dataset)
    predicted_data = copy_dataset.pop(i)
    rf = RandomForest(copy_dataset, max_depth, min_size, features_count_for_splitting, trees_count)
    prediction = rf.predict(predicted_data)
    if prediction == predicted_data[-1]:
        wins += 1

print(100 * wins/tries)