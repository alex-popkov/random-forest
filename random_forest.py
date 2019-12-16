# Random forest algorithm

from random import randrange


class DecisionTree():
    def __init__(self, dataset, max_depth, min_size, features_count):
        self.dataset = dataset
        self.max_depth = max_depth
        self.min_size = min_size
        self.features_count = features_count
        self.root = DecisionTreeNode(dataset, features_count, max_depth, min_size, 0)

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

    def init_node(self, dataset, features_count):  # M*N*sqrt(N) time, sqrt(N) + N + M^2*N*sqrt(N) space
        class_values = list(set(row[-1] for row in dataset))  # N time, N space
        features = []
        range_end = len(dataset[0]) - 1
        while len(features) < features_count:  # sqrt(N) time, sqrt(N) space
            random_id = randrange(range_end)
            if random_id not in features:
                features.append(random_id)
        for index in features:
            for row in dataset:
                tested_groups = get_potential_splitted_groups(index, row[index], dataset)  # N time, N*M space
                gini_index = get_gini_index(tested_groups, class_values)  # N time, C space
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


class CustomRandomForestClassifier():
    def __init__(self, dataset, max_depth, min_size, features_count, trees_count):
        self.treeList = []
        self.init_forest(dataset, max_depth, min_size, features_count, trees_count)

    def predict(self, predicted_data):
        prediction_list = []
        for tree in self.treeList:
            prediction_list.append(tree.predict(predicted_data))

        counter = 0
        most_frequent_class = prediction_list[0]

        for class_id in prediction_list:
            curr_frequency = prediction_list.count(class_id)
            if (curr_frequency > counter):
                counter = curr_frequency
                most_frequent_class = class_id

        return most_frequent_class

    def init_forest(self, dataset, max_depth, min_size, features_count, dataset_sample_count):
        sample_list = self.get_sample_list(dataset, dataset_sample_count)
        for sample in sample_list:
            self.treeList.append(DecisionTree(sample, max_depth, min_size, features_count))

    def get_sample_list(self, source_dataset, sample_count):
        sample_list = []
        dataset_length = len(source_dataset)
        for index in range(sample_count):
            sample = []
            while len(sample) < dataset_length:
                random_id = randrange(dataset_length - 1)
                sample.append(source_dataset[random_id])
            sample_list.append(sample)
        return sample_list


# Helper methods#################################################################################

def get_gini_index(groups, classes):  # N time, C space
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


def get_potential_splitted_groups(index, value, dataset):  # N time, N*M space
    left = []
    right = []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
