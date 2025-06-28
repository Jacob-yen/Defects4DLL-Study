import os
import openpyxl
from sklearn.cluster import KMeans
import random
import numpy as np
from sklearn.metrics import silhouette_score
from src.tools.logger_utils import LoggerUtils
logger_instance = LoggerUtils.get_instance()
logger = logger_instance.logger
import json
import sys

class FilterTool:
    def __init__(self, result_dicts, fail_cov_res_file):
        self.sorted_result_dicts = self.sort_dict(result_dicts)
        self.fail_cov_res_file = fail_cov_res_file
        self.clustering_tool = None
        self.clustered_dict_result = None

    def sort_dict(self, result_dicts):
        sorted_items = sorted(result_dicts.items(), key=lambda item: item[1], reverse=True)
        return dict(sorted_items)

    # 进行聚类
    def cluster(self):
        result_values = list(self.sorted_result_dicts.values())
        max_n_clusters = len(set(result_values))
        if max_n_clusters < 2:
            logger.error("The number of diversified test cases cannot support clustering! ")
            return False
        keys = list(self.sorted_result_dicts.keys())
        data = np.array(result_values).reshape(-1, 1)
        best_n_clusters, best_score = self.find_best_n_clusters(data, 2, max_n_clusters)
        self.clustering_tool = ClusteringTool(n_clusters=best_n_clusters)
        self.clustering_tool.fit(data)

        cluster_labels = self.clustering_tool.get_labels()
        clustered_dict = {}
        for key, label in zip(keys, cluster_labels):
            if label not in clustered_dict:
                clustered_dict[label] = {}
            clustered_dict[label][key] = self.sorted_result_dicts.get(key, 0)

        self.clustered_dict_result = clustered_dict
        return True

    # 选择最好的簇
    def find_best_n_clusters(self, data, k_min, k_max):
        best_score = -1
        best_n_clusters = k_min
        for n in range(k_min, k_max):
            clustering_tool = ClusteringTool(n_clusters=n)
            clustering_tool.fit(data)
            cluster_labels = clustering_tool.get_labels()
            score = silhouette_score(data, cluster_labels)

            if score > best_score:
                best_score = score
                best_n_clusters = n

        return best_n_clusters, best_score

    # 每个簇,选择一个测试用例
    def filter_reduction(self):
        # Randomly select one of the clusters with the highest similarity, and then perform a significance test
        clustered_keys = list(self.clustered_dict_result.keys())
        clustere_num = len(clustered_keys)
        file_list_selected = []
        for num in range(0, clustere_num):
            # Obtain the highest similarity of the current cluster and randomly select one
            files_list = list(self.clustered_dict_result.get(clustered_keys[num]).keys())
            # From sorted_ Result_ Filter out files from dicts_ Files and their scores in the list
            filtered_results = {file: self.sorted_result_dicts.get(file, 0) for file in files_list}
            # Find the highest score
            max_score = max(filtered_results.values())
            # Find the file set with the highest score
            best_files = [file for file, score in filtered_results.items() if score == max_score]
            selected_file = random.choice(best_files)
            # Merge the selected ones with the previous ones, and then calculate the results to record the results as significant as before
            file_list_selected.append(selected_file)
        return file_list_selected

    def stratified_sample(self, num_samples_ratio=0.6):
        file_num = len(list(self.sorted_result_dicts.keys()))
        num_samples = round(file_num * num_samples_ratio)
        logger.info(f"num_samples: {num_samples}")
        cluster_scores = {}
        min_cluster_score = sys.maxsize
        label_dict = {}
        for label in self.clustered_dict_result.keys():
            cluster_file_dict = self.clustered_dict_result[label]
            min_cluster_score = min_cluster_score if min_cluster_score < min(cluster_file_dict.values()) else min(cluster_file_dict.values())
            label_dict[label] = list(self.clustered_dict_result[label].keys())
            cluster_average_value = sum(cluster_file_dict.values()) / len(cluster_file_dict)
            cluster_scores[label] = cluster_average_value
        cluster_scores = {key: score - min_cluster_score for key, score in cluster_scores.items()}
        total_score = sum(cluster_scores.values())
        cluster_weights = {key: score / total_score for key, score in cluster_scores.items()}
        # Randomly select one of the clusters with the highest similarity, and then perform a significance test
        clustered_keys = list(self.clustered_dict_result.keys())
        clustere_num = len(clustered_keys)
        file_list_selected = []
        for num in range(0, clustere_num):
            file_num = round(num_samples * cluster_weights[clustered_keys[num]])
            files_list = list(self.clustered_dict_result.get(clustered_keys[num]).keys())[:file_num]
            file_list_selected.extend(files_list)
        return file_list_selected

    def weight_sample(self, num_samples_ratio=0.6):
        file_num = len(list(self.sorted_result_dicts.keys()))
        num_samples = round(file_num * num_samples_ratio)

        cluster_scores = {}
        min_cluster_score = sys.maxsize
        label_dict = {}
        for label in self.clustered_dict_result.keys():
            cluster_file_dict = self.clustered_dict_result[label]
            min_cluster_score = min_cluster_score if min_cluster_score < min(cluster_file_dict.values()) else min(cluster_file_dict.values())
            label_dict[label] = list(self.clustered_dict_result[label].keys())
            cluster_average_value = sum(cluster_file_dict.values()) / len(cluster_file_dict)
            cluster_scores[label] = cluster_average_value
        cluster_scores = {key: score - min_cluster_score for key, score in cluster_scores.items()}
        total_score = sum(cluster_scores.values())
        # logger.info(total_score)
        cluster_weights = {key: score / total_score for key, score in cluster_scores.items()}
        # logger.info(cluster_weights)

        samples = []
        index = 0
        while index < num_samples:
            random_value = np.random.rand()
            # logger.info(f"random_value: {random_value}")
            selected_point = None
            for label, weight in cluster_weights.items():
                if random_value <= weight:
                    selected_cluster = label_dict[label]
                    if len(selected_cluster) == 0:
                        continue
                    selected_point = random.choice(selected_cluster)
                    label_dict[label].remove(selected_point)
                    break
                else:
                    random_value -= weight
            if selected_point is not None:
                samples.append(selected_point)
            index = index + 1
        return samples

    def cover_reduction(self):
        result_dicts = {}
        for key in self.sorted_result_dicts.keys():
            logger.info(f"'{key}': {self.sorted_result_dicts[key]}\n")
            if len(result_dicts) == 0:
                result_dicts[key] = self.sorted_result_dicts[key]
            else:
                is_available = True
                for combine_key in result_dicts.keys():
                    if not self.dist_file_diversity(key, combine_key):
                        logger.info(f"{key} and {combine_key} are completely similar, without diversity")
                        is_available = False
                if is_available:
                    result_dicts[key] = self.sorted_result_dicts[key]
        return result_dicts


    def dist_file_diversity(self, available_file, combine_file):
        with open(available_file, 'r', encoding='utf8') as fp:
            available_data = json.load(fp)
        with open(combine_file, 'r', encoding='utf8') as fp:
            combine_data = json.load(fp)
        with open(self.fail_cov_res_file, 'r', encoding='utf8') as fp:
            fail_data = json.load(fp)
        fail_set = set(fail_data["files"].keys())
        for file in fail_set:
            fail_line_set = set(fail_data["files"][file]["executed_lines_frequency"].keys())
            available_line_set = set()
            combine_line_set = set()
            if file in available_data["files"].keys():
                available_line_set = set(available_data["files"][file]["executed_lines_frequency"].keys())
            if file in combine_data["files"].keys():
                combine_line_set = set(combine_data["files"][file]["executed_lines_frequency"].keys())
            available_intersection_set = fail_line_set.intersection(available_line_set)
            combine_intersection_set = fail_line_set.intersection(combine_line_set)
            if available_intersection_set != combine_intersection_set:
                return True
        return False


    def cluster_status(self):
        for cluster_label, cluster_keys in self.clustered_dict_result.items():
            average_value = 0
            for cluster_key in cluster_keys:
                average_value += self.sorted_result_dicts[cluster_key]
            average_value = average_value / len(cluster_keys)
            # logger.info(f"Cluster {cluster_label}-{cluster_keys} Average Value: {average_value}")
            log_message = f"Cluster {cluster_label}-\n"
            for key, value in cluster_keys.items():
                log_message += f"'{key}': {value}\n"
            log_message += f"Average Value: {average_value}"
            logger.info(log_message)


class ClusteringTool:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.cluster_model = KMeans(n_clusters=self.n_clusters)

    def fit(self, data):
        self.cluster_model.fit(data)

    def predict(self, data):
        return self.cluster_model.predict(data)

    def get_cluster_centers(self):
        return self.cluster_model.cluster_centers_

    def get_labels(self):
        return self.cluster_model.labels_
