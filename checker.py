#!/usr/bin/env python3
import sys

import numpy as np


def read_file(file_path):
    data = np.loadtxt(file_path)
    data = data[:, 1:]
    return data


def calculate_difference(expected, actual, threshold=1e-5):
    difference = np.abs(expected - actual)
    difference[difference < threshold] = 0
    return difference

def calculate_euclidean_distance(expected, actual, threshold=1e-5):
    euclidean_distance = np.sqrt(np.sum(np.square(expected - actual), axis=1))
    euclidean_distance[euclidean_distance < threshold] = 0
    return euclidean_distance
    

def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python compare_files.py <expected_file_path> <actual_file_path> [threshold]"
        )
        sys.exit(1)

    expected_file_path = sys.argv[1]
    actual_file_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5

    expected_data = read_file(expected_file_path)
    actual_data = read_file(actual_file_path)

    # check if dimensions match
    if expected_data.shape != actual_data.shape:
        print("Error: Dimensions of expected and actual files do not match.")
        sys.exit(1)

    difference_grid = calculate_difference(expected_data, actual_data, threshold)
    euclidean_distance = calculate_euclidean_distance(expected_data, actual_data, threshold)
    print("Difference Grid")
    print_grid(difference_grid)
    print("Euclidean Distance")
    for i in range(len(euclidean_distance)):
        # if euclidean_distance[i] == 0:
        #     print(f'Distance between expected[{i}] and actual[{i}] = Good Match')
        # else:
        print(f'Distance between expected[{i}] and actual[{i}] = {euclidean_distance[i]}')

if __name__ == "__main__":
    main()