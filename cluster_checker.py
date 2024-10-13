#!/usr/bin/env python3
import sys
import numpy as np
import argparse
from scipy.spatial.distance import euclidean

def read_centroids(file_path):
    """
    Reads centroids from a file. Each line is a centroid, with values separated by spaces.
    """
    with open(file_path, 'r') as file:
        centroids = [np.array(list(map(float, line.strip().split()[1:]))) for line in file]
    return centroids

def centroid_within_threshold(centroid1, centroid2, threshold):
    """
    Checks if the distance between two centroids is within the given threshold.
    """
    return euclidean(centroid1, centroid2) <= threshold

def centroid_distance(centroid1, centroid2):
    return euclidean(centroid1, centroid2)

def match_centroids(centroids1, centroids2, threshold):
    """
    Matches centroids from centroids1 to centroids2 based on the threshold.
    Each centroid must match exactly one centroid from the other set within the threshold.
    """
    matched = []
    unmatched_centroids2 = centroids2.copy()

    for i, centroid1 in enumerate(centroids1):
        matched_to = None
        closest_match_dis = sys.float_info.max
        closest_match = None
        for j, centroid2 in enumerate(unmatched_centroids2):
            if centroid_within_threshold(centroid1, centroid2, threshold):
                matched_to = j
                matched.append((i, centroid1, centroid2))  # Store the matching
                break
            elif centroid_distance(centroid1, centroid2) < closest_match_dis:
                closest_match = j
                closest_match_dis = centroid_distance(centroid1, centroid2)


        if matched_to is not None:
            unmatched_centroids2.pop(matched_to)
        elif closest_match is not None:
            print(f"Found no match for centroid {i}, closest match is off by a distance of {closest_match_dis}")
        # else:
        #     print(f"Centroid {i} from file 1 did not match any centroid in file 2 within the threshold.")

    if unmatched_centroids2:
        print(f"Unmatched centroids in file 2: {unmatched_centroids2}")
    else:
        print(f"All Centroids matched within the threshold: {threshold}")

    return matched

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Compare two sets of centroids within a given threshold.')
    parser.add_argument('file1', type=str, help='Path to the first file containing centroids')
    parser.add_argument('file2', type=str, help='Path to the second file containing centroids')
    parser.add_argument('threshold', type=float, help='Threshold distance to compare centroids')
    
    args = parser.parse_args()
    
    # Read centroids from the files
    centroids1 = read_centroids(args.file1)
    centroids2 = read_centroids(args.file2)
    
    # Check if the two files have the same number of centroids
    if len(centroids1) != len(centroids2):
        print(f"Error: The two files have different numbers of centroids (File 1: {len(centroids1)}, File 2: {len(centroids2)}).")
        return
    
    # Match centroids between the two sets
    matched_centroids = match_centroids(centroids1, centroids2, args.threshold)

    # Print the results
    # if matched_centroids:
    #     print("\nMatched centroids (within threshold):")
    #     for i, centroid1, centroid2 in matched_centroids:
    #         print(f"Centroid {i} from file 1 matches a centroid from file 2")
    #         print(f"Centroid 1: {centroid1}")
    #         print(f"Centroid 2: {centroid2}\n")

if __name__ == "__main__":
    main()
