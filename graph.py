#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Load points from file with updated format
def load_points(file_name):
    points = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        # Read number of points (first line)
        num_points = int(lines[0].strip())

        # Read each point from subsequent lines
        for line in lines[1:num_points + 1]:  # Skip the first line
            _, x, y = line.strip().split()
            points.append([float(x), float(y)])

    return np.array(points)

# Load clusters from file
def load_clusters(file_name):
    clusters = []
    with open(file_name, 'r') as file:
        for line in file:
            _, x, y = line.strip().split()
            clusters.append([float(x), float(y)])
    return np.array(clusters)

# Load points and initial centroids from respective files
points = load_points('input/random-n64-d2-c4.txt')
clusters = load_clusters('out.txt')

# Plotting the points and centroids
plt.figure(figsize=(8, 6))

# Plot points (same color)
plt.scatter(points[:, 0], points[:, 1], c='blue', label='Points', alpha=0.5)

# Plot clusters (different color, same color for all clusters)
plt.scatter(clusters[:, 0], clusters[:, 1], c='red', marker='X', s=200, label='Clusters')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Points and Clusters Visualization')
plt.legend()
plt.grid()
plt.show()
