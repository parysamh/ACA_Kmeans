import matplotlib.pyplot as plt
import re

# Read clustering_result.txt
with open("clustering_result.txt", "r") as file:
    lines = file.readlines()

points_by_cluster = {}
centroids = {}

# Parse each line
for line in lines:
    line = line.strip()
    match = re.match(r"Cluster (\d+) : \(([^,]+), ([^)\s]+)\) ---> (.+)", line)
    if match:
        cluster_id = int(match.group(1))
        centroid_x = float(match.group(2))
        centroid_y = float(match.group(3))
        point_section = match.group(4)

        # Store centroid
        centroids[cluster_id] = (centroid_x, centroid_y)

        # Find all points
        points = []
        for pt in re.findall(r"Point \(([^,]+), ([^)]+)\)", point_section):
            x = float(pt[0])
            y = float(pt[1])
            points.append((x, y))

        points_by_cluster[cluster_id] = points

# Set up plot
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# Plot points by cluster
for cluster_id, points in points_by_cluster.items():
    xs, ys = zip(*points)
    color = colors[cluster_id % len(colors)]
    plt.scatter(xs, ys, label=f"Cluster {cluster_id}", color=color, s=50)

# Plot centroids
for cluster_id, (cx, cy) in centroids.items():
    color = colors[cluster_id % len(colors)]
    plt.scatter(cx, cy, marker='X', s=200, edgecolors='black', color=color, label=f"Centroid {cluster_id}")

# Format plot
plt.title("K-Means Clustering Visualization")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
