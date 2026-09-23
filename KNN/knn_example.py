# K-Nearest Neighbors (KNN) regression without external libraries

# Five training rows: [student_id, hours_studied, internal_mark, final_mark]
training_data = [
    ["S1", 2, 40, 45],
    ["S2", 3, 45, 50],
    ["S3", 4, 50, 55],
    ["S4", 6, 60, 65],
    ["S5", 7, 65, 70],
]

query_point = [5, 55]
k = 3


def euclidean_distance(point_a, point_b):
    """Return the Euclidean distance between two numeric points."""
    squared_distance = 0
    for index in range(len(point_a)):
        difference = point_a[index] - point_b[index]
        squared_distance += difference * difference
    return squared_distance ** 0.5


def predict_final_mark(data, query, neighbors_count):
    distances = []

    for row in data:
        features = row[1:3]
        final_mark = row[3]
        distance = euclidean_distance(features, query)
        distances.append([distance, final_mark, row[0]])

    # Sort manually by distance so no machine-learning library is needed.
    for current_index in range(len(distances)):
        nearest_index = current_index
        for next_index in range(current_index + 1, len(distances)):
            if distances[next_index][0] < distances[nearest_index][0]:
                nearest_index = next_index
        distances[current_index], distances[nearest_index] = (
            distances[nearest_index],
            distances[current_index],
        )

    nearest_neighbors = distances[:neighbors_count]
    total_marks = 0
    for neighbor in nearest_neighbors:
        total_marks += neighbor[1]

    predicted_mark = total_marks / neighbors_count
    return predicted_mark, nearest_neighbors


prediction, nearest_neighbors = predict_final_mark(training_data, query_point, k)

print("Training rows:", len(training_data))
print("Query point [hours studied, internal mark]:", query_point)
print("K:", k)
print("Nearest neighbors:", nearest_neighbors)
print("Predicted final mark:", prediction)
