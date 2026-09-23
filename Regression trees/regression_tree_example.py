# Regression tree for predicting final marks without external libraries

training_data = [
    ["S1", 2, 40, 45],
    ["S2", 3, 45, 50],
    ["S3", 4, 50, 55],
    ["S4", 6, 60, 65],
    ["S5", 7, 65, 70],
]

query_point = [5, 55]


def mean(values):
    """Return the arithmetic mean of a list of numbers."""
    return sum(values) / len(values)


def variance(values):
    """Return the variance of a numeric list."""
    if len(values) < 2:
        return 0
    avg = mean(values)
    total = 0
    for value in values:
        total += (value - avg) ** 2
    return total / len(values)


def best_split(rows):
    """Find the feature and threshold that minimize the weighted target variance."""
    best_score = float("inf")
    best_feature = None
    best_threshold = None
    best_left = []
    best_right = []

    if not rows:
        return None

    target_values = [row[-1] for row in rows]
    current_variance = variance(target_values)

    for feature_index in range(len(rows[0]) - 1):
        values = sorted(set(row[feature_index] for row in rows))

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            left_rows = []
            right_rows = []

            for row in rows:
                if row[feature_index] <= threshold:
                    left_rows.append(row)
                else:
                    right_rows.append(row)

            if not left_rows or not right_rows:
                continue

            left_targets = [row[-1] for row in left_rows]
            right_targets = [row[-1] for row in right_rows]

            weighted_variance = (
                (len(left_rows) / len(rows)) * variance(left_targets)
                + (len(right_rows) / len(rows)) * variance(right_targets)
            )

            if weighted_variance < best_score:
                best_score = weighted_variance
                best_feature = feature_index
                best_threshold = threshold
                best_left = left_rows
                best_right = right_rows

    if best_feature is None:
        return None

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": best_left,
        "right": best_right,
        "score": best_score,
        "parent_variance": current_variance,
    }


class TreeNode:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


def build_tree(rows, max_depth=2, depth=0):
    """Recursively build a regression tree using a simple CART-style split."""
    targets = [row[-1] for row in rows]

    if not rows:
        return TreeNode(value=0)

    if depth >= max_depth or len(rows) <= 1 or len(set(targets)) == 1:
        return TreeNode(value=mean(targets))

    split = best_split(rows)
    if split is None:
        return TreeNode(value=mean(targets))

    left_child = build_tree(split["left"], max_depth=max_depth, depth=depth + 1)
    right_child = build_tree(split["right"], max_depth=max_depth, depth=depth + 1)

    return TreeNode(
        feature=split["feature"],
        threshold=split["threshold"],
        left=left_child,
        right=right_child,
    )


def predict(node, row):
    """Predict the target value for a row using the regression tree."""
    if node.value is not None:
        return node.value

    if row[node.feature] <= node.threshold:
        return predict(node.left, row)
    return predict(node.right, row)


# Use only the feature columns [hours_studied, internal_mark]
# but keep student_id for printing only.
feature_rows = [row[1:4] for row in training_data]

regression_tree = build_tree(feature_rows, max_depth=2)

predicted_final_mark = predict(regression_tree, [query_point[0], query_point[1]])

print("Training rows:", len(feature_rows))
print("Query point [hours studied, internal mark]:", query_point)
print("Regression tree prediction for final mark:", predicted_final_mark)
