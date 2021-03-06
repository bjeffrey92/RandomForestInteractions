import pickle
import json
import sys
from typing import List

from numpy import zeros
from sklearn.tree import DecisionTreeRegressor


class DecisionTree:
    def __init__(self, dt: DecisionTreeRegressor):
        self.n_nodes = dt.tree_.node_count
        self.features = dt.tree_.feature
        self.values = dt.tree_.value.squeeze()
        self.tree = {}  # type: ignore
        self.leaf_idx = zeros(shape=self.n_nodes, dtype=bool)

        children_left = dt.tree_.children_left
        children_right = dt.tree_.children_right
        stack = [0]  # start with the root node id (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id = stack.pop()

            children_left_ids = children_left[node_id]
            children_right_ids = children_right[node_id]

            is_split_node = children_left_ids != children_right_ids
            # If a split node, append left and right children to `stack`
            if is_split_node:
                stack.append(children_left_ids.tolist())
                stack.append(children_right_ids.tolist())

                # julia is indexed from 1
                jl_node_id = node_id + 1
                jl_children_left_ids = (children_left_ids + 1).tolist()
                jl_children_right_ids = (children_right_ids + 1).tolist()

                self.tree.setdefault(jl_node_id, []).append(
                    jl_children_left_ids
                )
                self.tree.setdefault(jl_node_id, []).append(
                    jl_children_right_ids
                )

            else:
                self.leaf_idx[node_id] = True

        self.internal_node_features = self.features[~self.leaf_idx]

        # convert from numpy array to list so can be json serialized
        self.features = self.features.tolist()
        self.values = self.values.tolist()
        self.leaf_idx = self.leaf_idx.tolist()
        self.internal_node_features = self.internal_node_features.tolist()


def write_json(model: List[DecisionTree], output_file: str):
    output = {}  # type: ignore
    output["estimators"] = []
    for tree in model:
        output["estimators"].append(
            {
                "tree": tree.tree,
                "features": tree.features,
                "values": tree.values,
                "leaf_idx": tree.leaf_idx,
                "internal_node_features": tree.internal_node_features,
            }
        )

    with open(output_file, "w") as a:
        json.dump(output, a)


def main():
    model_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(model_file, "rb") as a:
        model = pickle.load(a)

    trees = [DecisionTree(dt) for dt in model.estimators_]

    write_json(trees, output_file)


if __name__ == "__main__":
    main()
