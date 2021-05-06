module ParseRF

export parse_rf

using JSON
using Combinatorics


struct decision_tree
    features::Vector{Int}
    tree::Dict{Int,Vector{Int}}
    leaf_idx::Vector{Bool}
    internal_node_features::Vector{Int}
end


"""
    traverse_tree(1, 2, tree)

Walk down the tree from feature_1 and return true is ever encounter feature_2.
"""
function traverse_tree(
    feature_1::Int, 
    feature_2::Int, 
    tree::decision_tree
)::Bool
    function recursive_search(node)
        children = tree.tree[node]
        if any([i == feature_2 for i in children])
            return true
        end
        if all([tree.leaf_idx[i] for i in children])
            return false
        end
        children = filter(x -> !tree.leaf_idx[x], children) # remove leaves
        if length(children) > 0            
            return any([recursive_search(child) for child in children])
        else
            return false
        end
    end
    
    return recursive_search(feature_1)
end


"""
    linked_features(tree, [1, 2])

Determine whether the feature pair are linked in a tree.
"""
function linked_features(
    tree::decision_tree, 
    feature_pair::Vector{Int}, 
    both_permutations::Bool=false
)::Bool
    if !all([f in tree.internal_node_features for f in feature_pair])
        return false
    end

    feature_1_id = findall(tree.features .== feature_pair[1])[1]
    feature_2_id = findall(tree.features .== feature_pair[2])[1]

    # start from first node and walk down the tree
    same_path = traverse_tree(feature_1_id, feature_2_id, tree)
    if same_path
        return true
    end

    # start from second node and walk down the tree
    if both_permutations
        same_path = traverse_tree(feature_2_id, feature_1_id, tree)
        if same_path
            return true
        end
    end

    return false
end


"""
    relevant_trees(trees, [1, 2])

Get every tree in which the feature pair are linked.
"""
function relevant_trees(
    trees::Vector{decision_tree}, 
    feature_pairs::Vector{Vector{Int}}
)::Dict{Vector{Int},Vector{Bool}}
    d = Dict()
    for fp in feature_pairs
        d[fp] = [linked_features(tree, fp) for tree in trees]
    end
    return d
end


"""
    co_occuring_feature_pairs(trees, feature_pairs)

Determine which pairs of features are linked in the decision path of each tree.
"""
function co_occuring_feature_pairs(
    trees::Vector{decision_tree},
    feature_pairs::Vector{Vector{Int}}
)::Dict

    feature_pairs = [sort([i for i in j]) for j in feature_pairs]
    reverse_feature_pairs = [reverse(j) for j in feature_pairs]

    tree_idx_1 = relevant_trees(trees, feature_pairs)
    tree_idx_2 = relevant_trees(trees, reverse_feature_pairs)
    all_fp_tree_matches = merge(tree_idx_1, tree_idx_2) # combine

    return  Dict(
        i => j for (i, j) in all_fp_tree_matches if length(trees[j]) > 0
    )
end

    
"""
    valid_feature_pair(3, 6, alphabet_size=20)

Calculate whether two features are at the same position in a sequence.
"""
function valid_feature_pair(
    feature_1::Int,
    feature_2::Int;
    alphabet_size::Int=20
)::Bool
    feature_1_position = ceil((feature_1 + 1) / alphabet_size)
    feature_2_position = ceil((feature_2 + 1) / alphabet_size)
    return feature_1_position != feature_2_position
end
    

"""
    generate_feature_pairs(included_features, check_positions=true)

Get every valid pair of features
"""
function generate_feature_pairs(
    included_features::Vector;
    check_positions::Bool=true,
    alphabet_size::Int=20
)::Vector{Vector{Int}}
    feature_pairs = collect(
        combinations(included_features, 2)
        )
    if !check_positions
        return feature_pairs
    else
        return [
            i for i in feature_pairs
                if valid_feature_pair(i[1], i[2], alphabet_size=alphabet_size)
            ] 
    end
end
         

"""
    extract_rf_features(trees)

Get the ids of the features which were used by the random forest model.
"""
function extract_rf_features(trees::Vector{decision_tree})::Vector{Int}
    all_features = [
        tree.internal_node_features for tree in trees
    ]
    all_features = reduce(vcat, all_features) # flatten array
    return unique(all_features)
end


"""
    convert_json_rf(trees)

Convert json data to vector of decision_tree.
"""
function convert_json_rf(trees::Dict)::Vector{decision_tree}

    formatted_trees = Vector{decision_tree}()
    for tree in trees["estimators"]
        features = convert(Array{Int,1}, tree["features"])
        internal_node_features = convert(
            Array{Int,1}, tree["internal_node_features"]
        )
        leaf_idx = convert(Array{Bool,1}, tree["leaf_idx"])
        tree_ = Dict()
        for (k, v) in tree["tree"]
            tree_[parse(Int, k)] = convert(Array{Int,1}, v)
        end

        push!(formatted_trees,
            decision_tree(features, tree_, leaf_idx, internal_node_features)
        )
    end

    return formatted_trees
end


"""
    trees = load_rf_json("examples/rf.json")

Read in a random forest json file.
Should be the same format as examples/rf.json.
"""
function load_rf_json(file_path::String)::Dict
    trees = JSON.parsefile(file_path)
    return trees
end


"""
    parse_rf("rf.json")

Read in a json representation of a random forest model.
"""
function parse_rf(rf_json::String)
    trees = load_rf_json(rf_json)
    formatted_trees = convert_json_rf(trees)
    included_features = extract_rf_features(formatted_trees)
    feature_pairs = generate_feature_pairs(included_features)
    fp_trees = co_occuring_feature_pairs(formatted_trees, feature_pairs)
    return formatted_trees, included_features, feature_pairs, fp_trees
end
            
end # module
