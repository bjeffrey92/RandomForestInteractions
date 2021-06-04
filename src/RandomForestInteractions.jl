module RandomForestInteractions

include("ParseRF.jl")

using DataFrames
using HypothesisTests


"""
    paired_selection_frequency(trees, feature_pairs)

Calculate the paired selection frequency test p value for each pair of features in trees of the random forest.
"""
function paired_selection_frequency(
    trees::Vector{ParseRF.decision_tree},
    feature_pairs::Vector{Vector{Int}}
)::Vector

    all_features = unique(collect(Iterators.flatten(feature_pairs)))

    # dataframe to store trees in which each feature is present
    feature_tree_matches = Dict()
    for feat in all_features
        feature_tree_matches[string(feat)] = [
            feat in tree.internal_node_features for tree in trees
        ]
    end
    tree_features_df = DataFrame(feature_tree_matches)

    fisher_test_p_values = Dict(
        "feature_1" => Vector{Int}(),
        "feature_2" => Vector{Int}(),
        "P_value" => Vector{Float64}()
    )
    for fp in feature_pairs
        fp_1 = string(fp[1])
        fp_2 = string(fp[2])
        fp_df = tree_features_df[!,[fp_1, fp_2]]

        N_1 = nrow(
            fp_df[(fp_df[:,fp_1] .== true) .& (fp_df[:,fp_2] .== false),:]
        )
        N_2 = nrow(
            fp_df[(fp_df[:,fp_1] .== false) .& (fp_df[:,fp_2] .== true),:]
        )
        N_12 = nrow(
            fp_df[(fp_df[:,fp_1] .== true) .& (fp_df[:,fp_2] .== true),:]
        )
        N_neither = nrow(
            fp_df[(fp_df[:,fp_1] .== false) .& (fp_df[:,fp_2] .== false),:]
        )

        p_value = pvalue(
            FisherExactTest(N_12, N_1, N_2, N_neither), tail=:right
        )

        push!(fisher_test_p_values["feature_1"], fp[1])
        push!(fisher_test_p_values["feature_2"], fp[2])
        push!(fisher_test_p_values["P_value"], p_value)
    end

    df = DataFrame(fisher_test_p_values)
    sort!(df, order(:P_value)) # order by p value

    return df
end


function split_asymmetry(
    linked_fps::Dict
)::Vector
    # need at least five slopes to calcualte t statistic
    candidate_fps = Dict(
        i => j for (i, j) in linked_fps if length(j) >= 5 
    )

    function get_slope(fp, tree)
        
        # check that second feature appears at least twice
        if length(tree.features[tree.features .== fp[2]]) < 2
            return nothing
        end

        # assert that children of first node aren't leaves
        fp_1 = findall(tree.features .== fp[1])[1]
        left_child = tree.tree[fp_1][1]
        right_child = tree.tree[fp_1][2]
        if any([tree.leaf_idx[left_child], tree.leaf_idx[right_child]])
            return nothing
        end
        
        # nodes split on second feature
        all_fp_2 = findall(tree.features .== fp[2])

        # get nodes of second feature on left hand side of first feature
        left_fp_2_loc = Vector{Int}()
        for fp_2 in all_fp_2
            if ParseRF.traverse_tree(left_child, fp_2, tree)
                push!(left_fp_2_loc, fp_2)
            end
        end
        if length(left_fp_2_loc) == 0
            return nothing
        end

        # get nodes of second feature on right hand side of first feature
        all_fp_2 = [i for i in all_fp_2 if !(i in left_fp_2_loc)]
        right_fp_2_loc = []
        for fp_2 in all_fp_2
            if ParseRF.traverse_tree(right_child, fp_2, tree)
                push!(right_fp_2_loc, fp_2)
            end
        end
        if length(right_fp_2_loc) == 0
            return nothing
        end
        
    end

    fp_slopes = Dict()
    for (fp, linked_trees) in candidate_fps
        slopes = [get_slope(fp, tree) for tree in linked_trees]
        slopes = [i for i in slopes if !isnothing(i)]
        if length(slopes) >= 5
            fp_slopes[fp] = slopes    
        end
    end

end # module
