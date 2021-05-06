module RandomForestInteractions

include("ParseRF.jl")

using DataFrames
using HypothesisTests


function paired_selection_frequency(
    trees::Vector{ParseRF.decision_tree},
    included_features::Vector{Int},
    feature_pairs::Vector{Vector{Int}}
)::Vector

    # dataframe to store trees in which each feature is present
    feature_tree_matches = Dict()
    for feat in included_features
        feature_tree_matches[string(feat)] = [
            feat in tree.internal_node_features for tree in trees
        ]
    end
    tree_features_df = DataFrame(feature_tree_matches)

    fisher_test_p_values = Vector()
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

        push!(fisher_test_p_values, [fp[1], fp[2], p_value])
    end

    # sort by p value
    sort!(fisher_test_p_values, by=x -> x[3])

    return fisher_test_p_values
end

end # module
