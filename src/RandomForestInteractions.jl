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

)
    
end

end # module
