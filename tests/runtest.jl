using DecisionTree
using Random


function simulate_data(
    sample_size::Int,
    n_features::Int
)
    Random.seed!(1)
    features = randn(sample_size, n_features)
    weights = rand(-2:2, n_features)
    labels = features * weights
    return features, labels
end


function fit_random_forest(
    features::Matrix{Float64},
    labels::Vector{Float64},
    n_trees::Int
)    
    model = build_forest(labels, features, 2, n_trees, 0.7, 5)
    return model
end
