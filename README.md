# RandomForestInteractions
An efficient implementation of the statistical tests described in https://www.biorxiv.org/content/10.1101/353193v1.full for detecting evidence of interactions between pairs of features in a random forest model.

This is a julia package for assessing the importance of interactions between pairs of features in a random forest model. 
Contains code to process either, a native julia random forest model from the DecisionTree module (https://github.com/bensadeghi/DecisionTree.jl), or pickled random forest models fitted with the python machine learning module: scikit-learn (https://scikit-learn.org/stable/).
