#ifndef TREE_H_
#define TREE_H_

#include <limits>

class Tree {
public:
    // Configures the tree architecture.
    void Configure(const TreeConfig& config);
    // Set class weights (for unbalanced data).
    void ClassWeigths(std::vector<double> weigths);
    // Trains the classifier.
    void Fit(const TrackFeaturesVector& X, const TrackLabelVector& y);
    // Predicts the labels for X.
    TrackLabelVector Predict(const TrackFeaturesVector& X);
    // Predicts the labels for X, and returns probabilities instead of labels.
    std::vector<std::vector<double>> PredictProb(const TrackFeaturesVector& X);

    // Dumps the current tree configuration to a file.
    void Dump(std::string filename);
    // Loads the decision tree configuration from a file.
    void Load(std::string filename);

private:
    // Maximum depth of the tree.
    int max_depth_ = std::numeric_limits<int>::max();
    // Minimum samples in a split.
    int min_samples_split_ = 2;
    // Minimum samples per leaf node.
    int min_samples_leaf_ = 1;
    // Class weights.
    std::vector<double> class_weights_;
}

#endif  // TREE_H_
