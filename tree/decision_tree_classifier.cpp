#include "decision_tree_classifier.h"

#include <iostream>

void DecisionTreeClassifier::Configure(const TreeConfig& config) {
    tree_->setCVFolds(config.prune);

    if (config.max_depth >= 0) {
        tree_->setMaxDepth(config.max_depth);
    }
    if (config.min_samples_split >= 0) {
        tree_->setMinSampleCount(config.min_samples_split);
    }
    if (config.class_weights.size() > 0) {
        tree_->setPriors(cv::Mat(config.class_weights));
    }
}

bool DecisionTreeClassifier::Fit(const TrackFeaturesVector& X,
                                 const TrackLabelVector& y) {
    return tree_->train(X, cv::ml::ROW_SAMPLE, y);
}

void DecisionTreeClassifier::Predict(const TrackFeaturesVector& X, TrackLabelVector& y_hat) {
    tree_->predict(X, y_hat);
}

float DecisionTreeClassifier::Predict(const TrackFeaturesVector& X) {
    return tree_->predict(X);
}

std::vector<std::vector<double>> PredictProb(const TrackFeaturesVector& X) {
    throw std::logic_error("Function not yet implemented");
    return std::vector<std::vector<double>>();
}

void DecisionTreeClassifier::Save(std::string filename) {
    std::cout << filename << std::endl;
}
// Loads the decision tree configuration from a file.
void DecisionTreeClassifier::Load(std::string filename) {

}