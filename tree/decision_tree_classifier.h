#ifndef DECISION_TREE_CLASSIFIER_H_
#define DECISION_TREE_CLASSIFIER_H_

#include <opencv2/ml.hpp>
#include "tree_config.h"

using TrackFeaturesVector = cv::Mat;
using TrackLabelVector = cv::Mat;

using TreePredictFlags = cv::ml::DTrees::Flags;
using DataSampleTypes = cv::ml::SampleTypes;

class DecisionTreeClassifier {
public:
    DecisionTreeClassifier() : tree_(cv::ml::DTrees::create()) {}

    // Configures the tree architecture.
    void Configure(const TreeConfig& config);

    // Trains the classifier.
    bool Fit(const TrackFeaturesVector& X, const TrackLabelVector& y);
    // Predicts the labels for X.
    void Predict(const TrackFeaturesVector& X, TrackLabelVector& y_hat);
    float Predict(const TrackFeaturesVector& X);
    // Predicts the labels for X, and returns probabilities instead of labels.
    std::vector<std::vector<double>> PredictProb(const TrackFeaturesVector& X);

    // Dumps the current tree configuration to a file.
    void Save(std::string filename);
    // Loads the decision tree configuration from a file.
    void Load(std::string filename);

    // Getters.
    cv::Ptr<cv::ml::DTrees> tree() { return tree_; }
    bool is_trained() { return tree_->isTrained(); }

private:
    // OpenCV decision tree.
    cv::Ptr<cv::ml::DTrees> tree_;
};

#endif  // DECISION_TREE_CLASSIFIER_H_