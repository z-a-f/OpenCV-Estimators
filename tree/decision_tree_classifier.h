#ifndef DECISION_TREE_CLASSIFIER_H_
#define DECISION_TREE_CLASSIFIER_H_

#include <opencv/cv.hpp>
#include <opencv/ml.h>
#include "tree_config.h"

class DecisionTreeClassifier {
public:
    DecisionTreeClassifier() : tree_(cv::ml::DTrees::create()) {}

    // Configures the tree architecture.
    void Configure(const TreeConfig& config);

    // Trains the classifier.
    bool Fit(const cv::Mat& X, const cv::Mat& y);
    // Predicts the labels for X.
    void Predict(const cv::Mat& X, cv::Mat& y_hat);
    float Predict(const cv::Mat& X);
    // Predicts the labels for X, and returns probabilities instead of labels.
    std::vector<std::vector<double>> PredictProb(const cv::Mat& X);

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
