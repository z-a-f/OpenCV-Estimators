#include "decision_tree.h"

#include <opencv/cv.hpp>
#include <opencv/ml.h>

namespace estimator {
namespace {
// OpenCV 2 decision trees support different layouts for the data.
int kDataLayout = CV_ROW_SAMPLE;
}

// Trains the classifier.
bool DecisionTree::Fit(const cv::Mat& X, const cv::Mat& y) {
    if (params_.empty()) {
        trained_ = estimator_->train(X, kDataLayout, y);
    } else {
        trained_ = estimator_->train(X, kDataLayout, y,
            cv::Mat(),
            cv::Mat(),
            cv::Mat(),
            cv::Mat(),
            *params_);
    }
    return trained_;
}

// Predicts the labels for X.
void DecisionTree::Predict(const cv::Mat& X, cv::Mat& y_hat) {
    estimator_->predict(X, y_hat);
}
double DecisionTree::Predict(const cv::Mat& X) {
    return estimator_->predict(X)->value;
}

}  // namespace estimator
