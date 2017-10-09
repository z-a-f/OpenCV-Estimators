#include "svm.h"

#include <opencv/cv.hpp>
#include <opencv/ml.h>

namespace estimator {

// Trains the classifier.
bool SVM::Fit(const cv::Mat& X, const cv::Mat& y) {
    if (params_.empty()) {
        trained_ = estimator_->train(X, y);
    } else {
        trained_ = estimator_->train(X, y, cv::Mat(), cv::Mat(), *params_);
    }
    return trained_;
}

// Predicts the labels for X.
void SVM::Predict(const cv::Mat& X, cv::Mat& y_hat) {
    estimator_->predict(X, y_hat);
}
float SVM::Predict(const cv::Mat& X) {
    return estimator_->predict(X);
}

}  // namespace estimator
