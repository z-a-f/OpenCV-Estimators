#ifndef DECISION_TREE_H_
#define DECISION_TREE_H_

#include <opencv/cv.hpp>
#include <opencv/ml.h>

#include "../base/base_estimator.hpp"

namespace estimator {

class DecisionTree : public BaseEstimator<CvDTree, CvDTreeParams>{
public:
    DecisionTree() : BaseEstimator(new CvDTree) {}

    // Trains the classifier.
    bool Fit(const cv::Mat& X, const cv::Mat& y);

    // Predicts the labels for X.
    void Predict(const cv::Mat& X, cv::Mat& y_hat);
    double Predict(const cv::Mat& X);

    // Getters.
    bool is_trained() { return trained_; }
};
}  // namespace estimator


#endif  // DECISION_TREE_H_
