#ifndef SVM_H_
#define SVM_H_

#include <opencv/cv.hpp>
#include <opencv/ml.h>

#include "../base/base_estimator.hpp"

namespace estimator {

class SVM : public BaseEstimator<cv::SVM, cv::SVMParams>{
public:
    SVM() : BaseEstimator(new cv::SVM) {}

    // Trains the classifier.
    bool Fit(const cv::Mat& X, const cv::Mat& y);

    // Predicts the labels for X.
    void Predict(const cv::Mat& X, cv::Mat& y_hat);
    float Predict(const cv::Mat& X);
};
}  // namespace estimator

#endif  // SVM_H_
