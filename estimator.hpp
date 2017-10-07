#ifndef ESTIMATOR_HPP_
#define ESTIMATOR_HPP_

#include <type_traits>

#include <opencv/cv.h>
#include <opencv/ml.h>

template <class E>
class Estimator {
    // static_assert(std::is_base_of<BaseDerived, E>::value, 
    //     "Estimator template must be of BaseEstimator type!");
public:
    Estimator(cv::Ptr<E> estimator) : estimator_(estimator) {}
    Estimator() : estimator_(cv::Ptr<E>(new E)) {};

    virtual ~Estimator() {}

    virtual bool Fit(const cv::Mat& X, const cv::Mat& y) {
        return estimator_->train(X, y);
    }

    virtual void Predict(const cv::Mat& X, cv::Mat& y_hat) {
        estimator_->predict(X, y_hat);
    }

    virtual float Predict(const cv::Mat& X) {
        return estimator_->predict(X);
    }

    // Getters.
    cv::Ptr<E> estimator() { return estimator_; }

protected:
    cv::Ptr<E> estimator_;
};

#endif  // ESTIMATOR_HPP_
