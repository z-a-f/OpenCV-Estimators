#ifndef BASE_BASE_ESTIMATOR_HPP_
#define BASE_BASE_ESTIMATOR_HPP_

#include <opencv/cv.hpp>

namespace estimator{

// Templated abstract class for estimators. Every estimator should implement
// the 'Fit' and 'Predict' methods.
// Template parameters:
//    E -> Estimator type to be used.
//    P -> Parameter type.
template <class E, class P>
class BaseEstimator {
public:
    BaseEstimator(const cv::Ptr<E>& estimator)
        : estimator_(estimator), params_(0), trained_(false) {}

    virtual ~BaseEstimator() {}
    
    // Saves the configuration for the given estimator.
    virtual void Config(const cv::Ptr<P>& params) { params_ = params; }

    // Fits the estimator.
    virtual bool Fit(const cv::Mat& X, const cv::Mat& y) = 0;

    // Predicts the result of the.
    virtual void Predict(const cv::Mat& X, cv::Mat& y_hat) = 0;

    virtual float Predict(const cv::Mat& X) = 0;

        // Dumps the current tree configuration to a file.
    virtual void Save(const std::string& filename, const char* name = 0) {
        estimator_->save(filename.c_str(), name);
    }
    // Loads the decision tree configuration from a file.
    virtual void Load(const std::string& filename, const char* name = 0) {
        estimator_->load(filename.c_str(), name);
    }

    // Getters.
    virtual cv::Ptr<E> estimator() { return estimator_; }
    virtual bool is_trained() { return trained_; }

protected:
    // Shared pointer to the estimator.
    cv::Ptr<E> estimator_;

    // Shared pointer to the estimator parameters.
    cv::Ptr<P> params_;

    // Trained flag
    bool trained_;
};

}  // namespace estimator

#endif  // BASE_BASE_ESTIMATOR_HPP_