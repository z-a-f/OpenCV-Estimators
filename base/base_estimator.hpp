#ifndef BASE_BASE_ESTIMATOR_HPP_
#define BASE_BASE_ESTIMATOR_HPP_

#include <opencv/cv.hpp>

namespace estimator{

template <class E, class P>
class BaseEstimator {
public:
    BaseEstimator(const cv::Ptr<E>& estimator)
        : estimator_(estimator), params_(0), trained_(false) {}

    virtual ~BaseEstimator() {}

    virtual void Config(const cv::Ptr<P>& params) { params_ = params; }

    virtual bool Fit(const cv::Mat& X, const cv::Mat& y) = 0;

    virtual void Predict(const cv::Mat& X, cv::Mat& y_hat) = 0;

    virtual double Predict(const cv::Mat& X) = 0;

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

protected:
    cv::Ptr<E> estimator_;
    cv::Ptr<P> params_;

    bool trained_;
};

}  // namespace estimator

#endif  // BASE_BASE_ESTIMATOR_HPP_