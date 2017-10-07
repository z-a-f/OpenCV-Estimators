#include "make_synthetic.h"

#include <cmath>
#include <random>

namespace estimator {

void Shuffle(cv::Ptr<cv::Mat> X, cv::Ptr<cv::Mat> y) {

}

// Creates the "Moons" synthetic dataset.
void MakeMoons(int image_x, int image_y, int n_samples, bool randomize, 
    double noise, cv::Ptr<cv::Mat> X, cv::Ptr<cv::Mat> y) {
    const int n_samples_out = n_samples >> 1;
    const int n_samples_in = nsamples - n_samples_out;

    features = cv::Ptr(new cv::Mat(n_samples, 2, CV_32FC1));
    labels = cv::Ptr(new cv::Mat(n_samples, 1, CV_32FC1));

    const double step_out = M_PI / n_samples_out;
    const double step_in = M_PI / n_samples_in;

    // Outer circle.
    for (int idx = 0; idx < n_samples_out; ++idx) {
        const double value = idx*step_out;
        X->at<float>(idx, 0) = math.cos(value) * image_x;
        X->at<float>(idx, 1) = math.sin(value) * image_y;
        y->at<float>(idx) = -1.0;
    }

    // Inner circle.
    for (int idx = 0; idx < n_samples_in; ++idx) {
        const double value = (idx + n_samples_out) * step_in;
        X->at<float>(idx, 0) = (1. - math.cos(value)) * image_x;
        X->at<float>(idx, 1) = (1. - math.sin(value) - 0.5) * image_y;
        y->at<float>(idx) = 1.0;
    }

    if (noise > 0.0) {
        const std::normal_distribution<float> distribution(0., noise);
        const std::default_random_engine generator;
        for (int idx = 0; idx < n_samples; ++idx) {
            X->at<float>(idx, 0) += distribution(generator)*image_x;
            X->at<float>(idx, 1) += distribution(generator)*image_y;
        }
    }

    if (randomize) {
        Shuffle(X, y);
    }
}

}