#include "make_synthetic.h"

#include <cmath>
#include <random>

#include <iostream>

namespace estimator {

void Scale(float x_max, float y_max, cv::Ptr<cv::Mat> X) {
    // Find the minimums and maximums.
    float current_min_x = std::numeric_limits<float>::max();
    float current_min_y = std::numeric_limits<float>::max();
    float current_max_x = std::numeric_limits<float>::min();
    float current_max_y = std::numeric_limits<float>::min();
    for (int idx = 0; idx < X->rows; ++idx) {
        current_min_x = std::min(current_min_x, X->at<float>(idx, 0));
        current_min_y = std::min(current_min_y, X->at<float>(idx, 1));
        current_max_x = std::max(current_max_x, X->at<float>(idx, 0));
        current_max_y = std::max(current_max_y, X->at<float>(idx, 1));
    }

    // Scale.
    const float range_x = current_max_x - current_min_x;
    const float range_y = current_max_y - current_min_y;
    for (int idx = 0; idx < X->rows; ++idx) {
        X->at<float>(idx, 0) = x_max * (X->at<float>(idx, 0) - current_min_x) / range_x;
        X->at<float>(idx, 1) = y_max * (X->at<float>(idx, 1) - current_min_y) / range_y;
    }
}

void Shuffle(cv::Ptr<cv::Mat> X, cv::Ptr<cv::Mat> y) {
    // Implementation of the Fisher-Yates shuffling.
    const int n = X->rows; 
    for (int idx = 0; idx < n; ++idx) {
        const int jdx = std::rand() % (idx + 1);
        if (idx != jdx) {
            cv::Mat x_temp;
            X->row(idx).copyTo(x_temp);
            X->row(jdx).copyTo(X->row(idx));
            x_temp.copyTo(X->row(jdx));

            cv::Mat y_temp;
            y->row(idx).copyTo(y_temp);
            y->row(jdx).copyTo(y->row(idx));
            y_temp.copyTo(y->row(jdx));
        }
    }
}

// Creates the "Moons" synthetic dataset.
void MakeMoons(int image_x, int image_y, int n_samples, bool randomize, 
    double noise, cv::Ptr<cv::Mat> X, cv::Ptr<cv::Mat> y) {
    const int n_samples_out = n_samples >> 1;
    const int n_samples_in = n_samples - n_samples_out;

    const double step_out = M_PI / n_samples_out;
    const double step_in = M_PI / n_samples_in;

    // Outer circle.
    for (int idx = 0; idx < n_samples_out; ++idx) {
        const double value = idx * step_out;
        X->at<float>(idx, 0) = std::cos(value);
        X->at<float>(idx, 1) = std::sin(value);
        y->at<float>(idx) = -1.0;
    }

    // Inner circle.
    for (int idx = 0; idx < n_samples_in; ++idx) {
        const double value = idx * step_in;
        const int jdx = idx + n_samples_out;
        X->at<float>(jdx, 0) = (1. - std::cos(value));
        X->at<float>(jdx, 1) = (1. - std::sin(value) - 0.5);
        y->at<float>(jdx) = 1.0;
    }

    if (noise > 0.0) {
        std::normal_distribution<float> distribution(0., noise);
        std::default_random_engine generator;
        for (int idx = 0; idx < n_samples; ++idx) {
            X->at<float>(idx, 0) += (distribution(generator));
            X->at<float>(idx, 1) += (distribution(generator));
        }
    }

    if (randomize) {
        Shuffle(X, y);
    }

    Scale(image_x, image_y, X);
}

}
