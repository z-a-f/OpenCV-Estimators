#ifndef MAKE_SYNTHETIC_H_
#define MAKE_SYNTHETIC_H_

#include <opencv/cv.h>

namespace estimator {

// Scales the X to the range of (0, x_max) and (0, y_max).
void Scale(float x_max, float y_max, cv::Ptr<cv::Mat> X);

// Shuffles the rows in the X, y training pair.
void Shuffle(cv::Ptr<cv::Mat> X, cv::Ptr<cv::Mat> y);

// Creates the "Moons" synthetic dataset.
void MakeMoons(int image_x, int image_y, int n_samples, bool randomize, 
    double noise, cv::Ptr<cv::Mat> features, cv::Ptr<cv::Mat> labels);

} //  namespace estimator

#endif  // MAKE_SYNTHETIC_H_