#ifndef MAKE_SYNTHETIC_H_
#define MAKE_SYNTHETIC_H_

#include <opencv/cv.h>

namespace estimator {

// Creates the "Moons" synthetic dataset.
void MakeMoons(int image_x, int image_y, int n_samples, bool randomize, 
    double noise, cv::Ptr<cv::Mat> features, cv::Ptr<cv::Mat> labels);

} //  namespace estimator

#endif  // MAKE_SYNTHETIC_H_