
#include <cstdlib>
#include <iostream>
#include <time.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "svm/svm.h"
#include "tree/decision_tree.h"

#include "utils/make_synthetic.h"

using namespace cv;

// Demo for the classifiers wrapper.
int main() {
    // Parameters for the synthetic dataset
    const int width = 500;
    const int height = 400;
    const int samples = 50;
    const double noise = 0.1;
    const bool shuffle = true;

    Ptr<Mat> X (new Mat(samples, 2, CV_32FC1));
    Ptr<Mat> y (new Mat(samples, 1, CV_32FC1));
    srand(time(NULL));
    estimator::MakeMoons(height, width, samples, shuffle, noise, X, y);

    // Create two classifiers.
    estimator::SVM cls_svm;
    estimator::DecisionTree cls_dt;

    // Set up SVM's parameters.
    cv::Ptr<CvSVMParams> params_svm(new CvSVMParams);
    params_svm->svm_type    = CvSVM::C_SVC;
    params_svm->kernel_type = CvSVM::LINEAR;
    params_svm->term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Set up DT parameters.
    cv::Ptr<CvDTreeParams> params_dt(new CvDTreeParams);
    params_dt->max_depth = 10;
    params_dt->cv_folds = 1;
    params_dt->min_sample_count = 0;
    
    // Configure and fit.
    cls_svm.Config(params_svm);
    cls_svm.Fit(*X, *y);

    cls_dt.Config(params_dt);
    cls_dt.Fit(*X, *y);

    // Predictions as image backgrounds.
    Mat image_dt = Mat::zeros(height, width, CV_8UC3);
    Mat image_svm = Mat::zeros(height, width, CV_8UC3);

    const Vec3b light_green(115, 255, 115);
    const Vec3b light_blue(255, 115, 115);
    const Vec3b light_red(115, 115, 255);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const Mat sampleMat = (Mat_<float>(1,2) << i, j);
            const double response = cls_dt.Predict(sampleMat);
            if (response == 1) {
                image_dt.at<Vec3b>(i,j)  = light_red;

            } else if (response == -1) {
                image_dt.at<Vec3b>(i,j)  = light_blue;
             } else {
                image_dt.at<Vec3b>(i, j) = light_green;
             }
        }
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const Mat sampleMat = (Mat_<float>(1,2) << i, j);
            const double response = cls_svm.Predict(sampleMat);
            if (response == 1) {
                image_svm.at<Vec3b>(i,j)  = light_red;
            } else if (response == -1) {
                image_svm.at<Vec3b>(i,j)  = light_blue;
             } else {
                image_svm.at<Vec3b>(i, j) = light_green;
             }
        }
    }

    // Show true labels on top of the backgrounds.
    const int thickness = -1;
    const int lineType = 8;
    const Scalar green(0, 255, 0);
    const Scalar blue(255, 0, 0);
    const Scalar red(0, 0, 255);

    for (int row = 0; row < X->rows; ++row) {
        const int idx = int(X->at<float>(row, 0));
        const int jdx = int(X->at<float>(row, 1));
        if (y->at<float>(row, 0) > 0) {
            circle(image_dt, Point(jdx, idx), 5, red, thickness, lineType);
        } else if (y->at<float>(row, 0) < 0) {
            circle(image_dt, Point(jdx, idx), 5, blue, thickness, lineType);
        }
    }

    float correct = 0;
    for (int row = 0; row < X->rows; ++row) {
        const int idx = int(X->at<float>(row, 0));
        const int jdx = int(X->at<float>(row, 1));
        if (y->at<float>(row, 0) > 0) {
            circle(image_svm, Point(jdx, idx), 5, red, thickness, lineType);
        } else if (y->at<float>(row, 0) < 0) {
            circle(image_svm, Point(jdx, idx), 5, blue, thickness, lineType);
        }
        std::cout << y->at<float>(row, 0) << '\t' << cls_svm.Predict(X->row(row)) << std::endl;
        correct += abs(y->at<float>(row, 0) - cls_svm.Predict(X->row(row))) / 2.;
    }
    std::cout << 1. - correct / samples << std::endl;

    // Merge images.
    Mat image = Mat::zeros(height, width*2, CV_8UC3);
    Mat left(image, Rect(0, 0, width, height));
    image_dt.copyTo(left);
    Mat right(image, Rect(width, 0, width, height));
    image_svm.copyTo(right);

    imwrite("result.png", image);        // save the image
    // imshow("SVM Simple Example", image); // show it to the user
    // waitKey(0);
}
