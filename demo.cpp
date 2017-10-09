
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
// #include <opencv/ml.h>

#include "estimator.hpp"
#include "svm/svm.h"
#include "tree/decision_tree.h"

#include "utils/make_synthetic.h"

using namespace cv;

int main()
{
    // Data for visual representation
    int width = 500, height = 400;
    int samples = 50;

    Ptr<Mat> X (new Mat(samples, 2, CV_32FC1));
    Ptr<Mat> y (new Mat(samples, 1, CV_32FC1));
    estimator::MakeMoons(height, width, 50, true, 0.1, X, y);

    estimator::SVM cls_svm;
    estimator::DecisionTree cls_dt;

    // Set up SVM's parameters
    cv::Ptr<CvSVMParams> params_svm(new CvSVMParams);
    params_svm->svm_type    = CvSVM::C_SVC;
    params_svm->kernel_type = CvSVM::LINEAR;
    params_svm->term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    cv::Ptr<CvDTreeParams> params_dt(new CvDTreeParams);
    params_dt->max_depth = 10;
    params_dt->cv_folds = 1;
    params_dt->min_sample_count = 0;
    
    cls_svm.Config(params_svm);
    cls_svm.Fit(*X, *y);

    cls_dt.Config(params_dt);
    cls_dt.Fit(*X, *y);

    Vec3b green(0,255,0), blue (255,0,0), red(0, 0, 255);
    Vec3b light_green(115, 255, 115), light_blue (255, 115, 115), light_red(115, 115, 255);

    Mat image1 = Mat::zeros(height, width, CV_8UC3);
    Mat image2 = Mat::zeros(height, width, CV_8UC3);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const Mat sampleMat = (Mat_<float>(1,2) << i, j);
            const double response = cls_dt.Predict(sampleMat);
            if (response == 1) {
                image1.at<Vec3b>(i,j)  = light_red;

            } else if (response == -1) {
                image1.at<Vec3b>(i,j)  = light_blue;
             } else {
                image1.at<Vec3b>(i, j) = light_green;
             }
        }
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const Mat sampleMat = (Mat_<float>(1,2) << i, j);
            const double response = cls_svm.Predict(sampleMat);
            if (response == 1) {
                image2.at<Vec3b>(i,j)  = light_red;

            } else if (response == -1) {
                image2.at<Vec3b>(i,j)  = light_blue;
             } else {
                image2.at<Vec3b>(i, j) = light_green;
             }
        }
    }

    int thickness = -1;
    int lineType = 8;

    for (int row = 0; row < X->rows; ++row) {
        const int idx = int(X->at<float>(row, 0));
        const int jdx = int(X->at<float>(row, 1));
        if (y->at<float>(row, 0) > 0) {
            circle(image1, Point(jdx, idx), 5, Scalar(0, 0, 255), thickness, lineType);
        } else if (y->at<float>(row, 0) < 0) {
            circle(image1, Point(jdx, idx), 5, Scalar(255, 0, 0), thickness, lineType);
        }
    }

    for (int row = 0; row < X->rows; ++row) {
        const int idx = int(X->at<float>(row, 0));
        const int jdx = int(X->at<float>(row, 1));
        if (y->at<float>(row, 0) > 0) {
            circle(image2, Point(jdx, idx), 5, Scalar(0, 0, 255), thickness, lineType);
        } else if (y->at<float>(row, 0) < 0) {
            circle(image2, Point(jdx, idx), 5, Scalar(255, 0, 0), thickness, lineType);
        }
    }

    Mat image = Mat::zeros(height, width*2, CV_8UC3);
    Mat left(image, Rect(0, 0, width, height));
    image1.copyTo(left);
    Mat right(image, Rect(width, 0, width, height));
    image2.copyTo(right);

    // imwrite("result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
}
