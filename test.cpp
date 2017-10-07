
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
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    float labels[4] = {1.0, -1.0, -1.0, -1.0};
    Mat labelsMat(4, 1, CV_32FC1, labels);

    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    // cv::Ptr<CvSVMParams> params(new CvSVMParams);
    // params->svm_type    = CvSVM::C_SVC;
    // params->kernel_type = CvSVM::LINEAR;
    // params->term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // estimator::SVM cls;
    // cls.Load("test.model");

    // Set up DT's parameters
    estimator::DecisionTree cls;

    cv::Ptr<CvDTreeParams> params(new CvDTreeParams);
    params->max_depth = 10;
    params->cv_folds = 1;
    params->min_sample_count = 0;

    cls.Config(params);
    cls.Fit(trainingDataMat, labelsMat);

    Vec3b green(0,255,0), blue (255,0,0), red(0, 0, 255);
    // Show the decision regions given by the SVM
    int red_count = 0, green_count = 0, blue_count = 0;
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            double response = cls.Predict(sampleMat);
            // float response = SVM.predict(sampleMat);

            if (response == 1) {
                image.at<Vec3b>(i,j)  = green;
                ++green_count;
            } else if (response == -1) {
                image.at<Vec3b>(i,j)  = blue;
                ++blue_count;
             } else {
                image.at<Vec3b>(i, j) = red;
                ++red_count;
             }
        }
    std::cout << "Red: " << red_count << std::endl;
    std::cout << "Green: " << green_count << std::endl;
    std::cout << "Blues: " << blue_count << std::endl;

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

    // Show support vectors
    thickness = 2;
    lineType  = 8;
    // int c     = SVM->get_support_vector_count();
    // int c     = cls->get_support_vector_count();
    // int c = cls.estimator()->get_support_vector_count();

    // for (int i = 0; i < c; ++i)
    // {
    //     // const float* v = SVM->get_support_vector(i);
    //     const float* v = cls.estimator()->get_support_vector(i);
    //     circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    // }
    // delete cls;

    imwrite("result.png", image);        // save the image

    // imshow("SVM Simple Example", image); // show it to the user
    // waitKey(0);

}
