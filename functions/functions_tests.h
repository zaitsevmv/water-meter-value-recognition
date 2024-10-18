#ifndef LAB1_TESTS_FUNCTIONS_H
#define LAB1_TESTS_FUNCTIONS_H

#include <map>

#include "opencv2/core.hpp"

namespace cw_tests{

bool FoundMeter(const std::pair<cv::Point, int>& circle, int resizeCoefficient, const cv::Mat& targetMask);

double FROC(const std::map<int, std::pair<double, double>> &data, cv::Mat& outputGraph);

bool FoundNumbersMask(const cv::Mat& foundMask, const cv::Mat& targetMask, double thresh);

std::pair<int,int> TPNumbers(const std::pair<std::string, std::string>& foundNumbers,
                             const std::pair<std::string, std::string>& targetNumbers);

}

#endif //LAB1_TESTS_FUNCTIONS_H
