#include <iostream>
#include "opencv2/imgproc.hpp"

#include "../functions_tests.h"

constexpr int GRAPH_HEIGHT = 256, GRAPH_WIDTH = 256;

bool cw_tests::FoundMeter(const std::pair<cv::Point, int>& circle, const int resizeCoefficient,  const cv::Mat& targetMask){
    cv::Mat result = cv::Mat::zeros(targetMask.size(), CV_8UC1);
    cv::circle(result, circle.first*resizeCoefficient, circle.second*resizeCoefficient,255, -1);
    cv::Mat t;
    cv::bitwise_and(result, targetMask, t);
    return (cv::sum(t)[0] == cv::sum(targetMask)[0]);
}

double cw_tests::FROC(const std::map<int, std::pair<double, double>> &data, cv::Mat& outputGraph) {
    cv::Mat graph = cv::Mat(GRAPH_HEIGHT, GRAPH_WIDTH, CV_8UC1, 240);
    std::vector<std::pair<double, double>> tpr_fpr;
    for(const auto& a: data){
        tpr_fpr.push_back(a.second);
    }
    tpr_fpr.emplace_back(0,0);
    std::sort(tpr_fpr.begin(), tpr_fpr.end(),
              [](auto& a, auto& b){return a.second < b.second;});
    auto m = tpr_fpr[tpr_fpr.size()-1];
    cv::Point lastPoint = {static_cast<int>(GRAPH_WIDTH*(tpr_fpr[0].second/m.second)),
                           GRAPH_HEIGHT-static_cast<int>(GRAPH_HEIGHT*tpr_fpr[0].first)};
    double result = 0;
    for(int i = 1; i < tpr_fpr.size(); i++){
        cv::Point curPoint = {static_cast<int>(GRAPH_WIDTH*tpr_fpr[i].second/m.second),
                              GRAPH_HEIGHT-static_cast<int>(GRAPH_HEIGHT*tpr_fpr[i].first)};
        cv::line(graph, lastPoint, curPoint,0);
        result += (tpr_fpr[i].second-tpr_fpr[i-1].second)*(tpr_fpr[i].first + tpr_fpr[i-1].first)/2.0;
        lastPoint = curPoint;
    }
    if(result == 0){
        for(const auto& a: tpr_fpr){
            result = (result < a.first) ? a.first : result;
        }
    } else{
        result /= m.second;
    }
    outputGraph = graph.clone();
    return result;
}

bool cw_tests::FoundNumbersMask(const cv::Mat &foundMask, const cv::Mat &targetMask, const double thresh) {
    cv::Mat a, rAnd, rOr;
    cv::resize(foundMask, a, targetMask.size());
    cv::bitwise_and(targetMask, a, rAnd);
    cv::bitwise_or(targetMask, a, rOr);
    return cv::sum(rAnd)[0]/cv::sum(rOr)[0] >= thresh;
}

int lcs(std::string x, std::string y, const int m, const int n)
{
    if (m == 0 || n == 0) return 0;
    if (x[m - 1] == y[n - 1]) {
        return 1 + lcs(x, y, m - 1, n - 1);
    }
    else {
        return std::max(lcs(x, y, m, n - 1), lcs(x, y, m - 1, n));
    }
}

std::pair<int,int> cw_tests::TPNumbers(const std::pair<std::string, std::string> &foundNumbers,
                                      const std::pair<std::string, std::string> &targetNumbers) {
    std::string a = foundNumbers.first, b = targetNumbers.first;
    double score = 0;
    int l = lcs(a, b, a.size(), b.size());
    a = foundNumbers.second, b = targetNumbers.second;
    int r = lcs(a, b, a.size(), b.size());
    return std::make_pair(l, r);
}
