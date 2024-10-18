#ifndef LAB1_FUNCTIONS_LIB_H
#define LAB1_FUNCTIONS_LIB_H

#include <opencv2/core.hpp>

namespace wmd{

void ResizeImage(const cv::Mat& input, cv::Mat& output, int k);

void GetGray(const cv::Mat& image, cv::Mat& gray);

void GetRed(const cv::Mat& image, cv::Mat& red);
void GetRedAlt(const cv::Mat& image, cv::Mat& red);

void GetBlack(const cv::Mat& image, cv::Mat& black);

void WatershedImage(const cv::Mat& image, const cv::Mat& grayImage, cv::Mat& markers, cv::Mat mask=cv::Mat());

void AffineTransformImage(const cv::Mat& input, cv::Mat& output, const std::vector<cv::Vec4i>& lines);

std::vector<cv::Vec4i> FindLines(const cv::Mat &img);

void DrawLines(const cv::Mat &img, const std::vector<cv::Vec4i>& lines, cv::Mat& output);

cv::Vec4i FindAngle(const std::vector<cv::Vec4i>& lines);

cv::Vec4i StraightestLine(const std::vector<cv::Vec4i>& lines);

void MaskImage(const cv::Mat& input, cv::Mat& output, const cv::Mat& mask);

void Contrast(const cv::Mat& input, cv::Mat& output);

void Denoise(const cv::Mat& input, cv::Mat& output);

void GetCorners(const cv::Mat& input, cv::Mat& output, int blockSize, int kSize, double k);

void GetEdges(const cv::Mat& input, cv::Mat& output, int thresh);

std::pair<cv::Point, int> FindBiggestCircle(const cv::Mat& input, double dp, int minDist, int minRadius, double accuracy);

void RectangleImageByCircle(const cv::Mat& input, const std::pair<cv::Point, int>& circle, const cv::Size& sourceSize,
                            int k, cv::Mat& output);

void RectangleImageByCircleSingleChannel(const cv::Mat& input, const std::pair<cv::Point, int>& circle,
                                         const cv::Size& sourceSize, int k, cv::Mat& output);

void DrawLabels(const cv::Mat& labels, cv::Mat& output);

void DrawMarkers(const cv::Mat& markers, cv::Mat& output);

std::vector<cv::KeyPoint> FindKeyPoints(const cv::Mat& image, int count);

std::vector<std::vector<cv::KeyPoint>> ClusterKeyPoints(const cv::Mat& input, const std::vector<cv::KeyPoint>& kp,
                                                        cv::Mat& colorful, int bigK, cv::Point center = {0,0});

void DrawKeyPoints(const cv::Mat& image, const std::vector<cv::KeyPoint>& kp, cv::Mat& colorful);

cv::Rect BoundingRectByKeyPoints(const std::vector<cv::KeyPoint>& kp);
cv::RotatedRect BoundingRotatedRectByKeyPoints(const std::vector<cv::KeyPoint>& kp);

void DrawRect(cv::Mat& image, const cv::Rect& rect);
void DrawRect(cv::Mat& image, const cv::RotatedRect& rect);

void SegmentImage(const cv::Mat& image, cv::Mat& blackImage, cv::Mat& redImage, cv::Mat& mask);

void InvertImage(const cv::Mat& image, cv::Mat& output);

std::vector<cv::Mat> ExtractMasksFromMarkers(const cv::Mat& markers);

void ClusterMarkersByColorAmount(const std::vector<cv::Mat>& masks, const cv::Mat& grayImage, int bigK, cv::Mat& outputMarkers);

void DrawKeyPointsAsMarkers(const std::vector<cv::KeyPoint>& kp, cv::Size imageSize, cv::Mat& output);

void GetCircleMask(const cv::Mat& image, cv::Mat& output);

void AfterWatershedMasksThresh(const std::vector<cv::Mat>& masks, double thresh, cv::Mat& output);

void Dilate(const cv::Mat& input, cv::Mat& output);

void Erode(const cv::Mat& input, cv::Mat& output);

double GetMaskByContours(const cv::Mat& input, const std::vector<std::vector<cv::Point>>& contours, cv::Mat& output);

std::vector<std::vector<cv::Point>> GetContours(const cv::Mat& input);

std::vector<std::vector<cv::Point>> FilterContoursBySize(const cv::Mat& input,
                                                         const std::vector<std::vector<cv::Point>>& contours,
                                                         cv::Mat& output);

std::vector<std::vector<std::vector<cv::Point>>> ClusterContours(const cv::Mat& input,
                                                                 const std::vector<std::vector<cv::Point>>& contours,
                                                                 cv::Mat& colorful);

std::vector<std::vector<cv::Point>> FilterClusteredContours(const cv::Mat& input,
                                                   const std::vector<std::vector<cv::Point>>& contours,
                                                   cv::Mat& output);

cv::RotatedRect GetSmallerRect(const cv::Mat& input, cv::Mat& output, const double left, const double right);

}

namespace ocr {

void GetImageFromRect(const cv::Mat& image, cv::Mat& output, const cv::RotatedRect& rect, const cv::Size targetSize);
void GetImageFromRect(const cv::Mat& image, cv::Mat& output, const cv::Rect& rect, const cv::Size targetSize);

std::string ExtractNumbers(const std::vector<cv::Mat>& images, const std::string& recModelPath);

void SplitImageByColor(const cv::Mat& image, cv::Mat& notRed, cv::Mat& red);

std::vector<cv::Mat> GetDigits(const cv::Mat& image);

}

#endif //LAB1_FUNCTIONS_LIB_H