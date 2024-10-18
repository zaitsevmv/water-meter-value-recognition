#ifndef CW_METER_DETECTOR_H
#define CW_METER_DETECTOR_H

#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

class Detector {
public:
    explicit Detector(const std::string& path);
    Detector() = default;
    ~Detector() = default;

    void DetectMeter();
    void FindNumbersRect();
    std::string GetNumbers(const std::string& recModelPath);

    void ShowImage(const cv::Mat& img);
    void WriteImage(const cv::Mat& img, const std::string& path);
    cv::Mat GetImage();

    [[nodiscard]] double getPrecision() const{
        return resultPrecision;
    }

    void TestMeterDetection(const cv::Mat& targetMask, const std::string& path);

    void TestNumbersDetectionIoU(const cv::Mat& targetMask, const std::string& path);
    void TestNumbersDetectionPrecision(const cv::Mat& targetMask, const std::string& path, const std::string& graphOutput, double IoUThresh);

    static void TestResults(const std::string& foundNumbers, const std::string& targetNumbers, const std::string& path);

private:
    cv::Mat image;
    cv::Mat startImage;
    cv::Mat mask;

    cv::Mat redImage;
    cv::Mat blackImage;
    cv::Mat grayImage;

    int resizeCoefficient = 4;
    double resultPrecision = 0;

    std::pair<cv::Point, int> circle;
    cv::RotatedRect resultRect;
};


#endif //CW_METER_DETECTOR_H
