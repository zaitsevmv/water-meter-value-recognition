#include <fstream>

#include "opencv2/imgproc.hpp"

#include "meter_detector.h"
#include "functions_lib.h"
#include "functions_tests.h"

Detector::Detector(const std::string& path) {
    try{
        image = cv::imread(path);
        startImage = image.clone();
    } catch (...){
        CV_Error(cv::Error::StsParseError, "error reading image");
    }
}

void Detector::ShowImage(const cv::Mat& img) {
    cv::imshow("image", img);
    cv::waitKey(0);
}

void Detector::WriteImage(const cv::Mat& img, const std::string& path){
    cv::imwrite(path, img);
}

cv::Mat Detector::GetImage() {
    return image;
}

void Detector::DetectMeter() {
    wmd::ResizeImage(image, image, -resizeCoefficient);
    wmd::GetGray(image, grayImage);
    wmd::Contrast(grayImage, grayImage);
    wmd::Denoise(grayImage, grayImage);
    circle = wmd::FindBiggestCircle(grayImage, 2.7, 30, 40, 0.6);
    wmd::RectangleImageByCircle(startImage, circle, grayImage.size(), resizeCoefficient, image);
}

void Detector::FindNumbersRect() {
    if(circle.second == 0) return;
    cv::Mat colorful, circleMask;
//    wmd::ResizeImage(image, image, 2);
    wmd::Contrast(image, image);
    wmd::Denoise(image, image);
    wmd::GetCircleMask(image, circleMask);
    mask = circleMask.clone();
//    wmd::GetGray(image, grayImage);
    wmd::GetBlack(image, blackImage);
    auto kp = wmd::FindKeyPoints(blackImage, 120);
    cv::Mat g;
    wmd::DrawKeyPointsAsMarkers(kp, image.size(), g);
    cv::Mat markers;
    wmd::WatershedImage(image, g, markers, circleMask);
    auto ms = wmd::ExtractMasksFromMarkers(markers);
    wmd::AfterWatershedMasksThresh(ms, 0.1, colorful);
    wmd::AfterWatershedMasksThresh(ms, 0.05, colorful);
    wmd::AfterWatershedMasksThresh(ms, 0.02, colorful);
    wmd::AfterWatershedMasksThresh(ms, 0.01, colorful);
    wmd::AfterWatershedMasksThresh(ms, 0.005, colorful);
    wmd::Dilate(colorful, colorful);
    wmd::GetEdges(colorful, colorful, 51);
    wmd::Dilate(colorful, colorful);
    wmd::Erode(colorful, colorful);
    auto h = wmd::GetContours(colorful);
    auto contours = wmd::FilterContoursBySize(colorful, h, colorful);
//    cv::imshow("colorful", colorful);
//    cv::waitKey(0);
    cv::Mat res;
    if(contours.empty()) return;
    auto cl = wmd::ClusterContours(colorful, contours, res);
//    cv::imshow("clustered", res);
//    cv::imwrite("cw/results/clustered.png", res);
//    cv::imshow("image", image);
//    cv::waitKey(0);
//    return;
//    cv::waitKey(0);
    resultPrecision = 0;
    cv::Mat ruu = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat chh[3];
    cv::split(ruu, chh);
    for(const auto& con: cl){
        cv::Mat temp;
        auto cont = wmd::FilterClusteredContours(colorful, con, temp);
        if(cont.empty()) continue;
        auto precision = wmd::GetMaskByContours(colorful, cont, temp);
        if(precision > 0){
            chh[2] += temp;
        }
        if(precision > resultPrecision){
            mask = temp.clone();
            resultPrecision = precision;
        }
    }
    if(resultPrecision > 0){
        chh[2] -= mask;
        chh[1] += mask;
        cv::merge(chh, 3, ruu);
        WriteImage(image, "cw/results/final.png");
        wmd::InvertImage(circleMask, circleMask);
        wmd::MaskImage(mask, mask, circleMask);
        wmd::RectangleImageByCircle(startImage, circle, grayImage.size(), resizeCoefficient, image);
        resultRect = wmd::GetSmallerRect(mask, mask, 0.2, 0.1);
        wmd::MaskImage(image, image, mask);
        cv::imwrite("cw/results/rects_evaluate.png", ruu);
        cv::imshow("ruu", ruu);
        cv::waitKey(0);
    }
}

std::string Detector::GetNumbers(const std::string& recModelPath) {
    if(circle.second == 0) return "";
    ocr::GetImageFromRect(image, image, resultRect, {400, 100});
    cv::Mat a1, a2;
    ocr::SplitImageByColor(image, a1, a2);
    std::string result;
    if(!a1.empty()){
        auto mainDigits = ocr::GetDigits(a1);
        if(!mainDigits.empty()){
            result = ocr::ExtractNumbers(mainDigits, recModelPath);
        }
    }
    result.erase(0, std::min(result.find_first_not_of('0'), result.size()-1));
    result += "_";
    if(!a2.empty()){
        auto additionalDigits = ocr::GetDigits(a2);
        if(!additionalDigits.empty()){
            result += ocr::ExtractNumbers(additionalDigits, recModelPath);
        }
    }
    return result;
}

void Detector::TestMeterDetection(const cv::Mat& targetMask, const std::string& path) {
    auto fi = std::ifstream(path);
    if(!fi || fi.peek() == std::ifstream::traits_type::eof()){
        auto writeFile = std::ofstream(path);
        writeFile << "TruePositive: 0" << std::endl;
        writeFile << "FalseNegative: 0" << std::endl;
        fi = std::ifstream(path);
    }
    std::string a{}, b{};
    int tp{}, fn{};
    fi >> a >> tp >> b >> fn;
    auto fo = std::ofstream(path);
    auto testResult = cw_tests::FoundMeter(circle, resizeCoefficient, targetMask);
    tp += testResult;
    fn += !testResult;
    fo << a << ' ' << tp << std::endl << b << ' ' << fn << std::endl;
    fo << "Precision: " << (static_cast<double>(tp)/(fn+tp));
}

void Detector::TestNumbersDetectionIoU(const cv::Mat &targetMask, const std::string& path) {
    std::vector<std::pair<int, int>> tp_fpByIoU;
    std::vector<std::string> IoUThresh;
    auto fi = std::ifstream(path);
    if(!fi || fi.peek() == std::ifstream::traits_type::eof()){
        auto writeFile = std::ofstream(path);
        writeFile << "IoU TP FP" << std::endl;
        writeFile << 0.1 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.2 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.3 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.4 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.5 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.6 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.7 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.8 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.9 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 1.0 << ' ' << 0 << ' ' << 0 << std::endl;
        fi = std::ifstream(path);
    }
    std::string a{}, a1{}, a2{}, c{};
    fi >> a >> a1 >> a2;
    int tp{}, fn{};
    while(fi >> c){
        IoUThresh.push_back(c);
        fi >> tp >> fn;
        double thresh = std::stod(c);
        cv::Mat newTarget;
        wmd::RectangleImageByCircleSingleChannel(targetMask, circle, grayImage.size(), resizeCoefficient, newTarget);
//        cv::imwrite("cw/new_mask.png", newTarget);
//        cv::imwrite("cw/result.png", mask);
        auto f = cw_tests::FoundNumbersMask(mask, targetMask, thresh);
        tp_fpByIoU.emplace_back(tp+f, fn+!f);
    }
    auto fo = std::ofstream(path);
    fo << a << ' ' << a1 << ' ' << a2 << std::endl;
    for(int i = 0; i < IoUThresh.size(); i++){
        fo << IoUThresh[i] << ' ' << tp_fpByIoU[i].first << ' ' << tp_fpByIoU[i].second << ' ' << std::endl;
    }
}

void Detector::TestNumbersDetectionPrecision(const cv::Mat& targetMask, const std::string& path, const std::string& graphOutput,
                                             const double IoUThresh){
    std::map<int, std::pair<double, double>> data;
    std::vector<std::pair<double, double>> tp_fpByPrecision;
    std::vector<std::string> precisionThresh;
    auto fi = std::ifstream(path);
    if(!fi || fi.peek() == std::ifstream::traits_type::eof()){
        auto writeFile = std::ofstream(path);
        writeFile << "Total pictures: " << 0 << std::endl;
        writeFile << "PrecisionThresh TP FP" << std::endl;
        writeFile << 0.3 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.4 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.5 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.6 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.7 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.8 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 0.9 << ' ' << 0 << ' ' << 0 << std::endl;
        writeFile << 1.0 << ' ' << 0 << ' ' << 0 << std::endl;
        fi = std::ifstream(path);
    }
    std::string a{}, a1{}, a2{}, b1{}, b2{}, c{};
    double totalPictures{};
    int tp{}, fn{};
    fi >> b1 >> b2 >> totalPictures;
    fi >> a >> a1 >> a2;
    totalPictures++;
    while(fi >> c){
        if(c == "Precision:") break;
        precisionThresh.push_back(c);
        fi >> tp >> fn;
        if(resultPrecision == 0){
            tp_fpByPrecision.emplace_back(tp, fn);
            continue;
        }
        double thresh = std::stod(c);
        cv::Mat newTarget;
        wmd::RectangleImageByCircleSingleChannel(targetMask, circle, grayImage.size(), resizeCoefficient, newTarget);
        auto f = cw_tests::FoundNumbersMask(mask, targetMask, IoUThresh);
        if(resultPrecision <= thresh){
            tp_fpByPrecision.emplace_back(tp+f, fn+!f);
        } else{
            tp_fpByPrecision.emplace_back(tp, fn);
        }
    }
    auto fo = std::ofstream(path);
    fo << b1 << ' ' << b2 << ' ' << totalPictures << std::endl;
    fo << a << ' ' << a1 << ' ' << a2 << std::endl;
    for(int i = 0; i < precisionThresh.size(); i++){
        data[std::round( std::stod(precisionThresh[i])*10)] = std::make_pair(tp_fpByPrecision[i].first/totalPictures,
                                                                             tp_fpByPrecision[i].second/totalPictures);
        fo << precisionThresh[i] << ' ' << tp_fpByPrecision[i].first << ' ' << tp_fpByPrecision[i].second << ' ' << std::endl;
    }
    cv::Mat graph;
    auto res = cw_tests::FROC(data, graph);
    cv::imwrite(graphOutput, graph);
    fo << "Precision: " << res;
}

void Detector::TestResults(const std::string &foundNumbers, const std::string &targetNumbers, const std::string &path) {
    auto p = foundNumbers.find('_');
    std::pair<std::string, std::string> foundN = {foundNumbers.substr(0, p),
                                              foundNumbers.substr(p+1, foundNumbers.size()-p-1)};
    p = targetNumbers.find('_');
    std::pair<std::string, std::string> targetN = {targetNumbers.substr(0, p),
                                              targetNumbers.substr(p+1, targetNumbers.size()-p-1)};
    auto fi = std::ifstream(path);
    if(!fi || fi.peek() == std::ifstream::traits_type::eof()){
        auto writeFile = std::ofstream(path);
        writeFile << "TruePositives: " << 0 << ' ' << 0 << std::endl;
        writeFile << "TargetSizes: " << 0 << ' ' << 0 << std::endl;
        writeFile << "FoundSizes: " << 0 << ' ' << 0 << std::endl;
        fi = std::ifstream(path);
    }
    std::string a1{}, a2{}, a3{};
    int target1{0}, target2{0}, found1{0}, found2{0}, tp1{0}, tp2{0};
    fi >> a1 >> tp1 >> tp2;
    fi >> a2 >> target1 >> target2;
    fi >> a3 >> found1 >> found2;
    auto fo = std::ofstream(path);
    auto tp = cw_tests::TPNumbers(foundN, targetN);
    tp1 += tp.first;
    tp2 += tp.second;
    fo << a1 << ' ' << tp1 << ' ' << tp2 << std::endl;
    target1 += targetN.first.size();
    target2 += targetN.second.size();
    fo << a2 << ' ' << target1 << ' ' << target2 << std::endl;
    found1 += foundN.first.size();
    found2 += foundN.second.size();
    fo << a3 << ' ' << found1 << ' ' << found2 << std::endl;
    double score1, score2;
    score1 = 2.0 * static_cast<double>(tp1) / static_cast<double>(target1+found1);
    score2 = 2.0 * static_cast<double>(tp2) / static_cast<double>(target2+found2);
    fo << "Scores: " << score1 << ' ' << score2 << std::endl;
}