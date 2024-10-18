#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"

#include "../functions_lib.h"

double OOA(const cv::Mat& a, const cv::Mat& b){
    cv::Mat a1;
    cv::bitwise_and(a, b, a1);
    return static_cast<double>(cv::sum(a1)[0])/cv::sum(a)[0];
}

std::string ocr::ExtractNumbers(const std::vector<cv::Mat>& images, const std::string& recModelPath) {
    cv::Size recInputSize{100, 32};

    CV_Assert(!recModelPath.empty());
    cv::dnn::TextRecognitionModel recognizer(recModelPath);

    std::vector<std::string> vocabulary;
    for (char a = '0'; a <= '9'; a++) {
        vocabulary.emplace_back(1, a);
    }
    recognizer.setVocabulary(vocabulary);
    recognizer.setDecodeType("CTC-greedy");

    // Parameters for Recognition
    double recScale = 1.0 / 127.5;
    cv::Scalar recMean = cv::Scalar(127.5, 127.5, 127.5);
    recognizer.setInputParams(recScale, recInputSize, recMean);
    std::string result;
    cv::Mat recInput, cropped;
    for(const auto& image: images){
        cvtColor(image, recInput, cv::COLOR_BGR2GRAY);
        cv::resize(recInput, cropped, recInputSize);
        std::string recognitionResult = recognizer.recognize(cropped);
        if(!recognitionResult.empty()){
            result += recognitionResult[0];
        }
    }
    return result;
}

void ocr::SplitImageByColor(const cv::Mat &image, cv::Mat &notRed, cv::Mat &red) {
    cv::Mat t;
    wmd::GetRed(image, t);
    if(cv::sum(t)[0]/255.0 > 0.3*image.total()){
        wmd::Contrast(image, t);
        wmd::GetRed(t, t);
    }
    double coef = 0.003;
    if(cv::sum(t)[0]/255.0 > 0.2*image.total()){
        cv::erode(t,t,cv::Mat());
        coef *= 2;
    }
    std::vector<std::vector<cv::Point>> cont;
    cv::findContours(t, cont, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    t = cv::Mat::zeros(image.size(), CV_8UC1);
    for(int i = 0; i < cont.size(); i++){
        if(cv::contourArea(cont[i]) > coef*t.total()){
            cv::drawContours(t, cont, i, 255, -1);
        }
    }
    t = t(cv::Rect(0, image.rows/2-1, image.cols, 1));
    auto rect = cv::boundingRect(t);
    if(rect.x < 0.2*image.cols){
        cv::erode(t,t,cv::Mat());
        rect = cv::boundingRect(t);
    }
    rect.x *= 0.98;
    red = image(cv::Rect(rect.x, 0, image.cols - rect.x, image.rows));
    notRed = image(cv::Rect(0, 0, rect.x, image.rows));
}

std::vector<cv::Mat> ocr::GetDigits(const cv::Mat &image) {
    cv::Mat temp;
    cv::cvtColor(image, temp, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(temp, temp, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 13, 9);
    if(cv::sum(temp)[0]/255 > 0.5*temp.total()){
        wmd::InvertImage(temp, temp);
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(temp, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    temp = cv::Mat::zeros(temp.size(), CV_8UC1);
    for(int i = 0; i < contours.size(); i++){
        auto conArea = cv::boundingRect(contours[i]).area();
        if(conArea < 0.9*temp.total() && conArea > 0.02*temp.total()){
            cv::drawContours(temp, contours, i, 255, -1);
        }
    }
    cv::dilate(temp, temp, cv::Mat());
    cv::findContours(temp, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    std::vector<std::pair<int, int>> rectHeights;
    std::vector<cv::Rect> rects;
    for(const auto & contour : contours){
        auto rect = cv::boundingRect(contour);
        if(rect.width < rect.height){
            rects.push_back(rect);
            rectHeights.emplace_back(rect.height, rects.size()-1);
        }
    }
    std::sort(rectHeights.begin(), rectHeights.end());
    double diff = 1;
    int cur = 0, max = 0, maxPos = 0, pos = 0;
    for(int i = 1; i < rectHeights.size(); i++){
        diff = static_cast<double>(rectHeights[i-1].first)/rectHeights[i].first;
        if(diff > 0.85 && rectHeights[i].first > 0.4*image.rows){
            cur++;
            if(max < cur){
                max = cur;
                maxPos = pos;
            }
        } else{
            pos = i;
            cur = 0;
        }
    }
    if(max == 0){
        return std::vector<cv::Mat>();
    }
    std::vector<cv::Mat> result;
    std::vector<cv::Rect> additionalRects;
    cv::Mat testCoverage = cv::Mat::zeros(image.size(), CV_8UC1);
    for(int i = maxPos; i <= maxPos+max; i++){
        cv::Mat t = cv::Mat::zeros(image.size(), CV_8UC1);
        t(rects[rectHeights[i].second]) = 255;
        if(OOA(t, testCoverage) < 0.8){
            additionalRects.push_back(rects[rectHeights[i].second]);
            cv::bitwise_or(testCoverage, t, testCoverage);
        }
    }
    std::sort(additionalRects.begin(), additionalRects.end(), [](cv::Rect& a, cv::Rect& b){
        return a.x < b.x;
    });
    for(const auto& r: additionalRects){
        cv::Mat d;
        GetImageFromRect(image, d, r, {100, 32});
        result.push_back(d);
    }
    return result;
}

void ocr::GetImageFromRect(const cv::Mat &image, cv::Mat &output, const cv::RotatedRect &rect, const cv::Size targetSize) {
    cv::Point2f vertices2f[4];
    rect.points(vertices2f);
    cv::Point2f targetVertices[4];
    if(rect.size.width > rect.size.height){
        targetVertices[0] = cv::Point(0, targetSize.height - 1);
        targetVertices[1] = cv::Point(0, 0);
        targetVertices[2] = cv::Point(targetSize.width - 1, 0);
        targetVertices[3] = cv::Point(targetSize.width - 1, targetSize.height - 1);
    } else{
        targetVertices[0] = cv::Point(0, 0);
        targetVertices[1] = cv::Point(targetSize.width - 1, 0);
        targetVertices[2] = cv::Point(targetSize.width - 1, targetSize.height - 1);
        targetVertices[3] = cv::Point(0, targetSize.height - 1);
    }
    cv::Mat transformMatrix = getPerspectiveTransform(vertices2f, targetVertices);
    cv::warpPerspective(image, output, transformMatrix, targetSize);
}

void ocr::GetImageFromRect(const cv::Mat &image, cv::Mat &output, const cv::Rect& rect, const cv::Size targetSize) {
    cv::Point2f vertices[4];
    vertices[0] = cv::Point(rect.x, rect.y+rect.height);
    vertices[1] = cv::Point(rect.x, rect.y);
    vertices[2] = cv::Point(rect.x + rect.width, rect.y);
    vertices[3] = cv::Point(rect.x + rect.width, rect.y+rect.height);
    cv::Point2f targetVertices[4];
    targetVertices[0] = cv::Point(0, targetSize.height - 1);
    targetVertices[1] = cv::Point(0, 0);
    targetVertices[2] = cv::Point(targetSize.width - 1, 0);
    targetVertices[3] = cv::Point(targetSize.width - 1, targetSize.height - 1);
    cv::Mat transformMatrix = getPerspectiveTransform(vertices, targetVertices);
    cv::warpPerspective(image, output, transformMatrix, targetSize);
}
