#include <chrono>
#include <iostream>

#include "meter_detector.h"

int main(int argc, char *argv[]){
    cv::String keys =
            "{@path path   |     | image path }"
            "{@model model |     | image path}"
            "{time t       |     | measure time }"
            "{help h       |     | help}";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Detector");
    if(parser.has("help") || !parser.has("@path") || !parser.has("model")){
        parser.printMessage();
        return 0;
    }
    Detector det;
    std::string recModelPath = parser.get<std::string>("model");
    try{
        det = Detector(parser.get<std::string>("@path"));
    } catch (...){
        CV_Error(cv::Error::StsParseError, "error reading image");
    }
    auto startTime = std::chrono::steady_clock::now();
    det.DetectMeter();
    if(det.GetImage().size() == cv::Size(1,1)){
        std::cout << "No meter found" << std::endl;
        return 0;
    }
    det.FindNumbersRect();
    if(det.getPrecision() == 0){
        std::cout << "No numbers found" << std::endl;
        return 0;
    }
    auto result = det.GetNumbers(recModelPath);
    if(result != "_"){
        std::cout << result << std::endl;
    } else{
        std::cout << "No numbers recognised" << std::endl;
    }
    if(parser.has("time")){
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration(endTime-startTime);
        auto mil = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        std::cout << "Time elapsed: " << mil << " milliseconds." << std::endl << std::endl;
    }
    return 0;
}