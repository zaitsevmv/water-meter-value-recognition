#include <iostream>
#include <fstream>

#include "meter_detector.h"

int main(int argc, char *argv[]){
    cv::String keys =
            "{@images images         |     | path to images dataset directory }"
            "{@image_type image_type |     | image type (.png, .jpg, ...) }"
            "{@masks masks           |     | path to masks directory }"
            "{@mask_type mask_type   |     | mask type (.png, .jpg, ...) }"
            "{@model model           |     | path to text recognition model (like detector/model/DenseNet_BiLSTM_CTC.onnx) }"
            "{FROC                   |     | path to FROC graph image and data (like detector/results/FROC) }"
            "{IoU                    |     | IoU thresh for FROC }"
            "{IoU_path               |     | path to IoU evaluation (like detector/results/IoU) }"
            "{circles                |     | path to circle detection evaluation (like detector/results/circles) }"
            "{numbers                |     | path to numbers recognition evaluation (like detector/results/numbers) }"
            "{resume                 |     | don't override result files }"
            "{help h                 |     | help }";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Detector");
    if(parser.has("help") || !parser.has("images") || !parser.has("masks")
        || !parser.has("image_type") || !parser.has("mask_type") || !parser.has("model")){
        parser.printMessage();
        return 0;
    }
    std::string pathImages = parser.get<std::string>("images") + "/*" + parser.get<std::string>("image_type");
    std::string pathMasks = parser.get<std::string>("masks") + "/*" + parser.get<std::string>("mask_type");
    std::string recModelPath = parser.get<std::string>("model");
    std::vector<cv::String> images, masks;
    cv::glob(pathImages, images, false);
    cv::glob(pathMasks, masks, false);

    bool cleanFiles = !parser.has("resume");
    double IoU = 0.1;
    std::string circle_text, numbers_text, IoU_text, froc_text, froc_image;
    if(parser.has("circles")){
        circle_text = parser.get<std::string>("circles");
        if(cleanFiles){
            std::ofstream of(circle_text);
        }
    }
    if(parser.has("FROC")){
        froc_text = parser.get<std::string>("FROC");
        if(parser.has("IoU")){
            froc_text += "_"+parser.get<std::string>("IoU");
            IoU = parser.get<double>("IoU");
        } else{
            froc_text += "_0.1";
        }
        froc_image = froc_text + ".png";
        if(cleanFiles){
            std::ofstream of(froc_text);
        }
    }
    if(parser.has("IoU_path")){
        IoU_text = parser.get<std::string>("IoU_path");
        if(cleanFiles){
            std::ofstream of(IoU_text);
        }
    }
    if(parser.has("numbers")){
        numbers_text = parser.get<std::string>("numbers");
        if(cleanFiles){
            std::ofstream of(numbers_text);
        }
    }

    for (int i = 0; i < images.size(); i++) {
        Detector det{images[i]};
        std::cout << "Reading " << images[i] << std::endl;
        det.DetectMeter();
        cv::Mat targetMask = cv::imread(masks[i], cv::IMREAD_GRAYSCALE);
        if(!circle_text.empty()){
            det.TestMeterDetection(targetMask, circle_text);
        }
        if(det.GetImage().size() == cv::Size(1,1)) continue;
        det.FindNumbersRect();
        if(!froc_text.empty()){
            det.TestNumbersDetectionPrecision(targetMask, froc_text, froc_image, IoU);
        }
        if(det.getPrecision() == 0) continue;
        if(!IoU_text.empty()){
            det.TestNumbersDetectionIoU(targetMask, IoU_text);
        }
        auto result = det.GetNumbers(recModelPath);
        std::string realValue = images[i].substr(images[i].find("value_")+6,
                                                 images[i].find(".jpg")-images[i].find("value_")-6);
        if(!result.empty()){
            if(!numbers_text.empty()){
                det.TestResults(result, realValue, numbers_text);
            }
            std::cout << "Found: " << result << std::endl;
        }
    }
    return 0;
}