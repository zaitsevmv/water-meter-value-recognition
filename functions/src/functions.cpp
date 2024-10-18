#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/features2d.hpp"

#include "opencv2/highgui.hpp"

#include "../functions_lib.h"

constexpr uchar colorTab[]{0, 32, 64, 128, 160, 192, 224, 255};

const cv::Vec3d colorfulTab[]{ {0,0,255}, {255, 0, 0}, {0, 255, 0},
                               {255, 0, 255}, {0, 255, 255}, {255, 255, 0}};

float tg(cv::Vec4i l){
    return static_cast<float>(l[3]-l[1])/(l[2]-l[0]);
}

float distance2(cv::Point a, cv::Point b){
    float x = (a.x-b.x), y = (a.y-b.y);
    return std::sqrt(x*x+y*y);
}

float distance0(cv::Point a, cv::Point b){
    int x = abs(a.x-b.x), y = abs(a.y-b.y);
    return std::max(x,y);
}

float distance1(cv::Point a, cv::Point b){
    int x = abs(a.x-b.x), y = abs(a.y-b.y);
    return x + y;
}

float GetColorAmount(const cv::Mat &gray, const cv::Mat &mask) {
    if(cv::sum(mask)[0] == 0){
        return 99999999.0;
    }
    cv::Mat t;
    cv::bitwise_and(gray, mask, t);
    return cv::sum(t)[0]/cv::sum(mask)[0];
}

double IoU(const cv::Mat& a, const cv::Mat& b){
    cv::Mat a1, a2;
    cv::bitwise_and(a, b, a1);
    cv::bitwise_or(a,b,a2);
    return static_cast<double>(cv::sum(a1)[0])/cv::sum(a2)[0];
}

double OneOverAnother(const cv::Mat& a, const cv::Mat& b){
    cv::Mat a1;
    cv::bitwise_and(a, b, a1);
    return static_cast<double>(cv::sum(a1)[0])/cv::sum(a)[0];
}

double SizeDifference(float a, float b){
    if(a < b) std::swap(a, b);
    return a/b;
}

float mod(const float a, const int b){
    return a - (cvRound(a)/b)*b;
}

double AngleError(const float angle1, const float angle2, const float maxDiff){
    return abs(angle1-angle2)/maxDiff;
}

double HowSquareIsRect(const cv::RotatedRect& rect){
    auto w = rect.size.width;
    auto h = rect.size.height;
    if(w < h){
        std::swap(w,h);
    }
    return w/h;
}

void wmd::ResizeImage(const cv::Mat &input, cv::Mat &output, int k) {
    if(k < 0){
        cv::resize(input, output, input.size()/(-k));
    } else{
        cv::resize(input, output, input.size()*(k));
    }
}

void wmd::GetGray(const cv::Mat& image, cv::Mat& gray) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
}

void wmd::GetRed(const cv::Mat& image, cv::Mat& red) {
    cv::Mat test;
    cv::cvtColor(image, test, cv::COLOR_BGR2HSV);
    cv::Scalar lowerRed = {0,50,50};
    cv::Scalar upperRed = {10, 255, 255};
    cv::Mat m1, m2;
    cv::inRange(test, lowerRed, upperRed, m1);
    lowerRed = {170,50,50};
    upperRed = {180, 255, 255};
    cv::inRange(test, lowerRed, upperRed, m2);
    red = (m2 + m1);
}
void wmd::GetRedAlt(const cv::Mat& image, cv::Mat& red) {
    cv::Mat a1, a2;
    cv::Mat ch[3];
    cv::split(image, ch);
    cv::bitwise_and(ch[0], ch[2], a1);
    cv::bitwise_and(ch[1], ch[2], a2);
    red = (ch[2] - a1 - a2);
}

void wmd::GetBlack(const cv::Mat& image, cv::Mat& black) {
    cv::Mat colors[3];
    cv::split(image, colors);
    cv::Mat a, b;
    cv::bitwise_or(colors[0], colors[2], a);
    cv::bitwise_or(a,colors[1],a);
    cv::bitwise_not(a, black);
}

void wmd::WatershedImage(const cv::Mat& image, const cv::Mat& grayImage, cv::Mat& markers, cv::Mat mask){
    cv::Mat test;
    cv::threshold(grayImage, test, 128, 255, cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy, hh;
    cv::findContours(test, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    markers = cv::Mat::zeros(image.size(), CV_32SC1);
    int compCount = 0;
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0], compCount++)
    {
        drawContours(markers, contours, idx, cv::Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
    }
    watershed(image, markers);
    if(!mask.empty()){
        InvertImage(mask, mask);
        markers -= mask;
    }
}

std::vector<cv::Vec4i> wmd::FindLines(const cv::Mat &img) {
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(img, lines, 2.4, CV_PI/180, 20, img.cols/2, 15);
    return lines;
}

void wmd::DrawLines(const cv::Mat &img, const std::vector<cv::Vec4i>& lines, cv::Mat& output){
    if(output.empty())
        output = cv::Mat::zeros(img.size(), CV_8UC1);
    else if(output.channels() == 1)
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    for(const auto& l: lines){
        cv::line(output, {l[0], l[1]}, {l[2], l[3]}, {0,0,255},2);
    }
}

void wmd::AffineTransformImage(const cv::Mat& input, cv::Mat& output, const std::vector<cv::Vec4i>& lines){
    auto m = FindAngle(lines);
    cv::Point2f pt = {m[0]+static_cast<float>(m[2]-m[0])/2, m[1]+static_cast<float>(m[3]-m[1])/2};
    float mv = 30, newX = mv*tg(m);
    cv::Point2f normal = pt + cv::Point2f(newX, mv);
    cv::Point2f fin[3]{{static_cast<float>(m[0]), static_cast<float>(m[1])},
                       {static_cast<float>(m[2]), static_cast<float>(m[3])},
                       normal};
    cv::Point2f fout[3]{{static_cast<float>(m[0]), static_cast<float>(m[3])},
                        {static_cast<float>(m[2]), static_cast<float>(m[3])},
                        normal - cv::Point2f((normal.x-pt.x) + 10, -m[3]+pt.y)};
    auto p = cv::getAffineTransform(fin, fout);
    cv::warpAffine(input, output, p, input.size());
}

cv::Vec4i wmd::FindAngle(const std::vector<cv::Vec4i>& lines) {
    std::vector<std::pair<double, cv::Vec4i>> t;
    for(auto a: lines){
        t.emplace_back(tg(a), a);
    }
    std::sort(t.begin(), t.end(), [](auto a, auto b){return a.first > b.first;});
    int left = 0, mex = 0;
    double sum = t[left].first;
    cv::Vec4i mP = t[left].second;
    for(int right = 1; right < t.size(); right++){
        double m = sum/(right - left);
        if(abs(m*1.1)-abs(t[right].first) > 0 && abs(m*0.9)-abs(t[right].first) < 0){
            sum += t[right].first;
        } else{
            left = right;
            sum = t[left].first;
        }
        if(mex < right - left){
            mex = right - left;
            mP = t[left + (right-left)/2].second;
        }
    }
    return mP;
}

void wmd::MaskImage(const cv::Mat &input, cv::Mat &output, const cv::Mat &mask) {
    cv::Mat m;
    if(input.channels() == 3){
        cv::cvtColor(mask, m, cv::COLOR_GRAY2BGR);
    } else if(mask.channels() == 3){
        cv::cvtColor(mask, m, cv::COLOR_BGR2GRAY);
    } else{
        m = mask.clone();
    }
    cv::resize(m, m, input.size());
    cv::bitwise_and(input, m, output);
}

cv::Vec4i wmd::StraightestLine(const std::vector<cv::Vec4i> &lines){
    std::pair<double, cv::Vec4i> mex;
    for(const auto &a: lines){
        auto t = tg(a);
        if(t < mex.first){
            mex = {t, a};
        }
    }
    return mex.second;
}

void wmd::GetCorners(const cv::Mat& input, cv::Mat& output, const int blockSize, const int kSize, const double k) {
    cv::Mat test = input.clone();
    cv::GaussianBlur(input, test, {13, 13}, 0);
    cv::cornerHarris(test, output, blockSize, kSize, k);
}

void wmd::GetEdges(const cv::Mat &input, cv::Mat &output, const int thresh) {
    cv::Mat test;
    cv::GaussianBlur(input, test, {11,11}, 0);
    cv::Mat dx,dy;
    Scharr(test,dx,CV_16S,1,0);
    Scharr(test,dy,CV_16S,0,1);
    Canny(dx,dy, output, thresh, thresh*3);
}

void wmd::Contrast(const cv::Mat& input, cv::Mat& output) {
    if(input.channels() >= 3){
        cv::Mat lab_image;
        cv::cvtColor(input, lab_image, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_planes(3);
        cv::split(lab_image, lab_planes);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(10.0);
        cv::Mat dst;
        clahe->apply(lab_planes[0], dst);
        dst.copyTo(lab_planes[0]);
        cv::merge(lab_planes, lab_image);
        cv::cvtColor(lab_image, output, cv::COLOR_Lab2BGR);
        return;
    }
    auto c = cv::createCLAHE(10.0, {15,15});
    c->apply(input, output);
}

void wmd::Denoise(const cv::Mat& input, cv::Mat& output) {
    if(input.channels() == 3){
        cv::fastNlMeansDenoisingColored(input, output, 10, 10);
        return;
    }
    cv::fastNlMeansDenoising(input, output, 10);
}

std::pair<cv::Point, int> wmd::FindBiggestCircle(const cv::Mat &input, const double dp, const int minRadius, const int minDist,
                      const double accuracy) {
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(input, circles, cv::HOUGH_GRADIENT_ALT, dp, minDist, 32, accuracy, minRadius);
    std::pair<cv::Point, int> max {{0,0}, 0};
    for(const auto& circle : circles)
    {
        cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        if(radius > max.second){
            max = {center, radius};
        }
    }
    return max;
}

void wmd::RectangleImageByCircle(const cv::Mat &input, const std::pair<cv::Point, int> &circle, const cv::Size& sourceSize,
                                 const int k, cv::Mat &output) {
    if(circle == std::make_pair(cv::Point(0,0), 0)){
        output = cv::Mat::zeros(1,1, CV_8UC3);
        return;
    }
    cv::Point center = circle.first;
    int radius = circle.second;
    int x = std::max(0, center.x-radius);
    int y = std::max(0, center.y-radius);
    int width = 2*radius + std::min(0, center.x-radius) + std::min(0, sourceSize.width - center.x - radius);
    int height = 2*radius + std::min(0, center.y-radius) + std::min(0, sourceSize.height - center.y - radius);
    cv::Rect rect{k*x, k*y, k*width, k*height};
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC3);
    cv::circle(mask, k*center, radius*k, {255, 255, 255}, -1);
    cv::bitwise_and(input, mask, output);
    output = output(rect);
}

void wmd::RectangleImageByCircleSingleChannel(const cv::Mat &input, const std::pair<cv::Point, int> &circle,
                                              const cv::Size& sourceSize, const int k, cv::Mat &output) {
    if(circle == std::make_pair(cv::Point(0,0), 0)){
        output = cv::Mat::zeros(1,1, CV_8UC1);
        return;
    }
    cv::Point center = circle.first;
    int radius = circle.second;
    int x = std::max(0, center.x-radius);
    int y = std::max(0, center.y-radius);
    int width = 2*radius + std::min(0, center.x-radius) + std::min(0, sourceSize.width - center.x - radius);
    int height = 2*radius + std::min(0, center.y-radius) + std::min(0, sourceSize.height - center.y - radius);
    cv::Rect rect{k*x, k*y, k*width, k*height};
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::circle(mask, k*center, radius*k, 255, -1);
    cv::bitwise_and(input, mask, output);
    output = output(rect);
}

void wmd::DrawLabels(const cv::Mat &labels, cv::Mat &output) {
    int idx = 0;
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            int label = labels.at<int>(idx);
            output.at<uchar>(i, j) = colorTab[label];
            idx++;
        }
    }
}

void wmd::DrawMarkers(const cv::Mat &markers, cv::Mat &output) {
    output = cv::Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index >= 0) {
                output.at<cv::Vec3b>(i, j) = colorfulTab[index];
            }
        }
    }
}

std::vector<std::vector<cv::KeyPoint>> wmd::ClusterKeyPoints(const cv::Mat& input, const std::vector<cv::KeyPoint> &kp,
                                                             cv::Mat &colorful, const int bigK, cv::Point center) {
    int idx = 0;
    if(kp.empty()) return std::vector<std::vector<cv::KeyPoint>>();
    cv::Mat data = cv::Mat::zeros(kp.size(), 3, CV_32F);
    if(center == cv::Point(0,0)){
        center = input.size()/2;
    }
    for(const auto& a: kp){
        data.at<float>(idx, 0) = a.pt.x;
        data.at<float>(idx, 1) = a.pt.y;
        data.at<float>(idx, 2) = distance2(a.pt, center);
        idx++;
    }
    cv::Mat centers = cv::Mat(bigK, 2, CV_32F), labels;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS, 30, 1);
    cv::kmeans(data, std::min(bigK, static_cast<int>(kp.size())), labels, criteria, 8, cv::KMEANS_PP_CENTERS, centers);
    if(colorful.empty()){
        colorful = cv::Mat::zeros(input.size(), CV_8UC3);
    }
    std::vector<std::vector<cv::KeyPoint>> clusters(std::min(bigK, static_cast<int>(kp.size())));
    for(int i = 0; i < kp.size(); i++){
        int label = labels.at<int>(i);
        clusters[label].push_back(kp[i]);
    }
    return clusters;
}

std::vector<cv::KeyPoint> wmd::FindKeyPoints(const cv::Mat &image, const int count) {
    auto sift = cv::SIFT::create(count);
    std::vector<cv::KeyPoint> kp;
    sift->detect(image, kp);
    return kp;
}

void wmd::DrawKeyPoints(const cv::Mat &image, const std::vector<cv::KeyPoint>& kp, cv::Mat &colorful) {
    cv::drawKeypoints(image, kp, colorful, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

cv::Rect wmd::BoundingRectByKeyPoints(const std::vector<cv::KeyPoint> &kp) {
    cv::Rect rect;
    std::vector<cv::Point> pts(kp.size());
    int idx = 0;
    for(const auto& a: kp){
        pts[idx] = a.pt;
        idx++;
    }
    rect = cv::boundingRect(pts);
    return rect;
}

cv::RotatedRect wmd::BoundingRotatedRectByKeyPoints(const std::vector<cv::KeyPoint> &kp) {
    std::vector<cv::Point> pts(kp.size());
    int idx = 0;
    for(const auto& a: kp){
        pts[idx] = a.pt;
        idx++;
    }
    cv::RotatedRect rect = cv::minAreaRect(pts);
    return rect;
}

void wmd::DrawRect(cv::Mat &image, const cv::Rect &rect) {
    cv::rectangle(image, rect, 255, -1);
}
void wmd::DrawRect(cv::Mat &image, const cv::RotatedRect &rect) {
    cv::Point2f vertices2f[4];
    rect.points(vertices2f);
    cv::Point vertices[4];
    for(int i = 0; i < 4; ++i){
        vertices[i] = vertices2f[i];
    }
    cv::fillConvexPoly(image,vertices,4,255);
}

void wmd::InvertImage(const cv::Mat &image, cv::Mat &output) {
    output = (255 - image);
}

std::vector<cv::Mat> wmd::ExtractMasksFromMarkers(const cv::Mat &markers) {
    double mis{0}, mas{0};
    cv::minMaxLoc(markers, &mis, &mas);
    std::vector<cv::Mat> masks, notNull;
    for(int i = 0; i <= mas; i++){
        masks.push_back(cv::Mat::zeros(markers.size(), CV_8UC1));
    }
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index >= 0) {
                masks[index].at<uchar>(i,j) = 255;
            }
        }
    }
    for(const auto& a: masks){
        if(cv::sum(a)[0] > 0){
            notNull.push_back(a);
        }
    }
    return notNull;
}

void wmd::ClusterMarkersByColorAmount(const std::vector<cv::Mat>& masks, const cv::Mat& grayImage, const int bigK, cv::Mat& outputMarkers) {
    int idx = 0;
    if(masks.empty()) return;
    cv::Mat data = cv::Mat::zeros(masks.size(), 1, CV_32F);
    for(const auto& a: masks){
        data.at<float>(idx, 0) = GetColorAmount(grayImage, a);
        idx++;
    }
    cv::Mat centers = cv::Mat(bigK, 2, CV_32F), labels;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS, 300, 0.1);
    cv::kmeans(data, std::min(bigK, static_cast<int>(masks.size())), labels, criteria, 20, cv::KMEANS_RANDOM_CENTERS, centers);
    outputMarkers = cv::Mat::zeros(grayImage.size(), CV_32S);
    for(int i = 0; i < masks.size(); i++){
        int label = labels.at<int>(i);
        outputMarkers += label*(masks[i]/255);
    }

}

void wmd::DrawKeyPointsAsMarkers(const std::vector<cv::KeyPoint>& kp, cv::Size imageSize, cv::Mat &output) {
    output = cv::Mat::zeros(imageSize, CV_8UC1);
    for(const auto& a: kp){
        output.at<uchar>(a.pt) = 255;
    }
}

void wmd::GetCircleMask(const cv::Mat& image, cv::Mat &output) {
    output = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::circle(output, image.size()/2, image.rows/2, 255, -1);
}

void wmd::AfterWatershedMasksThresh(const std::vector<cv::Mat> &masks, double thresh, cv::Mat &output) {
    if(masks.empty()) return;
    size_t ttl = masks[0].total();
    double maxAllowed = ttl*thresh;
    if(output.empty())
        output = cv::Mat::zeros(masks[0].size(), CV_8UC1);
    for(const auto& m: masks){
        if(cv::sum(m)[0]/255 < maxAllowed) output += m/5;
    }
}

void wmd::Dilate(const cv::Mat &input, cv::Mat &output) {
    cv::dilate(input, output, cv::Mat());
}

void wmd::Erode(const cv::Mat &input, cv::Mat &output) {
    cv::erode(input, output, cv::Mat());
}

double Point2LineDistance(const cv::Point2f& point, const std::pair<cv::Point2f, cv::Point2f>& line) {
    double a = line.second.y - line.first.y;
    double b = line.second.x - line.first.x;
    double c = line.second.x * line.first.y - line.first.x * line.second.y;
    return std::abs(a * point.x - b * point.y + c) / std::sqrt(a * a + b * b);
}

double GetPrecision(const cv::Mat& image, const cv::RotatedRect& rect) {
    cv::Point2f imageCenter(image.cols/2.0, image.rows/2.0);
    cv::Point2f vertices[4];
    rect.points(vertices);
    double a1 = Point2LineDistance(imageCenter, {vertices[0], vertices[1]});
    double a2 = Point2LineDistance(imageCenter, {vertices[1], vertices[2]});
    double a3 = Point2LineDistance(imageCenter, {vertices[2], vertices[3]});
    double a4 = Point2LineDistance(imageCenter, {vertices[3], vertices[0]});

    double s1 = std::min(a1, a2), s2 = std::min(a3, a4);

    double radius = image.cols/2;
    double r1 = std::min(s1, s2), r2 = radius - std::max(s1, s2);
    r1 /= radius;
    r2 /= radius;
    double g = (std::max((1-std::abs(r1-0.25)*4), 0.2) + std::max((1-std::abs(r2-0.5)*2), 0.2)) / 2.0;
    return g;
}

double wmd::GetMaskByContours(const cv::Mat &input, const std::vector<std::vector<cv::Point>>& contours, cv::Mat &output) {
    std::vector<cv::RotatedRect> tempRects;
    std::vector<cv::Mat> tempMats;
    cv::Mat ggg = cv::Mat::zeros(input.size(), CV_8UC1);
    for(const auto& a: contours){
        auto rect = cv::minAreaRect(a);
        cv::Point2f vertices2f[4];
        rect.points(vertices2f);
        cv::Point vertices[4];
        for(int i = 0; i < 4; ++i){
            vertices[i] = vertices2f[i];
        }
        cv::Mat newRect = cv::Mat::zeros(input.size(), CV_8UC1);
        cv::fillConvexPoly(newRect,vertices,4, 255);
        cv::fillConvexPoly(newRect,vertices,4, 255);
        if(rect.size.width > rect.size.height && rect.angle == 90){
            std::swap(rect.size.width, rect.size.height);
            rect.angle = 0;
        }
        bool repeatable = false;
        for(const auto & tempMat : tempMats){
            if(IoU(newRect, tempMat) > 0.9){
                repeatable = true;
                break;
            }
        }
        if(!repeatable){
            tempRects.push_back(rect);
            tempMats.push_back(newRect.clone());
        }
    }
    if(tempRects.empty()) return 0;
    output = cv::Mat::zeros(input.size(), CV_8UC1);
    std::vector<cv::Point> centerPoints;
    std::vector<cv::RotatedRect> rects;
    std::vector<cv::Mat> rectMats;
    for(int i = 0; i < tempRects.size(); i++){
        bool repeatable = false;
        for(int j = 0; j < tempMats.size(); j++){
            if(i == j) continue;
            if(OneOverAnother(tempMats[i], tempMats[j]) > 0.95){
                repeatable = true;
                break;
            }
            if(IoU(tempMats[i], tempMats[j]) > 0.8){
                repeatable = true;
                break;
            }
        }
        if(!repeatable){
            cv::bitwise_or(output, tempMats[i], output);
            rects.push_back(tempRects[i]);
            centerPoints.push_back(tempRects[i].center);
            rectMats.push_back(tempMats[i].clone());
        }
    }
    if(centerPoints.empty()) return 0;
    tempRects.clear();
    tempMats.clear();
    cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    for(int i = 0; i < centerPoints.size()-1; i++){
        int newPoint = i+1;
        float minDist = distance2(centerPoints[i], centerPoints[i+1]);
        for(int j = i+1; j < centerPoints.size(); j++){
            auto d = distance2(centerPoints[i], centerPoints[j]);
            if(d < minDist){
                minDist = d;
                newPoint = j;
            }
        }
        std::swap(centerPoints[i+1], centerPoints[newPoint]);
        std::swap(rects[i+1], rects[newPoint]);
        std::swap(rectMats[i+1], rectMats[newPoint]);
    }
    std::vector<float> rectAngles(rects.size());
    std::vector<float> line2rectAngles(rects.size());
    std::vector<float> lineAngles(centerPoints.size()+1, 360);

    for(int i = 1; i < centerPoints.size(); i++){
        lineAngles[i] = cv::fastAtan2((centerPoints[i-1].y - centerPoints[i].y),
                                      (centerPoints[i-1].x - centerPoints[i].x));
        auto t = mod(lineAngles[i], 360);
        lineAngles[i] = std::min(t, 360-t);
    }

    for(int i = 0; i < rectAngles.size(); i++){
        rectAngles[i] = rects[i].angle;
        if (rects[i].size.width > rects[i].size.height
            && AngleError(lineAngles[i], lineAngles[i+1], 5) > 1.0) {
            rectAngles[i] = 90 + rectAngles[i];
        }
        if(abs(lineAngles[i] - rectAngles[i]) < abs(lineAngles[i+1] - rectAngles[i])){
            line2rectAngles[i] = lineAngles[i];
        } else{
            line2rectAngles[i] = lineAngles[i+1];
        }
    }
    double precision = 0;

    std::vector<int> goodRects(rects.size(), 0);
    std::vector<double> rectAngleErrors(rects.size(), 0);
    int maxGood = 0, lastGood = 0;
    for(int i = 1; i < goodRects.size(); i++){
        auto e = AngleError(rectAngles[i], rectAngles[i-1], 10);
        if(e < 1.0){
            goodRects[i] = goodRects[i-1]+1;
            rectAngleErrors[i] = e;
            if(goodRects[i] > maxGood){
                lastGood = i;
                maxGood = goodRects[i];
            }
        }
    }
    cv::Mat circleMask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::circle(circleMask, {circleMask.rows/2+1, circleMask.cols/2+1}, circleMask.rows/2, 255, -1);
    if(maxGood > 0) {
        int goodLines = 0;
        std::vector<int> rects100, rects50;
        for (int i = lastGood - maxGood + 1; i < lastGood; i++) {
            auto e = AngleError(lineAngles[i], lineAngles[i + 1], 10);
            if (e < 1.0) {
                goodLines++;
                rects100.push_back(i);
            }
        }
        int goodSizes = 0;
        for (int i = lastGood - maxGood; i < lastGood; i++) {
            auto e = SizeDifference(rects[i].size.area(), rects[i + 1].size.area());
            if (e < 1.5) {
                goodSizes++;
                rects50.push_back(i);
            }
        }

        if (goodLines >= 2 && goodSizes >= goodLines) {
            cv::Mat numbersMask = cv::Mat::zeros(input.size(), CV_8UC1);
            std::vector<cv::Point> interestingPoints;
            for (const auto &a: rects100) {
                auto rect = rects[a];
                cv::Point2f vertices2f[4];
                rect.points(vertices2f);
                cv::Point vertices[4];
                for (int i = 0; i < 4; ++i) {
                    vertices[i] = vertices2f[i];
                    interestingPoints.push_back(vertices[i]);
                }
                cv::fillConvexPoly(numbersMask, vertices, 4, 255);
                cv::fillConvexPoly(output, vertices, 4, {0, 255, 0});
            }
            auto res = cv::minAreaRect(interestingPoints);
            if(res.size.width > res.size.height){
                res.size.width *= 100;
            } else{
                res.size.height *= 100;
            }
            cv::Point2f vertices2f[4];
            res.points(vertices2f);
            cv::Point vertices[4];
            for (int i = 0; i < 4; ++i) {
                vertices[i] = vertices2f[i];
            }
            cv::fillConvexPoly(numbersMask, vertices, 4, 255);
            cv::fillConvexPoly(output, vertices, 4, {0, 255, 0});
            cv::line(output, vertices[0], vertices[1], {255, 0, 0});
            precision = 1;
            output = numbersMask;
            return precision;
        }

        if(rects50.size() == 1){
            rects50.push_back(rects50[0]+1);
            auto e1 = AngleError(mod(lineAngles[rects50[0]+1], 90), mod(rectAngles[rects50[0]], 90), 4);
            e1 = std::min(e1, AngleError(90 - mod(lineAngles[rects50[0]+1], 90), mod(rectAngles[rects50[0]], 90), 4));
            auto e2 = AngleError(mod(lineAngles[rects50[0]+1], 90), mod(rectAngles[rects50[1]], 90), 4);
            e2 = std::min(e2, AngleError(90 - mod(lineAngles[rects50[0]+1], 90), mod(rectAngles[rects50[1]], 90), 4));
            if(e1 < 1.0 && e2 < 1.0) {
                cv::Mat numbersMask = cv::Mat::zeros(input.size(), CV_8UC1);
                std::vector<cv::Point> interestingPoints;
                for (const auto &a: rects50) {
                    auto rect = rects[a];
                    cv::Point2f vertices2f[4];
                    rect.points(vertices2f);
                    cv::Point vertices[4];
                    for (int i = 0; i < 4; ++i) {
                        vertices[i] = vertices2f[i];
                        interestingPoints.push_back(vertices[i]);
                    }
                    cv::fillConvexPoly(numbersMask, vertices, 4, 255);
                    cv::fillConvexPoly(output, vertices, 4, {255, 255, 0});
                }
                auto res = cv::minAreaRect(interestingPoints);
                if(res.size.width > res.size.height){
                    res.size.width *= 100;
                } else{
                    res.size.height *= 100;
                }
                cv::Point2f vertices2f[4];
                res.points(vertices2f);
                cv::Point vertices[4];
                for (int i = 0; i < 4; ++i) {
                    vertices[i] = vertices2f[i];
                }
                cv::fillConvexPoly(output, vertices, 4, {0, 255, 0});
                cv::fillConvexPoly(numbersMask, vertices, 4, 255);
                output = numbersMask;
                precision = GetPrecision(numbersMask, res);
                return precision;
            }
        }
    }

    std::vector<int> bigCandidates;
    for(int i = 0; i < rects.size(); i++) {
        if(rects[i].size.area() > 0.07*input.total()
            && OneOverAnother(rectMats[i], circleMask) > 0.9
            && HowSquareIsRect(rects[i]) > 1.5){
            bigCandidates.push_back(i);
        }
    }
    int maxSize = 0, maxIdx = -1;
    if(bigCandidates.size() > 1){
        for (const auto &a: bigCandidates) {
            auto rect = rects[a];
            if(rect.size.area() > maxSize){
                maxIdx = a;
                maxSize = rect.size.area();
            }
        }
        auto rect = rects[maxIdx];
        if(rect.size.width > rect.size.height){
            rect.size.width *= 100;
        } else{
            rect.size.height *= 100;
        }
        cv::Point2f vertices2f[4];
        rect.points(vertices2f);
        cv::Point vertices[4];
        for (int i = 0; i < 4; ++i) {
            vertices[i] = vertices2f[i];
        }
        cv::fillConvexPoly(output, vertices, 4, {255, 0, 255});
        cv::Mat numbersMask = cv::Mat::zeros(input.size(), CV_8UC1);
        cv::fillConvexPoly(numbersMask, vertices, 4, 255);
        output = numbersMask;
        precision = GetPrecision(numbersMask, rect);
        return precision;
    } else if(bigCandidates.size() == 1){
        auto rect = rects[bigCandidates[0]];
        if(rect.size.width > rect.size.height){
            rect.size.width *= 100;
        } else{
            rect.size.height *= 100;
        }
        cv::Point2f vertices2f[4];
        rect.points(vertices2f);
        cv::Point vertices[4];
        for (int i = 0; i < 4; ++i) {
            vertices[i] = vertices2f[i];
        }
        cv::fillConvexPoly(output, vertices, 4, {255, 0, 255});
        cv::Mat numbersMask = cv::Mat::zeros(input.size(), CV_8UC1);
        cv::fillConvexPoly(numbersMask, vertices, 4, 255);
        output = numbersMask;
        precision = GetPrecision(numbersMask, rect);
        return precision;
    }
//    for(int i = 1; i < centerPoints.size(); i++){
//        cv::circle(output, centerPoints[i-1], 3, {0,0,255}, -1);
//        cv::line(output, centerPoints[i-1], centerPoints[i], {0,0,255}, 1);
//    }
    return 0;
}

std::vector<std::vector<cv::Point>> wmd::FilterContoursBySize(const cv::Mat &input,
                                                              const std::vector<std::vector<cv::Point>>& contours,
                                                              cv::Mat& output) {
    std::vector<std::vector<cv::Point>> rtnContours;
    output = cv::Mat::zeros(input.size(), CV_8UC1);
    for(int i = 0; i < contours.size(); i++){
        auto a = contours[i];
        if(a.size() < 100) {
//            cv::drawContours(output, contours, i, {255,0,0}, 1);
            continue;
        }
        auto rect = cv::minAreaRect(a);
        if(rect.size.area() > input.total()*0.3) {
//            cv::drawContours(output, contours, i, {0,255,0}, 1);
            continue;
        }
        rtnContours.push_back(contours[i]);
        cv::drawContours(output, contours, i, {255,255,255}, 1);
    }
    return rtnContours;
}

std::vector<std::vector<std::vector<cv::Point>>> wmd::ClusterContours(const cv::Mat &input,
                                                                      const std::vector<std::vector<cv::Point>>& contours,
                                                                      cv::Mat &colorful) {
    int bigK = 3;
    bigK = std::min(bigK, static_cast<int>(contours.size()));
    std::vector<std::vector<std::vector<cv::Point>>> result(bigK);
    colorful = cv::Mat::zeros(input.size(), CV_8UC3);
    cv::Mat data = cv::Mat::zeros(contours.size(), 2, CV_32F);
    for(int i = 0; i < contours.size(); i++){
        auto rect = cv::minAreaRect(contours[i]);
//        data.at<float>(i, 0) = cv::isContourConvex(contours[i]);
        data.at<float>(i, 1) = rect.angle/180.0;
        data.at<float>(i, 0) = std::min(rect.size.width/rect.size.height,rect.size.height/rect.size.width);
    }
    cv::Mat centers = cv::Mat(bigK, 2, CV_32F), labels;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS, 10, 0.01);
    cv::kmeans(data, bigK, labels, criteria, 8, cv::KMEANS_RANDOM_CENTERS, centers);
    for(int i = 0; i < contours.size(); i++){
        int label = labels.at<int>(i);
        result[label].push_back(contours[i]);
        cv::drawContours(colorful, contours, i, colorfulTab[label], 1);
    }
    return result;
}

std::vector<std::vector<cv::Point>> wmd::GetContours(const cv::Mat &input) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(input, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    return contours;
}

std::vector<std::vector<cv::Point>> wmd::FilterClusteredContours(const cv::Mat &input,
                                                        const std::vector<std::vector<cv::Point>>& contours,
                                                        cv::Mat &output) {
    std::vector<std::vector<cv::Point>> rtnContours;
    output = cv::Mat::zeros(input.size(), CV_8UC1);
    if(contours.empty()) return rtnContours;
    if(contours.size() == 1){
        auto rect = cv::minAreaRect(contours[0]);
        if(rect.size.area() > 0.1*input.total()) return contours;
        return rtnContours;
    }
    for(int i = 0; i < contours.size(); i++){
        auto rect = cv::minAreaRect(contours[i]);
        if(rect.size.area() > 0.2*input.total()) rtnContours.push_back(contours[i]);
        cv::Point2f vertices2f[4];
        rect.points(vertices2f);
        cv::Rect imageRect(cv::Point(), input.size());
        if(!imageRect.contains(vertices2f[0])) continue;
        if(!imageRect.contains(vertices2f[1])) continue;
        if(!imageRect.contains(vertices2f[2])) continue;
        if(!imageRect.contains(vertices2f[3])) continue;
        rtnContours.push_back(contours[i]);
    }
    return rtnContours;
}

cv::RotatedRect wmd::GetSmallerRect(const cv::Mat &input, cv::Mat &output, const double left, const double right) {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat t = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::findContours(input, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    std::vector<cv::Point> biggest;
    size_t m = 0;
    for(const auto& a: contours){
        if(a.size() > m){
            biggest = a;
            m = a.size();
        }
    }
    auto newRect = cv::minAreaRect(biggest);
    auto newCenter = newRect.center;
    float angleRad = newRect.angle * CV_PI / 180.0;
    double centerShift = (right - left)/2.0;

    if(newRect.size.height > newRect.size.width){
        cv::Point2f shift = cv::Point2f(-sin(angleRad), cos(angleRad)) * newRect.size.height*centerShift;
        newCenter += shift;
        newRect.size.height *= (1-2*std::min(left, right));
    } else{
        cv::Point2f shift = cv::Point2f(cos(angleRad), sin(angleRad)) * newRect.size.height*centerShift;
        newCenter -= shift;
        newRect.size.width *= (1-2*std::min(left, right));
    }

    newRect.center = newCenter;
    cv::Point2f vertices2f[4];
    newRect.points(vertices2f);
    cv::Point vertices[4];
    for (int i = 0; i < 4; ++i) {
        vertices[i] = vertices2f[i];
    }
    cv::fillConvexPoly(t, vertices, 4, 255);
    output = t.clone();
    return newRect;
}