add_library(detector STATIC meter_detector.h meter_detector.cpp)

target_link_libraries(detector
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
        functions_lib
        functions_tests
)
target_include_directories(detector PUBLIC ${PROJECT_SOURCE_DIR}/prj.cw/functions)