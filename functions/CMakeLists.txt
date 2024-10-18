add_library(functions_lib STATIC functions_lib.h src/functions.cpp src/OCR.cpp)
add_library(functions_tests STATIC functions_tests.h tests_src/tests.cpp)

target_link_libraries(functions_lib
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
        opencv_photo
        opencv_flann
        opencv_dnn
        opencv_features2d
        opencv_highgui
)

target_link_libraries(functions_tests
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
)