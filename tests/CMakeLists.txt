add_executable(cw_test cw_test.cpp)
target_include_directories(cw_test PRIVATE
        ${PROJECT_SOURCE_DIR}/prj.cw/detector
)

target_link_libraries(cw_test detector)

add_executable(cw_dataset_test dataset_test.cpp)
target_include_directories(cw_dataset_test PRIVATE
        ${PROJECT_SOURCE_DIR}/prj.cw/detector
)

target_link_libraries(cw_dataset_test detector)