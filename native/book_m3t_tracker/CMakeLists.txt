cmake_minimum_required(VERSION 3.16)
project(book_m3t_tracker LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(M3T_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/m3t-book-tracker/M3T" CACHE PATH "M3T source")
find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
set(GLFW_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/glfw-book/include")
set(GLFW_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv-book-reconstruction/lib/python3.12/site-packages/glfw/libglfw.3.dylib")
file(GLOB TRACKER_SOURCES "${M3T_SOURCE}/src/*.cpp")
list(FILTER TRACKER_SOURCES EXCLUDE REGEX "(azure_kinect_camera|realsense_camera|renderer_geometry|texture_modality)\\.cpp$")
# Upstream's hidden core-profile window needs forward compatibility on macOS.
# Keep upstream untouched; compile this documented generated copy instead.
file(READ "${M3T_SOURCE}/src/renderer_geometry.cpp" RENDERER)
string(REPLACE "glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);"
 "glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);\n    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);" RENDERER "${RENDERER}")
string(REPLACE "glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);"
 "glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);" RENDERER "${RENDERER}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/renderer_geometry_mac.cpp" "${RENDERER}")
add_library(book_m3t STATIC ${TRACKER_SOURCES} "${CMAKE_CURRENT_BINARY_DIR}/renderer_geometry_mac.cpp")
target_include_directories(book_m3t PUBLIC "${M3T_SOURCE}/include" "${M3T_SOURCE}/third_party"
 "/opt/homebrew/include/eigen3" "${GLFW_INCLUDE}" ${OpenCV_INCLUDE_DIRS} "/opt/homebrew/opt/libomp/include")
target_compile_options(book_m3t PRIVATE -Wno-deprecated-declarations -Xpreprocessor -fopenmp -include opencv2/geometry/3d.hpp)
target_link_libraries(book_m3t PUBLIC ${OpenCV_LIBS} GLEW::GLEW OpenGL::GL "${GLFW_LIBRARY}" "/opt/homebrew/opt/libomp/lib/libomp.dylib")
add_executable(book_m3t_tracker tracker.cpp)
target_link_libraries(book_m3t_tracker PRIVATE book_m3t)
