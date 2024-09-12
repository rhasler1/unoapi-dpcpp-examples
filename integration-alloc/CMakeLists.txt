# UnoAPI:CMakeLists-targetlibraries:begin
add_executable(integration_alloc main.cpp f.cpp trapezoid.cpp timestamps.cpp)
target_link_libraries(integration fmt::fmt spdlog::spdlog CLI11::CLI11)
# UnoAPI:CMakeLists-targetlibraries:end

