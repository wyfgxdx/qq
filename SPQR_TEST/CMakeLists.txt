cmake_minimum_required(VERSION 3.0)

project(spqr_test)


include_directories("/usr/include/eigen3")
include_directories("/home/robert/SuiteSparse-5.8.1/SPQR/Include")
include_directories("/home/robert/SuiteSparse-5.8.1/CHOLMOD/Include")
include_directories("/home/robert/SuiteSparse-5.8.1/include")
include_directories("/home/robert/SuiteSparse-5.8.1")
link_directories("/home/robert/SuiteSparse-5.8.1/SuiteSparse_config")
link_directories("/home/robert/SuiteSparse-5.8.1/CHOLMOD/Lib")
link_directories("/home/robert/SuiteSparse-5.8.1/lib")
link_directories("/home/robert/SuiteSparse-5.8.1/SPQR/Lib")


add_executable(spqr_test main.cpp)
target_link_libraries(spqr_test cholmod)
target_link_libraries(spqr_test /home/robert/SuiteSparse-5.8.1/SuiteSparse_config/SuiteSparse_config.o)
#target_link_libraries(spqr_test /home/robert/SuiteSparse-5.8.1/SPQR/Lib/SuiteSparseQR.o)
target_link_libraries(spqr_test spqr)

install(TARGETS spqr_test RUNTIME DESTINATION bin)

