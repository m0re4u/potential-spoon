set(CMAKE_SYSTEM_NAME Linux)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/gcc/include/
  ${CMAKE_CURRENT_LIST_DIR}/gcc/include/c++/5.2.0/
  ${CMAKE_CURRENT_LIST_DIR}/gcc/include/c++/5.2.0/i686-pc-linux-gnu/
  ${CMAKE_CURRENT_LIST_DIR}/clang/include/
)
# Nao target platform
set(triple i686-pc-linux-gnu)

# TODO: move -nostdlib to some cmake flag
set(CMAKE_C_COMPILER clang -nostdlib)
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

# TODO: move -nostdlib to some cmake flag
set(CMAKE_CXX_COMPILER clang++ -nostdlib)
set(CMAKE_CXX_COMPILER_TARGET ${triple})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/crt1.o  \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/crti.o  \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/crtbeginS.o \ \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/libstdc++.so.6.0.21 \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/libm-2.13.so \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/libgcc_s.so.1 \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/libc-2.13.so \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/libc_nonshared.a \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/libgcc_s.so.1 \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/libpthread-2.13.so \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/librt-2.13.so \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/ld-2.13.so \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/crtendS.o \
  ${CMAKE_CURRENT_LIST_DIR}/gcc/lib/crtn.o \
")
