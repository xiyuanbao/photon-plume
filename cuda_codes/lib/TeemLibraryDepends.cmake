# Generated by CMake 2.8.12.2

IF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.4)
  # Information for CMake 2.6 and above.
  SET("teem_LIB_DEPENDS" "general;/usr/lib64/libbz2.so;general;/usr/local/lib/libz.a;general;/usr/lib64/libpng.so;general;/usr/local/lib/libz.a;general;-lpthread;general;-lm;")
ELSE("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.4)
  # Information for CMake 2.4 and lower.
  SET("teem_LIB_DEPENDS" "/usr/lib64/libbz2.so;/usr/local/lib/libz.a;/usr/lib64/libpng.so;/usr/local/lib/libz.a;-lpthread;-lm;")
ENDIF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.4)
