# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/axel/Documents/code/LucasKanade

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/axel/Documents/code/LucasKanade/compile

# Include any dependencies generated for this target.
include CMakeFiles/exec.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/exec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/exec.dir/flags.make

CMakeFiles/exec.dir/source/main.cpp.o: CMakeFiles/exec.dir/flags.make
CMakeFiles/exec.dir/source/main.cpp.o: ../source/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/axel/Documents/code/LucasKanade/compile/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/exec.dir/source/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/exec.dir/source/main.cpp.o -c /home/axel/Documents/code/LucasKanade/source/main.cpp

CMakeFiles/exec.dir/source/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/exec.dir/source/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/axel/Documents/code/LucasKanade/source/main.cpp > CMakeFiles/exec.dir/source/main.cpp.i

CMakeFiles/exec.dir/source/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/exec.dir/source/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/axel/Documents/code/LucasKanade/source/main.cpp -o CMakeFiles/exec.dir/source/main.cpp.s

CMakeFiles/exec.dir/source/main.cpp.o.requires:
.PHONY : CMakeFiles/exec.dir/source/main.cpp.o.requires

CMakeFiles/exec.dir/source/main.cpp.o.provides: CMakeFiles/exec.dir/source/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/exec.dir/build.make CMakeFiles/exec.dir/source/main.cpp.o.provides.build
.PHONY : CMakeFiles/exec.dir/source/main.cpp.o.provides

CMakeFiles/exec.dir/source/main.cpp.o.provides.build: CMakeFiles/exec.dir/source/main.cpp.o

# Object files for target exec
exec_OBJECTS = \
"CMakeFiles/exec.dir/source/main.cpp.o"

# External object files for target exec
exec_EXTERNAL_OBJECTS =

exec: CMakeFiles/exec.dir/source/main.cpp.o
exec: CMakeFiles/exec.dir/build.make
exec: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_nonfree.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_nonfree.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
exec: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
exec: CMakeFiles/exec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable exec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/exec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/exec.dir/build: exec
.PHONY : CMakeFiles/exec.dir/build

CMakeFiles/exec.dir/requires: CMakeFiles/exec.dir/source/main.cpp.o.requires
.PHONY : CMakeFiles/exec.dir/requires

CMakeFiles/exec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/exec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/exec.dir/clean

CMakeFiles/exec.dir/depend:
	cd /home/axel/Documents/code/LucasKanade/compile && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/axel/Documents/code/LucasKanade /home/axel/Documents/code/LucasKanade /home/axel/Documents/code/LucasKanade/compile /home/axel/Documents/code/LucasKanade/compile /home/axel/Documents/code/LucasKanade/compile/CMakeFiles/exec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/exec.dir/depend

