# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /tyk/cmake-3.24.4/bin/cmake

# The command to remove a file.
RM = /tyk/cmake-3.24.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /tyk/MMHA/cutlass_learn/FC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tyk/MMHA/cutlass_learn/FC/build

# Include any dependencies generated for this target.
include CMakeFiles/FC.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/FC.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/FC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FC.dir/flags.make

CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o: /tyk/MMHA/cutlass_learn/FC/fc_bias_act_generate_example.cu
CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o: CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o.depend
CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o: CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/tyk/MMHA/cutlass_learn/FC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o"
	cd /tyk/MMHA/cutlass_learn/FC/build/CMakeFiles/FC.dir && /tyk/cmake-3.24.4/bin/cmake -E make_directory /tyk/MMHA/cutlass_learn/FC/build/CMakeFiles/FC.dir//.
	cd /tyk/MMHA/cutlass_learn/FC/build/CMakeFiles/FC.dir && /tyk/cmake-3.24.4/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/tyk/MMHA/cutlass_learn/FC/build/CMakeFiles/FC.dir//./FC_generated_fc_bias_act_generate_example.cu.o -D generated_cubin_file:STRING=/tyk/MMHA/cutlass_learn/FC/build/CMakeFiles/FC.dir//./FC_generated_fc_bias_act_generate_example.cu.o.cubin.txt -P /tyk/MMHA/cutlass_learn/FC/build/CMakeFiles/FC.dir//FC_generated_fc_bias_act_generate_example.cu.o.cmake

# Object files for target FC
FC_OBJECTS =

# External object files for target FC
FC_EXTERNAL_OBJECTS = \
"/tyk/MMHA/cutlass_learn/FC/build/CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o"

FC: CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o
FC: CMakeFiles/FC.dir/build.make
FC: /usr/local/cuda/lib64/libcudart_static.a
FC: /usr/lib/x86_64-linux-gnu/librt.so
FC: CMakeFiles/FC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tyk/MMHA/cutlass_learn/FC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FC"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FC.dir/build: FC
.PHONY : CMakeFiles/FC.dir/build

CMakeFiles/FC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FC.dir/clean

CMakeFiles/FC.dir/depend: CMakeFiles/FC.dir/FC_generated_fc_bias_act_generate_example.cu.o
	cd /tyk/MMHA/cutlass_learn/FC/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tyk/MMHA/cutlass_learn/FC /tyk/MMHA/cutlass_learn/FC /tyk/MMHA/cutlass_learn/FC/build /tyk/MMHA/cutlass_learn/FC/build /tyk/MMHA/cutlass_learn/FC/build/CMakeFiles/FC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FC.dir/depend
