PROJECT := caffedistri

CONFIG_FILE ?= ../caffe-public/Makefile.config
# Explicitly check for the config file, otherwise make -k will proceed anyway.
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found. See $(CONFIG_FILE).example.)
endif
include $(CONFIG_FILE)

BUILD_DIR_LINK := $(BUILD_DIR)
ifeq ($(RELEASE_BUILD_DIR),)
	RELEASE_BUILD_DIR := .$(BUILD_DIR)_release
endif
ifeq ($(DEBUG_BUILD_DIR),)
	DEBUG_BUILD_DIR := .$(BUILD_DIR)_debug
endif

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(DEBUG_BUILD_DIR)
	OTHER_BUILD_DIR := $(RELEASE_BUILD_DIR)
else
	BUILD_DIR := $(RELEASE_BUILD_DIR)
	OTHER_BUILD_DIR := $(DEBUG_BUILD_DIR)
endif

# All of the directories containing code.
SRC_DIRS := $(shell find * -type d -exec bash -c "find {} -maxdepth 1 \
	\( -name '*.cpp' -o -name '*.proto' \) | grep -q ." \; -print)

# The target shared library name
LIBRARY_NAME := $(PROJECT)
LIB_BUILD_DIR := $(BUILD_DIR)/lib
STATIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).a
DYNAMIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).so

##############################
# Get all source files
##############################
# CXX_SRCS are the source files excluding the tools directory.
CXX_SRCS := $(shell find src ! -path "*/tools/*" -name "*.cpp")
# TOOL_SRCS are the source files for the tool binaries
TOOL_SRCS := $(shell find src -path "*/tools/*" -name "*.cpp")
# BUILD_INCLUDE_DIR contains any generated header files we want to include.
BUILD_INCLUDE_DIR := $(BUILD_DIR)/src
# PROTO_SRC is the protocol buffer definition
PROTO_SRC_DIR := ../caffe-public/distribute/proto
PROTO_SRC := $(PROTO_SRC_DIR)/caffe.proto
# PROTO_BUILD_DIR will contain the .java files generated from PROTO_SRCS;
PROTO_BUILD_DIR := src/main/java
# NONGEN_CXX_SRCS includes all source/header files except those generated
# automatically (e.g., by proto).
NONGEN_CXX_SRCS := $(shell find \
	include \
	src \
	-name "*.cpp" -or -name "*.hpp")


##############################
# Derive generated files
##############################
# The javah generated jni files
JNI_CPP_FILES := $(shell find src/main/cpp/jni -name "Jni*.cpp" -exec basename {} ";")
JNI_CLASS_DIR := target/classes
JNI_JCAFFE_DIR := $(JNI_CLASS_DIR)/com/yahoo/ml/jcaffe
JNI_GEN_HEADER_DIR := include/jni
JNI_GEN_HEADER_PREFIX := com_yahoo_ml_jcaffe_
JNI_CLASS_NAME_PREFIX := com.yahoo.ml.jcaffe
JNI_GEN_HEADER_FILES := $(addprefix $(JNI_GEN_HEADER_DIR)/$(JNI_GEN_HEADER_PREFIX), ${JNI_CPP_FILES:Jni%.cpp=%.h})
# The generated files for protocol buffers
PROTO_GEN_JAVA := $(PROTO_BUILD_DIR)/caffe/Caffe.java
PROTO_GEN_JAVA_DIR := $(PROTO_BUILD_DIR)/caffe
# The objects corresponding to the source files
# These objects will be linked into the final shared library, so we
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
OBJS := $(CXX_OBJS)
# tool objects
TOOL_OBJS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o})
TOOL_BUILD_DIR := $(BUILD_DIR)
# Output files for automatic dependency generation
DEPS := ${CXX_OBJS:.o=.d}
# tool bins
TOOL_BINS := ${TOOL_OBJS:.o=.bin}

##############################
# Derive compiler warning dump locations
##############################
WARNS_EXT := warnings.txt
CXX_WARNS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o.$(WARNS_EXT)})
TOOL_WARNS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o.$(WARNS_EXT)})
ALL_CXX_WARNS := $(CXX_WARNS)

EMPTY_WARN_REPORT := $(BUILD_DIR)/.$(WARNS_EXT)
NONEMPTY_WARN_REPORT := $(BUILD_DIR)/$(WARNS_EXT)

##############################
# Derive include and lib directories
##############################

CUDA_INCLUDE_DIR ?= $(CUDA_DIR)/include

CUDA_LIB_DIR ?=
# add <cuda>/lib64 only if it exists
ifneq ("$(wildcard $(CUDA_DIR)/lib64)","")
       CUDA_LIB_DIR += $(CUDA_DIR)/lib64
endif
CUDA_LIB_DIR += $(CUDA_DIR)/lib

EXTRA_INCLUDE_DIRS := ../caffe-public/src
INCLUDE_DIRS += ${JAVA_HOME}/include
INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) ./include ../caffe-public/distribute/include $(EXTRA_INCLUDE_DIRS)
LIBRARY_DIRS += ../caffe-public/distribute/lib
ifneq ($(CPU_ONLY), 1)
	INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
	LIBRARY_DIRS += $(CUDA_LIB_DIR)
	LIBRARIES := cudart cublas curand
endif

LIBRARIES += glog gflags protobuf boost_system m caffe hdf5_hl hdf5

# handle IO dependencies
USE_OPENCV ?= 1

ifeq ($(USE_OPENCV), 1)
	LIBRARIES += opencv_core opencv_highgui opencv_imgproc 

	ifeq ($(OPENCV_VERSION), 3)
		LIBRARIES += opencv_imgcodecs
	endif
endif
WARNINGS := -Wall -Wno-sign-compare

##############################
# Set build directories
##############################

DISTRIBUTE_DIR ?= distribute
DISTRIBUTE_SUBDIRS := $(DISTRIBUTE_DIR)/lib
DIST_ALIASES := dist
ifneq ($(strip $(DISTRIBUTE_DIR)),distribute)
		DIST_ALIASES += distribute
endif

ALL_BUILD_DIRS := $(sort $(BUILD_DIR) $(addprefix $(BUILD_DIR)/, $(SRC_DIRS)) \
	$(LIB_BUILD_DIR) $(DISTRIBUTE_SUBDIRS) $(JNI_GEN_HEADER_DIR)) \
	$(PROTO_GEN_JAVA_DIR)

##############################
# Set directory for Doxygen-generated documentation
##############################
DOXYGEN_CONFIG_FILE ?= ./.Doxyfile
# should be the same as OUTPUT_DIRECTORY in the .Doxyfile
DOXYGEN_OUTPUT_DIR ?= ./doxygen
DOXYGEN_COMMAND ?= doxygen
# All the files that might have Doxygen documentation.
DOXYGEN_SOURCES := $(shell find \
	src \
	include \
	-name "*.cpp" -or -name "*.hpp")
DOXYGEN_SOURCES += $(DOXYGEN_CONFIG_FILE)


##############################
# Configure build
##############################

# Determine platform
UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LINUX := 1
else ifeq ($(UNAME), Darwin)
	OSX := 1
endif

# path for jni_md.h and jawt_md.h.
uname := $(shell echo ${UNAME} | tr A-Z a-z)
INCLUDE_DIRS += ${JAVA_HOME}/include/${uname}

# Linux
ifeq ($(LINUX), 1)
	CXX ?= /usr/bin/g++
	GCCVERSION := $(shell $(CXX) -dumpversion | cut -f1,2 -d.)
	# older versions of gcc are too dumb to build boost with -Wuninitalized
	ifeq ($(shell echo | awk '{exit $(GCCVERSION) < 4.6;}'), 1)
		WARNINGS += -Wno-uninitialized
	endif
	# boost::thread is reasonably called boost_thread (compare OS X)
	# We will also explicitly add stdc++ to the link target.
	LIBRARIES += boost_thread stdc++
endif

# OS X:
# clang++ instead of g++
# libstdc++ for NVCC compatibility on OS X >= 10.9 with CUDA < 7.0
ifeq ($(OSX), 1)
	CXX := /usr/bin/clang++
        CXXFLAGS +=  -fno-common -frtti
        LINKFLAGS +=  -undefined dynamic_lookup -dynamiclib
	ifneq ($(CPU_ONLY), 1)
		CUDA_VERSION := $(shell $(CUDA_DIR)/bin/nvcc -V | grep -o 'release \d' | grep -o '\d')
		ifeq ($(shell echo | awk '{exit $(CUDA_VERSION) < 7.0;}'), 1)
			CXXFLAGS += -stdlib=libstdc++
			LINKFLAGS += -stdlib=libstdc++
		endif
		# clang throws this warning for cuda headers
		WARNINGS += -Wno-unneeded-internal-declaration
	endif
	# gtest needs to use its own tuple to not conflict with clang
	COMMON_FLAGS += -DGTEST_USE_OWN_TR1_TUPLE=1
	# boost::thread is called boost_thread-mt to mark multithreading on OS X
	LIBRARIES += boost_thread-mt
        DYNAMIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).jnilib
	# we need to explicitly ask for the rpath to be obeyed
	DYNAMIC_FLAGS := -install_name @rpath/libcaffe.so
	ORIGIN := @loader_path
else
	ORIGIN := \$$ORIGIN
endif

# Custom compiler
ifdef CUSTOM_CXX
	CXX := $(CUSTOM_CXX)
endif

# Static linking
ifneq (,$(findstring clang++,$(CXX)))
	STATIC_LINK_COMMAND := -Wl,-force_load $(STATIC_NAME)
else ifneq (,$(findstring g++,$(CXX)))
	STATIC_LINK_COMMAND := -Wl,--whole-archive $(STATIC_NAME) -Wl,--no-whole-archive
else
  # The following line must not be indented with a tab, since we are not inside a target
  $(error Cannot static link with the $(CXX) compiler)
endif

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	NVCCFLAGS += -G
else
	COMMON_FLAGS += -DNDEBUG -O2
endif

# cuDNN acceleration configuration.
ifeq ($(USE_CUDNN), 1)
	LIBRARIES += cudnn
	COMMON_FLAGS += -DUSE_CUDNN
endif

# configure IO libraries
ifeq ($(USE_OPENCV), 1)
	COMMON_FLAGS += -DUSE_OPENCV
endif
ifeq ($(USE_LEVELDB), 1)
	COMMON_FLAGS += -DUSE_LEVELDB
endif
ifeq ($(USE_LMDB), 1)
	COMMON_FLAGS += -DUSE_LMDB
ifeq ($(ALLOW_LMDB_NOLOCK), 1)
	COMMON_FLAGS += -DALLOW_LMDB_NOLOCK
endif
endif

# CPU-only configuration
ifeq ($(CPU_ONLY), 1)
	OBJS := $(PROTO_OBJS) $(CXX_OBJS)
	TEST_OBJS := $(TEST_CXX_OBJS)
	TEST_BINS := $(TEST_CXX_BINS)
	ALL_WARNS := $(ALL_CXX_WARNS)
	TEST_FILTER := --gtest_filter="-*GPU*"
	COMMON_FLAGS += -DCPU_ONLY
endif

# Python layer support
ifeq ($(WITH_PYTHON_LAYER), 1)
	COMMON_FLAGS += -DWITH_PYTHON_LAYER
	LIBRARIES += $(PYTHON_LIBRARIES)
endif

ifeq ($(INFINIBAND), 1)
	COMMON_FLAGS += -DINFINIBAND
	LIBRARIES += ibverbs ibumad
endif

# BLAS configuration (default = ATLAS)
BLAS ?= atlas
ifeq ($(BLAS), mkl)
	# MKL
	LIBRARIES += mkl_rt
	COMMON_FLAGS += -DUSE_MKL
	MKL_DIR ?= /opt/intel/mkl
	BLAS_INCLUDE ?= $(MKL_DIR)/include
	BLAS_LIB ?= $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64
else ifeq ($(BLAS), open)
	# OpenBLAS
	LIBRARIES += openblas
else
	# ATLAS
	ifeq ($(LINUX), 1)
		ifeq ($(BLAS), atlas)
			# Linux simply has cblas and atlas
			LIBRARIES += cblas atlas
		endif
	else ifeq ($(OSX), 1)
		# OS X packages atlas as the vecLib framework
		LIBRARIES += cblas
		# 10.10 has accelerate while 10.9 has veclib
		XCODE_CLT_VER := $(shell pkgutil --pkg-info=com.apple.pkg.CLTools_Executables | grep 'version' | sed 's/[^0-9]*\([0-9]\).*/\1/')
		XCODE_CLT_GEQ_6 := $(shell [ $(XCODE_CLT_VER) -gt 5 ] && echo 1)
		ifeq ($(XCODE_CLT_GEQ_6), 1)
			BLAS_INCLUDE ?= /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
			LDFLAGS += -framework Accelerate
		else
			BLAS_INCLUDE ?= /System/Library/Frameworks/vecLib.framework/Versions/Current/Headers/
			LDFLAGS += -framework vecLib
		endif
	endif
endif
INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)

LIBRARY_DIRS += $(LIB_BUILD_DIR)

# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
# mex may invoke an older gcc that is too liberal with -Wuninitalized
MATLAB_CXXFLAGS := $(CXXFLAGS) -Wno-uninitialized
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)

USE_PKG_CONFIG ?= 0
ifeq ($(USE_PKG_CONFIG), 1)
	PKG_CONFIG := $(shell pkg-config opencv --libs)
else
	PKG_CONFIG :=
endif
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
		$(foreach library,$(LIBRARIES),-l$(library))

# Set the sub-targets of the 'everything' target.
EVERYTHING_TARGETS := all py$(PROJECT) test warn lint
# Only build matcaffe as part of "everything" if MATLAB_DIR is specified.
ifneq ($(MATLAB_DIR),)
	EVERYTHING_TARGETS += mat$(PROJECT)
endif

##############################
# Define build targets
##############################
.PHONY: all lib tools clean docs $(DIST_ALIASES) proto superclean \
	supercleanlist supercleanfiles warn everything

all: proto lib tools

proto: $(PROTO_GEN_JAVA)

$(PROTO_GEN_JAVA): $(PROTO_SRC) | $(PROTO_BUILD_DIR)
	@ echo PROTOC $<
	$(Q)protoc --proto_path=$(PROTO_SRC_DIR) --java_out=$(PROTO_BUILD_DIR) $<


$(JNI_GEN_HEADER_DIR)/$(JNI_GEN_HEADER_PREFIX)%.h : $(JNI_JCAFFE_DIR)/%.class \
	| $(JNI_GEN_HEADER_DIR)
	javah -d $(JNI_GEN_HEADER_DIR) -force -cp $(JNI_CLASS_DIR) \
	$(JNI_CLASS_NAME_PREFIX).$*

lib: $(JNI_GEN_HEADER_FILES) $(STATIC_NAME) $(DYNAMIC_NAME)

tools: $(TOOL_BINS) $(TOOL_BIN_LINKS)

warn: $(EMPTY_WARN_REPORT)

$(EMPTY_WARN_REPORT): $(ALL_WARNS) | $(BUILD_DIR)
	@ cat $(ALL_WARNS) > $@
	@ if [ -s "$@" ]; then \
		cat $@; \
		mv $@ $(NONEMPTY_WARN_REPORT); \
		echo "Compiler produced one or more warnings."; \
		exit 1; \
	  fi; \
	  $(RM) $(NONEMPTY_WARN_REPORT); \
	  echo "No compiler warnings!";

$(ALL_WARNS): %.o.$(WARNS_EXT) : %.o

$(BUILD_DIR_LINK): $(BUILD_DIR)/.linked

# Create a target ".linked" in this BUILD_DIR to tell Make that the "build" link
# is currently correct, then delete the one in the OTHER_BUILD_DIR in case it
# exists and $(DEBUG) is toggled later.
$(BUILD_DIR)/.linked:
	@ mkdir -p $(BUILD_DIR)
	@ $(RM) $(OTHER_BUILD_DIR)/.linked
	@ $(RM) -r $(BUILD_DIR_LINK)
	@ ln -s $(BUILD_DIR) $(BUILD_DIR_LINK)
	@ touch $@

$(ALL_BUILD_DIRS): | $(BUILD_DIR_LINK)
	@ mkdir -p $@

$(DYNAMIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo LD -o $@ $(OBJS)
	$(Q)$(CXX) -shared -o $@ $(OBJS) $(LINKFLAGS) $(LDFLAGS) $(DYNAMIC_FLAGS)

$(STATIC_NAME): $(OBJS) | $(LIB_BUILD_DIR)
	@ echo AR -o $@
	$(Q)ar rcs $@ $(OBJS)

$(BUILD_DIR)/%.o: %.cpp | $(ALL_BUILD_DIRS)
	@ echo CXX $<
	$(Q)$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o | $(DYNAMIC_NAME)
	@ echo CXX/LD -o $@
	$(Q)$(CXX) $< -o $@ $(LINKFLAGS) -l$(LIBRARY_NAME) $(LDFLAGS) \
		-Wl,-rpath,$(ORIGIN)/../lib
clean:
	mvn clean
	@- $(RM) -rf $(ALL_BUILD_DIRS)
	@- $(RM) -rf $(OTHER_BUILD_DIR)
	@- $(RM) -rf $(BUILD_DIR_LINK)
	@- $(RM) -rf $(DISTRIBUTE_DIR)

supercleanfiles:
	$(eval SUPERCLEAN_FILES := $(strip \
			$(foreach ext,$(SUPERCLEAN_EXTS), $(shell find . -name '*$(ext)' \
			-not -path './data/*'))))

supercleanlist: supercleanfiles
	@ \
	if [ -z "$(SUPERCLEAN_FILES)" ]; then \
		echo "No generated files found."; \
	else \
		echo $(SUPERCLEAN_FILES) | tr ' ' '\n'; \
	fi

superclean: clean supercleanfiles
	@ \
	if [ -z "$(SUPERCLEAN_FILES)" ]; then \
		echo "No generated files found."; \
	else \
		echo "Deleting the following generated files:"; \
		echo $(SUPERCLEAN_FILES) | tr ' ' '\n'; \
		$(RM) $(SUPERCLEAN_FILES); \
	fi

$(DIST_ALIASES): $(DISTRIBUTE_DIR)

$(DISTRIBUTE_DIR): all | $(DISTRIBUTE_SUBDIRS)
	# add include
	cp -r include $(DISTRIBUTE_DIR)/
	# add libraries
	cp $(STATIC_NAME) $(DISTRIBUTE_DIR)/lib
	cp $(DYNAMIC_NAME) $(DISTRIBUTE_DIR)/lib

-include $(DEPS)
