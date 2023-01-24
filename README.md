## Surfnet with mediapipe

This repository is a fork from [MediaPipe Github repository](https://github.com/google/mediapipe).

This is a work in progress repository, the final repository will be different.

### Installation

Follow instructions from [MediaPipe Github repository](https://github.com/google/mediapipe)

Be careful to update correctly the setup opencv script
```libdc1394-22-dev ``` to ```libdc1394-dev```

#### Android

Add to `.bashrc`
```
export ANDROID_HOME=$HOME/Android/Sdk
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/21.4.7075529
```

You need the android version 30, and the build tools 30.0.3 (nothing above). Installing them via Android Studio is the simplest.

```sh
source ~/.bashrc
sudo apt-get install openjdk-11-jdk
```

### Build

Based on [Basel](https://bazel.build/), the first build will be long as it builds the whole mediapipe framework, subsequent builds will be mostly cached.

```sh
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/surfnet/desktop:surfnet
bazel build -c opt --config=android_arm64 --define MEDIAPIPE_PROFILING=1 --linkopt="-s" mediapipe/surfnet/android/src/java/com/google/mediapipe/apps/surfnetmobile:surfnetmobile
```
### Run (desktop)

```mkdir /tmp/mediapipe```

```sh
export GLOG_logtostderr=1
./bazel-bin/mediapipe/surfnet/desktop/surfnet --calculator_graph_config_file=mediapipe/surfnet/graphs/surfnet.pbtxt
``` 

```sh
export GLOG_logtostderr=1
./bazel-bin/mediapipe/surfnet/desktop/surfnet --calculator_graph_config_file=mediapipe/surfnet/graphs/surfnet.pbtxt
``` 

./bazel-bin/mediapipe/surfnet/desktop/surfnettrack --calculator_graph_config_file=mediapipe/surfnet/graphs/tracking/surfnet_track_gpu.pbtxt --input_video_path=/media/charles/DATA/Programs/surfnet/vids/test_30sfullhd.mp4 --output_video_path=/media/charles/DATA/test.mp4


### Build AAR

```sh
bazel build -c opt --strip=ALWAYS     --host_crosstool_top=@bazel_tools//tools/cpp:toolchain     --fat_apk_cpu=arm64-v8a,x86_64     --legacy_whole_archive=0     --features=-legacy_whole_archive     --copt=-fvisibility=hidden     --copt=-ffunction-sections     --copt=-fdata-sections     --copt=-fstack-protector     --copt=-Oz     --copt=-fomit-frame-pointer     --copt=-DABSL_MIN_LOG_LEVEL=2     --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --linkopt=-Wl,--gc-sections,--strip-all     mediapipe/surfnet/android/src/java/com/google/mediapipe/apps/surfnetaar:surfnettrack
```

Building the graph
```sh
bazel build -c opt --strip=ALWAYS     --host_crosstool_top=@bazel_tools//tools/cpp:toolchain     --fat_apk_cpu=arm64-v8a,x86_64     --legacy_whole_archive=0     --features=-legacy_whole_archive     --copt=-fvisibility=hidden     --copt=-ffunction-sections     --copt=-fdata-sections     --copt=-fstack-protector     --copt=-Oz     --copt=-fomit-frame-pointer     --copt=-DABSL_MIN_LOG_LEVEL=2     --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --linkopt=-Wl,--gc-sections,--strip-all mediapipe/surfnet/graphs/tracking:surfnet_track_graph
```

### Run (Android studio)

copy the graph (.binarypb), model (.tflite) and labelmap (.txt) to the asset folder.
Make sure you have a compatible smartphone (need Android 10, OpenGL 3.1 minimum)
The android emulator doesn't work for now