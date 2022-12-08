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

You need the android version 33, and the build tools 30.0.3 (nothing above). Installing them via Android Studio is the simplest.

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

```mkdir /tmp/mediapipe```

```sh
export GLOG_logtostderr=1
./bazel-bin/mediapipe/surfnet/desktop/surfnet --calculator_graph_config_file=mediapipe/surfnet/graphs/surfnet.pbtxt
``` 

```
export GLOG_logtostderr=1
bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
mediapipe/examples/desktop/hello_world:hello_world
```

