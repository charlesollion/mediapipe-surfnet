## Surfnet with mediapipe

This repository is a fork from [MediaPipe Github repository](https://github.com/google/mediapipe).

This is a work in progress repository, the final repository will be different.

### Installation

Follow instructions from [MediaPipe Github repository](https://github.com/google/mediapipe)

Be careful to update correctly the opencv
```ssh
third_party/opencv_linux.BUILD
```

### Build

Based on [Basel](https://bazel.build/), the first build will be long as it builds the whole mediapipe framework, subsequent builds will be mostly cached.

```sh
export GLOG_logtostderr=1
bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
mediapipe/examples/desktop/hello_world:hello_world
```
