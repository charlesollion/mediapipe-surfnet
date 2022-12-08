# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//visibility:private"])


# do not change the name
# you need to change the gpu calculators vs standard calculator, or tracking
cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        # "//mediapipe/surfnet/graphs:mobile_calculators",
        #"//mediapipe/surfnet/graphs:mobile_gpu_calculators",
        "//mediapipe/surfnet/graphs/tracking:mobile_calculators",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)
# do not change the name
cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)

android_binary(
    name = "surfnetmobile",
    #srcs = glob(["*.java"]),
    srcs = ["MainActivity.java"],
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    assets = [
        "//mediapipe/surfnet/graphs:mobile_surfnet_cpu.binarypb",
        "//mediapipe/surfnet/models:yolov5s-fp16.tflite",
        #"//mediapipe/surfnet/models:yolov7-tiny-optim.tflite",
        "//mediapipe/surfnet/models:labelmap_yolo.txt",
    ],
    assets_dir = "",
    manifest_values = {
        "applicationId": "com.google.mediapipe.apps.surfnetmobile",
        "appName": "Surfnet",
        "mainActivity": ".MainActivity",
        "cameraFacingFront": "False",
        "binaryGraphName": "mobile_surfnet_cpu.binarypb",
        "inputVideoStreamName": "input_video",
        "outputVideoStreamName": "output_video",
        "flipFramesVertically": "True",
        "converterNumBuffers": "6",
    },
    multidex = "native",
    deps = [
        "//mediapipe/java/com/google/mediapipe/components:android_camerax_helper",
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "//third_party:androidx_appcompat",
        "//third_party:androidx_constraint_layout",
        "//third_party:opencv",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:com_google_guava_guava",
        ":mediapipe_jni_lib",
#        ":basic_lib",
    ],
)


android_binary(
    name = "surfnetmobilesimple",
    #srcs = glob(["*.java"]),
    srcs = ["MainActivity.java"],
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    assets = [
        "//mediapipe/surfnet/graphs:mobile_surfnet_cpu_simplified.binarypb",
        "//mediapipe/surfnet/models:yolov5s-fp16.tflite",
        "//mediapipe/surfnet/models:labelmap_yolo.txt",
    ],
    assets_dir = "",
    manifest_values = {
        "applicationId": "com.google.mediapipe.apps.surfnetmobile",
        "appName": "Surfnet",
        "mainActivity": ".MainActivity",
        "cameraFacingFront": "False",
        "binaryGraphName": "mobile_surfnet_cpu_simplified.binarypb",
        "inputVideoStreamName": "input_video",
        "outputVideoStreamName": "output_video",
        "flipFramesVertically": "True",
        "converterNumBuffers": "6",
    },
    multidex = "native",
    deps = [
        "//mediapipe/java/com/google/mediapipe/components:android_camerax_helper",
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "//third_party:androidx_appcompat",
        "//third_party:androidx_constraint_layout",
        "//third_party:opencv",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:com_google_guava_guava",
        ":mediapipe_jni_lib",
    ],
)

android_binary(
    name = "surfnetmobilegpu",
    # srcs = glob(["*.java"]),
    srcs = ["MainActivity.java"],
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    assets = [
        "//mediapipe/surfnet/graphs:mobile_surfnet_gpu.binarypb",
        #"//mediapipe/surfnet/models:yolov5s-fp16.tflite",
        "//mediapipe/surfnet/models:yolov5s-fp16-optimize.tflite",
        "//mediapipe/surfnet/models:labelmap_yolo.txt",
    ],
    assets_dir = "",
    manifest_values = {
        "applicationId": "com.google.mediapipe.apps.surfnetmobile",
        "appName": "Surfnet",
        "mainActivity": ".MainActivity",
        "cameraFacingFront": "False",
        "binaryGraphName": "mobile_surfnet_gpu.binarypb",
        "inputVideoStreamName": "input_video",
        "outputVideoStreamName": "output_video",
        "flipFramesVertically": "True",
        "converterNumBuffers": "4",
    },
    multidex = "native",
    deps = [
        "//mediapipe/java/com/google/mediapipe/components:android_camerax_helper",
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "//third_party:androidx_appcompat",
        "//third_party:androidx_constraint_layout",
        "//third_party:opencv",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:com_google_guava_guava",
        ":mediapipe_jni_lib",
    ],
)

android_binary(
    name = "surfnettrackgpu",
    srcs = ["MainActivity.java"],
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    assets = [
        "//mediapipe/surfnet/graphs/tracking:mobile_gpu.binarypb",
        "//mediapipe/surfnet/models:yolov5s-fp16-optimize.tflite",
        "//mediapipe/surfnet/models:labelmap_yolo.txt",
        # "//mediapipe/models:ssdlite_object_detection.tflite",
        # "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    assets_dir = "",
    manifest_values = {
        "applicationId": "com.google.mediapipe.apps.surfnetmobile",
        "appName": "Surfnet",
        "mainActivity": ".MainActivity",
        "cameraFacingFront": "False",
        "binaryGraphName": "mobile_gpu.binarypb",
        "inputVideoStreamName": "input_video",
        "outputVideoStreamName": "output_video",
        "flipFramesVertically": "True",
        "converterNumBuffers": "2",
    },
    multidex = "native",
    deps = [
        "//mediapipe/java/com/google/mediapipe/components:android_camerax_helper",
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "//third_party:androidx_appcompat",
        "//third_party:androidx_constraint_layout",
        "//third_party:opencv",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:com_google_guava_guava",
        ":mediapipe_jni_lib",
    ],
)