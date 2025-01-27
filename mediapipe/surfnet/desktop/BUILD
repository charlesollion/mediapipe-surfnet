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

package(default_visibility = ["//mediapipe/surfnet:__subpackages__"])

cc_binary(
    name = "example_surfnet",
    srcs = ["surfnet.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "run_graph_main",
    srcs = ["run_graph_main.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "run_graph_main_gpu",
    srcs = ["run_graph_main_gpu.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)


cc_binary(
    name = "surfnet",
    data = [
        "//mediapipe/surfnet/models:yolo_21062022-int8.tflite",
        "//mediapipe/surfnet/models:labelmap_yolo.txt",
    ],
    deps = [
        ":run_graph_main",
        "//mediapipe/surfnet/graphs:desktop_tflite_calculators",
    ],
)

cc_binary(
    name = "surfnettrack",
    data = [
        # "//mediapipe/surfnet/models:yolov5s-fp16-optimize.tflite",
        # "//mediapipe/surfnet/models:yolov5s-fp16-optimize.tflite",
        # "//mediapipe/surfnet/models:yolo_surfnet_fix.tflite",
        "//mediapipe/surfnet/models:yolo_pau-fp16.tflite",
        "//mediapipe/surfnet/models:labelmap_yolo.txt",
    ],
    deps = [
        # "//mediapipe/surfnet/graphs:desktop_tracking_tflite_calculators",
        # ":run_graph_main",
        ### uses mobile calculators for desktop GPU: does not work (error to be checked)
        ":run_graph_main_gpu",
        "//mediapipe/surfnet/graphs/tracking:mobile_calculators",
    ],
)