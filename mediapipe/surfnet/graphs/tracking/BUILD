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

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mobile_calculators",
    deps = [
        "//mediapipe/calculators/core:packet_resampler_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        # "//mediapipe/surfnet/graphs/tracking/subgraphs:yolo_gpu",
        # "//mediapipe/surfnet/graphs/tracking/subgraphs:object_detection_gpu",
        # "//mediapipe/surfnet/graphs/tracking/subgraphs:object_tracking_gpu",
        # "//mediapipe/surfnet/graphs/tracking/subgraphs:renderer_gpu",

        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/tflite:tflite_yolo_tensors_to_detections_calculator",

        "//mediapipe/calculators/video:box_tracker_calculator",
        "//mediapipe/calculators/video:flow_packager_calculator",
        "//mediapipe/calculators/video:motion_analysis_calculator",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",

        "//mediapipe/calculators/util:detection_unique_id_calculator",
        "//mediapipe/calculators/util:detections_to_timed_box_list_calculator",
        "//mediapipe/calculators/video:tracked_detection_manager_calculator",

        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_binary_graph(
    name = "mobile_gpu_binary_graph",
    graph = "object_detection_tracking_mobile_gpu.pbtxt",
    output_name = "mobile_gpu.binarypb",
    deps = [":mobile_calculators"],
)
