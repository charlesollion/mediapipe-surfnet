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

        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/tflite:tflite_yolo_tensors_to_detections_calculator",

        "//mediapipe/calculators/video:box_tracker_calculator",
        "//mediapipe/calculators/video:flow_packager_calculator",
        "//mediapipe/calculators/video:flow_to_image_calculator",
        "//mediapipe/calculators/video:motion_analysis_calculator",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",

        "//mediapipe/calculators/util:detection_unique_id_calculator",
        "//mediapipe/calculators/util:detections_to_timed_box_list_calculator",
        "//mediapipe/calculators/video:tracked_detection_manager_calculator",

        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",

        # "//mediapipe/calculators/core:packet_inner_join_calculator",
        # "//mediapipe/calculators/core:previous_loopback_calculator",
        # "//mediapipe/calculators/core:sequence_shift_calculator",
        # "//mediapipe/calculators/video:tvl1_optical_flow_calculator",
    ],
)

mediapipe_binary_graph(
    name = "surfnet_track_graph",
    graph = "surfnet_track.pbtxt",
    output_name = "surfnet_track.binarypb",
    deps = [":mobile_calculators"],
)

mediapipe_binary_graph(
    name = "surfnet_track_graph_gpu4",
    graph = "surfnet_track_gpu4.pbtxt",
    output_name = "surfnet_track4.binarypb",
    deps = [":mobile_calculators"],
)

mediapipe_binary_graph(
    name = "surfnet_detection4",
    graph = "surfnet_detection4.pbtxt",
    output_name = "surfnet_detection4.binarypb",
    deps = [":mobile_calculators"],
)

mediapipe_binary_graph(
    name = "surfnet_detection_landscape",
    graph = "surfnet_detection_landscape.pbtxt",
    output_name = "surfnet_detection_landscape.binarypb",
    deps = [":mobile_calculators"],
)

mediapipe_binary_graph(
    name = "surfnet_detection_coco",
    graph = "surfnet_detection_coco.pbtxt",
    output_name = "surfnet_detection_coco.binarypb",
    deps = [":mobile_calculators"],
)