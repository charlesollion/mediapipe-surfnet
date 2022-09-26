// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
// #include "mediapipe/calculators/tflite/tflite_yolo_tensors_to_dummy_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tflite/config.h"
#include "tensorflow/lite/interpreter.h"

#if MEDIAPIPE_TFLITE_GL_INFERENCE
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if MEDIAPIPE_TFLITE_METAL_INFERENCE
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif  // MEDIAPIPE_TFLITE_METAL_INFERENCE

namespace {
constexpr int kNumInputTensorsWithAnchors = 3;
constexpr int kNumCoordsPerBox = 4;

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
}  // namespace

namespace mediapipe {

#if MEDIAPIPE_TFLITE_GL_INFERENCE
using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
using ::tflite::gpu::gl::GlShader;
typedef ::tflite::gpu::gl::GlProgram GpuProgram;
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
typedef id<MTLComputePipelineState> GpuProgram;
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

namespace {

#if MEDIAPIPE_TFLITE_GPU_SUPPORTED
struct GPUData {
  GpuProgram decode_program;
  GpuProgram score_program;
  GpuTensor decoded_boxes_buffer;
  GpuTensor raw_boxes_buffer;
  GpuTensor raw_anchors_buffer;
  GpuTensor scored_boxes_buffer;
  GpuTensor raw_scores_buffer;
};
#endif  // MEDIAPIPE_TFLITE_GPU_SUPPORTED

}  // namespace

// Convert result TFLite tensors from object detection models into MediaPipe
// Detections.
//
// Input:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32. The vector of
//               tensors can have 2 or 3 tensors. First tensor is the predicted
//               raw boxes/keypoints. The size of the values must be (num_boxes
//               * num_predicted_values). Second tensor is the score tensor. The
//               size of the valuse must be (num_boxes * num_classes). It's
//               optional to pass in a third tensor for anchors (e.g. for SSD
//               models) depend on the outputs of the detection model. The size
//               of anchor tensor must be (num_boxes * 4).
//  TENSORS_GPU - vector of GlBuffer of MTLBuffer.
// Output:
//  DETECTIONS - Result MediaPipe detections.
//
// Usage example:
// node {
//   calculator: "TfLiteTensorsToDetectionsCalculator"
//   input_stream: "TENSORS:tensors"
//   input_side_packet: "ANCHORS:anchors"
//   output_stream: "DETECTIONS:detections"
//   options: {
//     [mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.ext] {
//       num_classes: 91
//       num_boxes: 1917
//       num_coords: 4
//       ignore_classes: [0, 1, 2]
//       x_scale: 10.0
//       y_scale: 10.0
//       h_scale: 5.0
//       w_scale: 5.0
//     }
//   }
// }
class TfLiteYoloTensorsToDummyCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status LoadOptions(CalculatorContext* cc);

  // ::mediapipe::TfLiteYoloTensorsToDummyCalculatorOptions options_;

#if MEDIAPIPE_TFLITE_GL_INFERENCE
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<GPUData> gpu_data_;
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  MPPMetalHelper* gpu_helper_ = nullptr;
  std::unique_ptr<GPUData> gpu_data_;
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  bool gpu_input_ = false;
};
REGISTER_CALCULATOR(TfLiteYoloTensorsToDummyCalculator);

absl::Status TfLiteYoloTensorsToDummyCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  bool use_gpu = false;

  if (cc->Inputs().HasTag(kTensorsTag)) {
    cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
  }

  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
    cc->Inputs().Tag(kTensorsGpuTag).Set<std::vector<GpuTensor>>();
    use_gpu |= true;
  }

  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
  }

  if (use_gpu) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  }

  return absl::OkStatus();
}

absl::Status TfLiteYoloTensorsToDummyCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
    gpu_input_ = true;
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  }

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  return absl::OkStatus();
}

absl::Status TfLiteYoloTensorsToDummyCalculator::Process(
    CalculatorContext* cc) {
  if ((!gpu_input_ && cc->Inputs().Tag(kTensorsTag).IsEmpty()) ||
      (gpu_input_ && cc->Inputs().Tag(kTensorsGpuTag).IsEmpty())) {
    return absl::OkStatus();
  }

  std::vector<Detection> output_detections;

  Detection detection;
  detection.add_score(0.2);
  detection.add_label_id(1);

  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(0.1);
  relative_bbox->set_ymin(0.1);
  relative_bbox->set_width(0.5);
  relative_bbox->set_height(0.5);
  detection.add_label("bonjour_bonjour");
  output_detections.push_back(detection);
  
  // Output
  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs()
        .Tag("DETECTIONS")
        .AddPacket(MakePacket<std::vector<Detection>>(output_detections)
          .At(cc->InputTimestamp()));
  }
      

  return absl::OkStatus();
}

absl::Status TfLiteYoloTensorsToDummyCalculator::Close(CalculatorContext* cc) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  gpu_helper_.RunInGlContext([this] { gpu_data_.reset(); });
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  gpu_data_.reset();
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  return absl::OkStatus();
}

absl::Status TfLiteYoloTensorsToDummyCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.

  return absl::OkStatus();
}

}  // namespace mediapipe
