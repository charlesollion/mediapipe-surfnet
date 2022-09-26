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
#include "mediapipe/calculators/tflite/tflite_yolo_tensors_to_dummy_calculator.pb.h"
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
  absl::Status DecodeTensor(const float* raw_tensor,
      std::vector<float>* boxes, std::vector<float>* scores, 
      std::vector<int>* classes);
  absl::Status ConvertToDetections(const float* detection_boxes,
                                   const float* detection_scores,
                                   const int* detection_classes,
                                   int num_boxes,
                                   std::vector<Detection>* output_detections);
  Detection ConvertToDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, float score, int class_id,
                               bool flip_vertically);

  ::mediapipe::TfLiteYoloTensorsToDummyCalculatorOptions options_;
  int max_detections_ = 100;
  int num_boxes_ = 25200;
  int num_coords_ = 85;
  float min_conf_thresh_ = 0.3;
  bool flip_vertically_ = false;

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

  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();
  const TfLiteTensor* raw_tensor = &input_tensors[0];
  // shape of raw tensor 1 25200 85
  // CHECK_EQ(raw_tensor->dims->size, 3);

  const float* float_data = raw_tensor->data.f;
  std::vector<float> boxes(max_detections_ * 4);
  std::vector<float> scores(max_detections_);
  std::vector<int> classes(max_detections_);

  MP_RETURN_IF_ERROR(DecodeTensor(float_data, &boxes, &scores, &classes));

  std::cerr<<"finished decoding nb boxes: " << boxes.size() / 4 << std::endl;
  // if(boxes.size() > 2) {
  //   std::cerr<< scores[0] << ", " << scores[1] << ", " << scores[2] << std::endl;
  //   std::cerr<< classes[0] << ", " << classes[1] << ", " << classes[2] << std::endl;
  // }
  std::vector<Detection> output_detections;

  MP_RETURN_IF_ERROR(ConvertToDetections(boxes.data(), scores.data(), classes.data(), scores.size(), &output_detections));

  std::cerr << "finished converting, found detections: " << output_detections.size() << std::endl;
  
  // Output
  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs()
        .Tag("DETECTIONS")
        .AddPacket(MakePacket<std::vector<Detection>>(output_detections)
          .At(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}

absl::Status TfLiteYoloTensorsToDummyCalculator::DecodeTensor(const float* raw_tensor,
    std::vector<float>* boxes, std::vector<float>* scores, 
    std::vector<int>* classes) {
    
  // yolo tensor output [1, 25200, 85]. The 85 comes from:
  // [x ,y ,w ,h , conf, class1, class2, ... ]
  
  int cur_idx = 0;
  for (int i = 0; i < num_boxes_; i++) {
    const int box_offset = i * num_coords_;

    // threshold confidences
    float confidence = raw_tensor[box_offset + 4];
    // (*confidences)[cur_idx] = confidence;

    // Find the top score for box i.
    int class_id = -1;
    float max_score = -std::numeric_limits<float>::max();
    for (int k = 5; k < num_coords_; ++k) {
      auto score = raw_tensor[box_offset + k];
      if (max_score < score) {
        max_score = score;
        class_id = k - 5;
      }
    }
    float final_score = max_score * confidence;
    if (final_score < min_conf_thresh_) {
      continue;
    }
    (*scores)[cur_idx] = final_score;
    (*classes)[cur_idx] = class_id;

    // get box
    float y_center = raw_tensor[box_offset];
    float x_center = raw_tensor[box_offset + 1];
    float h = raw_tensor[box_offset + 2];
    float w = raw_tensor[box_offset + 3];

    if (options_.reverse_output_order()) {
      x_center = raw_tensor[box_offset];
      y_center = raw_tensor[box_offset + 1];
      w = raw_tensor[box_offset + 2];
      h = raw_tensor[box_offset + 3];
    }

    // x_center =
    //     x_center / options_.x_scale() * anchors[i].w() + anchors[i].x_center();
    // y_center =
    //     y_center / options_.y_scale() * anchors[i].h() + anchors[i].y_center();

    // if (options_.apply_exponential_on_box_size()) {
    //   h = std::exp(h / options_.h_scale()) * anchors[i].h();
    //   w = std::exp(w / options_.w_scale()) * anchors[i].w();
    // } else {
    //   h = h / options_.h_scale() * anchors[i].h();
    //   w = w / options_.w_scale() * anchors[i].w();
    // }

    const float ymin = y_center - h / 2.f;
    const float xmin = x_center - w / 2.f;
    const float ymax = y_center + h / 2.f;
    const float xmax = x_center + w / 2.f;

    (*boxes)[cur_idx * 4 + 0] = ymin;
    (*boxes)[cur_idx * 4 + 1] = xmin;
    (*boxes)[cur_idx * 4 + 2] = ymax;
    (*boxes)[cur_idx * 4 + 3] = xmax;

    cur_idx += 1;
    if(cur_idx >= max_detections_) {
      break;
    }
  }
  boxes->resize(cur_idx * 4);
  scores->resize(cur_idx);
  classes->resize(cur_idx);

  return absl::OkStatus();
}


absl::Status TfLiteYoloTensorsToDummyCalculator::ConvertToDetections(
    const float* detection_boxes, const float* detection_scores,
    const int* detection_classes, int num_boxes, std::vector<Detection>* output_detections) {
    for (int i = 0; i < num_boxes; ++i) {
    const int box_offset = i * 4;
    Detection detection = ConvertToDetection(
        detection_boxes[box_offset + 0], detection_boxes[box_offset + 1],
        detection_boxes[box_offset + 2], detection_boxes[box_offset + 3],
        detection_scores[i], detection_classes[i], flip_vertically_);
    const auto& bbox = detection.location_data().relative_bounding_box();
    if (bbox.width() < 0 || bbox.height() < 0) {
      // Decoded detection boxes could have negative values for width/height due
      // to model prediction. Filter out those boxes since some downstream
      // calculators may assume non-negative values. (b/171391719)
      continue;
    }

    output_detections->emplace_back(detection);
  }
  return absl::OkStatus();
}

Detection TfLiteYoloTensorsToDummyCalculator::ConvertToDetection(
    float box_ymin, float box_xmin, float box_ymax, float box_xmax, float score,
    int class_id, bool flip_vertically) {
  Detection detection;
  detection.add_score(score);
  detection.add_label_id(class_id);

  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(box_xmin);
  relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
  relative_bbox->set_width(box_xmax - box_xmin);
  relative_bbox->set_height(box_ymax - box_ymin);
  return detection;
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
  options_ =
      cc->Options<::mediapipe::TfLiteYoloTensorsToDummyCalculatorOptions>();
  
  num_coords_ = options_.num_coords();
  max_detections_ = options_.max_detections();
  num_boxes_ = options_.num_boxes();
  min_conf_thresh_ = options_.min_conf_thresh();
  flip_vertically_ = options_.flip_vertically();

  return absl::OkStatus();
}

}  // namespace mediapipe
