/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/text/text_embedder/text_embedder.h"

#include <memory>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe::tasks::text::text_embedder {
namespace {

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";

// Note that these models use dynamic-sized tensors.
// Embedding model with BERT preprocessing.
constexpr char kMobileBert[] = "mobilebert_embedding_with_metadata.tflite";
// Embedding model with regex preprocessing.
constexpr char kRegexOneEmbeddingModel[] =
    "regex_one_embedding_with_metadata.tflite";

// Tolerance for embedding vector coordinate values.
constexpr float kEpsilon = 1e-4;
// Tolerancy for cosine similarity evaluation.
constexpr double kSimilarityTolerancy = 1e-6;

using ::mediapipe::file::JoinPath;
using ::testing::HasSubstr;
using ::testing::Optional;

class EmbedderTest : public tflite_shims::testing::Test {};

TEST_F(EmbedderTest, FailsWithMissingModel) {
  auto text_embedder =
      TextEmbedder::Create(std::make_unique<TextEmbedderOptions>());
  ASSERT_EQ(text_embedder.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      text_embedder.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  ASSERT_THAT(text_embedder.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(EmbedderTest, SucceedsWithMobileBert) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileBert);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      TextEmbedderResult result0,
      text_embedder->Embed("it's a charming and often affecting journey"));
  ASSERT_EQ(result0.embeddings.size(), 1);
  ASSERT_EQ(result0.embeddings[0].float_embedding.size(), 512);
  ASSERT_NEAR(result0.embeddings[0].float_embedding[0], 19.9016f, kEpsilon);

  MP_ASSERT_OK_AND_ASSIGN(
      auto result1, text_embedder->Embed("what a great and fantastic trip"));
  ASSERT_EQ(result1.embeddings.size(), 1);
  ASSERT_EQ(result1.embeddings[0].float_embedding.size(), 512);
  ASSERT_NEAR(result1.embeddings[0].float_embedding[0], 22.626251f, kEpsilon);

  // Check cosine similarity.
  MP_ASSERT_OK_AND_ASSIGN(
      double similarity, TextEmbedder::CosineSimilarity(result0.embeddings[0],
                                                        result1.embeddings[0]));
  EXPECT_NEAR(similarity, 0.969514, kSimilarityTolerancy);

  MP_ASSERT_OK(text_embedder->Close());
}

TEST(EmbedTest, SucceedsWithRegexOneEmbeddingModel) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kRegexOneEmbeddingModel);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(
      auto result0,
      text_embedder->Embed("it's a charming and often affecting journey"));
  EXPECT_EQ(result0.embeddings.size(), 1);
  EXPECT_EQ(result0.embeddings[0].float_embedding.size(), 16);

  EXPECT_NEAR(result0.embeddings[0].float_embedding[0], 0.0309356f, kEpsilon);

  MP_ASSERT_OK_AND_ASSIGN(
      auto result1, text_embedder->Embed("what a great and fantastic trip"));
  EXPECT_EQ(result1.embeddings.size(), 1);
  EXPECT_EQ(result1.embeddings[0].float_embedding.size(), 16);

  EXPECT_NEAR(result1.embeddings[0].float_embedding[0], 0.0312863f, kEpsilon);

  // Check cosine similarity.
  MP_ASSERT_OK_AND_ASSIGN(
      double similarity, TextEmbedder::CosineSimilarity(result0.embeddings[0],
                                                        result1.embeddings[0]));
  EXPECT_NEAR(similarity, 0.999937, kSimilarityTolerancy);

  MP_ASSERT_OK(text_embedder->Close());
}

TEST_F(EmbedderTest, SucceedsWithQuantization) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileBert);
  options->embedder_options.quantize = true;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      TextEmbedderResult result,
      text_embedder->Embed("it's a charming and often affecting journey"));
  ASSERT_EQ(result.embeddings.size(), 1);
  ASSERT_EQ(result.embeddings[0].quantized_embedding.size(), 512);

  MP_ASSERT_OK(text_embedder->Close());
}

}  // namespace
}  // namespace mediapipe::tasks::text::text_embedder
