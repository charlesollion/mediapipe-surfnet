// Copyright 2023 The MediaPipe Authors.
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

#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/components/containers/sources/MPPClassificationResult.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskResult.h"

NS_ASSUME_NONNULL_BEGIN

/** Represents the classification results generated by `MPPTextClassifier`. **/
NS_SWIFT_NAME(TextClassifierResult)
@interface MPPTextClassifierResult : MPPTaskResult

/** The `MPPClassificationResult` instance containing one set of results per classifier head. **/
@property(nonatomic, readonly) MPPClassificationResult *classificationResult;

/**
 * Initializes a new `MPPTextClassifierResult` with the given `MPPClassificationResult` and
 * timestamp (in milliseconds).
 *
 * @param classificationResult The `MPPClassificationResult` instance containing one set of results
 * per classifier head.
 * @param timestampMs The timestamp for this result.
 *
 * @return An instance of `MPPTextClassifierResult` initialized with the given
 * `MPPClassificationResult` and timestamp (in milliseconds).
 */
- (instancetype)initWithClassificationResult:(MPPClassificationResult *)classificationResult
                                 timestampMs:(NSInteger)timestampMs;

@end

NS_ASSUME_NONNULL_END
