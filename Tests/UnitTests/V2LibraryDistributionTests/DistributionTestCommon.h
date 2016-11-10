//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

void TrainSimpleDistributedFeedForwardClassifer(const DeviceDescriptor& device, DistributedTrainerPtr distributedTrainer, size_t rank, std::vector<double>* trainCE = nullptr, size_t ouputFreqMBInMinibatches = 20, size_t minibatchSize = 25);

void TestMinibatchSourceWarmStart(size_t numMBs, size_t minibatchSize, size_t warmStartSamples);