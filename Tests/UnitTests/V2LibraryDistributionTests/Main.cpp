//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "DistributionTestCommon.h"

bool Is1bitSGDAvailable()
{
    static bool is1bitSGDAvailable;
    static bool isInitialized = false;

    if (!isInitialized)
    {
        const char* p = getenv("TEST_1BIT_SGD");

        // Check the environment variable TEST_1BIT_SGD to decide whether to run on a CPU-only device.
        if (p != nullptr && 0 == strcmp(p, "0"))
        {
            is1bitSGDAvailable = false;
        }
        else
        {
            is1bitSGDAvailable = true;
        }
        isInitialized = true;
    }

    return is1bitSGDAvailable;
}

int main(int /*argc*/, char* /*argv*/[])
{

#if defined(_MSC_VER)
    // in case of asserts in debug mode, print the message into stderr and throw exception
    if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
        fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        return -1;
    }
#endif
    // make sure minibatch source works with distributed and no warm-start
    TestMinibatchSourceWarmStart(10, 64, 0);

    // make sure minibatch source works with full warm-start
    TestMinibatchSourceWarmStart(10, 64, 640);

    // make sure minibatch source works with non-zero warm-start in the middle
    TestMinibatchSourceWarmStart(10, 64, 64);
    TestMinibatchSourceWarmStart(10, 64, 100); // test unaligned warm-start wrt minibatch size

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

    {
        std::vector<double> CPUTrainCE;
        auto communicator = MPICommunicator();
        auto distributedTrainer = CreateDataParallelDistributedTrainer(communicator, false);
        TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank);

        if (IsGPUAvailable())
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank);
    }

    if (Is1bitSGDAvailable())
    {
        size_t ouputFreqMB = 20;
        size_t minibatchSize = 25;
        {
            size_t distributedAfterMB = 100;

            std::vector<double> nonDistCPUTrainCE;
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), nullptr, 0, &nonDistCPUTrainCE, ouputFreqMB, minibatchSize);

            std::vector<double> nonDistCPUTrainCE2;
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), nullptr, 0, &nonDistCPUTrainCE2, ouputFreqMB, minibatchSize);

            for (int i = 0; i < distributedAfterMB / ouputFreqMB; i++)
            {
                FloatingPointCompare(nonDistCPUTrainCE2[i], nonDistCPUTrainCE[i], "CPU training is not deterministic");
            }

            std::vector<double> CPUTrainCE;
            size_t distributedAfterSampleCount = distributedAfterMB * minibatchSize;
            auto communicator = QuantizedMPICommunicator(true, true, 1);
            auto distributedTrainer = CreateQuantizedDataParallelDistributedTrainer(communicator, false, distributedAfterSampleCount);
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank, &CPUTrainCE, ouputFreqMB, minibatchSize);
    
            for (int i = 0; i < distributedAfterMB / ouputFreqMB; i++)
            {
                FloatingPointCompare(CPUTrainCE[i], nonDistCPUTrainCE[i], "Warm start CE deviated from non-distributed");
            }

            if (IsGPUAvailable())
            {
                std::vector<double> nonDistGPUTrainCE;
                TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), nullptr, 0, &nonDistGPUTrainCE, ouputFreqMB, minibatchSize);

                std::vector<double> nonDistGPUTrainCE2;
                TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), nullptr, 0, &nonDistGPUTrainCE2, ouputFreqMB, minibatchSize);

                for (int i = 0; i < distributedAfterMB / ouputFreqMB; i++)
                {
                    FloatingPointCompare(nonDistGPUTrainCE2[i], nonDistGPUTrainCE[i], "GPU training is not deterministic");
                }

                std::vector<double> GPUTrainCE;
                TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank, &GPUTrainCE, ouputFreqMB, minibatchSize);

                for (int i = 0; i < distributedAfterMB / ouputFreqMB; i++)
                {
                    FloatingPointCompare(GPUTrainCE[i], nonDistGPUTrainCE[i], "Warm start CE deviated from non-distributed");
                }
            }
        }

        {
            auto communicator = MPICommunicator();
            auto distributedTrainer = CreateBlockMomentumDistributedTrainer(communicator, 1024);
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank);

            if (IsGPUAvailable())
                TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank);
        }
    }

    fprintf(stderr, "\nCNTKv2LibraryDistribution tests: Passed\n");
    fflush(stderr);

#if defined(_MSC_VER)
    _CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
#endif

    DistributedCommunicator::Finalize();
    return 0;
}