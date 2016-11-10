//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "DistributionTestCommon.h"

using namespace std::placeholders;

// TODO: Move to other file.
void TrainSimpleDistributedFeedForwardClassifer(const DeviceDescriptor& device, DistributedTrainerPtr distributedTrainer, size_t rank, std::vector<double>* trainCE, size_t ouputFreqMBInMinibatches, size_t minibatchSize)
{
    const size_t inputDim = 2;
    const size_t numOutputClasses = 2;
    const size_t hiddenLayerDim = 50;
    const size_t numHiddenLayers = 2;

    const size_t numSamplesPerSweep = 4000;
    const size_t numSweepsToTrainWith = 1;
    const size_t numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";
    auto minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } }, MinibatchSource::FullDataSweep, false);
    auto featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);

    std::unordered_map<StreamInformation, std::pair<NDArrayViewPtr, NDArrayViewPtr>> inputMeansAndInvStdDevs = { { featureStreamInfo, { nullptr, nullptr } } };
    ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, inputMeansAndInvStdDevs);

    auto nonLinearity = std::bind(Sigmoid, _1, L"Sigmoid");
    auto input = InputVariable({ inputDim }, DataType::Float, L"features");
    auto normalizedinput = PerDimMeanVarianceNormalize(input, inputMeansAndInvStdDevs[featureStreamInfo].first, inputMeansAndInvStdDevs[featureStreamInfo].second);
    auto classifierOutput = FullyConnectedDNNLayer(normalizedinput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedInput") );
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierOutput = FullyConnectedDNNLayer(classifierOutput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedHidden"));

    auto outputTimesParam = Parameter({ numOutputClasses, hiddenLayerDim }, DataType::Float, UniformInitializer(CNTK::DefaultParamInitScale, 1), device, L"outputTimesParam");
    auto outputBiasParam = Parameter({ numOutputClasses }, DataType::Float, UniformInitializer(CNTK::DefaultParamInitScale, 1), device, L"outputBiasParam");
    classifierOutput = Plus(outputBiasParam, Times(outputTimesParam, classifierOutput), L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    // Test save and reload of model
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;
        auto combinedNet = Combine({ trainingLoss, prediction, classifierOutput }, L"feedForwardClassifier");
        SaveAndReloadModel<float>(combinedNet, { &input, &labels, &trainingLossVar, &predictionVar, &classifierOutputVar }, device, rank);

        classifierOutput = classifierOutputVar;
        trainingLoss = trainingLossVar;
        prediction = predictionVar;
    }

    double learningRatePerSample = 0.02;
    size_t warmStartSamples = distributedTrainer ? distributedTrainer->GetDistributedAfterSampleCount() : MinibatchSource::InfiniteSamples;
    minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } }, MinibatchSource::InfinitelyRepeat, false, warmStartSamples);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { SGDLearner(classifierOutput->Parameters(), learningRatePerSample) }, distributedTrainer);
    size_t trainingCheckpointFrequency = 100;
    if (trainCE) trainCE->clear();
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        trainer.TrainMinibatch({ { input, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
        PrintTrainingProgress(trainer, i, ouputFreqMBInMinibatches);

        if ((i % ouputFreqMBInMinibatches) == 0 && trainCE)
        {
            trainCE->push_back(trainer.PreviousMinibatchLossAverage());
        }

        if ((i % trainingCheckpointFrequency) == (trainingCheckpointFrequency - 1))
        {
            const wchar_t* ckpName = L"feedForward.net";
            trainer.SaveCheckpoint(ckpName);
            trainer.RestoreFromCheckpoint(ckpName);
        }
    }
}