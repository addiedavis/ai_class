using System;
using System.Collections.Generic;
using System.Linq;
using Common;
using EnergyUsage.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace EnergyUsage
{
    internal class AnomalyLab : Lab<MeterData>
    {
        public override void LoadData(bool hasHeaders)
        {
            TrainingDataView = MlContext.Data.LoadFromTextFile<MeterData>(
                TrainDataPath,
                separatorChar: ',',
                hasHeader: true);
            TestDataView = MlContext.Data.LoadFromTextFile<MeterData>(
                TestDataPath,
                separatorChar: ',',
                hasHeader: true);
            FullDataView = MlContext.Data.LoadFromTextFile<MeterData>(
                FullDataPath,
                separatorChar: ',',
                hasHeader: true);
        }

        public override void Predict()
        {
            var transformedData = TrainedModel.Transform(FullDataView);

            // Getting the data of the newly created column as an IEnumerable
            IEnumerable<SpikePrediction> predictions =
                MlContext.Data.CreateEnumerable<SpikePrediction>(transformedData, false);

            // This algorythm needs standardized time segments 
            var colCDN = FullDataView.GetColumn<float>("ConsumptionDiffNormalized").ToArray();
            var colTime = FullDataView.GetColumn<DateTime>("time").ToArray();

            // Output the input data and predictions
            Console.WriteLine("======Displaying anomalies in the Power meter data=========");
            Console.WriteLine("Date              \tReadingDiff\tAlert\tScore\tP-Value");

            int i = 0;
            foreach (var p in predictions)
            {
                if (p.Prediction[0] == 1)
                {
                    Console.BackgroundColor = ConsoleColor.DarkYellow;
                    Console.ForegroundColor = ConsoleColor.Black;
                }
                Console.WriteLine("{0}\t{1:0.0000}\t{2:0.00}\t{3:0.00}\t{4:0.00}",
                    colTime[i], colCDN[i],
                    p.Prediction[0], p.Prediction[1], p.Prediction[2]);
                Console.ResetColor();
                i++;
            }
        }

        public override void SetupAndTrainModel()
        {
            // Configure th Estimator
            const int PValueSize = 30;
            const int SeasonalitySize = 30;
            const int TraingSize = 90;
            const double ConfidenceInterval = 98;

            string outputColumnName = nameof(SpikePrediction.Prediction);
            string inputColumnName = nameof(MeterData.ConsumptionDiffNormalized);

            // Train Pipeline with DetectSpikeBySsa
            var trainer = MlContext.Transforms.DetectSpikeBySsa(
                outputColumnName,
                inputColumnName,
                confidence: ConfidenceInterval, // Level of confidence in a spike detection
                pvalueHistoryLength: PValueSize, // Number of previous measurments to consider
                trainingWindowSize: TraingSize, // Subset of data to train with 
                seasonalityWindowSize: SeasonalitySize // Size of the largest season / shift
            );

            TrainedModel = trainer.Fit(FullDataView);
        }

        public SsaSpikeDetector TrainedModel { get; set; }

        public override void EvaluateModel()
        {
            // unsupervised
        }
    }
}