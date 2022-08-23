using System;
using System.Collections.Generic;
using System.Linq;
using IrisClassifier.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using Common;

namespace IrisClassifier
{
    internal class IrisClassifier : Lab<IrisData>
    {
        public override void Predict()
        {
            var predEngine = MlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(TrainedModel);

            // During prediction we will get Score column with 3 float values.
            // We need to find way to map each score to original label.
            // In order to do that we need to get TrainingLabelValues from Score column.
            // TrainingLabelValues on top of Score column represent original labels for i-th value in Score array.
            // Let's look how we can convert key value for PredictedLabel to original labels.
            // We need to read KeyValues for "PredictedLabel" column.
            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();

            // Since we apply MapValueToKey estimator with default parameters, key values
            // depends on order of occurence in data file. Which is "Iris-setosa", "Iris-versicolor", "Iris-virginica"
            // So if we have Score column equal to [0.2, 0.3, 0.5] that's mean what score for
            // Iris-setosa is 0.2
            // Iris-versicolor is 0.3
            // Iris-virginica is 0.5.
            //Add a dictionary to map the above float values to strings. 
            Dictionary<float, string> IrisFlowers = new Dictionary<float, string>();
            IrisFlowers.Add(0, "Setosa");
            IrisFlowers.Add(1, "versicolor");
            IrisFlowers.Add(2, "virginica");

            Console.WriteLine("=====Predicting using model====");
            //Score sample 1
            var resultprediction1 = predEngine.Predict(SampleIrisData.Iris1);

            Console.WriteLine($"Actual: setosa.     Predicted label and score:  {IrisFlowers[labelsArray[0]]}: {resultprediction1.Score[0]:0.####}");
            Console.WriteLine($"                                                {IrisFlowers[labelsArray[1]]}: {resultprediction1.Score[1]:0.####}");
            Console.WriteLine($"                                                {IrisFlowers[labelsArray[2]]}: {resultprediction1.Score[2]:0.####}");
            Console.WriteLine();

            //Score sample 2
            var resultprediction2 = predEngine.Predict(SampleIrisData.Iris2);

            Console.WriteLine($"Actual: Virginica.   Predicted label and score:  {IrisFlowers[labelsArray[0]]}: {resultprediction2.Score[0]:0.####}");
            Console.WriteLine($"                                                 {IrisFlowers[labelsArray[1]]}: {resultprediction2.Score[1]:0.####}");
            Console.WriteLine($"                                                 {IrisFlowers[labelsArray[2]]}: {resultprediction2.Score[2]:0.####}");
            Console.WriteLine();

            //Score sample 3
            var resultprediction3 = predEngine.Predict(SampleIrisData.Iris3);

            Console.WriteLine($"Actual: Versicolor.   Predicted label and score: {IrisFlowers[labelsArray[0]]}: {resultprediction3.Score[0]:0.####}");
            Console.WriteLine($"                                                 {IrisFlowers[labelsArray[1]]}: {resultprediction3.Score[1]:0.####}");
            Console.WriteLine($"                                                 {IrisFlowers[labelsArray[2]]}: {resultprediction3.Score[2]:0.####}");
            Console.WriteLine();
        }

        public override void SetupAndTrainModel()
        {
            // Setup data processing pipeline
            // 1: Map Key column to input label column
            // 2: Add features for the 4 measurements we have ( Sepal Width, Sepal Length, Petal Width, Petal Length )
            // 3: Cache (optional)
            var dataProcessPipeline = 
                MlContext.Transforms.Conversion
                    .MapValueToKey(outputColumnName: "KeyColumn", inputColumnName: nameof(IrisData.Label))
                .Append(MlContext.Transforms.Concatenate("Features",
                    nameof(IrisData.SepalLength),
                    nameof(IrisData.SepalWidth),
                    nameof(IrisData.PetalLength),
                    nameof(IrisData.PetalWidth)
                )
                .AppendCacheCheckpoint(MlContext));

            // Create Trainer
            // 1: SdcaMaxiumEntropy - preferred
            // 2: Map output column back to keycolumn
            // maximum entrophy 
            Trainer = MlContext.MulticlassClassification.Trainers
                .SdcaMaximumEntropy(labelColumnName: "KeyColumn", featureColumnName: "Features")
                .Append(MlContext.Transforms.Conversion
                    .MapKeyToValue(outputColumnName: nameof(IrisData.Label), inputColumnName: "KeyColumn"));

            // LBFGS Max Entropy 
            // Trainer = MlContext.MulticlassClassification.Trainers
            //     .LbfgsMaximumEntropy(labelColumnName: "KeyColumn", 
            //     featureColumnName: "Features", exampleWeightColumnName: null, 1, 2, 1, 5, true)
            //     .Append(MlContext.Transforms.Conversion
            //         .MapKeyToValue(outputColumnName: nameof(IrisData.Label), inputColumnName: "KeyColumn"));

            // Add trainer to pipeline
            var trainingPipeLine = dataProcessPipeline.Append(Trainer);


            // Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            TrainedModel = trainingPipeLine.Fit(TrainingDataView);
        }

        public override void EvaluateModel()
        {
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = TrainedModel.Transform(TestDataView);
            var metrics = MlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
            Common.ConsoleHelper.PrintMultiClassClassificationMetrics(Trainer.ToString(), metrics);

            MlContext.Model.Save(TrainedModel, TrainingDataView.Schema, "./model.zip");
            Console.WriteLine("The model is saved to {0}", Common.FilePath.GetAbsolutePath(typeof(Program), "models/model.zip"));
        }
    }
}