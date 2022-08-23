using System;
using Common;
using IrisClustering.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IrisClustering
{
    internal class IrisClustering : Lab<IrisData>
    {
        public override void Predict()
        {
            Console.WriteLine("=============== Predict a cluster for a single case (Single Iris data sample) ===============");

            // Test with one sample text 
            var sampleIrisData = new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            };

            // Create prediction engine related to the loaded trained model
            var predEngine = MlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(TrainedModel);

            //Score
            var resultprediction = predEngine.Predict(sampleIrisData);

            Console.WriteLine($"Cluster assigned for setosa flowers:" + resultprediction.SelectedClusterId);
        }

        public override void LoadData(bool hasHeaders)
        {
            Console.WriteLine("=============== Start of data load process ===============");

            FullDataView = MlContext.Data.LoadFromTextFile(path: FullDataPath,
                columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column(nameof(IrisData.SepalLength), DataKind.Single, 1),
                    new TextLoader.Column(nameof(IrisData.SepalWidth), DataKind.Single, 2),
                    new TextLoader.Column(nameof(IrisData.PetalLength), DataKind.Single, 3),
                    new TextLoader.Column(nameof(IrisData.PetalWidth), DataKind.Single, 4),
                },
                hasHeader: true,
                separatorChar: '\t');

            //Split dataset in two parts: TrainingDataset (80%) and TestDataset (20%)
            DataOperationsCatalog.TrainTestData trainTestData = MlContext.Data.TrainTestSplit(FullDataView, testFraction: 0.2);
            TrainingDataView = trainTestData.TrainSet;
            TestDataView = trainTestData.TestSet;
            Console.WriteLine("=============== End of data load process ===============");

        }

        public override void SetupAndTrainModel()
        {
            // (optional) peek data concatenated
            // Common.ConsoleHelper.PeekDataViewInConsole(MlContext, TrainingDataView, dataProcessPipeline, 10);
            // Common.ConsoleHelper.PeekVectorColumnDataInConsole(MlContext, "Features", TrainingDataView, dataProcessPipeline, 10);
            Console.WriteLine("=============== End of setup process ===============");
            Console.WriteLine("=============== Start of training process ===============");
            var dataProcessPipeline = MlContext.Transforms
                .Concatenate("Features", nameof(IrisData.SepalLength),
                nameof(IrisData.SepalWidth),
                nameof(IrisData.PetalLength),
                nameof(IrisData.PetalWidth)
                );
            
            Trainer = MlContext.Clustering.Trainers
                .KMeans(featureColumnName: "Features", numberOfClusters: 3);

            TrainingPipeLine = dataProcessPipeline.Append(Trainer);
            TrainedModel = TrainingPipeLine.Fit(TrainingDataView);
            // STEP 2: Create and train the model (KMeans Clustering on Features column)
        }

        public override void EvaluateModel()
        {
            // STEP3: Evaluate accuracy of the model
            // ConsoleHelper.PrintClusteringMetrics(Trainer.ToString(), metrics);

            // STEP4: SAVE Model
            IDataView predictions = TrainedModel.Transform(TestDataView);
            var metrics = MlContext.Clustering
                .Evaluate(predictions, scoreColumnName: "Score", featureColumnName: "Features");

            ConsoleHelper.PrintClusteringMetrics(Trainer.ToString(), metrics);

            Console.WriteLine("=============== End of training process ===============");
        }
    }
}