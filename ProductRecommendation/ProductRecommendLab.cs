using System;
using Common;
using ProductRecommendation.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace ProductRecommendation
{
    public class ProductRecommendLab : Lab<ProductEntry>
    {
        public override void LoadData(bool hasHeaders)
        {
            FullDataView = MlContext.Data.LoadFromTextFile(path: FullDataPath,
                columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column(name:nameof(ProductEntry.ProductID), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(0) }, keyCount: new KeyCount(262111)),
                    new TextLoader.Column(name:nameof(ProductEntry.CoPurchaseProductID), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(1) }, keyCount: new KeyCount(262111))
                },
                hasHeader: true,
                separatorChar: '\t');

            DataOperationsCatalog.TrainTestData trainTestData = MlContext.Data.TrainTestSplit(FullDataView, testFraction: 0.2);
            TrainingDataView = trainTestData.TrainSet;
            TestDataView = trainTestData.TestSet;
        }

        public override void Predict()
        {
            var predictionengine = MlContext.Model.CreatePredictionEngine<ProductEntry, Copurchase_prediction>(TrainedModel);
            var prediction = predictionengine.Predict(
                new ProductEntry()
                {
                    ProductID = 3,
                    CoPurchaseProductID = 63
                });

            Console.WriteLine("\n For ProductID = 3 and  CoPurchaseProductID = 63 the predicted score is " + Math.Round(prediction.Score, 1));
        }

        public override void SetupAndTrainModel()
        {
            //STEP 1: Your data is already encoded so all you need to do is specify options for MatrxiFactorizationTrainer with a few extra hyperparameters
            //        Alpha - 0.01, Lambda - 0.025
            
            //Step 2: Call the MatrixFactorization trainer by passing options.

            //STEP 3: Train the model fitting to the DataSet
        }
    }
}