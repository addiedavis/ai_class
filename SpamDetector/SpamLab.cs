using System;
using System.Collections.Generic;
using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using SpamDetector.DataStructures;

namespace SpamDetector
{
    internal class SpamLab : Lab<SpamInput>
    {
        public override void Predict()
        {
            var predictor = MlContext.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(TrainedModel);

            Console.WriteLine("=============== Predictions for below data===============");
            // Test a few examples
            ClassifyMessage(predictor, "That's a great idea. It should work.");
            ClassifyMessage(predictor, "free medicine winner! congratulations");
            ClassifyMessage(predictor, "Yes we should meet over the weekend!");
            ClassifyMessage(predictor, "you win pills and free entry vouchers");
            ClassifyMessage(predictor, "O frabjous day! Callooh! Callay!'");
        }

        public override void SetupAndTrainModel()
        {
            // Create the estimator which converts the text label to boolean, featurizes the text, and adds a linear trainer.
            // Data process configuration with pipeline data transformations 
            // 1: Map Input column to value ( MapValueToKey )
            // 2: Featurize Text using a TexturizingEstimator (WordBags, n-Grams)
            // 3: Copy FeaturesText to Features
            // 4: Cache ( optional )
            EstimatorChain<ColumnCopyingTransformer> dataProcessPipeline =
             MlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(MlContext.Transforms.Text.FeaturizeText
                    ("FeaturesText", new TextFeaturizingEstimator.Options
                    {
                        WordFeatureExtractor = new WordBagEstimator.Options
                        { NgramLength = 2, UseAllLengths = true },

                        CharFeatureExtractor = new WordBagEstimator.Options
                        { NgramLength = 3, UseAllLengths = false },

                        Norm = TextFeaturizingEstimator.NormFunction.L2,
                    },
                    "Message"))
                .Append(MlContext.Transforms.CopyColumns("Features", "FeaturesText"))
                .AppendCacheCheckpoint(MlContext); // Create Transformer pipeline
            
            // ConsoleHelper.PeekDataViewInConsole(MlContext,TrainingDataView, dataProcessPipeline, 4);

            // Set the training algorithm 
            // 1: Train the model using OvA ( OneVersusAll )
            // 2: Map output label back to Key (PredictedLabel)
            var trainer = MlContext.BinaryClassification.Trainers
                .AveragedPerceptron(
                    labelColumnName: "Label",
                    numberOfIterations: 10,
                    featureColumnName: "Features"
                );

            Trainer = MlContext.MulticlassClassification.Trainers
                .OneVersusAll(trainer, labelColumnName: "Label")
                .Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
           
            TrainingPipeLine = dataProcessPipeline.Append(Trainer);

            TrainedModel = TrainingPipeLine.Fit(FullDataView);

            MlContext.Model.Save(TrainedModel, TrainingDataView.Schema, "./model.zip");

        }

        public override void EvaluateModel()
        {
            // Evaluate the model using cross-validation.
            // Cross-validation splits our dataset into 'folds', trains a model on some folds and 
            // evaluates it on the remaining fold. We are using 5 folds so we get back 5 sets of scores.
            // Let's compute the average AUC, which should be between 0.5 and 1 (higher is better).
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = MlContext.MulticlassClassification.CrossValidate(data: FullDataView, estimator: TrainingPipeLine, numberOfFolds: 3);
            ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(Trainer.ToString(), crossValidationResults);
        }

        private void ClassifyMessage(PredictionEngine<SpamInput, SpamPrediction> predictor, string message)
        {
            var input = new SpamInput { Message = message };
            var prediction = predictor.Predict(input);

            Console.WriteLine("The message '{0}' is {1}", input.Message, prediction.isSpam == "spam" ? "spam" : "not spam");
        }
    }
}