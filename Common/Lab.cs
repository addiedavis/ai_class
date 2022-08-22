using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Common
{
    public abstract class Lab<T>
    {
        public ITransformer TrainedModel { get; set; }
        public IEstimator<ITransformer> Trainer { get; set; }
        public IEstimator<ITransformer> TrainingPipeLine { get; set; }

        protected IDataView TrainingDataView { get; set; }
        protected MLContext MlContext { get; set;  }
        protected IDataView TestDataView { get; set; }
        protected IDataView FullDataView { get; set; }

        public Lab()
        {
            MlContext = new MLContext(0);
        }

        public abstract void Predict();
        public abstract void SetupAndTrainModel();

        public virtual void EvaluateModel()
        {
            Console.WriteLine("Unlabeled data do not have evaluation statistics");
        }

        public virtual void LoadData(bool hasHeaders)
        {
            TrainingDataView = MlContext.Data.LoadFromTextFile<T>(TrainDataPath, hasHeader: hasHeaders);
            TestDataView = MlContext.Data.LoadFromTextFile<T>(TestDataPath, hasHeader: hasHeaders);
            FullDataView = MlContext.Data.LoadFromTextFile<T>(FullDataPath, hasHeader: hasHeaders);
        }

        public string TrainDataPath { get; } = Path.Combine(AppContext.BaseDirectory, "Data","train.txt");
        public string TestDataPath { get; } = Path.Combine(AppContext.BaseDirectory,"Data","test.txt");
        public string FullDataPath { get; } = Path.Combine(AppContext.BaseDirectory,"Data","full.txt");

    }
}