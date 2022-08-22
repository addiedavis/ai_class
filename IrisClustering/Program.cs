using System;
using Microsoft.ML;

namespace IrisClustering
{
    class Program
    {
        static void Main(string[] args)
        {
            var lab = new IrisClustering();
            lab.LoadData(true);
            lab.SetupAndTrainModel();
            lab.EvaluateModel();
            lab.Predict();



            Console.WriteLine("=============== End of process, hit any key to finish ===============");
        }
    }
}
