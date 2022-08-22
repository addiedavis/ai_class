using System;
using System.Diagnostics;
using System.IO;

namespace IrisClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            var lab = new IrisClassifier();
            lab.LoadData(true);
            lab.SetupAndTrainModel();
            lab.EvaluateModel();
            lab.Predict();
            
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.Read();
        }
    }
}
