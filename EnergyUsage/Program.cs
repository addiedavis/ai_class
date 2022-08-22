using System;

namespace EnergyUsage
{
    class Program
    {
        static void Main(string[] args)
        {
            var lab = new AnomalyLab();
            lab.LoadData(true);
            lab.SetupAndTrainModel();
            lab.EvaluateModel();
            lab.Predict();

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.Read();
        }
    }
}
