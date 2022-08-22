using System;

namespace SpamDetector
{
    class Program
    {
        static void Main(string[] args)
        {
            var lab = new SpamLab();
            lab.LoadData(false);
            lab.SetupAndTrainModel();
            lab.EvaluateModel();
            lab.Predict();

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.Read();
        }
    }
}
