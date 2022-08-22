using System;

namespace ProductRecommendation
{
    class Program
    {
        static void Main(string[] args)
        {
            var lab = new ProductRecommendLab();
            lab.LoadData(true);
            lab.SetupAndTrainModel();
            lab.EvaluateModel();
            lab.Predict();

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.Read();
        }
    }
}
