using Microsoft.ML.Data;

namespace EnergyUsage.DataStructures
{
    class SpikePrediction
    {
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
