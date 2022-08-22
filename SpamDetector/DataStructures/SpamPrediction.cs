using Common;
using Microsoft.ML.Data;

namespace SpamDetector.DataStructures
{
    class SpamPrediction
    {
        [ColumnName("PredictedLabel")]
        public string isSpam { get; set; }
    }

    class SpamPredictionFull : ScoredResult
    {
        
        [ColumnName("Message")]
        public string Message { get; set; }
        [ColumnName("PredictedLabel")]
        public string isSpam { get; set; }
    }
}
