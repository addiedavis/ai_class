using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public class ScoredResult
    {
        // Microsoft.ML.Data.VBuffer`1[System.Single
        [ColumnName("Score")]
        [KeyType(2)]
        public VBuffer<System.Single> Score { get; set; }
        [ColumnName("Label")]
        public VBuffer<float> Label { get; set; }
    }
}
