using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public class CrossValConfusionMatrix
    {
        private List<ConfusionMatrix> _list = new List<ConfusionMatrix>();
        private int _numClasses = 0;
        public void AddConfusionMatrix(ConfusionMatrix current)
        {
            if(!_list.Any())
            {
                _numClasses = current.NumberOfClasses;
            }

            _list.Add(current);
        }

        public List<List<double>> GetAveragedConfusionMatrix()
        {
            var totalList = new List<List<double>>();

            var allCounts = _list.Select(i => i.Counts);

            for (int i = 0; i < _numClasses; i++)
            {
                var current = new List<double>();
                for (int j = 0; j < _numClasses; j++)
                {
                    current.Add(_list.Select(x => x.GetCountForClassPair(j, i)).Average());
                }
                totalList.Add(current);
            }
            
            return totalList;
        }

        public double GetPrecision(int classIdx)
        {
            return _list.Select(i => i.PerClassPrecision[classIdx]).Average();
        }
        public double GetRecall(int classIdx)
        {
            return _list.Select(i => i.PerClassRecall[classIdx]).Average();
        }

        private IEnumerable<string> GetColumns(int classes)
        {
            if (!_list.Any())
                return Enumerable.Empty<string>();

            // hacky but convenient way to get class labels...
            var columns = _list.First().GetFormattedConfusionTable();
            var all = columns.Split(new string[] {" ","|","\r","\n","Confusion table", "PREDICTED","=" }, StringSplitOptions.RemoveEmptyEntries);
            return all.AsEnumerable().Take(classes);
        }

        public string GetPrettyConfusionMatrix()
        {
            var classes = GetColumns(_numClasses);

            
            string prefix = "";
            int numLabels = _numClasses;

            int colWidth = 10;//numLabels == 2 ? 8 : 5;
            //int maxNameLen = confusionMatrix.PredictedClassesIndicators.Max(name => name.Length);
            // If the names are too long to fit in the column header, we back off to using class indices
            // in the header. This will also require putting the indices in the row, but it's better than
            // the alternative of having ambiguous abbreviated column headers, or having a table potentially
            // too wide to fit in a console.
            bool useNumbersInHeader = false;//maxNameLen > colWidth;

            //int rowLabelLen = maxNameLen;
            int rowLabelLen = 15;
            int rowDigitLen = 0;

            // The "PREDICTED" in the table, at length 9, dictates the amount of additional padding that will
            // be necessary on account of label names.
            int paddingLen = 15;//Math.Max(9, rowLabelLen);
            string pad = new string(' ',2);//, paddingLen - 9);
            string rowLabelFormat = null;
            if (useNumbersInHeader)
            {
                int namePadLen = paddingLen - (rowDigitLen + 2);
                rowLabelFormat = string.Format("{{0,{0}}}. {{1,{1}}} ||", rowDigitLen, namePadLen);
            }
            else
                rowLabelFormat = string.Format("{{1,{0}}} ||", paddingLen);

            var confusionTable = GetAveragedConfusionMatrix();//confusionMatrix.Counts;
            var sb = new StringBuilder();
            //if (numLabels == 2)
            //{
            //    //var positiveCaps = confusionMatrix.PredictedClassesIndicators[0].ToString().ToUpper();

            //    var numTruePos = confusionTable[0][0];
            //    var numFalseNeg = confusionTable[0][1];
            //    var numTrueNeg = confusionTable[1][1];
            //    var numFalsePos = confusionTable[1][0];
            //    sb.AppendFormat("{0}TEST {1} RATIO:\t{2:N4} ({3:F1}/({3:F1}+{4:F1}))", prefix, "XXX",//positiveCaps,
            //        1.0 * (numTruePos + numFalseNeg) / (numTruePos + numTrueNeg + numFalseNeg + numFalsePos),
            //        numTruePos + numFalseNeg, numFalsePos + numTrueNeg);
            //}

            sb.AppendLine();
            sb.AppendFormat("{0}Averaged Confusion table", prefix);
            //if (confusionMatrix.IsSampled)
            //    sb.AppendLine(" (sampled)");
            //else
                sb.AppendLine();

            sb.AppendFormat("          {0}||", pad);
            for (int i = 0; i < numLabels; i++)
                sb.Append("===================");
            sb.AppendLine();
            sb.AppendFormat("PREDICTED {0}||", pad);
            string format = string.Format(" {{{0},{1}}} |", useNumbersInHeader ? 0 : 1, colWidth);
            for (int i = 0; i < numLabels; i++)
                sb.AppendFormat(format, i, classes.ElementAt(i));
            sb.AppendLine(" Recall");
            sb.AppendFormat("TRUTH     {0}||", pad);
            for (int i = 0; i < numLabels; i++)
                sb.Append("===================");

            sb.AppendLine();

            string format2 = string.Format(" {{0,{0}:{1}}} |", colWidth,
                string.IsNullOrWhiteSpace(prefix) ? "N0" : "F1");
            for (int i = 0; i < numLabels; i++)
            {
                sb.AppendFormat(rowLabelFormat, i, classes.ElementAt(i));//confusionMatrix.PredictedClassesIndicators[i]);
                for (int j = 0; j < numLabels; j++)
                    sb.AppendFormat(format2, confusionTable[i][j]);
                sb.AppendFormat(" {0,5:F4}", GetRecall(i));
                sb.AppendLine();
            }
            sb.AppendFormat("          {0}||", pad);
            for (int i = 0; i < numLabels; i++)
                sb.Append("===================");
            sb.AppendLine();
            sb.AppendFormat("Precision {0}||", pad);
            format = string.Format("{{0,{0}:N4}} |", colWidth + 1);
            for (int i = 0; i < numLabels; i++)
                sb.AppendFormat(format, GetPrecision(i));

            sb.AppendLine();
            return sb.ToString();

        }

    }
}
