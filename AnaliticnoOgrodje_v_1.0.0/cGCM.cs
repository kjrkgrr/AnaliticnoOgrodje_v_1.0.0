using Accord.Statistics.Analysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliticnoOgrodje_v_1_0
{
    public class cGCM
    {
        GeneralConfusionMatrix _matrix;
        private double _specificity;
        private double _falsePositiveRate;
        private double _sensitivity;
        private double _falseNegativeRate;
        private double _precision;

        public cGCM(GeneralConfusionMatrix pMatrix)
        {
            _matrix = pMatrix;
            _specificity = Convert.ToDouble(_matrix.Matrix[1, 1]) / (Convert.ToDouble(_matrix.Matrix[1, 1]) + Convert.ToDouble(_matrix.Matrix[0, 1]));
            _falsePositiveRate = 1 - _specificity;
            _sensitivity = Convert.ToDouble(_matrix.Matrix[0, 0]) / (Convert.ToDouble(_matrix.Matrix[0, 0]) + Convert.ToDouble(_matrix.Matrix[1, 0]));
            _falseNegativeRate = 1 - _sensitivity;
            _precision = Convert.ToDouble(_matrix.Matrix[0, 0]) / (Convert.ToDouble(_matrix.Matrix[0, 0]) + Convert.ToDouble(_matrix.Matrix[0, 1]));
        }

        public GeneralConfusionMatrix Matrix { get => _matrix; }
        public double Specificity { get => _specificity; }
        public double FalsePositiveRate { get => _falsePositiveRate; }
        public double Sensitivity { get => _sensitivity; }
        public double FalseNegativeRate { get => _falseNegativeRate; }
        public double Precision { get => _precision; }
    }
}
