using Accord.Statistics.Analysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliticnoOgrodje_v_1_0
{
    public class cEvaluator
    {
        //const double _failAucMax = 0.5;
        //const double _poorAucMax = 0.7;
        //const double _fairAucMax = 0.8;
        //const double _goodAucMax = 0.9;
        //const double _excellentAucMax = 1;



        //double _algEffectiveness;
        public List<cResultsEvaluation> List { get => _list; set => _list = value; }
        List<cResultsEvaluation> _list;

        public cEvaluator()
        {
            _list = new List<cResultsEvaluation>();
        }
        public void Evaluate(string pAlgName, double pAoc, GeneralConfusionMatrix pGcm, TimeSpan pTime, int[] pExpected, int[] pActual)
        {
            if(pExpected == null || pActual == null)
            {
                return;
            }
            cResultsEvaluation lRE = new cResultsEvaluation(pAlgName, pAoc, pGcm, pTime);
            lRE.Actual = pActual;
            lRE.Expected = pExpected;
            _list.Add(lRE);
        }
        public void Print()
        {
            if(_list != null)
            {
                foreach (cResultsEvaluation lRes in _list)
                {
                    Console.WriteLine("Name: " + lRes.AlgName + " ___ AOC: " + lRes.Aoc.ToString("#.##") + " ___ Accuracy: " + lRes.Accuracy.ToString("#.##") + " ___ Precision: " + lRes.Precision.ToString("#.##") + " ___ Execution Time: " + lRes.ExecutionTime);
                }
            }      
        }
        public static class cMeasure
        {
            public static TimeSpan MeasuresTime(Action pFunc)
            {
                DateTime lStart = DateTime.Now;

                pFunc();

                DateTime lEnd = DateTime.Now;

                return lEnd - lStart;
            }
        }
        public class cResultsEvaluation
        {
            private string _AlgName;
            private double _aoc;
            private int[,] _gcm;


            public string AlgName { get => _AlgName; }
            public double Aoc { get => _aoc; }
            public int[,] Gcm { get => _gcm; }
            public double Specificity { get => _specificity; }
            public double FalsePositiveRate { get => _falsePositiveRate; }
            public double Sensitivity { get => _sensitivity; }
            public double FalseNegativeRate { get => _falseNegativeRate; }
            public double Precision { get => _precision; }
            public double Accuracy { get => _accuracy; }
            public TimeSpan ExecutionTime { get => _executionTime; }
            public int[] Expected;
            public int[] Actual;


            double _specificity;
            double _falsePositiveRate;
            double _sensitivity;
            double _falseNegativeRate;
            double _precision;
            double _accuracy;
            TimeSpan _executionTime;
            public cResultsEvaluation(string pAlgName, double pAoc, GeneralConfusionMatrix pGcm, TimeSpan pTime)
            {
                _AlgName = pAlgName;
                _aoc = pAoc;
                _gcm = pGcm.Matrix;
                _accuracy = pGcm.Accuracy;
                if (_gcm != null)
                {
                    _specificity = Convert.ToDouble(pGcm.Matrix[1, 1]) / (Convert.ToDouble(pGcm.Matrix[1, 1]) + Convert.ToDouble(pGcm.Matrix[0, 1]));
                    _falsePositiveRate = 1 - Specificity;
                    _sensitivity = Convert.ToDouble(pGcm.Matrix[0, 0]) / (Convert.ToDouble(pGcm.Matrix[0, 0]) + Convert.ToDouble(pGcm.Matrix[1, 0]));
                    _falseNegativeRate = 1 - Sensitivity;
                    _precision = Convert.ToDouble(pGcm.Matrix[0, 0]) / (Convert.ToDouble(pGcm.Matrix[0, 0]) + Convert.ToDouble(pGcm.Matrix[0, 1]));
                    _executionTime = pTime;
                }
            }
        }
    }
}
