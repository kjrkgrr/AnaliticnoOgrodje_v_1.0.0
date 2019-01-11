using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliticnoOgrodje_v_1_0
{
    public class cStatisticTests
    {
        private double[] list;
        private double variance;
        private double mean;

        public double Variance { get => variance; }
        public int lenght()
        {
            if (list != null)
            {
                return list.Length;
            }
            return 0;
        }
        public cStatisticTests(double[] pList)
        {
            list = pList;
            variance = Varianca();
            mean = Mean();
        }
        public double Min()
        {
            return list.Min();
        }
        public double Max()
        {
            return list.Max();
        }
        public double Mean()
        {
            return list.Sum() / list.Length;
        }
        private double Varianca()
        {
            try
            {
                double lVar = 0;
                for (int i = 0; i < list.Length; i++)
                {
                    lVar += Math.Pow(list[i], 2);
                }
                return (lVar - list.Length * Math.Pow(Mean(), 2)) / (list.Length - 1);
                //return (lVar / (list.Length - 1)) - Math.Pow(Mean(), 2);
            }
            catch (Exception lEx)
            {
                return double.NaN;
            }
        }
        public static double Covarianca(cStatisticTests pTest_1, cStatisticTests pTest_2)
        {
            if (pTest_1.list.Length != pTest_2.list.Length) return double.NaN;
            double lSumMul = 0;
            for (int i = 0; i <= pTest_1.list.Length - 1; i++)
            {
                lSumMul += (pTest_1.list[i] * pTest_2.list[i]);
            }
            return (lSumMul - (pTest_1.list.Length * pTest_1.Mean() * pTest_2.Mean())) / (pTest_1.list.Length - 1);
        }
        public static double Covarianca2(cStatisticTests pTest_1, cStatisticTests pTest_2)
        {
            if (pTest_1.list.Length != pTest_2.list.Length) return double.NaN;
            double lSumMul = 0;
            double lMeanX = pTest_1.Mean();
            double lMeanY = pTest_2.Mean();
            for (int i = 0; i <= pTest_1.list.Length - 1; i++)
            {

                lSumMul += (pTest_1.list[i] - lMeanX) * (pTest_2.list[i] - lMeanY);
            }
            return lSumMul / (pTest_1.list.Length - 1);
        }
        /// <summary>
        /// measure of the linear correlation between two variables X and Y
        /// The closer r is to 0 the weaker the relationship and the closer to +1 or -1 the stronger the relationship 
        /// https://onlinecourses.science.psu.edu/stat200/book/export/html/237
        /// </summary>
        /// <param name="pTest_1"></param>
        /// <param name="pTest_2"></param>
        /// <returns></returns>
        public static double PearsonCorrelation(cStatisticTests pTest_1, cStatisticTests pTest_2)
        {
            return Covarianca(pTest_1, pTest_2) / (Math.Sqrt(pTest_1.Varianca()) * Math.Sqrt(pTest_2.Varianca()));
        }
        /// <summary>
        /// Coefficient of determination
        /// It is a statistic used in the context of statistical models whose main purpose 
        /// is either the prediction of future outcomes or the testing of hypotheses, 
        /// on the basis of other related information. It provides a measure of how well observed outcomes are replicated by the model, 
        /// based on the proportion of total variation of outcomes explained by the mode
        /// </summary>
        /// <param name="pTest_1"></param>
        /// <param name="pTest_2"></param>
        /// <returns></returns>
        public static double CoefficientOfdetermination(cStatisticTests pTest_1, cStatisticTests pTest_2)
        {
            return Math.Pow(PearsonCorrelation(pTest_1, pTest_2), 2);
        }
        //public static double LeastSquares(cStatisticTests pTest_1, cStatisticTests pTest_2)
        //{
        //    return cStatisticTests.Covarianca(pTest_1, pTest_2) / pTest_1.Varianca();
        //}
        public static double b(cStatisticTests pTest_1, cStatisticTests pTest_2)
        {
            return Covarianca(pTest_1, pTest_2) / pTest_1.variance;
        }
        /// <summary>
        /// http://mathworld.wolfram.com/LeastSquaresFitting.html
        /// </summary>
        /// <param name="pTest_1"></param>
        /// <param name="pTest_2"></param>
        /// <returns></returns>
        public static double a(cStatisticTests pTest_1, cStatisticTests pTest_2)
        {
            return pTest_2.Mean() - (b(pTest_1, pTest_2) * pTest_1.Mean());
        }
        /// <summary>
        /// https://onlinecourses.science.psu.edu/stat200/book/export/html/237
        /// https://www.codeproject.com/KB/cs/csstatistics.aspx
        /// </summary>
        /// <returns></returns>
        public static double[,] CorrelationMatrix()
        {
            return null;
        }
        public static double tValue(cStatisticTests pTest_1, cStatisticTests pTest_2)
        {
            double lPersons = Math.Abs(PearsonCorrelation(pTest_1, pTest_2));
            return (lPersons * Math.Sqrt(pTest_1.list.Length - 2)) / Math.Sqrt(1 - Math.Pow(lPersons, 2));
        }
        public static double tValue(double pPerson, int pLenght)
        {
            pPerson = /*Math.Abs(pPerson);*/ pPerson;
            return pPerson * Math.Sqrt(pLenght - 2) / Math.Sqrt(1 - Math.Pow(pPerson, 2));
        }
        public static double tValue2(cStatisticTests pTest_1, cStatisticTests pTest_2)
        {
            double lPersons = Math.Abs(PearsonCorrelation(pTest_1, pTest_2));
            return lPersons / Math.Sqrt(1 - Math.Pow(lPersons, 2) / (pTest_1.list.Length - 2));
        }
        public static double tValue2(double pPerson, int pLenght)
        {
            pPerson = Math.Abs(pPerson);
            return pPerson / Math.Sqrt((1 - Math.Pow(pPerson, 2)) / (pLenght - 2));
        }
        public static double pValue(double pPerson, double pT, int pLen)
        {
            double a = Math.Sqrt((1 - pPerson * pPerson) / (pLen - 2));
            return ((pT * -1) * a) + pPerson;
        }
        public static double TCall(double pVal)
        {
            //string lValTemp;
            double lValTemp;
            if (pVal > 0)
            {
                //lValTemp = "" + (pVal + 0.0000005).ToString();
                lValTemp = pVal + 0.0000005;
            }
            else
            {
                //lValTemp = "" + (pVal + 0.0000005).ToString();
                lValTemp = pVal - 0.0000005;
            }
            return lValTemp;
            //return lValTemp.Substring(0, lValTemp.IndexOf('.') + 7);
        }
        public static double Buzz(double pT, int pDF)
        {
            var t = Math.Abs(pT);
            var rt = t / Math.Sqrt(pDF);
            var fk = Math.Atan(rt);
            if (pDF == 1)
            {
                return (1 - fk) / (Math.PI / 2);
            }
            var ek = Math.Sin(fk);
            var dk = Math.Cos(fk);
            if (pDF % 2 == 1)
            {
                return 1 - (fk + ek * dk * Zip(dk * dk, 2, pDF - 3, -1)) / (Math.PI / 2);
            }
            else
            {
                return (1 - ek * Zip(dk * dk, 2, pDF - 3, -1));
            }
        }
        public static double Zip(double q, int i, int j, int b)
        {
            double zz = 1;
            double z = zz;
            double k = i;
            while (k <= j)
            {
                zz = (zz * q * k) / (k - b);
                z = z + zz;
                k = k + 2;
            }
            return z;
        }
    }
    //public class cProb
    //{

    //}
}


