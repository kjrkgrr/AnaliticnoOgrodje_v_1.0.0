using Accord.Math;
using Deedle;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliticnoOgrodje_v_1_0
{
    public class cConverter
    {
        /// <summary>
        /// Method converts one column of Frame to array of Type T (generic)
        /// </summary>
        /// <param name="pFrame">Type Frame<int,string></param>
        /// <param name="pColName">Type string</param>
        /// <returns>Type T array[]</returns>
        public static T[] ConvertOneColumnToArray<T>(Frame<int, string> pFrame, string pColName)
        {
            try
            {
                return pFrame.GetColumn<T>(pColName).Values.ToArray<T>();
            }
            catch (Exception lEx)
            {
                throw new System.Exception(lEx.Message + " Type not supported (Date, Time, etc...)");
            }

        }
        /// <summary>
        /// Method converts Frame to multidimensional array of Type T (generic)
        /// </summary>
        /// <param name="pFrame">Type Frame<int,string></param>
        /// <returns>Type T multidimensional array[][]</returns>
        public static T[][] ConvertFrameToArray<T>(Frame<int, string> pFrame)
        {
            var lDT = pFrame.ToDataTable(new string[1]);
            return lDT.ToJagged<T>(lDT.Columns.Cast<DataColumn>().Select(x => x.ColumnName).ToArray());
        }
        /// <summary>
        /// Method takes all names of parameteres (features) from Frame and returns string array
        /// </summary>
        /// <param name="pFrame">Type Frame<int,string></param>>
        /// <returns>Type string array[]</returns>
        public static string[] TakeFeaturesFromIFrame(Frame<int, string> pFrame)
        {
            string[] lFeatures = new string[pFrame.ColumnCount];

            for (int i = 0; i < pFrame.ColumnCount; i++)
            {
                lFeatures[i] = pFrame.Columns.GetKeyAt(i);
            }
            return lFeatures;
        }
        /// <summary>
        /// Method builds Jagged Array (Matrix) from double[,] array
        /// </summary>
        /// <param name="pArr2D">Type array double[,]</param>
        /// <param name="pRowCount">Type int, count of all rows</param>
        /// <param name="pColumnCount">Type int, count of all cols</param>
        /// <returns></returns>
        private static double[][] BuildJaggedArray(double[,] pArr2D, int pRowCount, int pColumnCount)
        {
            double[][] ary = new double[pRowCount][];
            for (int i = 0; i < pRowCount; i++)
            {
                ary[i] = new double[pColumnCount];
                for (int j = 0; j < pColumnCount; j++)
                {
                    ary[i][j] = double.IsNaN(pArr2D[i, j]) ? 0.0 : pArr2D[i, j];
                }
            }
            return ary;
        }
      
        /// <summary>
        /// Wrapps method BuildJaggedArray
        /// </summary>
        /// <param name="pFrame">Type array double[,]</param>
        /// <param name="pFeatures">Type string array[]</param>
        /// <returns></returns>
        public static double[][] BuildJaggedArrayWrapper(Frame<int, string> pFrame, string[] pFeatures)
        {
            return cConverter.BuildJaggedArray(
              pFrame.Columns[pFeatures].ToArray2D<double>(),
              pFrame.RowCount,
              pFeatures.Length
              );
        }
        public static int[][] BuildConfusionMatrix(int[] pActual, int[] pPreds, int pNumClass)
        {
            int[][] lMatrix = new int[pNumClass][];
            for (int i = 0; i < pNumClass; i++)
            {
                lMatrix[i] = new int[pNumClass];
            }

            for (int i = 0; i < pActual.Length; i++)
            {
                lMatrix[pActual[i]][pPreds[i]] += 1;
            }

            return lMatrix;
        }
    }
}
