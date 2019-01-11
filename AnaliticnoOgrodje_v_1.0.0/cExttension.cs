using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliticnoOgrodje_v_1_0
{
    public static class MyExtensions
    {
        public static T[] GetColumn<T>(T[,] pMatrix, int pColNum)
        {
            return Enumerable.Range(0, pMatrix.GetLength(0))
                    .Select(x => pMatrix[x, pColNum])
                    .ToArray();
        }

        public static T[] GetRow<T>(T[,] pMatrix, int pRowNum)
        {
            return Enumerable.Range(0, pMatrix.GetLength(1))
                    .Select(x => pMatrix[pRowNum, x])
                    .ToArray();
        }

        public static T[] GetColumn<T>(this T[][] pJaggedArray, int pWanted_col)
        {
            int lRowLen = pJaggedArray[0].Length;
            int lColLen = pJaggedArray.Length;

            if (lRowLen <= pWanted_col)
            {
                throw new IndexOutOfRangeException();
            }

            T[] rowArray = new T[lColLen];
            for (int i = 0; i < lColLen; i++)
            {
                rowArray[i] = pJaggedArray[i][pWanted_col];
            }
            return rowArray;
        }

        public static T[] GetRow<T>(this T[][] pJaggedArray, int pWanted_row)
        {
            int lRowLen = pJaggedArray[0].Length;    
            int lColLen = pJaggedArray.Length;

            if(lColLen <= pWanted_row)
            {
                throw new IndexOutOfRangeException();
            }

            T[] rowArray = new T[lRowLen];
            for (int i = 0; i < lRowLen; i++)
            {
                rowArray[i] = pJaggedArray[pWanted_row][i];
            }
            return rowArray;
        }
        public static IEnumerable<T> Replace<T>(this IEnumerable<T> pItems, Predicate<T> pCondition, Func<T, T> pRepace)
        {
            return pItems.Select(item => pCondition(item) ? pRepace(item) : item);
        }
    }
}
