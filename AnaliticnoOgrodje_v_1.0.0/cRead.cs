using Accord.Math;
using Deedle;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliticnoOgrodje_v_1_0
{
    public class cRead
    {
        private string path;
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="pPath">Type string</param>
        public cRead(string pPath)
        {
            path = pPath;
        }
        /// <summary>
        /// Method reads .cvs file and load's it in DataFrame
        /// </summary>
        /// <param name="pHasHeaders">optional parameter, Type bool</param>
        /// <param name="pSchema">optional parameter, Type string array</param>
        /// <returns>Type Deedle.Frame</returns>
        public Frame<int, string> Read(bool pHasHeaders = true, string pSchema = "", string pSeparator=";")
        {
            try
            {
                return Frame.ReadCsv(path, hasHeaders: pHasHeaders, inferTypes: true, schema: pSchema, separators:pSeparator);
            }
            catch(Exception lEx)
            {
                return null;
            }
            
        }
        /// <summary>
        /// Method splits DataFrame to two parts - Train set and Test set. Parameter pTrainProportion defines proportion of Train set and Test set
        /// </summary>
        /// <param name="pFrame">Type Frame<int,string></param>
        /// <param name="pTrainProportion">Type double</param>
        /// <returns>Type Deedle.Frame array[2]</returns>
        public static Frame<int, string>[] SplitData(Frame<int, string> pFrame, double pTrainProportion)
        {
            try
            {
               
                int lTrainProportionMax = (int)(pFrame.RowCount * pTrainProportion);
                int[] lShuffledIndexes = pFrame.RowKeys.ToArray();
                lShuffledIndexes.Shuffle();

                int lTrainSetIndexMax = (int)(pFrame.RowCount * pTrainProportion);

                int[] trainIndexes = lShuffledIndexes.Where(i => i < lTrainSetIndexMax).ToArray();
                int[] testIndexes = lShuffledIndexes.Where(i => i > lTrainSetIndexMax).ToArray();

                return new Frame<int, string>[2] { pFrame.Where(x => trainIndexes.Contains(x.Key)), pFrame.Where(x => testIndexes.Contains(x.Key)) };
                //return new Frame<int, string>[2] { pFrame.Where(x => x.Key <= lTrainProportionMax), pFrame.Where(x => x.Key > lTrainProportionMax) };
            }
            catch (Exception lEx)
            {
                return null;
            }
        }
    }
}
