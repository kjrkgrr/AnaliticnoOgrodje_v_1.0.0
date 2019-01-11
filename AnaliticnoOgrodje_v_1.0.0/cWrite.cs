using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliticnoOgrodje_v_1_0
{
    public class cWrite
    {
        public static void WriteDataTableToCsvFile(DataTable dataTable, string filePath)
        {
            StringBuilder fileContent = new StringBuilder();

            foreach (var col in dataTable.Columns)
            {
                fileContent.Append(col.ToString() + ",");
            }

            fileContent.Replace(",", System.Environment.NewLine, fileContent.Length - 1, 1);

            foreach (DataRow dr in dataTable.Rows)
            {
                foreach (var column in dr.ItemArray)
                {
                    double lval;
                    if (double.TryParse(column.ToString(), out lval))
                    {
                        fileContent.Append("\"" + lval.ToString("N16") + "\",");
                    }
                    else
                    {
                        fileContent.Append("\"" + column.ToString() + "\",");
                    }
                    
                }

                fileContent.Replace(",", System.Environment.NewLine, fileContent.Length - 1, 1);
            }

            System.IO.File.WriteAllText(filePath, fileContent.ToString());
        }

        internal static void WriteToCsvFile(IEnumerable<object> pItems, string pPath)
        {

            var list = pItems.ToList();
            int i = 0;
            using (StreamWriter lW = new StreamWriter(pPath))
            {
                foreach (object o in pItems)
                {
                    string lColumns = "";
                    string lLine = "";

                    System.Reflection.PropertyInfo[] lProperties = o.GetType().GetProperties();
                    foreach (System.Reflection.PropertyInfo linfo in lProperties)
                    {
                        string lPropertyName = linfo.Name.ToString();
                        if (i == 0)
                        {
                            lColumns += lPropertyName + ",";
                        }
                        string lVal = o.GetType().GetProperty(lPropertyName).GetValue(o, null).ToString();
                        if (lVal.Contains(','))
                        {
                            lVal = lVal.Replace(',', '.');
                        }
                        lLine += lVal + ",";

                    }
                   
                    if (i == 0)
                    {
                        lColumns = lColumns.Trim();
                        if (lColumns[lColumns.Length - 1] == ',')
                        {
                            lColumns = lColumns.Remove(lColumns.Length - 1);
                        }
                        lW.WriteLine(lColumns);
                    }
                    lLine = lLine.Trim();
                    if (lLine[lLine.Length - 1] == ',')
                    {
                        lLine = lLine.Remove(lLine.Length - 1);
                    }
                    lW.WriteLine(lLine);

                    i++;
                }
            }         
        }
    }
}
