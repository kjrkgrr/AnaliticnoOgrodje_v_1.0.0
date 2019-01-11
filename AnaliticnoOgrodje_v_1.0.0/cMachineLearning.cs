using Accord.Controls;
using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.Boosting;
using Accord.MachineLearning.Boosting.Learners;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Math.Optimization;
using Accord.Math.Optimization.Losses;
using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Statistics.Analysis;
using Accord.Statistics.Distributions.Fitting;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Filters;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression.Linear;
using Deedle;
using System;

using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using ZedGraph;

using System.Windows.Forms.DataVisualization.Charting;

namespace AnaliticnoOgrodje_v_1_0
{
    public abstract class cMachineLearning
    {
        private double[][] _original_DataX;
        private double[][] _original_DataXY;
        private double[] _original_DataY;
        private Frame<int, string> _frame;
        private string[] _Features;
        private string _TargetParameter;
        private double _TrainProportion;

        /// <summary>
        /// Saves train data in multidimensional array of numbers (double)
        /// </summary>
        private double[][] _trainX;
        /// <summary>
        /// Saves Tragged train parameter, predicted parameter in array of double -> INPUT PARAMETERS
        /// </summary>
        private double[] _trainY;
        /// <summary>
        /// Saves train data in multidimensional array of numbers (double) -> TARGET PARAMETERES
        /// </summary>
        private double[][] _testX;
        /// <summary>
        /// Saves Tragged test parameter, predicted parameter in array of double
        /// </summary>
        private double[] _testY;
        /// <summary>
        /// In sample predictions
        /// </summary>
        private string[] _TargetList;
        private double[] _InSamplePreds;
        /// <summary>
        /// Out sample predictions
        /// </summary>
        private double[] _OutSamplePreds;

        private double[] _OutPreds_ForROC_double;
        private int[] _OutPreds_ForROC_int;

        private double[][] _trainXOriginal;
        private double[][] _testXOriginal;

        public double[] Get_InSamplePreds { get => _InSamplePreds; set => _InSamplePreds = value; }
        public double[] Get_OutSamplePreds { get => _OutSamplePreds; set => _OutSamplePreds = value; }
        public Frame<int, string> Frame { get => _frame; set => _frame = value; }
        public string[] Features { get => _Features; set => _Features = value; }
        public string TargetParameter { get => _TargetParameter; }
        public double TrainProportion { get => _TrainProportion;}
        public double[][] TrainX { get => _trainX;}
        public double[] TrainY { get => _trainY; }
        public double[][] TestX { get => _testX;}
        public double[] TestY { get => _testY; }
        public string[] TargetList { get => _TargetList; }

        public double[][] TrainXOriginal { get => _trainXOriginal; }
        public double[][] TestXOriginal { get => _testXOriginal; }

        public double[] OutPreds_ForROC_double { get => _OutPreds_ForROC_double; set => _OutPreds_ForROC_double = value; }
        public int[] OutPreds_ForROC_int { get => _OutPreds_ForROC_int; set => _OutPreds_ForROC_int = value; }
        public double[][] Original_DataX { get => _original_DataX;}
        public double[] Original_DataY { get => _original_DataY; }
        public double[][] Original_DataXY { get => _original_DataXY; }

        public cMachineLearning(string pPathToData, string[] pFeatures, string pTargetParameter, double pTrainProportion = .5,
            bool pHasHeaders = true, string[] pTargetList = null, Normalization pNorm = Normalization.ZScore, string pSeparator = ";")
        {
            if(_frame == null){
                cRead lRead = new cRead(pPathToData);
                _frame = lRead.Read(pHasHeaders, pSeparator:pSeparator);
            }
            if(_frame == null)
            {
                return;
            }
            Features = pFeatures;
            _TargetParameter = pTargetParameter;
            _TrainProportion = pTrainProportion;
            _TargetList = pTargetList;
            //PrepData() - dok bum mel podatke;
        }
        /// <summary>
        /// Initialize train and test sets
        /// </summary>
        public void Init(Normalization pNorm)
        {
            var lList = cRead.SplitData(Frame, TrainProportion);

            _trainXOriginal = cConverter.BuildJaggedArrayWrapper(lList[0], Features);

            _testXOriginal = cConverter.BuildJaggedArrayWrapper(lList[1], Features);

            _original_DataX = cConverter.BuildJaggedArrayWrapper(Frame , Features);
            _original_DataY = Frame[TargetParameter].Values.ToArray();
          

            var lArr = Features.ToList();
            lArr.Add(TargetParameter);

            _original_DataXY = cConverter.BuildJaggedArrayWrapper(Frame, lArr.ToArray());
            SetNormalization(pNorm);

            //if (_TargetList != null)
            //{
            //   TODO
            //}

            _trainY = lList[0][TargetParameter].Values.ToArray();
            _testY = lList[1][TargetParameter].Values.ToArray();
        }
        private void SetNormalization(Normalization pNorm)
        {
            switch (pNorm)
            {
                case Normalization.ZScore:
                    _trainX = Accord.Statistics.Tools.ZScores(_trainXOriginal);
                    _testX = Accord.Statistics.Tools.ZScores(_testXOriginal);
                    return;
                case Normalization.Center:
                    _trainX = Accord.Statistics.Tools.Center(_trainXOriginal);
                    _testX = Accord.Statistics.Tools.Center(_testXOriginal);
                    return;
                case Normalization.Standardize:
                    _trainX = Accord.Statistics.Tools.Standardize(_trainXOriginal);
                    _testX = Accord.Statistics.Tools.Standardize(_testXOriginal);
                    return;

            }
            /// double[][] scores = Accord.Statistics.Tools.ZScores(inputs);
            /// double[][] centered = Accord.Statistics.Tools.Center(inputs);
            /// double[][] standard = Accord.Statistics.Tools.Standardize(inputs);
        }
        public  enum Normalization
        {
            ZScore = 1, Center= 2, Standardize= 3

        };

    }

    public class cLearningMethods : cMachineLearning
    {
        public cLearningMethods(string pPathToData, string[] pFeatures, string pTargetParameter, double pTrainProportion = .5, bool pHasHeaders = true, string[] pTargetList = null, Normalization pNorm = Normalization.ZScore, string pSeparator = ";")
            : base(pPathToData, pFeatures, pTargetParameter, pTrainProportion, pHasHeaders = true, pTargetList = null,  pNorm = Normalization.ZScore, pSeparator)
        {
            if (this.Frame == null)
            {
                return;
            }
            if (pFeatures == null)
            {
                Features = this.Frame.ColumnKeys.Where(x => !x.Equals(TargetParameter)).ToArray();
            }
            this.Init(pNorm);
           
        }
       

        public ReceiverOperatingCharacteristic CalculateROC(int pPoints)
        {
            ReceiverOperatingCharacteristic lRoc = null;
            var lTask = Task.Run(() =>
            {
                lRoc = new ReceiverOperatingCharacteristic(TestY, Get_OutSamplePreds);
                lRoc.Compute(pPoints);
            });
            bool lCompletedSuccessfully = lTask.Wait(TimeSpan.FromMilliseconds(3000));
            if (lCompletedSuccessfully)
            {
                return lRoc;
            }
            else
            {
                return null;
            }
        }
        public ReceiverOperatingCharacteristic CalculateROC(int pPoints, bool[] pExpected, int[] pActual)
        {
            ReceiverOperatingCharacteristic lRoc = null;
            var lTask = Task.Run(() =>
            { 
                lRoc = new ReceiverOperatingCharacteristic(pExpected, pActual);
                lRoc.Compute(pPoints);
            });
            bool lCompletedSuccessfully = lTask.Wait(TimeSpan.FromMilliseconds(3000));
            if (lCompletedSuccessfully)
            {
                return lRoc;
            }
            else
            {
                return null;
            }
            
        }
        public void PrintRoc(string pMethodName, ReceiverOperatingCharacteristic pRoc)
        {
            if (pRoc == null)
            {
                Console.WriteLine("ROC " + pMethodName + " : Cannot be measured");
            }
            else
            {
                Console.WriteLine("ROC " + pMethodName + " : " + pRoc.Area);
            }
        }
        /// <summary>
        /// Validation method,  takes 3 parameters, name of algorithm, and sample of predictions
        /// </summary>
        /// <param name="pModelName">Type string - name of method</param>
        /// <param name="pRegInSamplePreds">Type double -> In sample predictions (private vars of class - they are filled in learning methods)</param>
        /// <param name="pRegOutSamplePreds">Type double -> Out sample predictions (private vars of class - they are filled in learning methods)</param>

        public void ValidateModelResult(string pModelName, double[] pRegInSamplePreds, double[] pRegOutSamplePreds)
        {
            double regInSampleRMSE = System.Math.Sqrt(new SquareLoss(TrainX).Loss(pRegInSamplePreds));
            // RMSE for out-sample 
            double regOutSampleRMSE = System.Math.Sqrt(new SquareLoss(TestX).Loss(pRegOutSamplePreds));

            Console.WriteLine("RMSE: {0:0.0000} (Train) vs. {1:0.0000} (Test)", regInSampleRMSE, regOutSampleRMSE);

            // R^2 for in-sample 
            double regInSampleR2 = new RSquaredLoss(TrainX[0].Length, TrainX).Loss(pRegInSamplePreds);
            // R^2 for out-sample 
            double regOutSampleR2 = new RSquaredLoss(TestX[0].Length, TestX).Loss(pRegOutSamplePreds);

            Console.WriteLine("R^2: {0:0.0000} (Train) vs. {1:0.0000} (Test)", regInSampleR2, regOutSampleR2);

            // Scatter Plot of expected and actual
            //ScatterplotBox.Show(
            //    String.Format("Actual vs. Prediction ({0})", pModelName), _testY, pRegOutSamplePreds
            //    );
        }
        public class cRegression : cAlgs
        {
            cLearningMethods _lm;
            public cRegression(cLearningMethods pLm)
            {
                _lm = pLm;
            }
          
            /// <summary>
            /// Logistic regression model
            /// </summary>
            /// <param name="pTolerance"></param>
            /// <param name="pIteration"></param>
            /// <param name="pRegularization"></param>
            public void GeneralizedLinearRegression(double pTolerance = 1e-5, int pIteration = 100, int pRegularization = 0)
            {
                var lLearner = new IterativeReweightedLeastSquares<LogisticRegression>()
                {
                    Regularization = pRegularization,
                    Tolerance = pTolerance,
                    MaxIterations = pIteration
                };
                LogisticRegression lRegression = lLearner.Learn(_lm.TrainX, _lm.TrainY);
                _lm.Get_InSamplePreds = lRegression.Score(_lm.TrainX);
                _lm.Get_OutSamplePreds = lRegression.Score(_lm.TestX);

                bool[] lOutPreds = lRegression.Decide(_lm.TestX);
                bool[] lInPreds = lRegression.Decide(_lm.TrainX);

                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds.Select(x => Convert.ToInt32(x)).ToArray());
                _lm.OutPreds_ForROC_int = lOutPreds.Select(x => Convert.ToInt32(x)).ToArray();
                //ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds.Select(x => Convert.ToInt32(x)).ToArray());
                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
                //DrawROC(_testY.Distinct().ToArray().Length, lRoc);
                //[0,0] TP
                //[0,1] FP
                //[1,0] FN
                //[1,1] TN
            }
            /// <summary>
            /// Multiple Linear Regression!!!
            /// Learning method - OLS learning algorithm 
            /// Treshold: npr 0.5 za 2 var, 0.3  za 3, 0.25 za 4 itn
            /// </summary>
            public void MultipleLinearRegression(bool pUseIntercept = true, bool pIsRobust = true, double pTreshold = 0.5, bool pIsClassification=false)
            {
                //double a = DistanceCorrelationCoificient();
                // OLS learning algorithm
                var lOsl = new OrdinaryLeastSquares()
                {
                    UseIntercept = pUseIntercept,
                    IsRobust = pIsRobust,
                };
                // Fit a linear regression model
                MultipleLinearRegression lRegressionFit = lOsl.Learn(_lm.TrainX, _lm.TrainY);

                //.Transform - making predictions for model
                _lm.Get_InSamplePreds = lRegressionFit.Transform(_lm.TrainX);
                _lm.Get_OutSamplePreds = lRegressionFit.Transform(_lm.TestX);

                ReceiverOperatingCharacteristic lRoc= null;
                if (pIsClassification)
                {
                    int[] lOutPreds = _lm.Get_OutSamplePreds.Select(x => x >= pTreshold ? 1 : 0).ToArray();
                    int[] lInPreds = _lm.Get_InSamplePreds.Select(x => x >= pTreshold ? 1 : 0).ToArray();

                    Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);

                    double lErrorIn = new SquareLoss(_lm.TrainY).Loss(_lm.Get_InSamplePreds);

                    double lErrorOut = new SquareLoss(_lm.TestY).Loss(_lm.Get_OutSamplePreds);

                    _lm.OutPreds_ForROC_int = lOutPreds.Select(x => Convert.ToInt32(x)).ToArray();

                    //ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                    lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds.Select(x => Convert.ToInt32(x)).ToArray());
                    
                }
                else
                {
                    double errorIn = new SquareLoss(_lm.TrainY).Loss(_lm.Get_InSamplePreds);
                    double errorOut = new SquareLoss(_lm.TestY).Loss(_lm.Get_OutSamplePreds);
                    lRoc = new ReceiverOperatingCharacteristic(_lm.Get_OutSamplePreds, _lm.TestY);
                  
                }
                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
            }
            /// <summary>
            /// The multivariate linear regression is a generalization of
            /// the multiple linear regression. In the multivariate linear
            /// regression, not only the input variables are multivariate,
            /// but also are the output dependent variables.
            /// Learning method - OLS learning algorithm 

            /// </summary>
            public void MultivariateLinearRegression(double[][] pOutputsTrain = null, double[][] pOutputsTest = null, bool pUseIntercept = true, bool pIsRobust = true)
            {
                // OLS learning algorithm
                var lOsl = new OrdinaryLeastSquares()
                {
                    UseIntercept = pUseIntercept,
                    IsRobust = pIsRobust

                };
                // Fit a linear regression model
                if (pOutputsTrain == null)
                {
                    pOutputsTrain = new double[1][];
                    pOutputsTrain[0] = _lm.TrainY;
                }
                if(pOutputsTest == null)
                {
                    pOutputsTest = new double[1][];
                    pOutputsTest[0] = _lm.TestY;
                }
                
                MultivariateLinearRegression lRegressionFit = lOsl.Learn(_lm.TrainX, pOutputsTrain);

                //.Transform - making predictions for model
                double[][] lInSamplePreds = lRegressionFit.Transform(_lm.TrainX);
                double[][] lOutSamplePreds = lRegressionFit.Transform(_lm.TestX);

                //int[] lOutPreds = _lm.Get_OutSamplePreds.Select(x => x >= pTreshold ? 1 : 0).ToArray();
                //int[] lInPreds = _lm.Get_InSamplePreds.Select(x => x >= pTreshold ? 1 : 0).ToArray();

               // _Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);

                double lErrorIn = new SquareLoss(_lm.TrainY).Loss(lInSamplePreds);
                double lErrorOut = new SquareLoss(_lm.TestY).Loss(lOutSamplePreds);

                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
            }
        
        }
        public class cBoosting:cAlgs
        {
            cLearningMethods _lm;
            public cBoosting(cLearningMethods pLm)
            {
                _lm = pLm;
            }
            /// <summary>
            /// Boosting simple data
            /// </summary>
            /// <param name="pMaxIteration"></param>
            /// <param name="pTolerance"></param>
            public void AdaBoost(int pMaxIteration = 5, double pTolerance = 1e-3, double pThreshold = 0.5)
            {
                var lAdaboostLearner = new AdaBoost<DecisionStump>()
                {
                    Learner = (p) => new ThresholdLearning(),

                    // Train until:
                    MaxIterations = 5,
                    Tolerance = 1e-3,
                    Threshold = pThreshold
                };
                Boost<DecisionStump> lClassifier = lAdaboostLearner.Learn(_lm.TrainX, _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());

                //_InSamplePreds = lClassifier.Transform(_trainX);
                //_OutSamplePreds = lClassifier.Transform(_testX);


                int[] lInPreds = lClassifier.Decide(_lm.TrainX).Select(x => Convert.ToInt32(x)).ToArray();
                int[] lOutPreds = lClassifier.Decide(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray();
               Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);

                _lm.Get_OutSamplePreds = lClassifier.Transform(_lm.TestX).Select(x => Convert.ToDouble(x)).ToArray();

                _lm.OutPreds_ForROC_int = lClassifier.Transform(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray();

                //ReceiverOperatingCharacteristic lRoc = new ReceiverOperatingCharacteristic(_lm.TestY.Select(x => x == 1 ? true : false).ToArray(), lClassifier.Transform(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray());
                //lRoc.Compute(1000);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => x == 1 ? true : false).ToArray(), lClassifier.Transform(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray());

                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
            }
            /// <summary>
            /// This example shows how to use AdaBoost to train more complex
            /// </summary>
            /// <param name="pComputeStandardErrors"></param>
            /// <param name="pMaxIterationLearner"></param>
            /// <param name="pToleranceLearner"></param>
            /// <param name="pMaxIteration"></param>
            /// <param name="pTolerance"></param>
            public void AdaBoostLogisticRegression(bool pComputeStandardErrors = false, int pMaxIterationLearner = 50, double pToleranceLearner = 1e-3, int pMaxIteration = 50, double pTolerance = 1e-3, double pThreshold = 0.5)
            {
                var lAdaboostLearner = new AdaBoost<LogisticRegression>()
                {
                    // Here we can specify how each regression should be learned:
                    Learner = (param) => new IterativeReweightedLeastSquares<LogisticRegression>()
                    {
                        ComputeStandardErrors = pComputeStandardErrors,
                        MaxIterations = pMaxIterationLearner,
                        Tolerance = pToleranceLearner
                    },

                    // Train until:
                    MaxIterations = pMaxIteration,
                    Tolerance = pTolerance,
                    Threshold = pThreshold
                };
                Boost<LogisticRegression> lClassifier = lAdaboostLearner.Learn(_lm.TrainX, _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());

                int[] lInPreds = lClassifier.Decide(_lm.TrainX).Select(x => Convert.ToInt32(x)).ToArray();
                int[] lOutPreds = lClassifier.Decide(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray();
               
                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);

                _lm.OutPreds_ForROC_int = lClassifier.Transform(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray();

                _lm.Get_OutSamplePreds = lClassifier.Transform(_lm.TestX).Select(x => Convert.ToDouble(x)).ToArray();
                //ReceiverOperatingCharacteristic lRoc = new ReceiverOperatingCharacteristic(_lm.TestY.Select(x => x == 1 ? true : false).ToArray(), lClassifier.Transform(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray());
                //lRoc.Compute(1000);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => x == 1 ? true : false).ToArray(), lClassifier.Transform(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray());

                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc); 
            }
        }
        public class cDecisionTrees:cAlgs
        {
            cLearningMethods _lm;
            public cDecisionTrees(cLearningMethods pLm)
            {
                _lm = pLm;
            }
            public void DTC45Learning(int pJoin = 2, int pMaxHeight = 5)
            {
                var lDTC45Learning = new C45Learning()
                {
                  Join = pJoin,
                  MaxHeight = pMaxHeight,
                };
                DecisionTree lRes = lDTC45Learning.Learn(_lm.TrainX, _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());

                int[] lInPreds = lRes.Decide(_lm.TrainX).Select(x => Convert.ToInt32(x)).ToArray();
                int[] lOutPreds = lRes.Decide(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray();

                _lm.OutPreds_ForROC_int = lOutPreds;

                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);
                _lm.Get_OutSamplePreds = lRes.Transform(_lm.TestX).Select(x=>Convert.ToDouble(x)).ToArray();
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100,_lm.TestY.Select(x => x == 1 ? true : false).ToArray(), lRes.Transform(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray());
                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
            }
            public void RandomForest(int pNumberOfTree = 10, double pSampleRation = 1.0, double pCoverageRatio = 1)
            {
                var lRandomForestLearner = new RandomForestLearning()
                {
                    NumberOfTrees = pNumberOfTree,
                    SampleRatio = pSampleRation,// the proportion of samples used to train each of the trees
                    CoverageRatio = pCoverageRatio// the proportion of variables that can be used at maximum


                };

                RandomForest lRF = lRandomForestLearner.Learn(_lm.TrainX, _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());


                int[] lInPreds = lRF.Decide(_lm.TrainX).Select(x => Convert.ToInt32(x)).ToArray();
                int[] lOutPreds = lRF.Decide(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray();

                _lm.OutPreds_ForROC_int = lOutPreds;

                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);

                _lm.Get_OutSamplePreds = lRF.Transform(_lm.TestX).Select(x => Convert.ToDouble(x)).ToArray();

                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100,_lm.TestY.Select(x => x == 1 ? true : false).ToArray(), lRF.Transform(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray());
                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
            }
            
        }
        public class cVectorMachines :cAlgs
        {
            cLearningMethods _lm;
            public cVectorMachines(cLearningMethods pLm)
            {
                _lm = pLm;
            }

            /// <summary>
            /// SVM Multiclass
            /// </summary>
            /// <param name="pEpsilon"></param>
            /// <param name="pTolerance"></param>
            /// <param name="pComplexity"></param>
            /// <param name="pUserKernelEstimation"></param>
            public void MulticlassSupportVectorLearning(double pEpsilon = 0.01, double pTolerance = 1e-3, double pComplexity = 1e-4, bool pUseKernelEstimation = true)
            {
                var lSvmTeacer = new MulticlassSupportVectorLearning<Gaussian>()
                {
                    Learner = (param) => new SequentialMinimalOptimization<Gaussian>()
                    {
                        Epsilon = pEpsilon,
                        Tolerance = pTolerance,
                        Complexity = pComplexity,
                        UseKernelEstimation = pUseKernelEstimation,
                    }
                };
                var lSvmTrainModel = lSvmTeacer.Learn(_lm.TrainX, _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());

                _lm.Get_InSamplePreds = lSvmTrainModel.Score(_lm.TrainX);
                _lm.Get_OutSamplePreds = lSvmTrainModel.Score(_lm.TestX);

                var distinct = _lm.Get_OutSamplePreds.Distinct();

                int[] lInPreds = lSvmTrainModel.Decide(_lm.TrainX);
                int[] lOutPreds = lSvmTrainModel.Decide(_lm.TestX);
                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);


                //ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds);
                //_lm.PrintRoc("LogisticRegression", lRoc);
                setAuc(lRoc);
            }
            /// <summary>
            /// SVM
            /// </summary>
            /// <param name="pEpsilon">margin of tolerance where no penalty is given to errors</param>
            /// <param name="pTolerance"></param>
            /// <param name="pComplexity"></param>
            /// <param name="pUserKernelEstimation"></param>
            public void SVM(double pEpsilon = 2, double pTolerance = 1e-3, double pComplexity = 1e-4, bool pUseComplexHeuristics = true, bool pUseKernelEst = true)
            {
                var lLearn = new SequentialMinimalOptimization<Gaussian>()
                {
                    Epsilon = pEpsilon,
                    Tolerance = pTolerance,
                    Complexity = pComplexity,
                    UseComplexityHeuristic = pUseComplexHeuristics,
                    UseKernelEstimation = pUseKernelEst
                };
                var lSvmTrainModel = lLearn.Learn(_lm.TrainX, _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());

                //_lm.Get_InSamplePreds = lSvmTrainModel.Score(_lm.TrainX);
                //_lm.Get_OutSamplePreds = lSvmTrainModel.Score(_lm.TestX);

                int[] lInPreds = lSvmTrainModel.Decide(_lm.TrainX).Select(x => Convert.ToInt32(x)).ToArray();
                int[] lOutPreds = lSvmTrainModel.Decide(_lm.TestX).Select(x => Convert.ToInt32(x)).ToArray();

                _lm.OutPreds_ForROC_int = lOutPreds;

                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);

                //ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100,_lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds);
                setAuc(lRoc);
            }
            public void MultinomialLogisticLearning(int pMinBatchSize = 90)
            {
                var lLearner = new MultinomialLogisticLearning<BroydenFletcherGoldfarbShanno>()
                {
                    //MiniBatchSize = pMinBatchSize
                };
                var lModel= lLearner.Learn(_lm.TrainX, _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());

                _lm.Get_InSamplePreds = lModel.Score(_lm.TrainX);
                _lm.Get_OutSamplePreds = lModel.Score(_lm.TestX);

                int[] lInPreds = lModel.Decide(_lm.TrainX);
                int[] lOutPreds = lModel.Decide(_lm.TestX);
                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);

                //ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds.Select(x => Convert.ToInt32(x) == 1 ? 1 : 0).ToArray());
                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
            }
            /// <summary>
            /// Gaussian SVM
            /// </summary>
            /// <param name="pEpsilon"></param>
            /// <param name="pTolerance"></param>
            /// <param name="pComplexity"></param>
            /// <param name="pUserKernelEstimation"></param>
            public void FanChenLinSupportVectorRegressionGaussian(double pEpsilon = 0.1, double pTolerance = 1e-5, double pComplexity = 1e-6, bool pUserKernelEstimation = true, bool pUseComplexityHeuristic=false)
            {
                var lGaussianSVMLearner = new FanChenLinSupportVectorRegression<Gaussian>()
                {
                //    Epsilon = pEpsilon,
                //    Tolerance = pTolerance,
                //    Complexity = pComplexity,
                //    UseComplexityHeuristic = pUseComplexityHeuristic,
                //    UseKernelEstimation= pUserKernelEstimation,
                    Kernel = new Gaussian()
                };
                var lGaussianSVM = lGaussianSVMLearner.Learn(_lm.TrainX, _lm.TrainY);
                _lm.Get_InSamplePreds = lGaussianSVM.Score(_lm.TrainX);
                _lm.Get_OutSamplePreds = lGaussianSVM.Score(_lm.TestX);

                bool[] lOutPreds = lGaussianSVM.Decide(_lm.TestX);
                bool[] lInPreds = lGaussianSVM.Decide(_lm.TrainX);
                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds.Select(x => Convert.ToInt32(x)).ToArray());

                //ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds.Select(x => Convert.ToInt32(x) == 1 ? 1 : 0).ToArray());
                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
            }
            /// <summary>
            /// // Linear SVM Learning Algorithm /svR SUPPORT VECTOR REGRESSION
            /// </summary>
            /// <param name="pEpsilon">Type double</param>
            /// <param name="pTolerance">Type double</param>
            /// <param name="pUseComplexityHeuristic">Type bool</param>
            public void LinearRegressionNewtonMethod(double pEpsilon = 2.1, double pTolerance = 1e-5, bool pUseComplexityHeuristic = true)
            {
                // Linear SVM Learning Algorithm
                var lTeacher = new LinearRegressionNewtonMethod()
                {
                    Epsilon = pEpsilon, //Najvecji auc 3 za eurusd-features
                    Tolerance = pTolerance,
                    UseComplexityHeuristic = pUseComplexityHeuristic
                    

                };
                var lLinearSvm = lTeacher.Learn(_lm.TrainX, _lm.TrainY);
                _lm.Get_InSamplePreds = lLinearSvm.Score(_lm.TrainX);
                _lm.Get_OutSamplePreds = lLinearSvm.Score(_lm.TestX);

                bool[] lOutPreds = _lm.Get_OutSamplePreds.Select(x => x > 0.5 ? true : false).ToArray(); //lLinearSvm.Decide(_lm.TestX); 
                bool[] lInPreds = _lm.Get_InSamplePreds.Select(x => x > 0.5 ? true : false).ToArray();//lLinearSvm.Decide(_lm.TrainX);

                _lm.OutPreds_ForROC_int = lOutPreds.Select(x => Convert.ToInt32(x)).ToArray();

                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds.Select(x => Convert.ToInt32(x)).ToArray());

                // ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds.Select(x => Convert.ToInt32(x) == 1 ? 1 : 0).ToArray());
                setAuc(lRoc);
                //_lm.PrintRoc("LogisticRegression", lRoc);
                //Console.WriteLine("ACC: " + lGcm.Accuracy);
            }

        }
        public class cNeuralNet:cAlgs
        {
            cLearningMethods _lm;
            public cNeuralNet(cLearningMethods pLm)
            {
                _lm = pLm;
            }
            private ActivationNetwork lNNet;
            public void RestrictedBoltzmannMachine(double pAlpha = 0.5, int pInputsCount = 6, int pHidenCounts = 2, double pMomentum = 0, double pDecay = 0, double pLearningRate = 0.1, int lLearnIteration = 5000)
            {
                BernoulliFunction lBfunc = new BernoulliFunction(pAlpha);
                RestrictedBoltzmannMachine lRbm = new RestrictedBoltzmannMachine(lBfunc, pInputsCount, pHidenCounts);

                ContrastiveDivergenceLearning lTeacher = new ContrastiveDivergenceLearning(lRbm)
                {
                    Momentum = pMomentum,
                    Decay = pDecay,
                    LearningRate = pLearningRate
                };
                for (int i = 0; i < 5; i++)
                {
                    lTeacher.RunEpoch(_lm.TrainX);
                }
                _lm.Get_InSamplePreds = lRbm.Compute(_lm.TrainY);
                _lm.Get_OutSamplePreds = lRbm.Compute(_lm.TestY);
                
            }
            /// <summary>
            /// Always normalize data!
            /// double[][] scores = Accord.Statistics.Tools.ZScores(inputs);
            /// double[][] centered = Accord.Statistics.Tools.Center(inputs);
            /// double[][] standard = Accord.Statistics.Tools.Standardize(inputs);
            /// </summary>
            /// <param name="pAf">Activation function</param>
            /// <param name="pLayers">int[0] number of hidden layers, int[1] number of target class</param>
            /// <param name="pNumInputs">Must be lenght of features</param>
            public void NeuralNetBackPropagation(IActivationFunction pAf, int pEpoch, int pNumInputs, int[] pLayers)
            {
                if(pNumInputs==0 || pNumInputs != _lm.Features.Length)
                {
                    pNumInputs = _lm.Features.Length;
                }
                // sigmoid activation function, 3 inputs, 2 layers, 4 neuron in first layer,1 neuron in second layer,  = new int[2]{ 4,1}
                lNNet = new ActivationNetwork(pAf, pNumInputs, pLayers);

             

                var lTeacher = new BackPropagationLearning(lNNet);

                int[]  lOutPreds = Predict(lTeacher, pEpoch);

                _lm.OutPreds_ForROC_int = lOutPreds;

                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds.Select(x => Convert.ToInt32(x)).ToArray());

                //ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100)
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds); ;

                setAuc(lRoc);
            }
            /// <summary>
            /// Always normalize data!
            /// </summary>
            /// <param name="pAf"></param>
            /// <param name="pEpoch"></param>
            /// <param name="pNumInputs"></param>
            /// <param name="pLayers"></param>
            public void NeuralNetLevenbergMarquardtLearning(IActivationFunction pAf, int pEpoch, int pNumInputs, int[] pLayers)
            {
                if (pNumInputs == 0 || pNumInputs != _lm.Features.Length)
                {
                    pNumInputs = _lm.Features.Length;
                }
                // sigmoid activation function, 3 inputs, 2 layers, 4 neuron in first layer,1 neuron in second layer,  = new int[2]{ 4,1}
                lNNet = new ActivationNetwork(pAf, pNumInputs, pLayers);

                var lTeacher = new LevenbergMarquardtLearning(lNNet);

                int[] lOutPreds = Predict(lTeacher, pEpoch);

                _lm.OutPreds_ForROC_int = lOutPreds;

                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds.Select(x => Convert.ToInt32(x)).ToArray());

                //ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100);
                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(), lOutPreds);

                setAuc(lRoc);
            }
            private int[] Predict(ISupervisedLearning lTeacher, int pEpoch)
            {

                double[][] lOutputs = Accord.Math.Jagged.OneHot(_lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());

                double[] error = new double[pEpoch];
                for (int i = 0; i < pEpoch; i++)
                {
                    error[i] = lTeacher.RunEpoch(_lm.TrainX, lOutputs);
                    //Console.WriteLine("* Epoch: " + i + "  - error: " + error[i].ToString());
                }
                _lm.Get_InSamplePreds = new double[_lm.TrainX.Length];
                List<int> lInSamplePredsList = new List<int>();
                for (int i = 0; i < _lm.TrainX.Length; i++)
                {
                    double[] lOutput = lNNet.Compute(_lm.TrainX[i]);
                    _lm.Get_InSamplePreds[i] = lOutput.Max();

                    int lPrediction = lOutput.ToList().IndexOf(lOutput.Max());
                    lInSamplePredsList.Add(lPrediction);
                }
                _lm.Get_OutSamplePreds = new double[_lm.TestX.Length];
                List<int> lOutSamplePredsList = new List<int>();
                for (int i = 0; i < _lm.TestX.Length; i++)
                {
                    double[] lOutput = lNNet.Compute(_lm.TestX[i]);
                    _lm.Get_OutSamplePreds[i] = lOutput.Max();

                    int lPrediction = lOutput.ToList().IndexOf(lOutput.Max());
                    lOutSamplePredsList.Add(lPrediction);
                }
                int[] lInPreds = lInSamplePredsList.ToArray();
                int[] lOutPreds = lOutSamplePredsList.ToArray();

                return lOutPreds;
            }
        }
        public class cClassification:cAlgs
        {
            cLearningMethods _lm;
            public cClassification(cLearningMethods pLm)
            {
                _lm = pLm;
            }

            /// <summary>
            /// Naive Bayes classifiers
            /// https://www.r-bloggers.com/understanding-naive-bayes-classifier-using-r/
            /// </summary>
            /// <param name="pRegularization"></param>
            /// <param name="pDiagonal"></param>
            /// <param name="pRobust"></param>
            /// <param name="pShared"></param>
            public void NaiveBayesLearning(double pRegularization = 1e-5, bool pDiagonal = true, bool pRobust = true, bool pShared = true)
            {
                if(_lm.TrainX == null || _lm.TestX == null)
                {
                    return;
                }
                var lnbTeacher = new NaiveBayesLearning<NormalDistribution>();
                lnbTeacher.Options.InnerOption = new NormalOptions
                {
                    Regularization = pRegularization,
                    Diagonal = pDiagonal,
                    Robust = pRobust,
                    Shared = pShared,

                };

                var lnbTrainedModel = lnbTeacher.Learn(_lm.TrainX, _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray());

                _lm.Get_InSamplePreds = lnbTrainedModel.Score(_lm.TrainX);
                _lm.Get_OutSamplePreds = lnbTrainedModel.Score(_lm.TestX);

                int[] lInPreds = lnbTrainedModel.Decide(_lm.TrainX);
                int[] lOutPreds = lnbTrainedModel.Decide(_lm.TestX);

                _lm.OutPreds_ForROC_int = lOutPreds;

                var aaaa = new GeneralConfusionMatrix(expected: _lm.TrainY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lInPreds);
                Tempgcm = new GeneralConfusionMatrix(expected: _lm.TestY.Select(x => Convert.ToInt32(x)).ToArray(), predicted: lOutPreds);

                ReceiverOperatingCharacteristic lRoc = _lm.CalculateROC(100, _lm.TestY.Select(x => Convert.ToBoolean(x)).ToArray(),lOutPreds);

                setAuc(lRoc);
                //_lm.PrintRoc("NaiveBayesLearning", lRoc);

            }
        } 
        public class cAlgs
        {
            private GeneralConfusionMatrix _Tempgcm;
            private double _Tempauc;

            public GeneralConfusionMatrix Tempgcm { get => _Tempgcm; set => _Tempgcm = value; }
            public double Tempauc { get => _Tempauc; set => _Tempauc = value; }

            public void setAuc(ReceiverOperatingCharacteristic pRoc)
            {
                if (pRoc != null)
                {
                    Tempauc = pRoc.Area;
                }
                else
                {
                    Tempauc = 0;
                }
            }
        }
        public class cRegularization
        {
            cLearningMethods _lm;
            public cRegularization(cLearningMethods plm)
            {
                _lm = plm;
            }
            public void L1(double[] pInput, double pT = 0.00001, double pC = 0.00001)
            {
                ProbabilisticCoordinateDescent lPCD = new ProbabilisticCoordinateDescent()
                {
                    Tolerance = pT,
                    Complexity = pC
                };
            }
            public void L1(double[][] pInput)
            {

            }
        }
        public class cStatistics
        {
            cLearningMethods _lm;
            public cStatistics(cLearningMethods pLm)
            {
                _lm = pLm;
            }
            public void TMG_Pearson()
            {
                //v10= 0.1dm / td
                //v90 = 0.9dm / (td+tc)
                var originalData_1 = _lm.Original_DataXY.Where(x=>x[2]==1);
                var v10_1 = originalData_1.Select(x => (0.1 * x[8]) / x[4]);
                var v90_1 = originalData_1.Select(x => (0.1 * x[8]) / (x[4]+x[5]));
                var sila_norm_1 = originalData_1.Select(x => x[10] / x[9]);
                var originalData_2= _lm.Original_DataXY.Where(x => x[2] == 2);
                var v10_2 = originalData_2.Select(x => (0.1 * x[8]) / x[4]);
                var v90_2 = originalData_2.Select(x => (0.1 * x[8]) / (x[4] + x[5]));
                var sila_norm_2 = originalData_2.Select(x => x[10]/x[9]);

                cStatisticTests lStat_InputTd_1 = new cStatisticTests(originalData_1.Select(x => x[4]).ToArray());
                cStatisticTests lStat_InputTc_1 = new cStatisticTests(originalData_1.Select(x => x[5]).ToArray());
                cStatisticTests lStat_InputDm_1 = new cStatisticTests(originalData_1.Select(x => x[8]).ToArray());
                cStatisticTests lStat_V10_1 = new cStatisticTests(v10_1.Select(x=>x).ToArray());
                cStatisticTests lStat_v90_1 = new cStatisticTests(v90_1.Select(x=>x).ToArray());

                cStatisticTests lStat_a_1 = new cStatisticTests(sila_norm_1.Select(x=>x).ToArray());

                double lCor_DT_1 = cStatisticTests.PearsonCorrelation(lStat_InputTd_1, lStat_a_1);
                double lCor_Tc_1 = cStatisticTests.PearsonCorrelation(lStat_InputTc_1, lStat_a_1);
                double lCor_Dm_1 = cStatisticTests.PearsonCorrelation(lStat_InputDm_1, lStat_a_1);
                double lCor_V10_1 = cStatisticTests.PearsonCorrelation(lStat_V10_1, lStat_a_1);
                double lCor_V90_1 = cStatisticTests.PearsonCorrelation(lStat_v90_1, lStat_a_1);

                cStatisticTests lStat_InputTd_2 = new cStatisticTests(originalData_2.Select(x => x[4]).ToArray());
                cStatisticTests lStat_InputTc_2 = new cStatisticTests(originalData_2.Select(x => x[5]).ToArray());
                cStatisticTests lStat_InputDm_2 = new cStatisticTests(originalData_2.Select(x => x[8]).ToArray());
                cStatisticTests lStat_V10_2 = new cStatisticTests(v10_2.Select(x => x).ToArray());
                cStatisticTests lStat_v90_2 = new cStatisticTests(v90_2.Select(x => x).ToArray());

                cStatisticTests lStat_a_2 = new cStatisticTests(sila_norm_2.Select(x => x).ToArray());

                double lCor_DT_2 = cStatisticTests.PearsonCorrelation(lStat_InputTd_2, lStat_a_2);
                double lCor_Tc_2 = cStatisticTests.PearsonCorrelation(lStat_InputTc_2, lStat_a_2);
                double lCor_Dm_2 = cStatisticTests.PearsonCorrelation(lStat_InputDm_2, lStat_a_2);
                double lCor_V10_2 = cStatisticTests.PearsonCorrelation(lStat_V10_2, lStat_a_2);
                double lCor_V90_2 = cStatisticTests.PearsonCorrelation(lStat_v90_2, lStat_a_2);
                //TODO ali nema korelacije
            }
            public double DistanceCorrelationCoificient()
            {
                //cStatisticTests lTest1 = new cStatisticTests(_lm.TrainXOriginal.Select(x=>x[4]).ToArray());



                var originalData = _lm.Original_DataXY.Where(x => x[0] == 0);
                originalData = originalData.Where(x => x[1] == 0);
                cStatisticTests lStat_InputTd = new cStatisticTests(originalData.Select(x => x[4]).ToArray());
                cStatisticTests lStat_InputTc = new cStatisticTests(originalData.Select(x => x[5]).ToArray());
                cStatisticTests lStat_InputDm = new cStatisticTests(originalData.Select(x => x[8]).ToArray());

                var teza = originalData.Select(x => x[9]).ToArray();
                var target = originalData.Select(x => x[10]).ToArray();
                for (int i = 0; i < teza.Length; i++)
                {
                    target[i] = target[i] / teza[i];
                }

                cStatisticTests lStat_Output = new cStatisticTests(target);

                int lLen = lStat_InputTd.lenght();
                //double lCov = cStatisticTests.Covarianca(lTest1, lStat_Output);

                //double lCor = cStatisticTests.PearsonCorrelation(lTest1, lStat_Output); //Accord.Math.Distances.PearsonCorrelation lp = new Accord.Math.Distances.PearsonCorrelation();
                //double lCorD = cStatisticTests.CoefficientOfdetermination(lTest1, lStat_Output);

                //double lCor3 = Accord.Statistics.Tools.Determination( _lm.TrainY, _lm.TrainXOriginal.Select(x => x[4]).ToArray());

                //var aaaaaa = Accord.Statistics.Measures.Correlation(_lm.TrainXOriginal.ToMatrix());

                double b = cStatisticTests.b(lStat_InputTd, lStat_Output);
                double a = cStatisticTests.a(lStat_InputTd, lStat_Output);


                double lCorTD = cStatisticTests.PearsonCorrelation(lStat_InputTd, lStat_Output);
                double l_t_TD = cStatisticTests.tValue(lCorTD, lLen);
                double l_p_TD = cStatisticTests.TCall(cStatisticTests.Buzz(l_t_TD, lLen - 2));

                double lCorTC = cStatisticTests.PearsonCorrelation(lStat_InputTc, lStat_Output);
                double l_t_TC = cStatisticTests.tValue(lCorTC, lLen);
                double l_p_TC = cStatisticTests.TCall(cStatisticTests.Buzz(l_t_TC, lLen - 2));

                double lCorDM = cStatisticTests.PearsonCorrelation(lStat_InputDm, lStat_Output);
                double l_t_DM = cStatisticTests.tValue(lCorDM, lLen);
                double l_p_DM = cStatisticTests.TCall(cStatisticTests.Buzz(l_t_DM, lLen - 2));

                double lCorTDTC = cStatisticTests.PearsonCorrelation(lStat_InputTd, lStat_InputTc);
                double l_t_TDTC = cStatisticTests.tValue(lCorTDTC, lLen);
                double l_p_TDTC = cStatisticTests.TCall(cStatisticTests.Buzz(l_t_TDTC, lLen - 2));

                double lCorTCDM = cStatisticTests.PearsonCorrelation(lStat_InputTc, lStat_InputDm);
                double l_t_TCDM = cStatisticTests.tValue(lCorTCDM, lLen);
                double l_p_TCDM = cStatisticTests.TCall(cStatisticTests.Buzz(l_t_TCDM, lLen - 2));

                double lCorDMTD = cStatisticTests.PearsonCorrelation(lStat_InputDm, lStat_InputTd);
                double l_t_DMTD = cStatisticTests.tValue(lCorDMTD, lLen);
                double l_p_DMTD = cStatisticTests.TCall(cStatisticTests.Buzz(l_t_DMTD, lLen - 2));



                Accord.Statistics.Testing.TwoSampleTTest lTT = new Accord.Statistics.Testing.TwoSampleTTest(_lm.Original_DataX.Select(x => x[4]).ToArray(), _lm.Original_DataY);

                DataTable lDT = new DataTable();
                lDT.Columns.Add("-");
                lDT.Columns.Add("TC");
                lDT.Columns.Add("TD");
                lDT.Columns.Add("DM");
                //lDT.Columns.Add("Sila");

                DataRow lR = lDT.NewRow();
                lR[0] = "TD";
                lR[1] = lCorTDTC;

                lDT.Rows.Add(lR);

                lR = lDT.NewRow();
                lR[0] = "p";
                lR[1] = l_p_TDTC;

                lDT.Rows.Add(lR);

                lR = lDT.NewRow();
                lR[0] = "DM";
                lR[1] = lCorTCDM;
                lR[2] = lCorDMTD;

                lDT.Rows.Add(lR);

                lR = lDT.NewRow();
                lR[0] = "p";
                lR[1] = l_p_TCDM;
                lR[2] = l_p_DMTD;

                lDT.Rows.Add(lR);

                lR = lDT.NewRow();
                lR[0] = "Sila";
                lR[1] = lCorTC;
                lR[2] = lCorTD;
                lR[3] = lCorDM;

                lDT.Rows.Add(lR);

                lR = lDT.NewRow();
                lR[0] = "p";
                lR[1] = l_p_TC;
                lR[2] = l_p_TD;
                lR[3] = l_p_DM;

                lDT.Rows.Add(lR);

                cWrite.WriteDataTableToCsvFile(lDT, "C:/Users/Gemma/Desktop/T-Test.csv");
                return double.NaN;
            }
        }
    }
}
