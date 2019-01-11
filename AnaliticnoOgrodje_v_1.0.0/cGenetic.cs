using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnaliticnoOgrodje_v_1_0
{
    class cGenetic
    {
        //Feature selection - (or dimensionality reduction) and instance selection (or record reduction)
        public static void Fib()
        {
            int i = 0;
            List<double> lList = new List<double>();
            while (i != 100)
            {
                if (i == 0 && lList.Count() == 0)
                {
                    lList.Add(i);
                }
                else
                {
                    if (lList.Count() == 1)
                    {
                        lList.Add(i);
                    }
                    else
                    {
                        lList.Add(lList[i - 1] + lList[i - 2]);
                    }
                }
                i++;
            }
            List<double> lList2 = new List<double>();
            for (i = 1; i < lList.Count(); i++)
            {
                lList2.Add(lList[i - 1] / lList[i]);
            }
        }
        //genetski algoritam za odabir parametra za naive bayes npr, fit funkcija kak dobro baYES klasificira, dummy mnozica nekva, diabetes
        public class cGenome<T>
        {
            private T[] genes;
            private Func<T> selectFeature;
            private Func<int, double> fitness;
            private int size;
            float mutationRate;
            float crossover;
            Random random;

            public T[] Genes { get => genes; set => genes = value; }
            public Func<int, double> Fitness { get => fitness; }

            //public float Fitness()
            //{

            //}

            public cGenome(int pSize, Random pRandom, Func<T> pSelectFeature, Func<int, double> pFitness, float pMutationRate, float pCrossover, bool pInit = true, bool pAllowDuplicates = true)
            {
                size = pSize;
                random = pRandom;
                genes = new T[pSize];
                this.selectFeature = pSelectFeature;
                this.fitness = pFitness;
                mutationRate = pMutationRate;
                crossover = pCrossover;
                if (pInit)
                {
                    Init(pAllowDuplicates);
                }
            }
            private void Init(bool pAllowDuplicates = false)
            {
                for (int i = 0; i < size; i++)
                {
                    var lGen = InitGenes();

                    if (!pAllowDuplicates)
                    {
                        genes[i] = Compare(lGen);
                    }
                    else
                    {
                        genes[i] = lGen;
                    }
                    

                }
            }
            public double CalculateFitness(int pI)
            {
                return fitness(pI);
            }
            private T Compare(T pGen)
            {
                for (int i = 0; i < genes.Length; i++)
                {
                    if (genes[i].GetHashCode() == pGen.GetHashCode())
                    {
                        return Compare(InitGenes());
                    }
                }
                return pGen;
            }
            private T InitGenes()
            {
                return selectFeature();
            }
            public cGenome<T> Crossover(cGenome<T> pParent, bool pAllowDuplicates = false)
            {
                cGenome<T> lChiled = new cGenome<T>(size, random, selectFeature, fitness, mutationRate,crossover,false, pAllowDuplicates);
                for(int i =0; i < genes.Length; i++)
                {
                    var lGen = random.NextDouble() < crossover ? genes[i] : pParent.genes[i];

                    if (!pAllowDuplicates)
                    {
                        lChiled.genes[i] = lChiled.Compare(lGen);
                    }
                    else
                    {
                        lChiled.genes[i] = lGen;
                    }
                    
                }
                lChiled.Mutate(pAllowDuplicates);
                return lChiled;

            }
            public void Mutate(bool pAllowDuplicates = false)
            {
                if(random.NextDouble() < mutationRate)
                {
                    int lPos = random.Next(0, genes.Length-1);

                    if (!pAllowDuplicates)
                    {
                        genes[lPos] = Compare(selectFeature());
                    }
                    else
                    {
                        genes[lPos] = selectFeature();
                    }
                    //genes[lPos] = Compare(selectFeature());
                    
                }
            }
        }
        public class cGA<T>
        {
            private List<cGenome<T>> population;
            internal List<cGenome<T>> Population { get => population; set => population = value; }
            public double BestFitness { get => bestFitness; set => bestFitness = value; }

            private int populationSize;
            private int genomSize;
            Random random;
            private Func<T> selectFeature;
            private Func<int, double> fitness;
            private float mutationRate;
            private float crossover;

            private double bestFitness;

            public cGA(int pPopulationSize, int pGenomSize, Random pRandom, Func<T> pSelectFeature, Func<int, double> pFitness, float pMutationRate = 0.01f, float pCrossover=0.8f, bool pAllowDuplicates = true)
            {
                populationSize = pPopulationSize;
                genomSize = pGenomSize;
                random = pRandom;
                this.selectFeature= pSelectFeature;
                this.fitness = pFitness;
                mutationRate = pMutationRate;
                crossover = pCrossover;

                population = new List<cGenome<T>>(pPopulationSize);
                for(int i = 0; i<population.Capacity; i++)
                {
                    population.Add(new cGenome<T>(pGenomSize,random, SelectFeature, pFitness, pMutationRate, pCrossover,true, pAllowDuplicates));
                }
            }

            public void NewGeneration(double pProportion=0.5, bool pAllowDuplicates = true)
            {
                bestFitness = SortPopulationByFitness();
                if(bestFitness == 1)
                {
                    return;
                }
                List<cGenome<T>> lListParents = ChooseParents(pProportion);
                List<cGenome<T>> lListOffsprings = new List<cGenome<T>>();

                //var a = from p in lListParents where p != null from c in lListParents where c != null select new {prvo = p,drugo = c};

                for (int i = 0; i<lListParents.Count(); i++)
                {
                    if (i < lListParents.Count() - 1)
                    {
                        for (int j = i + 1; j < lListParents.Count(); j++)
                        {
                            lListOffsprings.Add(lListParents[i].Crossover(lListParents[j],pAllowDuplicates));
                        }
                    }         
                }
                population = null;
                population = lListOffsprings.Take(populationSize).ToList();
            }
            private double SortPopulationByFitness()
            {
                List<KeyValuePair<cGenome<T>, double>> lList = new List<KeyValuePair<cGenome<T>, double>>();
                for(int i =0; i<population.Count; i++)
                {
                    var lFit = population[i].Fitness(i);
                    lList.Add(new KeyValuePair<cGenome<T>, double>(population[i], lFit));
                }
                lList = lList.OrderByDescending(o => o.Value).ToList() ;
                for(int i = 0; i < population.Count; i++)
                {
                    population[i] = lList[i].Key;
                }
                return lList[0].Value;
            }

            private T SelectFeature()
            {
                return selectFeature();
            }
            private List<cGenome<T>> ChooseParents(double pProportion)
            {
                List<cGenome<T>> lList = new List<cGenome<T>>();
                if ((population.Count() * pProportion) / 2 < 2)
                {
                    lList.Add(population[0]);
                    lList.Add(population[1]);
                }
                else
                {
                    for (int i = 0; i < Math.Ceiling(populationSize * pProportion); i++)
                    {
                        lList.Add(population[i]);
                    }
                }
                return lList;
            }
        }
    }
}
