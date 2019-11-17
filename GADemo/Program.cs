using System;
using System.Collections.Generic;
using System.Linq;
using MLLib.AI.GA;
using MLLib.AI.OBNN;

namespace MLLib.GADemo
{
    internal static class Program
    {
        public static double FitnessFunc(Genome gen)
        {
            var x = gen.Genes[0];
            var y = gen.Genes[1];

            return -(x * x / 10.0 + y * y / 6.0) + 12.0;
        }

        public static void SearchingMaximum()
        {
            var random = new Random();
            var population = new Population(50, i => new Genome(
                new List<double>
                {
                    random.Next() % 2 == 0 ? random.NextDouble() * 10 - 20 : random.NextDouble() * 10 + 10,
                    random.Next() % 2 == 0 ? random.NextDouble() * 10 - 20 : random.NextDouble() * 10 + 10,
                },
                FitnessFunc));

            Genome bestCreature = null;
            while (bestCreature == null || Math.Abs(bestCreature.Fitness - 12) > 1e-4)
            {
                population.EvaluateFitness();

                bestCreature = population.BestCreature(false);
                Console.WriteLine("X: {0:F3}, Y : {1:F3}, Fitness: {2:F5}",
                    bestCreature.Genes[0],
                    bestCreature.Genes[1],
                    bestCreature.Fitness);

                population.Selection(false);
                population.Crossover(CrossoverAlgorithm.Blend);
                population.Mutate(0.2);
            }
        }

        public static void CalculateXOR()
        {
            var population = new Population(1000, i =>
            {
                var nn = new NeuralNetwork(new[] {2, 10, 10, 2});
                nn.FillRandom();

                return NeuroEvolution.NNToGenome(nn, new XORSolver(nn));
            });

            var start = DateTime.Now;
            Genome bestCreature = null;
            var generation = 0;
            while (bestCreature == null || Math.Abs(bestCreature.Fitness) > 1e-5)
            {
                population.MultiThreadEvaluateFitness(8); //12seconds
                //population.EvaluateFitness(); //136seconds

                bestCreature = population.BestCreature(true);
                Console.WriteLine("Generation {0}. Best: {1:F9}. Average: {2:F4}",
                    generation++,
                    bestCreature.Fitness,
                    population.AverageFitness());

                population.Selection(true, 3);
                population.Crossover(CrossoverAlgorithm.Blend);
                population.Mutate(1);
            }

            Console.WriteLine("Done in: {0}ms", (DateTime.Now - start).TotalMilliseconds);
            while (true)
            {
                Console.Write("Enter two bits: ");
                var bits = Console.ReadLine().Split(' ');
                if(bits.Length != 2)
                    break;

                var a = double.Parse(bits[0]);
                var b = double.Parse(bits[1]);

                var output = (bestCreature.Creature as XORSolver).NeuralNetwork
                    .ForwardPass(new[] {a, b}).ToList();
                var max = output.IndexOf(output.Max());

                Console.WriteLine("{0} ^ {1} = {2} [{3:F4}, {4:F4}]",
                    a, b, max == 1 ? 1 : 0, output[0], output[1]);
            }
        }

        public static void Main(string[] args)
        {
           CalculateXOR();
           //SearchingMaximum();
        }
    }
}