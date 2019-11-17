using System;
using System.Threading;
using MLLib.AI.GA;
using MLLib.AI.OBNN;

namespace MLLib.GADemo
{
    public class XORSolver : ICreature
    {
        public NeuralNetwork NeuralNetwork;
        public const double MinTest = -10;
        public const double MaxTest =  10;

        private double _fitness;

        public XORSolver(NeuralNetwork neuralNetwork)
        {
            NeuralNetwork = neuralNetwork;
        }

        public void Reset()
        {
            _fitness = 0;
        }

        public object GetState() { return null; }

        private void DoTest(int a, int b)
        {
            var output = NeuralNetwork.ForwardPass(new double[] {a, b});
            var result = a ^ b;

            _fitness += Math.Pow(output[0] - (result == 1 ? 1 : 0), 2);
            _fitness += Math.Pow(output[1] - (result == 0 ? 1 : 0), 2);
        }

        public bool Step(int time)
        {
            DoTest(0, 0);
            DoTest(0, 1);
            DoTest(1, 0);
            DoTest(1, 1);

            //Thread.Sleep(5);
            return false;
        }

        public double GetFitness()
        {
            return _fitness;
        }

        public ICreature CreatureChild()
        {
            return new XORSolver((NeuralNetwork)NeuralNetwork.Clone());
        }

        public void Update(Genome genome)
        {
            NeuralNetwork = NeuroEvolution.GenomeToNN(NeuralNetwork, genome);
        }
    }
}