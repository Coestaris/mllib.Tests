using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using MLLib.AI;
using MLLib.AI.OBNN;
using OpenTK.Graphics;
using OpenTK.Graphics.ES11;
using MLLib.WindowHandler;

namespace MLLib.Tests.XORCalculator
{
    internal class Program
    {
        private static NeuralNetwork _network;

        public class XORTask : TrainSample
        {
            private static Random _random = new Random();
            private double[] _trainData;
            private double[] _expected;

            public override double[] ToTrainData()
            {
                return _trainData;
            }

            public override double[] ToExpected()
            {
                return _expected;
            }

            public override bool CheckAssumption(double[] output)
            {
                var a = Math.Abs(_expected[0] - 1) < 0.01;
                return a
                    ? output[0] > .90 && output[1] < 0.10
                    : output[1] > .90 && output[0] < 0.10;
            }

            public XORTask()
            {
                _trainData = new double[] {_random.Next() % 2, _random.Next() % 2};
                var result = ((int) _trainData [0] ^ (int) _trainData [1]) == 1;
                _expected = new double[] {result ? 1 : 0, result ? 0 : 1};
            }
        }

        public static void Main(string[] args)
        {
            var win = new Window(1000, 700, "XORCalculator")
            {
                BackgroundColor = new Color4(94f / 255f, 91f / 255f, 102f / 255f, 0)
            };

            var tasks = new List<XORTask>();
            for(var i = 0; i < 1000; i++)
                tasks.Add(new XORTask());

            _network = new NeuralNetwork(new []{ 2, 4, 4, 2} );
            _network.FillGaussianRandom();
            var teacher = new Teacher(_network, 1, 1, tasks.Cast<TrainSample>().ToList())
            {
                SilentMode = true
            };

            var handler = new NNVisualizer(win, _network, teacher, () => _network.FillGaussianRandom());

            handler.Start();
        }
    }
}