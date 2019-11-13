using System;
using System.Collections.Generic;
using System.Linq;
using MLLib.AI;
using MLLib.AI.MBNN;
using MLLib.AI.OBNN;
using NeuralNetwork = MLLib.AI.OBNN.NeuralNetwork;

namespace MLLib.Tests.TennisClassifier
{
    internal class Program
    {
        private static Random _random = new Random((int)DateTime.Now.TimeOfDay.TotalMilliseconds);
        private static List<WeatherCondition> _weatherConditions;
        private const int DataCount  = 50000;
        private const int TestsCount = 10000;
        private const int ValidationCount = 10000;

        public static void Main(string[] args)
        {
            _weatherConditions = new List<WeatherCondition>();
            for (int i = 0; i < DataCount + TestsCount + ValidationCount; i++)
            {
                _weatherConditions.Add(new WeatherCondition(
                    _random.NextDouble(),
                    _random.NextDouble(),
                    _random.NextDouble(),
                    _random.NextDouble()));

                /*
                var outlook = Outlook.Overcast;
                var humidity = Humidity.High;
                var temperature = Temperature.Cool;

                switch (_random.Next(0, 2))
                {
                    case 0: outlook = Outlook.Sunny; break;
                    case 1: outlook = Outlook.Overcast; break;
                    case 2: outlook = Outlook.Rain; break;
                }
                switch (_random.Next(0, 2))
                {
                    case 0: humidity = Humidity.High; break;
                    case 1: humidity = Humidity.Normal; break;
                }
                var windy = _random.Next() % 2 == 0;
                switch (_random.Next(0, 3))
                {
                    case 0: temperature = Temperature.Hot; break;
                    case 1: temperature = Temperature.Mild; break;
                    case 2: temperature = Temperature.Cool; break;
                }

                _weatherConditions.Add(new WeatherCondition(
                    outlook, humidity, windy, temperature));*/
            }

            var network = new ImprovedNeuralNetwork(new[] {4, 16, 16, 1}, new CrossEntropyCostFunction())
            {
                LearningRate = .5,
                RegularizationLambda = 0.1
            };
            network.FillGaussianRandom();

            var teacher = new Teacher(
                network,
                30, 4,
                _weatherConditions.Take(DataCount).Cast<TrainSample>().ToList(),
                _weatherConditions.Skip(DataCount).Take(ValidationCount).Cast<TrainSample>().ToList(),
                _weatherConditions.Skip(DataCount + ValidationCount).Cast<TrainSample>().ToList())
            {
                MonitorTrainingCost = true,
                MonitorTrainingAccuracy = true,
                MonitorValidationAccuracy = true
            };

            teacher.Teach();
            teacher.Test(20, true);
            Console.WriteLine("Test data cost: {0:F3}, Test data accuracy: {1:F3}",
                teacher.TestDataCost,
                teacher.TestDataAccuracy * 100);
        }
    }
}