using System;
using System.Drawing.Imaging;
using System.Linq;
using System.Threading;
using ml.AI;
using ml.AI.OBNN;

namespace HWDRecognizer
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            var dataset = new Dataset(
                    "data/dataset.data",
                    "data/datasetLabels.data",
                    "data/test.data",
                    "data/testLabels.data"
                );

            Console.WriteLine("Loaded database ({1} images) in {0} ms.",
                dataset.LoadTime,
                dataset.DatasetImages.Count + dataset.TestImages.Count);
            var inputLayerSize = dataset.ImageSize.Width * dataset.ImageSize.Height;

            var network = new ImprovedNeuralNetwork(new[] {inputLayerSize, 30, 10},
                new CrossEntropyCostFunction())
            {
                LearningRate = 0.5,
                RegularizationLambda = 5
            };

            network.FillGaussianRandom();

            var teacher = new Teacher(
                network,
                2, 10,
                dataset.DatasetImages.Take(50000).Cast<TrainSample>().ToList(),
                dataset.TestImages.Cast<TrainSample>().ToList(),
                dataset.DatasetImages.Skip(50000).Cast<TrainSample>().ToList())
            {
                MonitorTrainingCost = true,
                MonitorTrainingAccuracy = true,
                MonitorValidationAccuracy = true
            };


            teacher.Teach();
            teacher.Test(100, true);
            Console.WriteLine("Test data cost: {0:F3}, Test data accuracy: {1:F3}",
                teacher.TestDataCost,
                teacher.TestDataAccuracy * 100);
        }
    }
}