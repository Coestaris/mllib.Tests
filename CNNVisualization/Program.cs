using System.Collections.Generic;
using System.Drawing;
using MLLib.AI.CNN;
using MLLib.AI.CNN.Layers;
using OpenTK.Graphics;
using MLLib.WindowHandler;

namespace MLLib.Tests.CNNVisualization
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            var network = JSONParser.Parse("net.json");
            /*var network = new ConvolutionalNeuralNetwork();
            var layers = new List<CNNLayer>
            {
                new InputLayer(new Size(128, 128), true),
                new ConvolutionalLayer(8, 5, 2, 1),
                new ReLuLayer(),
                new SubsamplingLayer(2, 2, 0),
                new ConvolutionalLayer(16, 5, 2, 1),
                new ReLuLayer(),
                new SubsamplingLayer(3, 3, 0),
                new FullyConnectedLayer(10),
                new SoftmaxLayer(),
            };
            network.PushLayers(layers);
*/

            var window = new Window(1400, 700, "CNNVisualization");
            var handler = new Visualizer(window, network);
            handler.Start();

        }
    }
}