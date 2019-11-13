using System;
using System.Drawing;
using System.Linq;
using MLLib.AI.CNN;
using MLLib.AI.CNN.Layers;
using OpenTK;
using MLLib.WindowHandler;

namespace MLLib.Tests.CNNVisualization.Objects
{
    internal struct LayerLabel
    {
        public string[] Strings;
        public SizeF[] Sizes;
        public Vector2[] Positions;
    }

    public class NameRenderer : DrawableObject
    {
        private StringRenderer _renderer;

        public ConvolutionalNeuralNetwork Network;
        public LayerThumb[][] Thumbs;
        public int DrawSize;

        private LayerLabel[] layerLabels;

        public NameRenderer(
            Vector2 position,
            StringRenderer renderer,
            ConvolutionalNeuralNetwork network,
            LayerThumb[][] thumbs,
            int drawSize,
            Size windowSize) : base(position)
        {
            _renderer = renderer;
            Network = network;
            Thumbs = thumbs;
            DrawSize = drawSize;

            layerLabels = new LayerLabel[Thumbs.Length + 1];

            layerLabels[0] = new LayerLabel();
            layerLabels[0].Strings = new[] {"Input"};
            layerLabels[0].Sizes = new[] {renderer.MeasureString(layerLabels[0].Strings[0])};
            layerLabels[0].Positions = new[] {
                new Vector2(
                    DrawSize / 4.0f + DrawSize / 2.0f - layerLabels[0].Sizes[0].Width / 2.0f,
                    windowSize.Height - 60),
            };

            var index = 1;
            var up = false;
            foreach (var layer in Network.Layers)
            {
                var name = "";
                if (layer is InputLayer) name = "Downscaled\ninput";
                else if (layer is ConvolutionalLayer) name = "Convolutional\nlayer";
                else if (layer is SubsamplingLayer) name = "Subsampling\nlayer";
                else if (layer is FullyConnectedLayer) name = "Fully connected\nlayer";
                else if (layer is ReLuLayer) name = "ReLu\nlayer";
                else if (layer is SigmoidLayer) name = "Sigmoid\nlayer";
                else if (layer is SoftmaxLayer) name = "Softmax\nlayer";
                else name = "";

                layerLabels[index] = new LayerLabel();
                layerLabels[index].Strings = name.Split('\n');
                layerLabels[index].Sizes = new SizeF[layerLabels[index].Strings.Length];
                layerLabels[index].Positions = new Vector2[layerLabels[index].Strings.Length];
                var strIndex = 0;
                foreach (var str in layerLabels[index].Strings)
                {
                    layerLabels[index].Sizes[strIndex] = renderer.MeasureString(str);

                    var thumbPos = thumbs[index - 1].Last().Position;
                    layerLabels[index].Positions[strIndex] =
                        new Vector2(
                            thumbPos.X + thumbs[index - 1].Last().Texture.Size.Width / 2.0f - layerLabels[index].Sizes[strIndex].Width / 2.0f + 5,
                            windowSize.Height - 40 + layerLabels[index].Sizes[strIndex].Height * strIndex + (up ? 5 : -20));
                    strIndex++;
                }

                index++;
                up = !up;
            }
        }

        public override void Draw()
        {
            for (var i = 0; i < layerLabels.Length; i++)
            for (var str = 0; str < layerLabels[i].Strings.Length; str++)
            {
                _renderer.DrawString(
                    layerLabels[i].Strings[str],
                    layerLabels[i].Positions[str]);
            }
        }
    }
}