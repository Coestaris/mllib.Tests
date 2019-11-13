using System;
using System.Drawing;
using System.Threading;
using MLLib.AI;
using MLLib.AI.OBNN;
using MLLib.Tests.XORCalculator.Objects;
using OpenTK;
using OpenTK.Graphics.ES11;
using MLLib.WindowHandler;
using MLLib.WindowHandler.Controls;

namespace MLLib.Tests.XORCalculator
{
    internal static class TextureIds
    {
        public const int Button = 0;
        public const int ButtonActive = 1;

        public const int Checkbox = 2;
        public const int CheckboxActive = 3;
        public const int CheckboxChecked = 4;
        public const int CheckboxCheckedActive = 5;
    }

    public sealed class NNVisualizer : WindowHandler.WindowHandler
    {
        //Visualization parameters
        private const int StepsPerFrame = 2;
        private const int ErrorResetFactor = 0;
        private const int StepsDelay = 0;

        private readonly NeuralNetwork _network;
        private readonly Teacher _teacher;

        //Scene objects
        private Checkbox _checkbox1;
        private Checkbox _checkbox2;
        private Neuron[][] _neurons;
        private Axon[][] _axons;
        private InfoRenderer _infoRenderer;

        private StringRenderer _neuronStringRenderer;
        private StringRenderer _buttonStringRenderer;
        private StringRenderer _textRenderer;

        private bool _working;
        private Action _resetFunc;

        public NNVisualizer(Window window, NeuralNetwork network, Teacher teacher, Action ResetFunc) : base(window)
        {
            _resetFunc = ResetFunc;
            _network = network;
            _teacher = teacher;
        }

        protected override void OnUpdate()
        {
            if (_working)
                for (var i = 0; i < StepsPerFrame; i++)
                {
                    Step();
                    Thread.Sleep(StepsDelay);
                }
        }

        private void Step()
        {
            _teacher.Teach();

            DisplayValues();

            if (ErrorResetFactor != 0 && _infoRenderer.Step % ErrorResetFactor == 0)
                _teacher.ResetError();

            _infoRenderer.Error = _teacher.Error;
            _infoRenderer.Step++;
        }

        private void Reset()
        {
            _working = false;

            _teacher.ResetError();
            _resetFunc();
            DisplayValues();

            _infoRenderer.Error = _teacher.Error;
            _infoRenderer.Step = 0;
        }

        private void DisplayValues()
        {
            for(var l = 0; l < _network.Layers.Count; l++)
            for (var n = 0; n < _network.Layers[l].Size; n++)
            {
                _neurons[l][n].Activation = (float) _network.Layers[l].Activations[n];
                _neurons[l][n].Bias = (float) _network.Layers[l].Biases[n];
            }

            for(var l = 0; l < _network.Layers.Count - 1; l++)
            {
                var layer = _network.Layers[l];
                var nextLayer = _network.Layers[l + 1];
                for (var n = 0; n < layer.Size; n++)
                {
                    for (var j = 0; j < nextLayer.Size; j++)
                        _axons[l][n * nextLayer.Size + j].Weight =
                            (float)_network.Layers[l].Weights[n * nextLayer.Size + j];
                }
            }
        }

        private void Manual()
        {
            if (!_working)
            {
                _network.ForwardPass(new double[] {_checkbox1.Checked ? 1 : 0, _checkbox2.Checked ? 1 : 0});
                DisplayValues();
            }
        }

        protected override void OnStart()
        {
            ResourceManager.PushTexture(TextureIds.Button, "button.png");
            ResourceManager.PushTexture(TextureIds.ButtonActive, "buttonActive.png");
            ResourceManager.PushTexture(TextureIds.Checkbox, "checkBox.png");
            ResourceManager.PushTexture(TextureIds.CheckboxActive, "checkBoxActive.png");
            ResourceManager.PushTexture(TextureIds.CheckboxChecked, "checkBoxChecked.png");
            ResourceManager.PushTexture(TextureIds.CheckboxCheckedActive, "checkBoxCheckedActive.png");

            ResourceManager.PushRenderer(_neuronStringRenderer = new StringRenderer(
                StringRenderer.NumericCharSet,
                new Font("DejaVu Sans Mono", 12, FontStyle.Regular),
                Brushes.Black));

            ResourceManager.PushRenderer(_buttonStringRenderer = new StringRenderer(
                StringRenderer.FullCharSet,
                new Font("DejaVu Sans Mono", 16, FontStyle.Regular),
                Brushes.White));

            ResourceManager.PushRenderer(_textRenderer = new StringRenderer(
                StringRenderer.FullCharSet,
                new Font("DejaVu Sans Mono", 16),
                Brushes.White));

            _neurons = new Neuron[_network.Layers.Count][];
            _axons   = new Axon  [_network.Layers.Count][];

            var xStep = Window.Width / (float)(_network.Layers.Count + 1);
            var x = xStep / 2;

            xStep += xStep / (_network.Layers.Count - 1);

            var layerCount = 0;
            foreach (var layer in _network.Layers)
            {
                _neurons[layerCount] = new Neuron[layer.Size];
                if(layer.Weights != null)
                    _axons[layerCount] = new Axon[layer.Weights.Length];

                var yStep = Window.Height / (float) (layer.Size + 1);
                var y = yStep;

                for (var i = 0; i < layer.Size; i++)
                {
                    _neurons[layerCount][i] = new Neuron(30, new Vector2(x, y))
                    {
                        _renderer = _neuronStringRenderer
                    };
                    y += yStep;
                }

                x += xStep;
                layerCount++;
            }

            for(int l = 0; l < _network.Layers.Count - 1; l++)
            {
                var layer = _network.Layers[l];
                var nextLayer = _network.Layers[l + 1];

                for (var i = 0; i < layer.Size; i++)
                {
                    for (var j = 0; j < nextLayer.Size; j++)
                    {
                        _axons[l][i * nextLayer.Size + j] = new Axon(
                            _neurons[l][i].Position,
                            _neurons[l + 1][j].Position,
                            (float) layer.Weights[i * nextLayer.Size + j]
                        );
                        AddObject(_axons[l][i * nextLayer.Size + j]);
                    }
                }
            }

            foreach (var neurons in _neurons)
                foreach (var neuron in neurons)
                    AddObject(neuron);

            AddObject(
                new Button(
                    TextureIds.ButtonActive,
                    TextureIds.Button,
                    new Vector2(65, 30),
                () => _working = true,
                    _buttonStringRenderer,
                    "Start"));

            AddObject(
                new Button(
                    TextureIds.ButtonActive,
                    TextureIds.Button,
                    new Vector2(190, 30),
                    () => _working = false,
                    _buttonStringRenderer,
                    "Stop"));


            AddObject(
                new Button(
                    TextureIds.ButtonActive,
                    TextureIds.Button,
                    new Vector2(315, 30),
                    () => Reset(),
                    _buttonStringRenderer,
                    "Reset"));

            AddObject(
                new Button(
                    TextureIds.ButtonActive,
                    TextureIds.Button,
                    new Vector2(440, 30),
                    () => Step(),
                    _buttonStringRenderer,
                    "Step"));

            AddObject(_checkbox1 = new Checkbox(
                "Input1",
                TextureIds.Checkbox, TextureIds.CheckboxActive,
                TextureIds.CheckboxChecked, TextureIds.CheckboxCheckedActive,
                new Vector2(20, Window.Height - 50),
                (b) => Manual(), _buttonStringRenderer));

            AddObject(_checkbox2 = new Checkbox(
                "Input2",
                TextureIds.Checkbox, TextureIds.CheckboxActive,
                TextureIds.CheckboxChecked, TextureIds.CheckboxCheckedActive,
                new Vector2(20, Window.Height - 20),
                (b) => Manual(), _buttonStringRenderer));

            AddObject(_infoRenderer = new InfoRenderer(_textRenderer, Vector2.One));

            Reset();
        }
    }
}