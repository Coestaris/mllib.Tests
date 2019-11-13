using System.Drawing;
using System.Linq;
using CNNVisualization.Objects;
using ml.AI;
using ml.AI.CNN;
using ml.AI.CNN.Layers;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using WindowHandler;
using WindowHandler.Controls;

namespace CNNVisualization
{
    public class Visualizer : WindowHandler.WindowHandler
    {
        public ConvolutionalNeuralNetwork Network;
        public LayerThumb[][] LayerThumbs;
        public InfoRenderer InfoRenderer;
        public Drawable Drawable;

        public const float DefaultScale = 2.5f;
        public const float IconSize = 24f;
        public const int DrawSize = 400;

        public Visualizer(Window window, ConvolutionalNeuralNetwork network) : base(window)
        {
            Network = network;
        }

        private void RebuildTextures()
        {
            for (var i = 0; i <  Network.Layers.Count; i++)
            {
                var rawVolume = Network.Layers[i].OutVolume.WeightsRaw;
                var min = double.MaxValue;
                var max = double.MinValue;
                for (var j = 0; j < rawVolume.Length; j++)
                {
                    if (rawVolume[j] > max) max = rawVolume[j];
                    if (rawVolume[j] < min) min = rawVolume[j];
                }

                foreach (var layerThumb in LayerThumbs[i])
                    layerThumb.RebuildTexture(new MinMax(min, max));
            }
        }

        private void UpdateInfoRenderer(Volume output)
        {
            var index = 0;
            var sorted = output.WeightsRaw.Select(p => new {index = index++, value = p})
                .OrderByDescending(p => p.value).Take(3).ToList();

            InfoRenderer.Guesses     = sorted.Select(p => p.index).ToArray();
            InfoRenderer.GuessValues = sorted.Select(p => p.value).ToArray();
        }

        public void DrawFunc()
        {
            var volume = Drawable.ToVolume(24);
            var output = Network.ForwardPass(volume);
            RebuildTextures();
            UpdateInfoRenderer(output);
        }

        protected override void OnStart()
        {
            GL.Enable(EnableCap.Texture2D);
            GL.Enable(EnableCap.Blend);

            StringRenderer renderer;
            ResourceManager.PushRenderer(renderer = new StringRenderer(
                StringRenderer.FullCharSet,
                new Font("DejaVu Sans Mono", 12, FontStyle.Regular),
                Brushes.Black));

            var globalBitmap = new Bitmap(Window.Width, Window.Height);
            var ax = DrawSize + Window.Width / Network.Layers.Count;
            var dx = ax - ax / (float)Network.Layers.Count - DrawSize;

            LayerThumbs = new LayerThumb[Network.Layers.Count][];
            for (var i = 0; i < Network.Layers.Count; i++)
            {
                LayerThumbs[i] = new LayerThumb[Network.Layers[i].OutDepth];

                float scale;
                if (Network.Layers[i].OutSize.Width == 1 &&
                    Network.Layers[i].OutSize.Height == 1) scale = DefaultScale * 2;
                else scale = IconSize / Network.Layers[0].OutSize.Width * DefaultScale;

                var dy = Window.Height / 2.0f -
                         (Network.Layers[i].OutSize.Height * scale / 2) / Network.Layers[i].OutDepth;
                for (var j = 0; j < Network.Layers[i].OutDepth; j++)
                {
                    LayerThumbs[i][j] = new LayerThumb(
                        new Vector2(
                            ax + dx * i,
                            (float) (dy + (j - Network.Layers[i].OutDepth / 2) *
                                     (Network.Layers[i].OutSize.Height * scale / 2 * 2.5))),
                        new Vector2(scale, scale),
                        Network.Layers[i], j);

                    AddObject(LayerThumbs[i][j]);

                }
            }

            var texID = ResourceManager.TexturesCount;
            AddObject(Drawable = new Drawable(
                new Vector2(
                    DrawSize / 4.0f,
                    Window.Height / 2.0f - DrawSize / 2.0f),
                DrawSize));
            ResourceManager.PushTexture(texID++, Drawable.Texture);
            Drawable.DrawCallback = DrawFunc;

            using (var gr = Graphics.FromImage(globalBitmap))
            {
                gr.FillRegion(new SolidBrush(Color.FromArgb(255, 94, 91,102)),
                    new Region(new Rectangle(0, 0, Window.Width, Window.Height)));

                gr.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

                var drawPen = new Pen(new SolidBrush(Color.FromArgb(255, 30, 25, 30)));
                float prevScale = 0;

                for (var i = 0; i < LayerThumbs.Length; i++)
                {
                    float scale;

                    if (Network.Layers[i].OutSize.Width == 1 &&
                        Network.Layers[i].OutSize.Height == 1) scale = DefaultScale * 2;
                    else scale = IconSize / Network.Layers[0].OutSize.Width * DefaultScale;


                    var tex = LayerThumbs[i];
                    var prevTex = i != 0 ? LayerThumbs[i - 1] : null;

                    if (prevTex != null)
                    {
                        if (Network.Layers[i] is FullyConnectedLayer ||
                            Network.Layers[i] is ConvolutionalLayer ||
                            Network.Layers[i] is InputLayer)
                        {
                            for (var cur = 0; cur < tex.Length; cur++)
                            for (var prev = 0; prev < prevTex.Length; prev++)
                                gr.DrawLine(drawPen,
                                    new PointF(
                                        tex[cur].Position.X + Network.Layers[i].OutSize.Height * 1.25f * scale / 2,
                                        tex[cur].Position.Y + Network.Layers[i].OutSize.Width * 1.25f * scale / 2),
                                    new PointF(
                                        prevTex[prev].Position.X + Network.Layers[i - 1].OutSize.Height * 1.25f * prevScale / 2,
                                        prevTex[prev].Position.Y + Network.Layers[i - 1].OutSize.Width * 1.25f * prevScale / 2));
                        }
                        else
                        {
                            for (var cur = 0; cur < tex.Length; cur++)
                            {
                                gr.DrawLine(drawPen,
                                    new PointF(
                                        tex[cur].Position.X + Network.Layers[i].OutSize.Height * 1.25f * scale / 2,
                                        tex[cur].Position.Y + Network.Layers[i].OutSize.Width * 1.25f * scale / 2),
                                    new PointF(
                                        prevTex[cur].Position.X + Network.Layers[i - 1].OutSize.Height * 1.25f * prevScale / 2,
                                        prevTex[cur].Position.Y + Network.Layers[i - 1].OutSize.Width * 1.25f * prevScale / 2));
                            }
                        }
                    }


                    prevScale = scale;
                }

                var firstTex = LayerThumbs[0][0];
                gr.DrawLine(drawPen,
                        new PointF(Drawable.Position.X + DrawSize, Drawable.Position.Y),
                        new PointF(firstTex.Position.X, firstTex.Position.Y));

                gr.DrawLine(drawPen,
                    new PointF(Drawable.Position.X + DrawSize, Drawable.Position.Y + DrawSize),
                    new PointF(firstTex.Position.X, firstTex.Position.Y + firstTex.Texture.Size.Width * DefaultScale));
            }

            Texture globalTexture;
            InsertObject(0, new Picture(Vector2.Zero, globalTexture = new Texture(globalBitmap), Vector2.One));

            ResourceManager.PushTexture(texID++, globalTexture);

            InfoRenderer = new InfoRenderer(renderer);
            ResourceManager.PushTexture(texID++, new Texture("button.png"));
            ResourceManager.PushTexture(texID++, new Texture("buttonActive.png"));

            AddObject(new Button(
                    texID - 1, texID - 2,
                    Drawable.Position + new Vector2(DrawSize / 2.0f - 5, DrawSize + 30),
                    Drawable.Reset,
                    renderer, "Reset"));

            AddObject(new NameRenderer(Vector2.Zero, renderer, Network, LayerThumbs, DrawSize,
                new Size(Window.Width, Window.Height)));

            AddObject(InfoRenderer);
            RebuildTextures();
        }
    }
}