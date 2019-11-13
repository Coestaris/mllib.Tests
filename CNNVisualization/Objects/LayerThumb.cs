using System;
using System.Drawing;
using MLLib.AI.CNN;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using MLLib.WindowHandler;

namespace MLLib.Tests.CNNVisualization.Objects
{
    internal struct MinMax
    {
        public MinMax(double min, double max)
        {
            Min = min;
            Max = max;
            Delta = max - min;
        }

        public double Delta;
        public double Min;
        public double Max;
    }

    public class LayerThumb : DrawableObject
    {
        public Texture Texture;
        public Vector2 Scale;

        public CNNLayer Layer;
        public int Depth;

        private int _index;
        private int[] PBOs;
        public int _dataSize;

        private static byte NormalizeColor(double d)
        {
            var b = d * 255;
            if (b > 255) return 255;
            if (b < 0) return 0;
            return (byte) b;
        }

        private unsafe void UpdatePixels(byte* ptr, MinMax layerPeak)
        {
            var rawVolume = Layer.OutVolume.WeightsRaw;
            for (var i = 0; i < _dataSize; i += 3)
            {
                var index = i / 3;
                var v = rawVolume[index * Layer.OutDepth + Depth];

                if (Math.Abs(layerPeak.Delta) <= 1) v -= layerPeak.Min;
                else v = (v - layerPeak.Min) / layerPeak.Delta;

                var b = NormalizeColor(v);
                ptr[i] = b;
                ptr[i + 1] = b;
                ptr[i + 2] = b;
            }
        }

        internal unsafe void RebuildTexture(MinMax layerPeak)
        {
            var pbo1 = PBOs[(_index) % PBOs.Length];
            var pbo2 = PBOs[(_index + 1) % PBOs.Length];
            _index++;

            GL.BindTexture(TextureTarget.Texture2D, Texture.ID);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, pbo2);
            GL.BufferData(BufferTarget.PixelUnpackBuffer, _dataSize, IntPtr.Zero, BufferUsageHint.StreamDraw);
            var ptr = (byte*) GL.MapBuffer(BufferTarget.PixelUnpackBuffer, BufferAccess.WriteOnly);
            if (ptr != null)
            {
                UpdatePixels(ptr, layerPeak);
                GL.UnmapBuffer(BufferTarget.PixelUnpackBuffer);
            }

            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, pbo1);
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0,
                Texture.Size.Width, Texture.Size.Height, PixelFormat.Rgb, PixelType.UnsignedByte, IntPtr.Zero);


            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);
        }

        public LayerThumb(Vector2 position, Vector2 scale, CNNLayer layer, int depth) : base(position)
        {
            Scale = scale;
            Layer = layer;
            Depth = depth;

            var bmp = Layer.ToBitmap(Depth, Color.Cornsilk, Color.Black);
            Texture = new Texture(bmp);
            Window.ResourceManager.PushTexture(Texture);

            PBOs = new int[1];

            _dataSize = Texture.Size.Width * Texture.Size.Height * 3;
            GL.GenBuffers(PBOs.Length, PBOs);

            foreach (var id in PBOs)
            {
                GL.BindBuffer(BufferTarget.PixelUnpackBuffer, id);
                GL.BufferData(BufferTarget.PixelUnpackBuffer, _dataSize, IntPtr.Zero, BufferUsageHint.StreamDraw);
            }

            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);

        }

        public override void Draw()
        {
            DrawTexture(Texture, Position.X, Position.Y, Scale.X, Scale.Y);
        }
    }
}