using System;
using System.Drawing;
using System.Globalization;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using MLLib.WindowHandler;

namespace MLLib.Tests.XORCalculator.Objects
{
    public class Neuron : DrawableObject
    {
        public float Activation;
        public float Radius;
        public float Bias;

        internal StringRenderer _renderer;

        public Neuron(float radius, Vector2 position) : base(position)
        {
            Radius = radius;
        }

        public override void Draw()
        {
            const double angleDelta = 0.01;

            GL.Begin(PrimitiveType.TriangleFan);
            if(Bias < 0)
                GL.Color3(LerpColor(Color.Black, Color.Red, -Bias / 10));
            else
                GL.Color3(LerpColor(Color.Black, Color.Green, Bias / 10));

            GL.Vertex2(Position);
            for (double a = 0; a < Math.PI * 2; a += angleDelta)
                GL.Vertex2(
                    Position.X + Math.Cos(a) * Radius,
                    Position.Y + Math.Sin(a) * Radius);

            GL.End();

            GL.LineWidth(1);
            GL.Begin(PrimitiveType.LineStrip);
            GL.Color3(Color.Black);
            for (double a = 0; a <= Math.PI * 2; a += angleDelta)
            {
                GL.Vertex2(
                    Position.X + Math.Cos(a) * Radius,
                    Position.Y + Math.Sin(a) * Radius);
            }

            GL.End();

            GL.Color3(LerpColor(Color.Red, Color.Green, Activation));
            DrawCenteredString(Activation.ToString("F3"), _renderer, true, true);
        }
    }
}