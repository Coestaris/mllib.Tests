using System;
using System.Drawing;
using ml.AI;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using WindowHandler;

namespace XORCalculator.Objects
{
    public class Axon : DrawableObject
    {
        public Vector2 Position2;
        public float Weight;

        private const float scaleFactor = 5;

        public Axon(Vector2 position, Vector2 position2, float weight) : base(position)
        {
            Position2 = position2;
            Weight = weight;
        }

        public override void Draw()
        {
            GL.LineWidth(Weight * 2);
            if(Weight < 0)
                GL.Color3(LerpColor(Color.Red, Color.Black, - Weight / scaleFactor));
            else
                GL.Color3(LerpColor(Color.Green, Color.Black, Weight / scaleFactor));

            GL.Begin(BeginMode.Lines);

            GL.Vertex2(Position);
            GL.Vertex2(Position2);

            GL.End();

        }
    }
}