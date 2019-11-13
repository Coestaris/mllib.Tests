using System;
using System.Drawing;
using MLLib.AI;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using MLLib.WindowHandler;

namespace MLLib.Tests.XORCalculator.Objects
{
    public class InfoRenderer : DrawableObject
    {
        internal StringRenderer _renderer;

        internal int Step;
        internal double Error;

        internal Teacher Teacher;

        public InfoRenderer(StringRenderer renderer, Vector2 position) : base(position)
        {
            _renderer = renderer;
        }

        public override void Draw()
        {
            GL.Color3(Color.White);
            _renderer.DrawString("Step: " + Step, new Vector2(5, 50));
            _renderer.DrawString("Error: ", new Vector2(5, 70));
            _renderer.DrawString("Press ESC to Exit", new Vector2(Parent.Width - 220, Parent.Height - 30));

            if (double.IsNaN(Error))
            {
                GL.Color3(Color.White);
                _renderer.DrawString("none", new Vector2(80, 72));
            }
            else
            {
                GL.Color3(LerpColor(Color.Red, Color.Green, (float)Error));
                _renderer.DrawString(Error.ToString("F4"), new Vector2(80, 72));
            }
        }
    }
}