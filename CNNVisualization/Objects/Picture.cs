using MLLib.AI.OBNN;
using OpenTK;
using MLLib.WindowHandler;

namespace MLLib.Tests.CNNVisualization.Objects
{
    public class Picture : DrawableObject
    {
        public Texture Texture;
        public Vector2 Scale;

        public Picture(Vector2 position, Texture texture, Vector2 scale) : base(position)
        {
            Texture = texture;
            Scale = scale;
        }

        public override void Draw()
        {
            DrawTexture(Texture, Position.X, Position.Y, Scale.X, Scale.Y);
        }
    }
}