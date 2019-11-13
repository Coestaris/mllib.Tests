using ml.AI.OBNN;
using OpenTK;
using WindowHandler;

namespace CNNVisualization.Objects
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