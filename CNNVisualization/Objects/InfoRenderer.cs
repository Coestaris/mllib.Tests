using OpenTK;
using WindowHandler;

namespace CNNVisualization.Objects
{
    public class InfoRenderer : DrawableObject
    {
        private StringRenderer _renderer;
        public int[] Guesses;
        public double[] GuessValues;

        public InfoRenderer(StringRenderer renderer) : base(Vector2.One)
        {
            _renderer = renderer;
        }

        public override void Draw()
        {
            if (Guesses == null || GuessValues == null)
            {
                _renderer.DrawString("Nothing to show =\\", Vector2.Zero);
            }
            else
            {
                _renderer.DrawString($"Most possible guess: {Guesses[0]} ({GuessValues[0]:F3})", new Vector2(0, 0));
                for(int i = 1; i < Guesses.Length; i++)
                    _renderer.DrawString($"Guess #{i + 1}: {Guesses[i]} ({GuessValues[i]:F3})", new Vector2(0, 20 * i));
            }
        }
    }
}