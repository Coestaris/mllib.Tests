using System;
using System.IO;
using System.Linq;
using ml.AI.RNN;

namespace RNNDemo
{
    internal class Program
    {
        public static Random Random;

        //RNN params
        public const int HiddenCount = 100;
        public const int TrainOutputLen = 100;
        public const double LearningRate = 0.05;

        //Training params
        public const int EpochsCount = 10000;
        public const int CallBackActivation = 100;
        public const int SampleCount = 50;

        public static string EscapeString(string s)
        {
            var str = "";
            foreach (var ch in s)
            {
                switch (ch)
                {
                    case '\n': str += "[\\n]"; break;
                    case '\t': str += "[\\t]"; break;
                    default:   str +=     ch; break;
                }
            }

            return str;
        }

        public static void Main(string[] args)
        {
            Random = new Random();

            var input = File.ReadAllText("input.txt");
            var rnn = new RecurrentNeuralNetwork(input, HiddenCount, TrainOutputLen, LearningRate);
            Console.WriteLine("Input has {0} char in total and {1} distinct chars",
                input.Length, rnn.Vocab.Length);

            var n = 0;
            rnn.Train(EpochsCount, CallBackActivation, (i, loss) =>
            {
                var max = (int) Math.Floor(EpochsCount / (double)CallBackActivation) - 1;

                Console.WriteLine("[{0}/{1}] Generated sample: \"{2}\"", n + 1, max,
                    EscapeString(rnn.Sample(input[Random.Next(0, input.Length)], SampleCount)));

                Console.WriteLine("[{0}/{1}] Iterations: {2}, Loss: {3:F4}", n + 1, max, i, loss);
                n++;
                return false;
            });

            Console.WriteLine("==== LEARNING COMPLETE ====");
            Console.WriteLine("Input empty string to exit");
            while (true)
            {
                Console.Write("Input some character: ");
                var c = Console.ReadLine();
                if(c.Length == 0) return;

                char ch;
                switch (c)
                {
                    case "\\n": ch = '\n'; break;
                    case "\\t": ch = '\t'; break;
                    default: ch = c[0]; break;
                }

                if (!rnn.Vocab.Contains(ch))
                {
                    Console.WriteLine("Vocab doesnt contain character '{0}'. Vocab: [{1}]",
                        ch, string.Join(",", rnn.Vocab.Select(p => $"'{EscapeString(p.ToString())}'")));

                    continue;
                }

                char[] outCh;
                var probability = rnn.GetNextCharProbability(ch, out outCh);
                for (var i = 0; i < 3; i++)
                {
                    var displayChar = EscapeString(outCh[i].ToString());
                    Console.Write("({0:F3}:{1}){2}", probability[i], displayChar, i != 2 ? ", " : "\n");
                }
            }
        }
    }
}