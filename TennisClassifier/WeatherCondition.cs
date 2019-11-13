using System;
using MLLib.AI;

namespace MLLib.Tests.TennisClassifier
{
    public enum Temperature
    {
        Cool,
        Mild,
        Hot
    }

    public enum Outlook
    {
        Sunny,
        Overcast,
        Rain
    }

    public enum Humidity
    {
        High,
        Normal
    }

    public class WeatherCondition : TrainSample
    {
        private static Random _random = new Random((int)DateTime.Now.TimeOfDay.TotalMilliseconds);

        public double Outlook; //0 - sunny, .5 - overcast, 1 - rain
        public double Humidity; //0 - normal, 1 - high
        public double Windy; //0 - false, 1 - true
        public double Temperature; //0 - cool, .5 - mild, 1 - hot

        public Outlook GetOutlook()
        {
            if (Outlook > 2 / 3.0) return TennisClassifier.Outlook.Rain;
            if (Outlook > 1 / 3.0) return TennisClassifier.Outlook.Overcast;
            return TennisClassifier.Outlook.Sunny;
        }

        public Humidity GetHumidity()
        {
            if (Humidity > .5) return TennisClassifier.Humidity.High;
            else return TennisClassifier.Humidity.Normal;
        }

        public bool GetWindy()
        {
            return Windy > .5;
        }

        public Temperature GetTemperature()
        {
            if (Temperature > 2 / 3.0) return TennisClassifier.Temperature.Hot;
            if (Temperature > 1 / 3.0) return TennisClassifier.Temperature.Mild;
            return TennisClassifier.Temperature.Cool;
        }

        public WeatherCondition(double outlook, double humidity, double windy, double temperature)
        {
            Outlook = outlook;
            Humidity = humidity;
            Windy = windy;
            Temperature = temperature;
        }

        public WeatherCondition(Outlook outlook, Humidity humidity, bool windy, Temperature temperature)
        {
            Outlook = _random.NextDouble() / 3;
            switch (outlook)
            {
                case TennisClassifier.Outlook.Sunny:
                    break;
                case TennisClassifier.Outlook.Overcast:
                    Outlook += 1 / 3.0;
                    break;
                case TennisClassifier.Outlook.Rain:
                    Outlook += 2 / 3.0;
                    break;
            }

            Humidity = _random.NextDouble() / 2;
            if(humidity == TennisClassifier.Humidity.Normal)
                    Humidity += .5;

            Windy = _random.NextDouble() / 2;
            if (windy)
                Windy += .5;

            Temperature = _random.NextDouble() / 3;
            switch (temperature)
            {
                case TennisClassifier.Temperature.Cool:
                    break;
                case TennisClassifier.Temperature.Mild:
                    Outlook += 1 / 3.0;
                    break;
                case TennisClassifier.Temperature.Hot:
                    Outlook += 2 / 3.0;
                    break;
            }
        }

        public bool ShouldPlay()
        {
            switch (GetTemperature())
            {
                case TennisClassifier.Temperature.Cool:
                    switch (GetOutlook())
                    {
                        case TennisClassifier.Outlook.Sunny:
                            return true;
                        case TennisClassifier.Outlook.Overcast:
                            return true;
                        case TennisClassifier.Outlook.Rain:
                            return !GetWindy();
                    }
                    break;
                case TennisClassifier.Temperature.Mild:
                    switch (GetOutlook())
                    {
                        case TennisClassifier.Outlook.Sunny:
                            return GetWindy();
                        case TennisClassifier.Outlook.Overcast:
                            return true;
                        case TennisClassifier.Outlook.Rain:
                        {
                            switch (GetHumidity())
                            {
                                case TennisClassifier.Humidity.High:
                                    return true;
                                case TennisClassifier.Humidity.Normal:
                                    return !GetWindy();
                            }
                        }
                        break;
                    }
                    break;
                case TennisClassifier.Temperature.Hot:
                    if (GetWindy())
                        return true;
                    else
                    {
                        if (GetHumidity() == TennisClassifier.Humidity.High)
                        {
                            switch (GetOutlook())
                            {
                                case TennisClassifier.Outlook.Sunny:
                                    return false;
                                case TennisClassifier.Outlook.Overcast:
                                    return true;
                                case TennisClassifier.Outlook.Rain:
                                    return false;
                            }
                        }
                        else return true;
                    }
                    break;
            }

            return false;
        }

        private double[] _trainData;
        private double[] _expectedData;

        public override double[] ToTrainData()
        {
            if(_trainData == null)
                _trainData = new[] {Outlook, Humidity, Windy, Temperature};

            return _trainData;
        }

        public override double[] ToExpected()
        {
            if (_expectedData == null)
            {
                var s = ShouldPlay();
                _expectedData = new double[] {s ? 1 : 0};
            }

            return _expectedData;
        }

        public override bool CheckAssumption(double[] output)
        {
            if (ShouldPlay())
                return output[0] > 0.85;

            return output[0] < 0.15;
        }
    }
}