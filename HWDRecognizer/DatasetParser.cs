using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using ml.AI;

namespace HWDRecognizer
{
    public class HWImage : TrainSample
    {
        public byte[] Data;
        public Size Size;
        public int Index;
        public bool IsTest;
        public int Number;

        private double[] _expectedData;
        private double[] _trainData;

        internal HWImage(byte[] data, Size size, int index, bool isTest, int number)
        {
            Data = data;
            Size = size;
            Index = index;
            IsTest = isTest;
            Number = number;
        }

        public override double[] ToExpected()
        {
            if (_expectedData == null)
            {
                _expectedData = new double[10];
                _expectedData [Number] = 1;
            }
            return _expectedData;
        }

        public override bool CheckAssumption(double[] output)
        {
            var counter = 0;
            var sortedOutput = output
                .Select(p => new { value = p, index = counter++ })
                .OrderByDescending(p => p.value)
                .ToArray();

            return sortedOutput[0].index == Number;
        }

        public Volume ToVolume()
        {
            return new Volume(ToBitmap(), false);
        }

        public override double[] ToTrainData()
        {
            if (_trainData == null)
            {
                _trainData = new double[Data.Length];
                for (var i = 0; i < Data.Length; i++)
                    _trainData[i] = Data[i] / 256.0;
            }

            return _trainData;
        }

        public Bitmap ToBitmap()
        {
            var bmp = new Bitmap(Size.Width, Size.Height, PixelFormat.Format8bppIndexed);

            var palette = bmp.Palette;
            var entries = palette.Entries;
            for (var i = 0; i < 256; i++)
            {
                var b = Color.FromArgb((byte)i, (byte)i, (byte)i);
                entries[i] = b;
            }
            bmp.Palette = palette;


            var data = bmp.LockBits(new Rectangle(Point.Empty, Size),
                ImageLockMode.WriteOnly, bmp.PixelFormat);

            var targetStride = data.Stride;
            var scan0 = data.Scan0.ToInt64();
            for (var y = 0; y < Size.Height; y++)
                Marshal.Copy(Data, y * Size.Width, new IntPtr(scan0 + y * targetStride),  Size.Width);

            bmp.UnlockBits(data);
            return bmp;
        }
    }

    public class Dataset
    {
        public List<HWImage> DatasetImages;
        public List<HWImage> TestImages;

        private const UInt32 labelMagicNumber = 0x00000801;
        private const UInt32 datasetMagicNumber = 0x00000803;

        private TimeSpan _loadTime;
        public double LoadTime => _loadTime.TotalMilliseconds;
        public Size ImageSize
        {
            get
            {
                if(DatasetImages.Count == 0 && TestImages.Count == 0)
                    return Size.Empty;

                if (DatasetImages.Count != 0)
                    return DatasetImages[0].Size;

                return TestImages[0].Size;
            }
        }

        public Dataset(
            string datasetFilename,
            string datasetLabelsFilename,
            string testFilename,
            string testLabelsFilename)
        {
            DatasetImages = new List<HWImage>();
            TestImages = new List<HWImage>();

            var startTime = DateTime.Now;
            ReadDataset(DatasetImages, datasetFilename, false);
            ReadDataset(TestImages, testFilename, true);

            ReadLabels(DatasetImages, datasetLabelsFilename);
            ReadLabels(TestImages, testLabelsFilename);

            _loadTime = TimeSpan.FromMilliseconds((DateTime.Now - startTime).TotalMilliseconds);
        }

        private UInt32 GetUInt32(byte[] bytes)
        {
            return (UInt32)
                (bytes[0] << 24 |
                bytes[1] << 16 |
                bytes[2] << 8 |
                bytes[3]);
        }

        private void ReadDataset(ICollection<HWImage> dest, string fileName, bool isTest)
        {
            if(!File.Exists(fileName))
                throw new FileNotFoundException("Unable to open file", fileName);

            var imageSize = Size.Empty;
            using (var reader = new BinaryReader(new FileStream(fileName, FileMode.Open)))
            {
                var int32Buffer = new byte[4];
                reader.Read(int32Buffer, 0, 4);

                if (GetUInt32(int32Buffer) != datasetMagicNumber)
                    throw new InvalidDataException();

                reader.Read(int32Buffer, 0, 4);
                var count = GetUInt32(int32Buffer);

                reader.Read(int32Buffer, 0, 4);
                imageSize.Width = (int) GetUInt32(int32Buffer);

                reader.Read(int32Buffer, 0, 4);
                imageSize.Height = (int) GetUInt32(int32Buffer);

                var dataBuffer = new byte[imageSize.Width * imageSize.Height];
                for (int i = 0; i < count; i++)
                {
                    reader.Read(dataBuffer, 0, dataBuffer.Length);
                    dest.Add(new HWImage(
                        (byte[]) (dataBuffer.Clone()), imageSize, i, isTest, -1));
                }
            }
        }

        private void ReadLabels(List<HWImage> target, string fileName)
        {
            if(!File.Exists(fileName))
                throw new FileNotFoundException("Unable to open file", fileName);

            using (var reader = new BinaryReader(new FileStream(fileName, FileMode.Open)))
            {
                var int32Buffer = new byte[4];
                reader.Read(int32Buffer, 0, 4);

                if (GetUInt32(int32Buffer) != labelMagicNumber)
                    throw new InvalidDataException();

                reader.Read(int32Buffer, 0, 4);
                var count = (int) GetUInt32(int32Buffer);
                if(count != target.Count)
                    throw new DataException("Lengths of target images and labels are not equal");

                for (int i = 0; i < count; i++)
                    target[i].Number = reader.ReadByte();
            }
        }
    }
}