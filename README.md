# Ml.Tests

Small applications using self-written [Machine Leaning Library](https://github.com/Coestaris/ml).
All projects work with both Microsoft Visual Studio and Mono.
For older commit history check library repo.

#### Project roadmap:
```
ML.Tests
└──┬── CNNVisualization  - Visualization of recognizing handwritten numbers using CNN
   ├── HWDRecognizer     - Recognizing MNIST images using NN
   ├── RNNDemo           - Generating text using RNN
   ├── GADemo            - Searching minimum of function and solving XOR using GA
   ├── TennisClassifier  - Classifying task solving with NN
   └── XORCalculator     - Visualization of NN learning to calculate XOR 
```

#### Build and run projects
```bash
git clone --recurse-submodules -j8 https://github.com/Coestaris/ml.Tests
cd ml.Tests
xbuild
```

##### XORCalculator examples:
![](https://raw.githubusercontent.com/Coestaris/ml.Tests/master/images/xor1.gif)
![](https://raw.githubusercontent.com/Coestaris/ml.Tests/master/images/xor2.gif)
##### CNNVisualization example:
![](https://raw.githubusercontent.com/Coestaris/ml.Tests/master/images/cnn.gif)
