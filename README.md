

 VB.NET:

markdown

# MicroGrad - An Autograd Library for VB.NET

MicroGrad is a powerful autograd library for VB.NET, specifically designed for machine learning and deep learning tasks. Whether you are a seasoned machine learning engineer or a beginner taking your first steps into the world of neural networks, MicroGrad simplifies the process of building and training models.

## Key Features

- **Automatic Differentiation:** MicroGrad's core feature is automatic differentiation, which allows you to compute gradients of mathematical expressions. This is crucial for training machine learning models, as it enables gradient-based optimization techniques like stochastic gradient descent.

- **Neural Network Building Blocks:** MicroGrad provides a collection of neural network building blocks, including layers, activation functions, and attention mechanisms. These components can be easily combined to create complex neural architectures.

- **Flexibility:** With MicroGrad, you have the flexibility to define custom neural network layers and loss functions. This means you can experiment with novel architectures and loss functions tailored to your specific use case.

- **Ease of Use:** MicroGrad is designed to be user-friendly. You can define and train models using familiar VB.NET syntax, making it accessible to both newcomers and experienced developers.

## Installation

You can add MicroGrad to your VB.NET project by cloning this repository or downloading the source code. You may also compile it as a standalone assembly and reference it in your project.

## Usage


Here are some examples of how to use MicroGrad's core components:

### Scalars and Vectors

MicroGrad provides `Scaler` and `Vector` classes for numerical computations. You can create and manipulate scalars and vectors with ease:

```vbnet
Dim a As New Scaler(2.0)
Dim b As New Scaler(3.0)
Dim c As Scaler = a + b
Dim d As Scaler = a * b

Dim v1 As New Vector(New List(Of Double)() From {1.0, 2.0, 3.0})
Dim v2 As New Vector(New List(Of Double)() From {4.0, 5.0, 6.0})
Dim v3 As Vector = v1 + v2


Neural Networks

You can build neural networks using MicroGrad's neural network components. Here's an example of creating a simple feedforward neural network:

``` vbnet

Dim model As New LayerBlock()
model.AddLayer(New LinearLayer(inputSize:=64, outputSize:=128, applyNonLinearActivation:=True))
model.AddLayer(New LinearLayer(inputSize:=128, outputSize:=10))

' Forward pass
Dim input As New Vector(...) ' Your input data
Dim output As Vector = model.Compute(input)
```

## Multi-Head Attention

MicroGrad includes multi-head attention layers, a critical component in transformer-based models:

``` vbnet

Dim attentionLayer As New MultiheadAttentionLayer(inputSize:=256, querySize:=64, keySize:=64, valueSize:=64, heads:=4)

' Attend to queries, keys, and values
Dim queries As New List(Of Scaler)(...)
Dim keys As New List(Of Scaler)(...)
Dim values As New List(Of Scaler)(...)
Dim attendedOutputs As List(Of Scaler) = attentionLayer.Attend(queries, keys, values)
```



### Scaler

The `Scaler` class represents a single scalar value and supports mathematical operations. Here's an example of how to use it:

```vb
Imports MicroGrad

' Create scalars
Dim a As New Scaler(2.0)
Dim b As New Scaler(3.0)

' Perform operations
Dim c As Scaler = a + b
Dim d As Scaler = a * b

' Print results
Console.WriteLine("a + b = " & c.data)
Console.WriteLine("a * b = " & d.data)
```

## Vector

The Vector class represents a collection of scalar values and supports element-wise operations. Here's an example of how to use it:

```vbnet

Imports MicroGrad

' Create vectors
Dim values1 As New List(Of Double)() From {1.0, 2.0, 3.0}
Dim values2 As New List(Of Double)() From {4.0, 5.0, 6.0}
Dim vector1 As New Vector(values1)
Dim vector2 As New Vector(values2)

' Perform element-wise operations
Dim resultAdd As Vector = vector1 + vector2
Dim resultMultiply As Vector = vector1 * vector2

' Print results
Console.WriteLine("vector1 + vector2 = " & String.Join(", ", resultAdd.data))
Console.WriteLine("vector1 * vector2 = " & String.Join(", ", resultMultiply.data))
```
## Models
MicroGrad provides several pre-built models, including Transformer models and MLPs with attention. Here's an example of how to create and use the TransformerModel:

``` vbnet

Imports MicroGrad.Models

' Create a TransformerModel with specified parameters
Dim numBlocks As Integer = 2
Dim encoderHeads As Integer = 2
Dim decoderHeads As Integer = 2
Dim inputSize As Integer = 64
Dim querySize As Integer = 16
Dim keySize As Integer = 16
Dim valueSize As Integer = 16

Dim transformerModel As New TransformerModel(numBlocks, encoderHeads, decoderHeads, inputSize, querySize, keySize, valueSize)

' Define input data as a list of Scalers (example)
Dim inputData As New List(Of Scaler)()
' ... Populate inputData with your data ...

' Compute the output of the TransformerModel
Dim output As List(Of Scaler) = transformerModel.Compute(inputData)

' Access model parameters if needed
Dim modelParameters As List(Of Scaler) = transformerModel.Parameters()
```


Here, we import the TransformerModel from the MicroGrad.Models namespace and create an instance of it with specified parameters. We then define input data as a list of Scaler objects (you should populate it with your data) and compute the output of the model. If you need to access the model's parameters, you can do so using the Parameters() method.

## Contributing

We welcome contributions to MicroGrad. Feel free to open issues, submit pull requests, or provide feedback to help improve the library.

## License

MicroGrad is licensed under the MIT License. See the LICENSE file for details.
