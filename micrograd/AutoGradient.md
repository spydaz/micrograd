### Creating a Vector of Values:


You can create a vector of Value objects representing a list of scalars as follows:

```vbnet

Dim vectorData As New List(Of Double) From {1.0, 2.0, 3.0}
Dim vectorValue As New Value(vectorData)
```

This creates a Value  object vectorValue that contains a vector of three scalar values.

### Activation Functions:


You can apply activation functions to scalar or vector Value objects as needed. For example:

```vbnet

Dim scalarValue As New Value(2.0)
Dim sigmoidResult As Value = scalarValue.Sigmoid() ' Apply sigmoid activation to a scalar

Dim reluResult As Value = vectorValue.Relu() ' Apply ReLU activation to a vector
```

### Mathematical Operations:

You can perform mathematical operations on scalar or vector Value objects using overloaded operators. For example:

```vbnet

Dim scalarValue1 As New Value(2.0)
Dim scalarValue2 As New Value(3.0)

Dim additionResult As Value = scalarValue1 + scalarValue2 ' Scalar addition

Dim vectorValue1 As New Value(New List(Of Double) From {1.0, 2.0, 3.0})
Dim vectorValue2 As New Value(New List(Of Double) From {4.0, 5.0, 6.0})

Dim multiplicationResult As Value = vectorValue1 * vectorValue2 ' Element-wise vector multiplication
```

### Loss Function:

You can use the MeanSquaredError function to compute the mean squared error loss:

```vbnet

Dim prediction As New Value(2.5) ' Example prediction
Dim target As Double = 3.0 ' Example target value

Dim mseLoss As Value = prediction.MeanSquaredError(target)
```

### Softmax and Log-Softmax:

You can apply softmax and log-softmax functions to a vector of logits as follows:

```vbnet

Dim logits As New Value(New List(Of Double) From {1.0, 2.0, 3.0})

Dim softmaxResult As Value = logits.Softmax()
Dim logSoftmaxResult As Value = logits.LogSoftmax()
```

These operations will return Value objects containing the softmax or log-softmax values for each element in the input vector.

### Backpropagation:

After performing forward operations, you can calculate gradients and perform backpropagation using the backward method:

```vbnet

' Assuming you've computed a loss and want to backpropagate
mseLoss.grad = 1.0 ' Set the gradient of the loss
mseLoss.backward() ' Perform backpropagation to compute gradients for all involved Value objects
```

### Softmax and LogSoftmax
To use the Softmax and LogSoftmax functions in your Value class, you can follow these steps:
Create Value Objects for Logits: First, create Value objects for your input logits. These logits represent the unnormalized scores for different classes in a classification problem. Each element in the logits corresponds to a class.

```vbnet
Dim logits As New List(Of Value)()
logits.Add(New Value(2.0)) ' Example logits for class 1
logits.Add(New Value(1.0)) ' Example logits for class 2
logits.Add(New Value(0.1)) ' Example logits for class 3
```

### Apply Softmax or LogSoftmax: 
You can apply the Softmax or LogSoftmax functions to the logits as follows:
```vbnet
Dim softmaxResult As Value = logits(0).Softmax()
Dim logSoftmaxResult As Value = logits(0).LogSoftmax()
In this example, we've applied the functions to the first element of the logits list, but you can apply them to any element depending on your specific use case.
```
### Access the Result: 
The softmaxResult and logSoftmaxResult objects now contain the softmax or log-softmax values for each element in the input vector. You can access these values as follows:
```vbnet
Dim softmaxValues As List(Of Double) = softmaxResult.data
Dim logSoftmaxValues As List(Of Double) = logSoftmaxResult.data
```
The softmaxValues and logSoftmaxValues lists will contain the softmax or log-softmax values for each class, respectively.
Here's a complete example that demonstrates how to use the Softmax and LogSoftmax functions with a list of logits:
```vbnet
Dim logits As New List(Of Value)()
logits.Add(New Value(2.0)) ' Example logits for class 1
logits.Add(New Value(1.0)) ' Example logits for class 2
logits.Add(New Value(0.1)) ' Example logits for class 3

' Apply softmax and log-softmax
Dim softmaxResult As Value = logits(0).Softmax()
Dim logSoftmaxResult As Value = logits(0).LogSoftmax()

' Access the results
Dim softmaxValues As List(Of Double) = softmaxResult.data
Dim logSoftmaxValues As List(Of Double) = logSoftmaxResult.data

' Print the results
Console.WriteLine("Softmax Values:")
For Each value In softmaxValues
    Console.WriteLine(value)
Next

Console.WriteLine("Log-Softmax Values:")
For Each value In logSoftmaxValues
    Console.WriteLine(value)
Next
```
This example demonstrates how to use these functions to compute softmax and log-softmax values for a list of logits, which is common in classification tasks in machine learning.

### SequenceCrossEntropy loss:

SequenceCrossEntropy loss function for sequence tasks, apply the exponentiation and division operators, and use the GELU activation function on Value objects as needed in your Visual Basic projects.

Here's an example of how to use these extended functionalities:

```vbnet

' Create a Value object
Dim x As New Value(2.0)

' Calculate GELU
Dim geluResult As Value = x.GELU()
Console.WriteLine($"GELU(x) = {geluResult.data}")

' Calculate x^3
Dim xCubed As Value = x ^ 3.0
Console.WriteLine($"x^3 = {xCubed.data}")

' Calculate Sequence Cross-Entropy
Dim predictedSequence As New List(Of Double) From {0.2, 0.7, 0.5}
Dim targetSequence As New List(Of Double) From {0.3, 0.6, 0.4}
Dim sequenceLoss As Value = x.SequenceCrossEntropy(targetSequence)
Console.WriteLine($"Sequence Cross-Entropy Loss = {sequenceLoss.data}")
```
