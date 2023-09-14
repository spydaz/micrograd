
Imports System.Windows.Forms
Imports Micrograd.AutoGradient.engine

Namespace AutoGradient
    'Autograd engine (with a bite! :)).
    'Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG
    'and a small neural networks library on top of it with a PyTorch-like API. 
    'Model By Andrew karpathy
    Namespace engine

        ''' <summary>
        ''' Represents a scalar value and its gradient used for autodifferentiation.
        ''' </summary>
        Public Class Value
            Public _backward As Action = Sub()
                                         End Sub

            Public data As Double
            Public grad As Double
            Private _op As String
            Private _prev As HashSet(Of Value)
            ''' <summary>
            ''' Initializes a new instance of the <see cref="Value"/> class with the specified data value and optional child nodes.
            ''' </summary>
            ''' <param name="data">The initial value of the scalar.</param>
            ''' <param name="_children">Optional. A collection of child nodes in the computation graph.</param>
            ''' <param name="_op">Optional. The operation that produced this node.</param>
            Public Sub New(data As Double, Optional _children As IEnumerable(Of Value) = Nothing, Optional _op As String = "New")
                Me.data = data
                Me.grad = 0
                Me._op = _op
                Me._prev = If(_children IsNot Nothing, New HashSet(Of Value)(_children), New HashSet(Of Value)())
            End Sub
            Public Sub New(data As Value, Optional _children As IEnumerable(Of Value) = Nothing, Optional _op As String = "New")
                Me.data = data.data
                Me.grad = 0
                Me._op = _op
                Me._prev = If(_children IsNot Nothing, New HashSet(Of Value)(_children), New HashSet(Of Value)())
            End Sub
            Public Property SGD As Double
            ''' <summary>
            ''' Calculates the exponential (e^x) of the data.
            ''' </summary>
            ''' <returns>A new <see cref="Value"/> containing the exponential value.</returns>
            Public Shared Function Exp(x As Value) As Value
                Dim expValue As Double = Math.Exp(x.data)
                ' The gradient update for the exponential function is exp(x) * gradient
                Dim gradientUpdate As Double = expValue * x.grad
                Return New Value(expValue, x, "Exp") With {.SGD = gradientUpdate}
            End Function
            Public Function MeanSquaredError(target As Value) As Value
                Dim loss As Value = Me - target
                Return loss * loss
            End Function



            Public Shared Operator -(left As Value, right As Value) As Value
                Dim out = New Value(left.data - right.data, {left, right}, "-") ' Use "-" operation
                out._backward = Sub()
                                    left.grad += 1                ' Gradient of left with respect to the output is 1
                                    right.grad += -1              ' Gradient of right with respect to the output is -1
                                End Sub
                Return out
            End Operator


            Public Shared Operator -(left As Value, right As Double) As Value
                Dim out = New Value(left.data - right, {left, New Value(right)}, "-") ' Use "-" operation
                out._backward = Sub()
                                    left.grad += 1              ' Gradient of left with respect to the output is 1
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator

            Public Shared Operator -(left As Value, right As Integer) As Value
                Dim out = New Value(left.data - right, {left, New Value(right)}, "-") ' Use "-" operation
                out._backward = Sub()
                                    left.grad += 1              ' Gradient of left with respect to the output is 1
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator


            Public Shared Operator -(left As Value, right As List(Of Value)) As List(Of Value)
                Dim result As New List(Of Value)()

                For Each v In right
                    Dim newValue As New Value(left - v, {left, v}, "-") ' Use "-" operation

                    ' Update gradients correctly
                    newValue._backward = Sub()
                                             left.grad += v.data - newValue.grad     ' Gradient of left with respect to the output is 1
                                             v.grad += -left.data - newValue.grad    ' Gradient of v with respect to the output is -1
                                         End Sub

                    result.Add(newValue)
                Next

                Return result
            End Operator


            '*
            Public Shared Operator *(left As Value, right As Value) As Value
                Dim out = New Value(left.data * right.data, {left, right}, "*") ' Use "*" operation
                out._backward = Sub()
                                    left.grad += right.data * out.grad    ' Gradient of left with respect to the output is right.data
                                    right.grad += left.data * out.grad    ' Gradient of right with respect to the output is left.data
                                End Sub
                Return out
            End Operator


            Public Shared Operator *(left As Value, right As Double) As Value
                Dim out = New Value(left.data * right, {left, New Value(right)}, "*") ' Use "*" operation
                out._backward = Sub()
                                    left.grad += right * out.grad   ' Gradient of left with respect to the output is right
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator


            Public Shared Operator *(left As Value, right As Integer) As Value
                Dim out = New Value(left.data * right, {left, New Value(right)}, "*") ' Use "*" operation
                out._backward = Sub()
                                    left.grad += right * out.grad   ' Gradient of left with respect to the output is right
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator


            Public Shared Operator *(left As Value, right As List(Of Value)) As List(Of Value)
                Dim result As New List(Of Value)()

                For Each v In right
                    Dim newValue As New Value(left * v, {left, v}, "*")
                    newValue._backward = Sub()
                                             left.grad += v.data * newValue.grad
                                             v.grad += left.data * newValue.grad
                                         End Sub
                    result.Add(newValue)
                Next

                Return result
            End Operator

            '/
            Public Shared Operator /(left As Value, right As Value) As Value
                Dim out = New Value(left.data / right.data, {left, right}, "/") ' Use "/" operation
                out._backward = Sub()
                                    left.grad += 1 / right.data * out.grad     ' Gradient of left with respect to the output is 1 / right.data
                                    right.grad += -left.data / (right.data * right.data) * out.grad ' Gradient of right with respect to the output is -left.data / (right.data^2)
                                End Sub
                Return out
            End Operator


            Public Shared Operator /(left As Value, right As Double) As Value
                Dim out = New Value(left.data / right, {left, New Value(right)}, "/") ' Use "/" operation
                out._backward = Sub()
                                    left.grad += 1 / right * out.grad ' Gradient of left with respect to the output is 1 / right
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator

            Public Shared Operator /(left As Value, right As Integer) As Value
                Dim out = New Value(left.data / right, {left, New Value(right)}, "/") ' Use "/" operation
                out._backward = Sub()
                                    left.grad += 1 / right * out.grad ' Gradient of left with respect to the output is 1 / right
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator


            Public Shared Operator /(left As Value, right As List(Of Value)) As List(Of Value)
                Dim result As New List(Of Value)()

                For Each v In right
                    Dim newValue As New Value(left / v, {left, v}, "/") ' Use "/" operation

                    ' Update gradients correctly
                    newValue._backward = Sub()
                                             left.grad += 1 / v.data * newValue.grad  ' Gradient of left with respect to the output is 1 / v.data
                                             v.grad += -left.data / (v.data * v.data) * newValue.grad  ' Gradient of v with respect to the output is -left.data / (v.data * v.data)
                                         End Sub

                    result.Add(newValue)
                Next

                Return result
            End Operator


            Public Shared Operator ^(left As Value, exponent As Double) As Value

                Dim out As New Value(Math.Pow(left.data, exponent), {left}, "Power")
                Dim gradientFactor As Double = exponent * Math.Pow(left.data, exponent - 1)

                out._backward = Sub()
                                    left.grad += exponent * out.grad
                                    exponent += left.data * out.grad
                                End Sub




                Return New Value(out, left, $"^{exponent}") With {.grad = left.grad * gradientFactor}
            End Operator

            Public Shared Operator ^(left As Value, exponent As Integer) As Value
                ' Exponentiation operator (^)

                Dim out As New Value(Math.Pow(left.data, exponent), {left}, "Power")
                Dim gradientFactor As Double = exponent * Math.Pow(left.data, exponent - 1)

                out._backward = Sub()
                                    left.grad += exponent * out.grad
                                    exponent += left.data * out.grad
                                End Sub




                Return New Value(out, left, $"^{exponent}") With {.grad = left.grad * gradientFactor}
            End Operator

            Public Shared Operator ^(left As Value, exponent As Value) As Value
                ' Exponentiation operator (^)
                Dim out As New Value(Math.Pow(left.data, exponent.data), {left, exponent}, "Power")
                Dim gradientFactor As Double = exponent.data * Math.Pow(left.data, exponent.data - 1)

                out._backward = Sub()
                                    left.grad += exponent.data * out.grad
                                    exponent.grad += left.data * out.grad
                                End Sub




                Return New Value(out, left, $"^{exponent}") With {.grad = left.grad * gradientFactor}
            End Operator

            '+
            Public Shared Operator +(left As Value, right As Value) As Value
                Dim out = New Value(left.data + right.data, {left, right}, "+")
                out._backward = Sub()
                                    left.grad += out.grad
                                    right.grad += out.grad
                                End Sub
                Return out
            End Operator
            Public Shared Operator +(left As Value, right As Double) As Value
                Dim out = New Value(left.data + right, {left, New Value(right)}, "+")
                out._backward = Sub()
                                    left.grad += 1 * out.grad
                                End Sub
                Return out
            End Operator
            Public Shared Operator +(left As Value, right As Integer) As Value
                Dim out = New Value(left.data + right, {left, New Value(right)}, "+")
                out._backward = Sub()
                                    left.grad += 1 * out.grad
                                End Sub
                Return out
            End Operator
            Public Shared Operator +(left As Value, right As List(Of Value)) As List(Of Value)
                Dim result As New List(Of Value)()

                For Each v In right
                    Dim newValue As New Value(left + v, {left, v}, "+") ' Use "+" operation

                    ' Update gradients correctly
                    newValue._backward = Sub()
                                             left.grad += 1 * newValue.grad  ' Gradient of left with respect to the output is 1
                                             v.grad += 1 * newValue.grad    ' Gradient of v with respect to the output is 1
                                         End Sub

                    result.Add(newValue)
                Next

                Return result
            End Operator
            ''' <summary>
            ''' Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG.
            ''' Computes the gradients using the backward pass.
            ''' </summary>
            Public Sub backward()
                ' Topological order all of the children in the graph
                Dim topo As New List(Of Value)()
                Dim visited As New HashSet(Of Value)()

                BuildTopology(Me, topo, visited)

                ' Go one variable at a time and apply the chain rule to get its gradient
                Me.grad = 1
                topo.Reverse()

                For Each v In topo
                    v._backward()
                Next

            End Sub

            ''' <summary>
            ''' Applies the Gaussian Error Linear Unit (GELU) activation function to this value.
            ''' </summary>
            ''' <returns>A new <see cref="Value"/> containing the result of the GELU activation.</returns>
            Public Function GELU() As Value
                ' Gaussian Error Linear Unit (GELU) activation function
                Dim x As Double = data
                Dim coefficient As Double = 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.Pow(x, 3.0))))
                Dim geluValue As Double = x * coefficient
                Return New Value(geluValue, Me, "GELU")
            End Function
            ''' <summary>
            ''' Initializes a TreeView to visualize the computation graph starting from this Value object.
            ''' </summary>
            ''' <param name="rootValue">The root Value object of the computation graph.</param>
            ''' <returns>A TreeView representing the computation graph.</returns>
            Public Function InitializeGraphTreeView(rootValue As Value) As Windows.Forms.TreeView
                ' Create the root node for the TreeView
                Dim rootTreeNode As New TreeNode(rootValue.ToString())

                ' Populate the TreeView starting from the root node
                PopulateGraphTreeView(rootValue, rootTreeNode)

                ' Create a new TreeView and add the root node to it
                Dim graphTreeView As New Windows.Forms.TreeView
                graphTreeView.Nodes.Add(rootTreeNode)

                Return graphTreeView
            End Function

            Public Function LOGSigmoid() As Value
                ' Logarithmic Sigmoid activation function
                Dim logSigmoidValue = -Math.Log(1 + Math.Exp(-data))
                Return New Value(logSigmoidValue, Me, "LOGSigmoid")
            End Function

            ''' <summary>
            ''' Applies the Log-Softmax operation to this vector of values.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the log-softmax values.</returns>
            Public Function LogSoftmax() As Vector
                ' Log-Softmax operation
                ' Assuming this is used in a vector context, where data represents logits
                Dim expValues As List(Of Double) = _prev.Select(Function(child) Math.Exp(child.data)).ToList()
                Dim expSum = expValues.Sum()

                ' Compute log-softmax values for each element in the vector
                Dim logSoftmaxValues As List(Of Double) = expValues.Select(Function(exp) Math.Log(exp / expSum)).ToList()

                ' Construct a new Value object for the log-softmax result
                Return New Vector(logSoftmaxValues, Me, "LogSoftmax")
            End Function

            ''' <summary>
            ''' Applies the Logarithmic Softmin activation function to this vector of values.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the log-softmin values.</returns>
            Public Function LogSoftmin() As Vector
                ' Logarithmic Softmin activation function
                Dim expValues As List(Of Double) = _prev.Select(Function(child) Math.Exp(child.data)).ToList()
                Dim expSum = expValues.Sum()

                ' Compute log-softmin values for each element in the vector
                Dim logSoftminValues As List(Of Double) = expValues.Select(Function(exp) -Math.Log(exp / expSum)).ToList()

                ' Construct a new Value object for the log-softmin result
                Return New Vector(logSoftminValues, Me, "LogSoftmin")
            End Function
            ''' <summary>
            ''' Computes the Mean Squared Error (MSE) loss with a target value.
            ''' </summary>
            ''' <param name="target">The target value for computing MSE.</param>
            ''' <returns>A new <see cref="Value"/> containing the MSE loss.</returns>
            Public Function MeanSquaredError(target As Double) As Value
                ' Mean Squared Error loss function
                Dim loss = 0.5 * Math.Pow(data - target, 2)
                Return New Value(loss, Me, "MSE Loss")
            End Function

            Public Sub PopulateGraphTreeView(rootValue As Value, parentNode As TreeNode)
                Dim currentNode As New TreeNode(rootValue.ToString())
                parentNode.Nodes.Add(currentNode)

                ' Recursively traverse child nodes (children of the current Value)
                For Each childValue In rootValue._prev
                    PopulateGraphTreeView(childValue, currentNode)
                Next
            End Sub
            ''' <summary>
            ''' Applies the Rectified Linear Unit (ReLU) activation function to this value.
            ''' </summary>
            ''' <returns>A new <see cref="Value"/> containing the result of the ReLU activation.</returns>
            Public Function Relu() As Value
                ' Rectified Linear Unit (ReLU) activation function
                Return New Value(Math.Max(0, data), Me, "ReLU")
            End Function
            ''' <summary>
            ''' Applies the Sigmoid activation function to this value.
            ''' </summary>
            ''' <returns>A new <see cref="Value"/> containing the result of the sigmoid activation.</returns>
            Public Function Sigmoid() As Value
                ' Sigmoid activation function
                Dim sigmoidValue = 1 / (1 + Math.Exp(-data))
                Return New Value(sigmoidValue, Me, "Sigmoid")
            End Function

            ''' <summary>
            ''' Applies the Softmax operation to this vector of values.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the softmax values.</returns>
            Public Function Softmax() As Vector
                ' Softmax operation
                ' Assuming this is used in a vector context, where data represents logits
                Dim expValues As List(Of Double) = _prev.Select(Function(child) Math.Exp(child.data)).ToList()
                Dim expSum = expValues.Sum()

                ' Compute softmax values for each element in the vector
                Dim softmaxValues As List(Of Double) = expValues.Select(Function(exp) exp / expSum).ToList()

                ' Construct a new Value object for the softmax result
                Return New Vector(softmaxValues, Me, "Softmax")
            End Function

            ''' <summary>
            ''' Applies the Softmin activation function to this vector of values.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the softmin values.</returns>
            Public Function Softmin() As Vector
                ' Softmin activation function
                Dim expValues As List(Of Double) = _prev.Select(Function(child) Math.Exp(child.data)).ToList()
                Dim expSum = expValues.Sum()

                ' Compute softmin values for each element in the vector
                Dim softminValues As List(Of Double) = expValues.Select(Function(exp) exp / expSum).ToList()

                ' Construct a new Value object for the softmin result
                Return New Vector(softminValues, Me, "Softmin")
            End Function

            Public Sub StochasticGradientDescent(Optional learningRate As Double = 0.001, Optional Inplace As Boolean = False)

                ' Topological order all of the children in the graph
                Dim topo As New List(Of Value)()
                Dim visited As New HashSet(Of Value)()

                BuildTopology(Me, topo, visited)

                ' Go one variable at a time and apply the chain rule to get its gradient

                topo.Reverse()

                For Each v In topo
                    v._backward()
                    If Inplace - True Then

                        ' Update each Value object with its gradient using stochastic gradient descent (SGD)
                        v.data -= learningRate * v.grad
                    Else

                        ' Update each Value object with its gradient using stochastic gradient descent (SGD)
                        v.SGD -= learningRate * v.grad
                    End If

                Next

            End Sub

            Public Function SumWeightedInputsList(x As List(Of Value), w As List(Of Value), b As Value) As Value



                ' Ensure that the number of inputs matches the number of weights
                If x.Count <> w.Count Then
                    Throw New ArgumentException("Number of inputs must match the number of weights.")
                End If

                Dim weightedSum As Value = b

                For i As Integer = 0 To x.Count - 1
                    weightedSum += x(i) * w(i)
                Next

                Return weightedSum
            End Function
            ''' <summary>
            ''' Applies the hyperbolic tangent (tanh) activation function to this value.
            ''' </summary>
            ''' <returns>A new <see cref="Value"/> containing the result of the tanh activation.</returns>
            Public Function Tanh() As Value
                ' Hyperbolic tangent (tanh) activation function
                Dim tanhValue = Math.Tanh(data)
                Return New Value(tanhValue, Me, "Tanh")
            End Function

            Public Overrides Function ToString() As String
                Return $"Value(data={Me.data}, grad={Me.grad})"
            End Function

            Private Sub BuildTopology(v As Value, ByRef topo As List(Of Value), ByRef visited As HashSet(Of Value))
                If Not visited.Contains(v) Then
                    visited.Add(v)
                    For Each child In v._prev
                        BuildTopology(child, topo, visited)
                    Next
                    topo.Add(v)
                End If
            End Sub
        End Class




    End Namespace
    Public Module Example

        Sub Main()
            ' Define a simple neural network architecture
            Dim inputSize As Integer = 2
            Dim hiddenSize As Integer = 2
            Dim outputSize As Integer = 1
            Dim learningRate As Double = 0.1
            Dim epochs As Integer = 1000

            ' Create input data
            Dim inputData As New Value(0.2)
            Dim target As New Value(0.8)

            ' Define model parameters (weights and biases)
            Dim w1 As New Value(0.5)
            Dim b1 As New Value(0.1)
            Dim w2 As New Value(0.3)
            Dim b2 As New Value(-0.2)

            ' Training loop
            For epoch = 1 To epochs
                ' Forward pass
                Dim z1 As Value = inputData * w1 + b1
                Dim a1 As Value = z1.Relu() ' Apply ReLU activation
                Dim z2 As Value = a1 * w2 + b2
                Dim output As Value = z2.Sigmoid() ' Apply Sigmoid activation

                ' Calculate loss (mean squared error)
                Dim loss As Value = output.MeanSquaredError(target.data)

                ' Backpropagation
                loss.backward()

                ' Update model parameters using SGD
                w1.SGD = -learningRate * w1.grad
                b1.SGD = -learningRate * b1.grad
                w2.SGD = -learningRate * w2.grad
                b2.SGD = -learningRate * b2.grad

                ' Reset gradients for the next iteration
                w1.grad = 0
                b1.grad = 0
                w2.grad = 0
                b2.grad = 0

                ' Print loss for monitoring training progress
                If epoch Mod 100 = 0 Then
                    Console.WriteLine($"Epoch {epoch}, Loss: {loss.data}")
                End If
            Next

            ' After training, perform inference
            Dim testInput As New Value(0.4)
            Dim z1Test As Value = testInput * w1 + b1
            Dim a1Test As Value = z1Test.Relu()
            Dim z2Test As Value = a1Test * w2 + b2
            Dim outputTest As Value = z2Test.Sigmoid()

            Console.WriteLine($"Inference Result: {outputTest.data}")

            Console.ReadLine()
        End Sub



        ' Helper function to generate random weight matrices
        Function RandomMatrix(rows As Integer, cols As Integer) As Double(,)
            Dim irand As New Random
            Dim matrix(rows - 1, cols - 1) As Double

            For i = 0 To rows - 1
                For j = 0 To cols - 1
                    matrix(i, j) = irand.NextDouble()
                Next
            Next

            Return matrix
        End Function
    End Module



End Namespace

