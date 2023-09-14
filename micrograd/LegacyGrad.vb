Imports SpydazWebAI.AutoGradient.engine

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


        ''' <summary>
        ''' Represents a vector of values and provides operations for element-wise arithmetic and activation functions.
        ''' </summary>
        Public Class Vector
            Inherits Value
            Public Shadows data As List(Of Value)
            ''' <summary>
            ''' Initializes a new instance of the <see cref="Vector"/> class with the specified data values and optional child nodes.
            ''' </summary>
            ''' <param name="data">The list of initial values for the vector.</param>
            ''' <param name="_children">Optional. A collection of child nodes in the computation graph.</param>
            ''' <param name="_op">Optional. The operation that produced this node.</param>
            Public Sub New(data As List(Of Double), Optional _children As IEnumerable(Of Value) = Nothing, Optional _op As String = "")
                MyBase.New(New Double, _children, _op)
                For Each item In data
                    Me.data.Add(New Value(item, If(_children IsNot Nothing, New HashSet(Of Value)(_children), New HashSet(Of Value)()), _op))
                Next
            End Sub

            Public Shared Operator -(left As Vector, right As Vector) As Vector
                ' Overload the subtraction operator to perform element-wise subtraction between two vectors of Values
                If left.data.Count <> right.data.Count Then
                    Throw New ArgumentException("Both vectors must have the same number of elements.")
                End If

                Dim resultData As New List(Of Double)()

                For i As Integer = 0 To left.data.Count - 1
                    resultData.Add(left.data(i).data - right.data(i).data)
                Next

                ' Create a new Vector with the result data
                Dim resultVector As New Vector(resultData, {left, right}, "-")

                ' Define the backward function to update gradients
                resultVector._backward = Sub()
                                             For i As Integer = 0 To left.data.Count - 1
                                                 left.data(i).grad += 1.0 * resultVector.grad
                                                 right.data(i).grad -= 1.0 * resultVector.grad
                                             Next
                                         End Sub

                Return resultVector
            End Operator

            Public Shared Function OuterProduct(vector1 As Vector, vector2 As Vector) As List(Of Value)
                ' Compute the outer product of two vectors
                Dim result As New List(Of Value)()

                For Each v1 In vector1.data
                    For Each v2 In vector2.data
                        Dim newValue As New Value(v1.data * v2.data)
                        result.Add(newValue)
                    Next
                Next

                Return result
            End Function

            Public Shared Function ApplyFunction(inputVector As Vector, func As Func(Of Value, Value)) As Vector
                ' Apply a function element-wise to a vector
                Dim result As New Vector(New List(Of Double))

                For Each item In inputVector.data
                    Dim newValue As Value = func(item)
                    result.data.Add(newValue)
                Next

                Return result
            End Function

            Public Shared Shadows Operator -(left As Vector, right As Value) As List(Of Value)
                ' Overload the subtraction operator to subtract a Value from each element in a vector of Values
                Dim result As New List(Of Value)()

                For Each v In left.data
                    Dim newValue As New Value(v.data - right.data, {v, right}, "-")

                    ' Calculate gradients correctly
                    newValue._backward = Sub()
                                             v.grad += 1.0 * newValue.grad
                                             right.grad -= 1.0 * newValue.grad
                                         End Sub

                    result.Add(newValue)
                Next

                Return result
            End Operator




            Public Shared Shadows Operator -(left As Vector, right As List(Of Double)) As List(Of Value)
                ' Overload the subtraction operator to perform element-wise subtraction between a vector of Values and a List(Of Double)
                If left.data.Count <> right.Count Then
                    Throw New ArgumentException("Both lists must have the same number of elements.")
                End If

                Dim result As New List(Of Value)()

                For i As Integer = 0 To left.data.Count - 1
                    Dim newValue As New Value(left.data(i) - right(i), {left}, "-")

                    ' Calculate gradients correctly
                    newValue._backward = Sub()
                                             left.data(i).grad += 1.0 * newValue.grad
                                         End Sub

                    result.Add(newValue)
                Next

                Return result
            End Operator

            Public Shared Shadows Operator *(left As Vector, right As Value) As List(Of Value)
                ' Overload the multiplication operator to multiply a vector of Values by a Value
                Dim result As New List(Of Value)()

                For Each v In left.data
                    Dim newValue As New Value(v.data * right.data, {v, right}, "*") ' Update the gradient calculation
                    newValue._backward = Sub()
                                             v.grad += right.data * newValue.grad ' Gradient update for left vector
                                             right.grad += v.data * newValue.grad ' Gradient update for right value
                                         End Sub
                    result.Add(newValue)
                Next

                Return result
            End Operator

            Public Shared Shadows Operator *(left As Vector, right As List(Of Double)) As List(Of Value)
                If left.data.Count <> right.Count Then
                    Throw New ArgumentException("Both lists must have the same number of elements.")
                End If

                Dim result As New List(Of Value)()

                For i As Integer = 0 To left.data.Count - 1
                    Dim newValue As New Value(left.data(i) * right(i), {left, New Value(right(i))}, "*") ' Update the gradient calculation
                    newValue._backward = Sub()
                                             left.data(i).grad += right(i) * newValue.grad ' Gradient update for left vector element
                                             right(i) += left.data(i).data * newValue.grad ' Gradient update for right list element
                                         End Sub
                    result.Add(newValue)
                Next

                Return result
            End Operator

            Public Shared Shadows Operator *(left As Vector, right As List(Of Value)) As List(Of Value)
                If left.data.Count <> right.Count Then
                    Throw New ArgumentException("Both lists must have the same number of elements.")
                End If

                Dim result As New List(Of Value)()

                For i As Integer = 0 To left.data.Count - 1
                    ' Multiply the elements and create a new Value object
                    Dim newValue As New Value(left.data(i) * right(i).data, {left, New Value(right(i).data)}, "*")

                    ' Define the backward function to update gradients
                    newValue._backward = Sub()
                                             left.data(i).grad += right(i).data * newValue.grad ' Gradient update for the left vector element
                                             right(i).grad += left.data(i).data * newValue.grad ' Gradient update for the right list element
                                         End Sub

                    ' Add the result to the output list
                    result.Add(newValue)
                Next

                Return result
            End Operator



            Public Shared Shadows Operator /(left As Vector, right As Value) As List(Of Value)
                ' Overload the division operator to divide each element in a vector of Values by a Value
                Dim result As New List(Of Value)()

                For Each v In left.data
                    Dim newValue As New Value(v.data / right.data, {v, right}, "/")

                    ' Calculate gradients correctly
                    newValue._backward = Sub()
                                             v.grad += 1.0 / right.data * newValue.grad
                                             right.grad -= v.data / (right.data * right.data) * newValue.grad
                                         End Sub

                    result.Add(newValue)
                Next

                Return result
            End Operator

            Public Shared Shadows Operator /(left As Vector, right As List(Of Double)) As List(Of Value)
                ' Overload the division operator to perform element-wise division between a vector of Values and a List(Of Double)
                If left.data.Count <> right.Count Then
                    Throw New ArgumentException("Both lists must have the same number of elements.")
                End If

                Dim result As New List(Of Value)()

                For i As Integer = 0 To left.data.Count - 1
                    Dim newValue As New Value(left.data(i) / right(i), {left, New Value(right(i))}, "/")

                    ' Calculate gradients correctly
                    newValue._backward = Sub()
                                             left.data(i).grad += 1.0 / right(i) * newValue.grad
                                             right(i) -= left.data(i).data / (right(i) * right(i)) * newValue.grad
                                         End Sub

                    result.Add(newValue)
                Next

                Return result
            End Operator

            Public Shared Shadows Operator +(left As Vector, right As Value) As List(Of Value)
                ' Overload the addition operator to add a vector of Values to a Value
                Dim result As New List(Of Value)()

                For Each v In left.data
                    Dim newValue As New Value(v.data + right.data, {v, right}, "+") ' Update the gradient calculation
                    newValue._backward = Sub()
                                             v.grad += 1 * newValue.grad ' Gradient update for left vector
                                             right.grad += 1 * newValue.grad ' Gradient update for right value
                                         End Sub
                    result.Add(newValue)
                Next

                Return result
            End Operator

            Public Shared Shadows Operator +(left As Vector, right As List(Of Double)) As List(Of Value)
                If left.data.Count <> right.Count Then
                    Throw New ArgumentException("Both lists must have the same number of elements.")
                End If

                Dim result As New List(Of Value)()

                For i As Integer = 0 To left.data.Count - 1
                    Dim newValue As New Value(left.data(i) + right(i), {left, New Value(right(i))}, "+") ' Update the gradient calculation
                    newValue._backward = Sub()
                                             left.data(i).grad += 1 * newValue.grad ' Gradient update for left vector element
                                             right(i) += 1 * newValue.grad ' Gradient update for right list element
                                         End Sub
                    result.Add(newValue)
                Next

                Return result
            End Operator

            Public Shared Function ApplyMask(ByVal vector As Vector, ByVal mask As Vector) As Vector
                If vector Is Nothing OrElse mask Is Nothing Then
                    Throw New ArgumentNullException("Vector and mask cannot be null.")
                End If

                If vector.data.Count <> mask.data.Count Then
                    Throw New ArgumentException("Vector and mask must have the same size.")
                End If

                Dim resultVector As New Vector(New List(Of Double))

                For i = 0 To vector.data.Count - 1
                    If mask.data(i).data = 1 Then
                        resultVector.data.Add(vector.data(i))
                    Else
                        resultVector.data.Add(New Value(0))
                    End If
                Next

                Return resultVector
            End Function

            Public Shared Function Concatenate(ByVal vector1 As Vector, ByVal vector2 As Vector) As Vector
                If vector1 Is Nothing OrElse vector2 Is Nothing Then
                    Throw New ArgumentNullException("Vectors cannot be null.")
                End If

                Dim resultVector As New Vector(New List(Of Double))
                resultVector.data.AddRange(vector1.data)
                resultVector.data.AddRange(vector2.data)

                Return resultVector
            End Function
            ''' <summary>
            ''' Returns the dot product of two vectors.
            ''' </summary>
            ''' <param name="Left">The first vector.</param>
            ''' <param name="Right">The second vector.</param>
            ''' <returns>The dot product of the input vectors.</returns>
            Public Shared Function DotProduct(ByVal Left As Vector, ByVal Right As Vector) As Value
                If Left Is Nothing Or Right Is Nothing Then
                    Throw New ArgumentNullException("Vectors cannot be null.")
                End If

                If Left.data.Count <> Right.data.Count Then
                    Throw New ArgumentException("Vectors must have the same size for dot product operation.")
                End If
                Dim Product As Double = 0

                For i = 0 To Left.data.Count - 1
                    Product += Left.data(i).data * Right.data(i).data
                Next

                Return New Value(Product, {Left, Right}, "DotProduct")
            End Function

            ''' <summary>
            ''' Square each value of the vector.
            ''' </summary>
            ''' <param name="vect">The vector to be squared.</param>
            ''' <returns>A new vector containing squared values.</returns>
            Public Shared Function SquareValues(ByVal vect As Vector) As Vector
                If vect Is Nothing Then
                    Throw New ArgumentNullException("Vector cannot be null.")
                End If

                Dim squaredValues As List(Of Double) = vect.data.Select(Function(value) value.data * value.data).ToList()

                Return New Vector(squaredValues, vect, "Square Series")
            End Function

            ''' <summary>
            ''' Returns the sum of squares of the values in the vector.
            ''' </summary>
            ''' <param name="vect">The vector whose sum of squares is to be calculated.</param>
            ''' <returns>The sum of squares of the vector values.</returns>
            Public Shared Function SumOfSquares(ByVal vect As Vector) As Value
                If vect Is Nothing Then
                    Throw New ArgumentNullException("Vector cannot be null.")
                End If
                Dim Product = 0


                For Each iValue As Value In vect.data
                    Product += iValue.data * iValue.data
                Next

                Return New Value(Product, vect, "SumOfSquares")
            End Function

            Public Sub Display()
                For Each ivalue In data
                    Console.Write(ivalue.data & " ")
                Next
                Console.WriteLine()
            End Sub

            ''' <summary>
            ''' Returns the dot product of this vector with another vector.
            ''' </summary>
            ''' <param name="vect">The vector to calculate the dot product with.</param>
            ''' <returns>The dot product of the two vectors.</returns>
            Public Function DotProduct(ByVal vect As Vector) As Vector
                If vect Is Nothing Then
                    Throw New ArgumentNullException("Vector cannot be null.")
                End If
                Dim product = 0

                For i = 0 To Math.Min(data.Count, vect.data.Count) - 1
                    product += data(i).data * vect.data(i).data
                Next

                Return New Value(product, {Me, vect}, "DotProduct")
            End Function

            ''' <summary>
            ''' Multiply each value in the vector together to produce a final value.
            ''' a * b * c * d ... = final value
            ''' </summary>
            ''' <returns>The result of multiplying all values in the vector together.</returns>
            Public Function ScalarProduct() As Value
                Dim total As Double = 1
                For Each item In data
                    total *= item.data
                Next
                Return New Value(total,, "ScalarProduct")
            End Function

            Public Function SequenceCrossEntropy(targetSequence As Vector) As Value
                ' Custom sequence cross-entropy loss function
                Dim loss As Double = 0
                For i As Integer = 0 To data.Count - 1
                    loss -= targetSequence.data(i).data * Math.Log(data(i).data)
                Next
                Return New Value(loss, Me, "SequenceCrossEntropy")
            End Function

            Public Shared Function SumWeightedInputs(x As Vector, w As List(Of Value), b As Value) As Value
                ' Ensure that the number of inputs matches the number of weights
                If x.data.Count <> w.Count Then
                    Throw New ArgumentException("Number of inputs must match the number of weights.")
                End If

                Dim weightedSum As Value = b

                For i As Integer = 0 To x.data.Count - 1
                    weightedSum += x.data(i) * w(i)
                Next

                Return weightedSum
            End Function
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

    ' Define the Value and Vector classes here as you provided

    ' Define a simple Feedforward Neural Network class
    Public Class NeuralNetwork
        Private inputLayerSize As Integer
        Private hiddenLayerSize As Integer
        Private outputLayerSize As Integer
        Private learningRate As Double
        Private weightsInputHidden As List(Of Value)
        Private biasHidden As Value
        Private weightsHiddenOutput As List(Of Value)
        Private biasOutput As Value

        Public Sub Train(inputData As Vector, targetData As Vector, epochs As Integer)
            For epoch As Integer = 1 To epochs
                ' Forward pass
                Dim output As Vector = Forward(inputData)

                ' Calculate loss (error) using Mean Squared Error
                Dim loss As Value = output.MeanSquaredError(targetData)

                ' Backward pass to compute gradients
                loss.backward()

                ' Update weights and biases automatically using autograd
                For Each weight In weightsInputHidden
                    weight.data = weight.data - learningRate * weight.grad
                Next
                biasHidden.data = biasHidden.data - learningRate * biasHidden.grad

                For Each weight In weightsHiddenOutput
                    weight.data = weight.data - learningRate * weight.grad
                Next
                biasOutput.data = biasOutput.data - learningRate * biasOutput.grad
            Next
        End Sub



        Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, learningRate As Double)
            inputLayerSize = inputSize
            hiddenLayerSize = hiddenSize
            outputLayerSize = outputSize
            Me.learningRate = learningRate

            ' Initialize weights and biases
            weightsInputHidden = New List(Of Value)()
            weightsHiddenOutput = New List(Of Value)()

            For i As Integer = 0 To hiddenLayerSize - 1
                weightsInputHidden.Add(New Value(New Random().NextDouble())) ' Initialize with random weights
            Next

            For i As Integer = 0 To outputLayerSize - 1
                weightsHiddenOutput.Add(New Value(New Random().NextDouble())) ' Initialize with random weights
            Next

            biasHidden = New Value(0.0)
            biasOutput = New Value(0.0)
        End Sub


        Private Function Sigmoid(x As Value) As Value
            x = Value.Exp(x) + 1
            ' Sigmoid activation function
            Return New Value(1 / x.data)
        End Function
        Public Function Forward(inputData As Vector) As Vector
            ' Forward pass through the network
            Dim hiddenInput As Vector = Vector.SumWeightedInputs(inputData, weightsInputHidden, biasHidden)
            Dim hiddenOutput As Vector = Vector.ApplyFunction(hiddenInput, AddressOf Sigmoid)
            Dim outputInput As Vector = Vector.SumWeightedInputs(hiddenOutput, weightsHiddenOutput, biasOutput)
            Dim output As Vector = Vector.ApplyFunction(outputInput, AddressOf Sigmoid)
            Return output
        End Function

        Sub Main()
            ' Define your input, hidden, and output layer sizes
            Dim inputSize As Integer = 2
            Dim hiddenSize As Integer = 4
            Dim outputSize As Integer = 1

            ' Create a neural network with the specified sizes and learning rate
            Dim neuralNetwork As New NeuralNetwork(inputSize, hiddenSize, outputSize, learningRate:=0.1)

            ' Define your training data and target data as Vectors
            Dim trainingData As New Vector(New List(Of Double) From {0.0, 1.0}) ' Adjust as needed
            Dim targetData As New Vector(New List(Of Double) From {1.0}) ' Adjust as needed

            ' Train the neural network for a certain number of epochs
            neuralNetwork.Train(trainingData, targetData, epochs:=1000)

            ' Perform predictions using the trained network
            Dim prediction As Vector = neuralNetwork.Forward(trainingData)

            ' Display the prediction
            Console.WriteLine("Prediction: " & prediction.data(0).ToString)
        End Sub

    End Class

    Namespace NN
        Public Class AutoGradientParameters
            ''' <summary>
            ''' Returns a list of parameters in the module.
            ''' </summary>
            ''' <returns>A list of parameters.</returns>
            Public Overridable Function Parameters() As List(Of Value)
                Return New List(Of Value)()
            End Function

            ''' <summary>
            ''' Zeroes out the gradients of all parameters in the module.
            ''' </summary>
            Public Sub ZeroGrad()
                For Each p In Parameters()
                    p.grad = 0
                Next
            End Sub
        End Class

        Public Class Layer
            Inherits AutoGradientParameters

            Private neurons As List(Of Neuron)

            Public Sub New(nin As Integer, nout As Integer)
                neurons = New List(Of Neuron)()

                For i As Integer = 1 To nout
                    neurons.Add(New Neuron(nin))
                Next
            End Sub

            Public Sub New(nin As Integer, nout As Integer, applyNonLinearActivation As Boolean)
                neurons = New List(Of Neuron)()

                For i As Integer = 1 To nout
                    neurons.Add(New Neuron(nin, applyNonLinearActivation))
                Next
            End Sub

            Public Function Compute(x As List(Of Value)) As Object
                Dim out As New List(Of Value)

                For Each n As Neuron In neurons
                    out.Add(n.Compute(x))
                Next

                If out.Count = 1 Then
                    Return out(0)
                Else
                    Return out
                End If
            End Function


            Public Overrides Function Parameters() As List(Of Value)
                Parameters = New List(Of Value)()

                For Each n As Neuron In neurons
                    Parameters.AddRange(n.Parameters())
                Next

                Return Parameters
            End Function

            Public Overrides Function ToString() As String
                Dim neuronDescriptions As New List(Of String)()

                For Each n As Neuron In neurons
                    neuronDescriptions.Add(n.ToString())
                Next

                Return $"Layer of [{String.Join(", ", neuronDescriptions)}]"
            End Function
        End Class

        Public Class MLP
            Inherits AutoGradientParameters

            Private layers As List(Of Layer)

            Public Sub New(nin As Integer, nouts As List(Of Integer))
                layers = New List(Of Layer)()
                Dim sz As List(Of Integer) = New List(Of Integer)()
                sz.Add(nin)
                sz.AddRange(nouts)

                For i As Integer = 0 To nouts.Count - 1
                    Dim isLastLayer As Boolean = (i = nouts.Count - 1)
                    Dim applyNonLinearActivation As Boolean = Not isLastLayer
                    layers.Add(New Layer(sz(i), sz(i + 1), applyNonLinearActivation))
                Next

            End Sub

            Public Function Compute(x As List(Of Value)) As List(Of Value)
                For Each layer As Layer In layers
                    x = layer.Compute(x)
                Next

                Return x
            End Function

            Public Overrides Function Parameters() As List(Of Value)
                Parameters = New List(Of Value)()

                For Each layer As Layer In layers
                    Parameters.AddRange(layer.Parameters())
                Next

                Return Parameters()
            End Function

            Public Overrides Function ToString() As String
                Dim layerDescriptions As New List(Of String)()

                For Each layer As Layer In layers
                    layerDescriptions.Add(layer.ToString())
                Next

                Return $"MLP of [{String.Join(", ", layerDescriptions)}]"
            End Function
        End Class

        Public Class Neuron
            Inherits AutoGradientParameters

            Private b As Value
            Private nonlin As Boolean
            Private w As List(Of Value)
            Public Sub New(nin As Integer, Optional nonlin As Boolean = True)
                Me.nonlin = nonlin
                w = New List(Of Value)()

                For i As Integer = 1 To nin
                    w.Add(New Value(Rnd() * 2 - 1)) ' Random values between -1 and 1
                Next

                b = New Value(0)
            End Sub

            Public Function Compute(x As List(Of Value)) As Value
                Dim act As Value = SumWeightedInputs(x)

                If nonlin Then
                    Return act.Tanh()
                Else
                    Return act
                End If
            End Function

            Public Overrides Function Parameters() As List(Of Value)
                Parameters = New List(Of Value)()
                Parameters.AddRange(w)
                Parameters.Add(b)
                Return Parameters
            End Function

            Public Overrides Function ToString() As String
                Dim activationType As String = If(nonlin, "ReLU", "Linear")
                Return $"{activationType}Neuron({w.Count})"
            End Function

            Private Function SumWeightedInputs(x As List(Of Value)) As Value
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
        End Class
    End Namespace
End Namespace

