Imports System.Windows.Forms
Imports Micrograd.AutoGradient.Engine.NN.Models.Blocks
Imports Micrograd.AutoGradient.Engine.NN.Models.Layers

Namespace AutoGradient
    'Autograd engine (with a bite! :)).
    'Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG
    'and a small neural networks library on top of it with a PyTorch-like API. 
    'Model By Andrew karpathy
    Namespace Engine
        Public Class Scaler
            Public _backward As Action = Sub()
                                         End Sub
            Public data As Double
            Public grad As Double
            Private _op As String
            Public _prev As HashSet(Of Scaler)
            ''' <summary>
            ''' Initializes a new instance of the <see cref="Scaler"/> class with the specified data Scaler and optional child nodes.
            ''' </summary>
            ''' <param name="data">The initial Scaler of the scalar.</param>
            ''' <param name="_children">Optional. A collection of child nodes in the computation graph.</param>
            ''' <param name="_op">Optional. The operation that produced this node.</param>
            Public Sub New(data As Double, Optional _children As IEnumerable(Of Scaler) = Nothing, Optional _op As String = "New")
                Me.data = data
                Me.grad = 0
                Me._op = _op
                Me._prev = If(_children IsNot Nothing, New HashSet(Of Scaler)(_children), New HashSet(Of Scaler)())
            End Sub
            Public Sub New(data As Scaler, Optional _children As IEnumerable(Of Scaler) = Nothing, Optional _op As String = "New")
                Me.data = data.data
                Me.grad = 0
                Me._op = _op
                Me._prev = If(_children IsNot Nothing, New HashSet(Of Scaler)(_children), New HashSet(Of Scaler)())
            End Sub
            ''' <summary>
            ''' Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG.
            ''' Computes the gradients using the backward pass.
            ''' </summary>
            Public Sub backward()
                ' Topological order all of the children in the graph
                Dim topo As New List(Of Scaler)()
                Dim visited As New HashSet(Of Scaler)()

                BuildTopology(Me, topo, visited)

                ' Go one variable at a time and apply the chain rule to get its gradient
                Me.grad = 1
                topo.Reverse()

                For Each v In topo
                    v._backward()
                Next

            End Sub
            Private Sub BuildTopology(v As Scaler, ByRef topo As List(Of Scaler), ByRef visited As HashSet(Of Scaler))
                If Not visited.Contains(v) Then
                    visited.Add(v)
                    For Each child In v._prev
                        BuildTopology(child, topo, visited)
                    Next
                    topo.Add(v)
                End If
            End Sub

            'ScalerVsScaler
            Public Shared Operator -(left As Scaler, right As Scaler) As Scaler
                Dim out = New Scaler(left.data - right.data, {left, right}, "-") ' Use "-" operation
                out._backward = Sub()
                                    left.grad += 1                ' Gradient of left with respect to the output is 1
                                    right.grad += -1              ' Gradient of right with respect to the output is -1
                                End Sub
                Return out
            End Operator
            Public Shared Operator *(left As Scaler, right As Scaler) As Scaler
                Dim out = New Scaler(left.data * right.data, {left, right}, "*") ' Use "*" operation
                out._backward = Sub()
                                    left.grad += right.data * out.grad    ' Gradient of left with respect to the output is right.data
                                    right.grad += left.data * out.grad    ' Gradient of right with respect to the output is left.data
                                End Sub
                Return out
            End Operator
            Public Shared Operator +(left As Scaler, right As Scaler) As Scaler
                Dim out = New Scaler(left.data + right.data, {left, right}, "+")
                out._backward = Sub()
                                    left.grad += out.grad
                                    right.grad += out.grad
                                End Sub
                Return out
            End Operator
            Public Shared Operator /(left As Scaler, right As Scaler) As Scaler
                Dim out = New Scaler(left.data / right.data, {left, right}, "/") ' Use "/" operation
                out._backward = Sub()
                                    left.grad += 1 / right.data * out.grad     ' Gradient of left with respect to the output is 1 / right.data
                                    right.grad += -left.data / (right.data * right.data) * out.grad ' Gradient of right with respect to the output is -left.data / (right.data^2)
                                End Sub
                Return out
            End Operator
            Public Shared Operator ^(left As Scaler, exponent As Scaler) As Scaler
                ' Exponentiation operator (^)
                Dim out As New Scaler(Math.Pow(left.data, exponent.data), {left, exponent}, "Power")
                Dim gradientFactor As Double = exponent.data * Math.Pow(left.data, exponent.data - 1)

                out._backward = Sub()
                                    left.grad += exponent.data * out.grad
                                    exponent.grad += left.data * out.grad
                                End Sub




                Return New Scaler(out, left, $"^{exponent}") With {.grad = left.grad * gradientFactor}
            End Operator
            'ScalerVsList
            Public Shared Operator +(left As Scaler, right As List(Of Scaler)) As List(Of Scaler)
                Dim result As New List(Of Scaler)()

                For Each v In right
                    Dim newScaler As New Scaler(left + v, {left, v}, "+") ' Use "+" operation

                    ' Update gradients correctly
                    newScaler._backward = Sub()
                                              left.grad += 1 * newScaler.grad  ' Gradient of left with respect to the output is 1
                                              v.grad += 1 * newScaler.grad    ' Gradient of v with respect to the output is 1
                                          End Sub

                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Operator -(left As Scaler, right As List(Of Scaler)) As List(Of Scaler)
                Dim result As New List(Of Scaler)()

                For Each v In right
                    Dim newScaler As New Scaler(left - v, {left, v}, "-") ' Use "-" operation

                    ' Update gradients correctly
                    newScaler._backward = Sub()
                                              left.grad += v.data - newScaler.grad     ' Gradient of left with respect to the output is 1
                                              v.grad += -left.data - newScaler.grad    ' Gradient of v with respect to the output is -1
                                          End Sub

                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Operator *(left As Scaler, right As List(Of Scaler)) As List(Of Scaler)
                Dim result As New List(Of Scaler)()

                For Each v In right
                    Dim newScaler As New Scaler(left * v, {left, v}, "*")
                    newScaler._backward = Sub()
                                              left.grad += v.data * newScaler.grad
                                              v.grad += left.data * newScaler.grad
                                          End Sub
                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Operator /(left As Scaler, right As List(Of Scaler)) As List(Of Scaler)
                Dim result As New List(Of Scaler)()

                For Each v In right
                    Dim newScaler As New Scaler(left / v, {left, v}, "/") ' Use "/" operation

                    ' Update gradients correctly
                    newScaler._backward = Sub()
                                              left.grad += 1 / v.data * newScaler.grad  ' Gradient of left with respect to the output is 1 / v.data
                                              v.grad += -left.data / (v.data * v.data) * newScaler.grad  ' Gradient of v with respect to the output is -left.data / (v.data * v.data)
                                          End Sub

                    result.Add(newScaler)
                Next

                Return result
            End Operator
            'ScalerVsDouble
            Public Shared Operator -(left As Scaler, right As Double) As Scaler
                Dim out = New Scaler(left.data - right, {left, New Scaler(right)}, "-") ' Use "-" operation
                out._backward = Sub()
                                    left.grad += 1              ' Gradient of left with respect to the output is 1
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator
            Public Shared Operator +(left As Scaler, right As Double) As Scaler
                Dim out = New Scaler(left.data + right, {left, New Scaler(right)}, "+")
                out._backward = Sub()
                                    left.grad += 1 * out.grad
                                End Sub
                Return out
            End Operator
            Public Shared Operator /(left As Scaler, right As Double) As Scaler
                Dim out = New Scaler(left.data / right, {left, New Scaler(right)}, "/") ' Use "/" operation
                out._backward = Sub()
                                    left.grad += 1 / right * out.grad ' Gradient of left with respect to the output is 1 / right
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator
            Public Shared Operator *(left As Scaler, right As Double) As Scaler
                Dim out = New Scaler(left.data * right, {left, New Scaler(right)}, "*") ' Use "*" operation
                out._backward = Sub()
                                    left.grad += right * out.grad   ' Gradient of left with respect to the output is right
                                    ' No need to update the gradient of right because it's a constant (integer)
                                End Sub
                Return out
            End Operator
            Public Shared Operator ^(left As Scaler, exponent As Double) As Scaler

                Dim out As New Scaler(Math.Pow(left.data, exponent), {left}, "Power")
                Dim gradientFactor As Double = exponent * Math.Pow(left.data, exponent - 1)

                out._backward = Sub()
                                    left.grad += exponent * out.grad
                                    exponent += left.data * out.grad
                                End Sub




                Return New Scaler(out, left, $"^{exponent}") With {.grad = left.grad * gradientFactor}
            End Operator
            Public Function MeanSquaredError(target As Double) As Scaler
                Dim loss As Scaler = Me - target
                Return loss * loss
            End Function
            Public Function MeanSquaredError(target As Scaler) As Scaler
                Dim loss As Scaler = Me - target
                Return loss * loss
            End Function
            Public Shared Function MeanSquaredError(Predicted As Scaler, target As Double) As Scaler
                Dim loss As Scaler = Predicted - target
                Return loss * loss
            End Function
            Public Function Difference(target As Double) As Scaler
                Dim iError As Scaler = Me - target
                Return iError
            End Function
            Public Function Difference(target As Scaler) As Scaler
                Dim iError As Scaler = Me - target
                Return iError
            End Function
            Public Shared Function Difference(ByRef Prediction As Scaler, target As Double) As Scaler
                Dim iError As Scaler = Prediction - target
                Return iError
            End Function
            Public Shared Function Difference(ByRef Prediction As Scaler, target As Scaler) As Scaler
                Dim iError As Scaler = Prediction - target
                Return iError
            End Function
            'To
            Public Overrides Function ToString() As String
                Return $"Scaler(data={Me.data}, grad={Me.grad})"
            End Function
            Public Function ToDouble() As Double
                Return Me.data
            End Function
            Public Function ToVector() As Vector
                Dim Ivect As New Vector(New List(Of Double), Me, "ScalerToVector")
                Ivect.ADD(Me)
                Return Ivect
            End Function
            Public Shared Function ToDouble(x As Scaler) As Double
                Return x.data
            End Function
            Private Shared Sub PopulateGraphTreeView(rootScaler As Scaler, parentNode As TreeNode)
                Dim currentNode As New TreeNode(rootScaler.ToString())
                parentNode.Nodes.Add(currentNode)

                ' Recursively traverse child nodes (children of the current Scaler)
                For Each childScaler In rootScaler._prev
                    PopulateGraphTreeView(childScaler, currentNode)
                Next

            End Sub
            ''' <summary>
            ''' Initializes a TreeView to visualize the computation graph starting from this Scaler object.
            ''' </summary>
            ''' <param name="rootScaler">The root Scaler object of the computation graph.</param>
            ''' <returns>A TreeView representing the computation graph.</returns>
            Public Shared Function ToTreeView(rootScaler As Scaler) As Windows.Forms.TreeView
                ' Create the root node for the TreeView
                Dim rootTreeNode As New TreeNode(rootScaler.ToString())

                ' Populate the TreeView starting from the root node
                PopulateGraphTreeView(rootScaler, rootTreeNode)

                ' Create a new TreeView and add the root node to it
                Dim graphTreeView As New Windows.Forms.TreeView
                graphTreeView.Nodes.Add(rootTreeNode)

                Return graphTreeView
            End Function
            ''' <summary>
            ''' Initializes a TreeView to visualize the computation graph starting from this Scaler object.
            ''' </summary>
            ''' <returns>A TreeView representing the computation graph.</returns>
            Public Function ToTreeView() As Windows.Forms.TreeView
                ' Create the root node for the TreeView
                Dim rootTreeNode As New TreeNode(Me.ToString())

                ' Populate the TreeView starting from the root node
                PopulateGraphTreeView(Me, rootTreeNode)

                ' Create a new TreeView and add the root node to it
                Dim graphTreeView As New Windows.Forms.TreeView
                graphTreeView.Nodes.Add(rootTreeNode)

                Return graphTreeView
            End Function


            ''' <summary>
            ''' Calculates the exponential (e^x) of the data.
            ''' </summary>
            ''' <returns>A new <see cref="scaler"/> containing the exponential Scaler.</returns>
            Public Shared Function Exp(x As Scaler) As Scaler
                ' Exponential activation function
                Dim expScaler As Double = Math.Exp(x.data)

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(expScaler, New Scaler(x, , "Double"), "Exp")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' The gradient of the exponential function with respect to its input is simply the exponential itself.
                ' Therefore, you can set the gradient of this result to the gradient provided during backpropagation.
                ' The gradient update for the exponential function is exp(x) * gradient
                Dim gradientUpdate As Double = expScaler * x.grad

                ' Set the gradient for this operation
                result.grad = gradientUpdate

                Return result
            End Function
            Public Shared Function Exp(x As Double) As Scaler
                ' Exponential activation function
                Dim expScaler As Double = Math.Exp(x)

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(expScaler, New Scaler(x, , "Double"), "Exp")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' The gradient of the exponential function with respect to its input is simply the exponential itself.
                ' Therefore, you can set the gradient of this result to the gradient provided during backpropagation.
                ' The gradient update for the exponential function is exp(x) * gradient
                Dim gradientUpdate As Double = expScaler * x

                ' Set the gradient for this operation
                result.grad = gradientUpdate

                Return result
            End Function
            Public Function Exp() As Scaler
                ' Exponential activation function
                Dim expScaler As Double = Math.Exp(data)

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(expScaler, Me, "Exp")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' The gradient of the exponential function with respect to its input is simply the exponential itself.
                ' Therefore, you can set the gradient of this result to the gradient provided during backpropagation.
                ' The gradient update for the exponential function is exp(x) * gradient
                Dim gradientUpdate As Double = expScaler * grad

                ' Set the gradient for this operation
                result.grad = gradientUpdate

                Return result
            End Function
            Public Function Dot(ByVal other As Scaler) As Scaler
                ' Calculate the dot product between this Scaler and another Scaler
                Dim result As New Scaler(0, {Me, other}, "Dot")
                result.data = Me.data * other.data
                Return result
            End Function

            Public Shared Function Dot(vect As Scaler, ByVal other As Scaler) As Scaler
                ' Calculate the dot product between this Scaler and another Scaler
                Dim result As New Scaler(0, {vect, other}, "Dot")
                result.data = vect.data * other.data
                Return result
            End Function

            Public Shared Function Sum(ByVal scalers As List(Of Scaler)) As Scaler
                ' Calculate the sum of a list of Scaler objects
                Dim result As New Scaler(0, scalers, "Sum")

                For Each scaler As Scaler In scalers
                    result.data += scaler.data
                Next

                Return result
            End Function

            'Activations
            ''' <summary>
            ''' Applies the hyperbolic tangent (tanh) activation function to this Scaler.
            ''' </summary>
            ''' <returns>A new <see cref="Scaler"/> containing the result of the tanh activation.</returns>
            Public Function Tanh() As Scaler
                ' Hyperbolic tangent (tanh) activation function
                Dim tanhScaler As Double = Math.Tanh(data)

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(tanhScaler, Me, "Tanh")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' The gradient of the tanh function with respect to its input is (1 - tanh(x)^2).
                ' Therefore, you need to set the gradient of this result using this formula.

                ' Set the gradient for this operation
                result.grad = 1 - tanhScaler * tanhScaler

                Return result
            End Function
            Public Shared Function Tanh(x As Scaler) As Scaler
                ' Hyperbolic tangent (tanh) activation function
                Dim tanhScaler As Double = Math.Tanh(x.data)

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(tanhScaler, x, "Tanh")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' The gradient of the tanh function with respect to its input is (1 - tanh(x)^2).
                ' Therefore, you need to set the gradient of this result using this formula.

                ' Set the gradient for this operation
                result.grad = 1 - tanhScaler * tanhScaler

                Return result
            End Function

            ''' <summary>
            ''' Applies the Sigmoid activation function to this Scaler.
            ''' </summary>
            ''' <returns>A new <see cref="Scaler"/> containing the result of the sigmoid activation.</returns>
            Public Function Sigmoid() As Scaler
                ' Sigmoid activation function
                Dim sigmoidScaler As Double = 1 / (1 + Math.Exp(-data))

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(sigmoidScaler, Me, "Sigmoid")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' The gradient of the Sigmoid function with respect to its input is sigmoid(x) * (1 - sigmoid(x)).
                ' Therefore, you need to set the gradient of this result using this formula.

                ' Set the gradient for this operation
                result.grad = sigmoidScaler * (1 - sigmoidScaler)

                Return result
            End Function
            Public Shared Function Sigmoid(x As Scaler) As Scaler
                ' Sigmoid activation function
                Dim sigmoidScaler As Double = 1 / (1 + Math.Exp(-x.data))

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(sigmoidScaler, x, "Sigmoid")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' The gradient of the Sigmoid function with respect to its input is sigmoid(x) * (1 - sigmoid(x)).
                ' Therefore, you need to set the gradient of this result using this formula.

                ' Set the gradient for this operation
                result.grad = sigmoidScaler * (1 - sigmoidScaler)

                Return result
            End Function

            ''' <summary>
            ''' Applies the Rectified Linear Unit (ReLU) activation function to this Scaler.
            ''' </summary>
            ''' <returns>A new <see cref="Scaler"/> containing the result of the ReLU activation.</returns>

            Public Function Relu() As Scaler
                ' Rectified Linear Unit (ReLU) activation function
                Dim reluResult As Double = Math.Max(0, data)

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(reluResult, Me, "ReLU")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' Since it's a scalar function, you'll typically set its gradient to 1 for positive input values and 0 for negative input values in the backpropagation process.

                ' Set the gradient for this operation
                If data > 0 Then
                    result.grad = 1
                Else
                    result.grad = 0
                End If

                Return result
            End Function
            Public Shared Function Relu(x As Scaler) As Scaler
                ' Rectified Linear Unit (ReLU) activation function
                Dim reluResult As Double = Math.Max(0, x.data)

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(reluResult, x, "ReLU")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' Since it's a scalar function, you'll typically set its gradient to 1 for positive input values and 0 for negative input values in the backpropagation process.

                ' Set the gradient for this operation
                If x.data > 0 Then
                    result.grad = 1
                Else
                    result.grad = 0
                End If

                Return result
            End Function

            ''' <summary>
            ''' Applies the Gaussian Error Linear Unit (GELU) activation function to this Scaler.
            ''' </summary>
            ''' <returns>A new <see cref="Scaler"/> containing the result of the GELU activation.</returns>
            Public Function GELU() As Scaler
                ' Gaussian Error Linear Unit (GELU) activation function
                Dim x As Double = data

                ' Calculate the GELU coefficient
                Dim coefficient As Double = 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.Pow(x, 3.0))))

                ' Calculate the GELU result
                Dim geluScaler As Double = x * coefficient

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(geluScaler, Me, "GELU")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' Since it's a scalar function, you'll typically set its gradient to 1 in the backpropagation process.

                ' Set the gradient for this operation
                result.grad = 1

                Return result
            End Function
            Public Shared Function GELU(ByRef Valx As Scaler) As Scaler
                ' Gaussian Error Linear Unit (GELU) activation function
                Dim x As Double = Valx.data

                ' Calculate the GELU coefficient
                Dim coefficient As Double = 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.Pow(x, 3.0))))

                ' Calculate the GELU result
                Dim geluScaler As Double = x * coefficient

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(geluScaler, Valx, "GELU")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' Since it's a scalar function, you'll typically set its gradient to 1 in the backpropagation process.

                ' Set the gradient for this operation
                result.grad = 1

                Return result
            End Function


            Public Function LOGSigmoid() As Scaler
                ' Logarithmic Sigmoid activation function
                Dim logSigmoidScaler As Double = -Math.Log(1 + Math.Exp(-data))

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(logSigmoidScaler, Me, "LOGSigmoid")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' Since it's a scalar function, you'll typically set its gradient to 1 in the backpropagation process.

                ' Set the gradient for this operation
                result.grad = 1

                Return result
            End Function
            Public Shared Function LOGSigmoid(x As Scaler) As Scaler
                ' Logarithmic Sigmoid activation function
                Dim logSigmoidScaler As Double = -Math.Log(1 + Math.Exp(-x.data))

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(logSigmoidScaler, x, "LOGSigmoid")

                ' The gradient computation for this function depends on the gradients of the subsequent operations in the computational graph.
                ' Since it's a scalar function, you'll typically set its gradient to 1 in the backpropagation process.

                ' Set the gradient for this operation
                result.grad = 1

                Return result
            End Function
            'Important
            Public Function DotSum(inputs As List(Of Scaler)) As Scaler
                ' Compute the dot product of the scaler with each element in the list and sum them up
                Dim dotProduct As Double = 0

                For Each nInput In inputs
                    dotProduct += data * nInput.data
                Next

                Return New Scaler(dotProduct, inputs, "DotSum")
            End Function
            Public Shared Function DotSum(scaler As Scaler, inputs As List(Of Scaler)) As Scaler
                ' Compute the dot product of the scaler with each element in the list and sum them up
                Dim dotProduct As Double = 0

                For Each nInput In inputs
                    dotProduct += scaler.data * nInput.data
                Next

                Return New Scaler(dotProduct, inputs, "DotSum")
            End Function
            Public Shared Function Dot(scaler As Scaler, inputs As List(Of Scaler)) As List(Of Scaler)
                ' Perform element-wise multiplication of scaler with each element in the list
                Dim result As New List(Of Scaler)

                For Each nInput In inputs
                    Dim product As New Scaler(scaler * nInput, inputs, "element-wise multiplication")
                    result.Add(product)
                Next

                Return result
            End Function
            Public Function Dot(inputs As List(Of Scaler)) As List(Of Scaler)
                ' Perform element-wise multiplication of scaler with each element in the list
                Dim result As New List(Of Scaler)()

                For Each nInput In inputs
                    Dim product As New Scaler(Me * nInput, inputs, "element-wise multiplication")
                    result.Add(product)
                Next

                Return result
            End Function
            Public Sub ApplyStochasticGradientDescent(Optional learningRate As Double = 0.001, Optional Inplace As Boolean = False)

                ' Topological order all of the children in the graph
                Dim topo As New List(Of Scaler)()
                Dim visited As New HashSet(Of Scaler)()

                BuildTopology(Me, topo, visited)

                ' Go one variable at a time and apply the chain rule to get its gradient

                topo.Reverse()

                For Each v In topo
                    v._backward()
                    If Inplace - True Then

                        ' Update each Scaler object with its gradient using stochastic gradient descent (SGD)
                        v.data -= learningRate * v.grad
                    Else

                        ' Update each Scaler.grad object with its gradient using stochastic gradient descent (SGD)
                        v.grad -= learningRate * v.grad
                    End If

                Next

            End Sub
            Public Function SumWeightedInputsList(x As List(Of Scaler), w As List(Of Scaler), b As Scaler) As Scaler



                ' Ensure that the number of inputs matches the number of weights
                If x.Count <> w.Count Then
                    Throw New ArgumentException("Number of inputs must match the number of weights.")
                End If

                Dim weightedSum As Scaler = b

                For i As Integer = 0 To x.Count - 1
                    weightedSum += x(i) * w(i)
                Next

                Return weightedSum
            End Function
            Public Shared Function WeightedLoss(predicted As Scaler, target As Scaler, weight As Double) As Scaler
                Dim ierror As Scaler = predicted - target
                Dim loss As Scaler = New Scaler(0.5 * weight * ierror.data * ierror.data, {target, predicted}, "WeightedLoss")
                Return loss
            End Function
            Public Function WeightedLoss(target As Scaler, weight As Double) As Scaler
                Dim ierror As Scaler = Me - target
                Dim loss As Scaler = New Scaler(0.5 * weight * ierror.data * ierror.data, {target, Me}, "WeightedLoss")
                Return loss
            End Function
            Public Shared Function HuberLoss(predicted As Scaler, target As Scaler, delta As Double) As Scaler
                Dim ierror As Scaler = predicted - target
                Dim loss As Scaler
                If Math.Abs(ierror.data) <= delta Then
                    loss = New Scaler(0.5 * ierror.data * ierror.data, {predicted, target}, "HuberLoss")
                Else
                    loss = New Scaler(delta * (Math.Abs(ierror.data) - 0.5 * delta), {predicted, target, New Scaler(delta)}, "HuberLoss")
                End If
                Return loss
            End Function
            Public Function HuberLoss(target As Scaler, delta As Double) As Scaler
                Dim ierror As Scaler = Me - target
                Dim loss As Scaler
                If Math.Abs(ierror.data) <= delta Then
                    loss = New Scaler(0.5 * ierror.data * ierror.data, {Me, target}, "HuberLoss")
                Else
                    loss = New Scaler(delta * (Math.Abs(ierror.data) - 0.5 * delta), {Me, target, New Scaler(delta)}, "HuberLoss")
                End If
                Return loss
            End Function
        End Class
        Public Class Vector
            Public _backward As Action = Sub()
                                         End Sub

            Private _op As String
            Public _prev As HashSet(Of Vector)
            Public data As List(Of Scaler)
            Public grads As List(Of Double)
            ''' <summary>
            ''' Initializes a new instance of the <see cref="Vector"/> class with the specified data Scalers and optional child nodes.
            ''' </summary>
            ''' <param name="data">The list of initial Scalers for the vector.</param>
            ''' <param name="_children">Optional. A collection of child nodes in the computation graph.</param>
            ''' <param name="_op">Optional. The operation that produced this node.</param>
            Public Sub New(data As List(Of Double), Optional _children As IEnumerable(Of Vector) = Nothing, Optional _op As String = "")
                For Each item In data
                    Me.data.Add(New Scaler(item, If(_children IsNot Nothing, New HashSet(Of Vector)(_children), New HashSet(Of Scaler)()), _op))
                Next
                grads = New List(Of Double)
                Me._op = _op
                Me._prev = If(_children IsNot Nothing, New HashSet(Of Vector)(_children), New HashSet(Of Vector)())

            End Sub
            Public Sub New(data As List(Of Scaler), Optional _children As IEnumerable(Of Vector) = Nothing, Optional _op As String = "")
                data = data
                'Gradients will be in thier respective Scalers
                'We will keep a record here
                grads = New List(Of Double)
                Me._op = _op
                Me._prev = If(_children IsNot Nothing, New HashSet(Of Vector)(_children), New HashSet(Of Scaler)())

            End Sub
            'to
            Public Overrides Function ToString() As String
                Dim str As String = ""
                For Each item In data
                    str &= item.ToString & vbNewLine

                Next
                Return str
            End Function
            Public Function ToScalerList() As List(Of Scaler)
                Return data
            End Function
            Public Function ToDoubleLst() As List(Of Double)
                Dim Lst As New List(Of Double)
                For Each item In data
                    Lst.Add(item.ToDouble)
                Next
                Return Lst
            End Function
            Public Sub backward()
                'zeor grads
                grads = New List(Of Double)
                'execute back for each scaler and grab gradient
                For Each item In data
                    item.backward()
                    grads.Add(item.grad)
                Next
            End Sub
            'VectorVsVector
            Public Shared Operator -(left As Vector, right As Vector) As Vector
                ' Overload the subtraction operator to perform element-wise subtraction between two vectors of Scalers
                If left.data.Count <> right.data.Count Then
                    Throw New ArgumentException("Both vectors must have the same number of elements.")
                End If

                Dim resultData As New List(Of Double)()

                For i As Integer = 0 To left.data.Count - 1
                    resultData.Add(left.data(i).data - right.data(i).data)
                Next

                ' Create a new Vector with the result data
                Dim resultVector As New Vector(resultData, {left, right}, "-")



                Return resultVector
            End Operator
            Public Shared Operator +(left As Vector, right As Vector) As Vector
                ' Overload the Add operator to perform element-wise  between two vectors of Scalers
                If left.data.Count <> right.data.Count Then
                    Throw New ArgumentException("Both vectors must have the same number of elements.")
                End If

                Dim resultData As New List(Of Double)()

                For i As Integer = 0 To left.data.Count - 1
                    resultData.Add(left.data(i).data + right.data(i).data)
                Next

                ' Create a new Vector with the result data
                Dim resultVector As New Vector(resultData, {left, right}, "+")



                Return resultVector
            End Operator
            Public Shared Operator *(left As Vector, right As Vector) As Vector
                ' Overload the * operator to perform element-wise  between two vectors of Scalers
                If left.data.Count <> right.data.Count Then
                    Throw New ArgumentException("Both vectors must have the same number of elements.")
                End If

                Dim resultData As New List(Of Double)()

                For i As Integer = 0 To left.data.Count - 1
                    resultData.Add(left.data(i).data * right.data(i).data)
                Next

                ' Create a new Vector with the result data
                Dim resultVector As New Vector(resultData, {left, right}, "*")



                Return resultVector
            End Operator
            Public Shared Operator /(left As Vector, right As Vector) As Vector
                ' Overload the / operator to perform element-wise  between two vectors of Scalers
                If left.data.Count <> right.data.Count Then
                    Throw New ArgumentException("Both vectors must have the same number of elements.")
                End If

                Dim resultData As New List(Of Double)()

                For i As Integer = 0 To left.data.Count - 1
                    resultData.Add(left.data(i).data / right.data(i).data)
                Next

                ' Create a new Vector with the result data
                Dim resultVector As New Vector(resultData, {left, right}, "*")



                Return resultVector
            End Operator
            'VectorVsScaler
            Public Shared Shadows Operator +(left As Vector, right As Scaler) As List(Of Scaler)
                ' Overload the addition operator to add a vector of Scalers to a Scaler
                Dim result As New List(Of Scaler)()

                For Each v In left.data
                    Dim newScaler As New Scaler(v.data + right.data, {v, right}, "+") ' Update the gradient calculation
                    newScaler._backward = Sub()
                                              v.grad += 1 * newScaler.grad ' Gradient update for left vector
                                              right.grad += 1 * newScaler.grad ' Gradient update for right Scaler
                                          End Sub
                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Shadows Operator -(left As Vector, right As Scaler) As List(Of Scaler)
                ' Overload the subtraction operator to subtract a Scaler from each element in a vector of Scalers
                Dim result As New List(Of Scaler)()

                For Each v In left.data
                    Dim newScaler As New Scaler(v.data - right.data, {v, right}, "-")

                    ' Calculate gradients correctly
                    newScaler._backward = Sub()
                                              v.grad += 1.0 * newScaler.grad
                                              right.grad -= 1.0 * newScaler.grad
                                          End Sub

                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Shadows Operator *(left As Vector, right As Scaler) As List(Of Scaler)
                ' Overload the multiplication operator to multiply a vector of Scalers by a Scaler
                Dim result As New List(Of Scaler)()

                For Each v In left.data
                    Dim newScaler As New Scaler(v.data * right.data, {v, right}, "*") ' Update the gradient calculation
                    newScaler._backward = Sub()
                                              v.grad += right.data * newScaler.grad ' Gradient update for left vector
                                              right.grad += v.data * newScaler.grad ' Gradient update for right Scaler
                                          End Sub
                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Shadows Operator /(left As Vector, right As Scaler) As List(Of Scaler)
                ' Overload the division operator to divide each element in a vector of Scalers by a Scaler
                Dim result As New List(Of Scaler)()

                For Each v In left.data
                    Dim newScaler As New Scaler(v.data / right.data, {v, right}, "/")

                    ' Calculate gradients correctly
                    newScaler._backward = Sub()
                                              v.grad += 1.0 / right.data * newScaler.grad
                                              right.grad -= v.data / (right.data * right.data) * newScaler.grad
                                          End Sub

                    result.Add(newScaler)
                Next

                Return result
            End Operator
            'VectorVsDouble
            Public Shared Shadows Operator +(left As Vector, right As Double) As List(Of Scaler)
                ' Overload the addition operator to add a vector of Scalers to a Scaler
                Dim result As New List(Of Scaler)()

                For Each v In left.data
                    Dim newScaler As New Scaler(v.data + right, {v}, "+") ' Update the gradient calculation
                    newScaler._backward = Sub()
                                              v.grad += 1 * newScaler.grad ' Gradient update for left vector
                                          End Sub
                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Shadows Operator -(left As Vector, right As Double) As List(Of Scaler)
                ' Overload the subtraction operator to subtract a Scaler from each element in a vector of Scalers
                Dim result As New List(Of Scaler)()

                For Each v In left.data
                    Dim newScaler As New Scaler(v.data - right, {v}, "-")

                    ' Calculate gradients correctly
                    newScaler._backward = Sub()
                                              v.grad += 1.0 * newScaler.grad

                                          End Sub

                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Shadows Operator *(left As Vector, right As Double) As List(Of Scaler)
                ' Overload the multiplication operator to multiply a vector of Scalers by a Scaler
                Dim result As New List(Of Scaler)()

                For Each v In left.data
                    Dim newScaler As New Scaler(v.data * right, {v}, "*") ' Update the gradient calculation
                    newScaler._backward = Sub()
                                              v.grad += right * newScaler.grad ' Gradient update for left vector
                                          End Sub
                    result.Add(newScaler)
                Next

                Return result
            End Operator
            Public Shared Shadows Operator /(left As Vector, right As Double) As List(Of Scaler)
                ' Overload the division operator to divide each element in a vector of Scalers by a Scaler
                Dim result As New List(Of Scaler)()

                For Each v In left.data
                    Dim newScaler As New Scaler(v.data / right, {v}, "/")

                    ' Calculate gradients correctly
                    newScaler._backward = Sub()
                                              v.grad += 1.0 / right * newScaler.grad
                                          End Sub

                    result.Add(newScaler)
                Next

                Return result
            End Operator

            Public Shared Function ApplyFunction(inputVector As Vector, func As Func(Of Scaler, Scaler)) As Vector
                If inputVector Is Nothing OrElse func Is Nothing Then
                    Throw New ArgumentNullException("Input vector and function cannot be null.")
                End If

                Dim result As New Vector(New List(Of Scaler), {inputVector}, "Function")

                For Each item In inputVector.data
                    Dim newScaler As Scaler = func(item)
                    result.data.Add(newScaler)

                    ' Calculate gradients for the input vector
                    item.grad += newScaler.grad
                Next

                Return result
            End Function
            Public Function ApplyFunction(func As Func(Of Scaler, Scaler)) As Vector


                Dim result As New Vector(New List(Of Scaler), {Me}, "Function")

                For Each item In data
                    Dim newScaler As Scaler = func(item)
                    result.data.Add(newScaler)

                    ' Calculate gradients for the input vector
                    item.grad += newScaler.grad
                Next

                Return result
            End Function
            'ListControl
            'ADD
            Public Sub ADD(ByRef Item As Scaler)
                data.Add(Item)
            End Sub
            Public Shared Sub ADD(Vect As Vector, ByRef Item As Scaler)
                Vect.data.Add(Item)
            End Sub
            Public Shared Sub ADD(Vect As Vector, ByRef Item As Double)
                Vect.data.Add(New Scaler(Item))
            End Sub
            Public Shared Sub ADD(Vect As Vector, ByRef Item As Integer)
                Vect.data.Add(New Scaler(Item))
            End Sub
            Public Sub ADD(ByRef Item As Double)
                data.Add(New Scaler(Item))
            End Sub
            Public Sub ADD(ByRef Item As Integer)
                data.Add(New Scaler(Item))
            End Sub
            'REMOVE
            Public Sub REMOVE(ByRef Item As Scaler)
                data.Remove(Item)
            End Sub
            Public Shared Sub REMOVE(Vect As Vector, ByRef Item As Scaler)
                Vect.data.Remove(Item)
            End Sub
            Public Shared Sub REMOVE(Vect As Vector, ByRef Item As Double)
                Vect.data.Remove(New Scaler(Item))
            End Sub
            Public Shared Sub REMOVE(Vect As Vector, ByRef Item As Integer)
                Vect.data.Remove(New Scaler(Item))
            End Sub
            Public Sub REMOVE(ByRef Item As Double)
                data.Remove(New Scaler(Item))
            End Sub
            Public Sub REMOVE(ByRef Item As Integer)
                data.Remove(New Scaler(Item))
            End Sub
            Public Function Index(ByRef idx As Integer) As Scaler
                Return data(idx)
            End Function
            Public Shared Function Index(ByRef Vect As Vector, ByRef idx As Integer) As Scaler
                Return Vect.data(idx)
            End Function
            Public Shared Function IndexAsDouble(ByRef Vect As Vector, ByRef idx As Integer) As Double
                Return Vect.data(idx).ToDouble
            End Function
            Public Function IndexAsDouble(ByRef idx As Integer) As Double
                Return data(idx).ToDouble
            End Function
            Public Function IndexAsTree(ByRef idx As Integer) As TreeView
                Return data(idx).ToTreeView
            End Function
            Public Shared Function IndexAsTree(ByRef Vect As Vector, ByRef idx As Integer) As TreeView
                Return Vect.data(idx).ToTreeView
            End Function
            'Functions
            ''' <summary>
            ''' Multiply each Scaler in the vector together to produce a final Scaler.
            ''' a * b * c * d ... = final Scaler
            ''' </summary>
            ''' <returns>The result of multiplying all Scalers in the vector together.</returns>
            Public Function Multiply() As Scaler
                Dim total As Double = 0 ' Initialize total to 1

                For Each item In data
                    total *= item.data
                Next

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(total, Me, "Multiply")

                ' Set the gradient for the result
                result.grad = 1

                ' Calculate and store gradients for the input vector elements
                For Each item In data
                    item.grad += total / item.data
                Next

                Return result
            End Function

            Public Shared Function Multiply(vect As Vector) As Scaler
                Dim total As Double = 0 ' Initialize total to 1

                For Each item In vect.data
                    total *= item.data
                Next

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(total, vect, "Multiply")

                ' Set the gradient for the result
                result.grad = 1

                ' Calculate and store gradients for the input vector elements
                For Each item In vect.data
                    item.grad += total / item.data
                Next

                Return result
            End Function
            Public Function Sum() As Scaler
                Dim total As Double = 0 ' Initialize total to 0

                For Each item In data
                    total += item.data
                Next

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(total, Me, "Sum")

                Return result
            End Function
            Public Shared Function Sum(vect As Vector) As Scaler
                Dim total As Double = 0

                For Each item In vect.data
                    total += item.data
                Next

                ' Create a Scaler object for the result and store the gradients
                Dim result As New Scaler(total, vect, "Multiply")



                Return result
            End Function
            Public Shared Function OuterProduct(vector1 As Vector, vector2 As Vector) As List(Of Scaler)
                ' Compute the outer product of two vectors
                Dim result As New List(Of Scaler)()

                For Each v1 In vector1.data
                    For Each v2 In vector2.data
                        Dim newScaler As New Scaler(v1.data * v2.data)
                        result.Add(newScaler)

                        ' Calculate gradients for the original vectors
                        v1.grad += v2.data * newScaler.grad
                        v2.grad += v1.data * newScaler.grad

                        ' Reset the gradient of the new Scaler
                        newScaler.grad = 0
                    Next
                Next

                Return result
            End Function
            Public Function OuterProduct(vector2 As Vector) As List(Of Scaler)
                ' Compute the outer product of two vectors
                Dim result As New List(Of Scaler)()

                For Each v1 In data
                    For Each v2 In vector2.data
                        Dim newScaler As New Scaler(v1.data * v2.data)
                        result.Add(newScaler)

                        ' Calculate gradients for the original vectors
                        v1.grad += v2.data * newScaler.grad
                        v2.grad += v1.data * newScaler.grad

                        ' Reset the gradient of the new Scaler
                        newScaler.grad = 0
                    Next
                Next

                Return result
            End Function
            Public Shared Function Concatenate(ByVal vector1 As Vector, ByVal vector2 As Vector) As Vector
                If vector1 Is Nothing OrElse vector2 Is Nothing Then
                    Throw New ArgumentNullException("Vectors cannot be null.")
                End If

                Dim resultVector As New Vector(New List(Of Double), {vector1, vector2}, "concatenate")
                resultVector.data.AddRange(vector1.data)
                resultVector.data.AddRange(vector2.data)

                ' Calculate gradients for the input vectors
                Dim numElementsVector1 = vector1.data.Count
                Dim numElementsVector2 = vector2.data.Count

                For i = 0 To numElementsVector1 - 1
                    vector1.data(i).grad += resultVector.data(i).grad
                Next

                For i = 0 To numElementsVector2 - 1
                    vector2.data(i).grad += resultVector.data(numElementsVector1 + i).grad
                Next

                Return resultVector
            End Function
            Public Function Concatenate(ByVal vector2 As Vector) As Vector
                If vector2 Is Nothing Then
                    Throw New ArgumentNullException("Vectors cannot be null.")
                End If

                Dim resultVector As New Vector(New List(Of Double), {Me, vector2}, "concatenate")
                resultVector.data.AddRange(Me.data)
                resultVector.data.AddRange(vector2.data)

                ' Calculate gradients for the input vectors
                Dim numElementsVector1 = Me.data.Count
                Dim numElementsVector2 = vector2.data.Count

                For i = 0 To numElementsVector1 - 1
                    Me.data(i).grad += resultVector.data(i).grad
                Next

                For i = 0 To numElementsVector2 - 1
                    vector2.data(i).grad += resultVector.data(numElementsVector1 + i).grad
                Next

                Return resultVector
            End Function
            Public Shared Function SumWeightedInputs(x As Vector, w As List(Of Scaler), b As Scaler) As Scaler
                ' Ensure that the number of inputs matches the number of weights
                If x.data.Count <> w.Count Then
                    Throw New ArgumentException("Number of inputs must match the number of weights.")
                End If

                Dim weightedSum As Scaler = b

                For i As Integer = 0 To x.data.Count - 1
                    Dim weightedInput = x.data(i) * w(i)
                    weightedSum += weightedInput

                    ' Calculate gradients for inputs, weights, and bias
                    x.data(i).grad += w(i).data ' Gradient for input
                    w(i).grad += x.data(i).data ' Gradient for weight
                    b.grad += 1 ' Gradient for bias (constant 1)

                    ' Chain rule: d(weightedInput)/d(x.data(i)) = w(i).data
                    x.data(i).grad += weightedInput.grad * w(i).data ' Gradient for input
                    w(i).grad += weightedInput.grad * x.data(i).data ' Gradient for weight
                    b.grad += weightedInput.grad ' Gradient for bias
                Next

                Return weightedSum
            End Function
            Public Function SumWeightedInputs(w As List(Of Scaler), b As Scaler) As Scaler
                ' Ensure that the number of inputs matches the number of weights
                If data.Count <> w.Count Then
                    Throw New ArgumentException("Number of inputs must match the number of weights.")
                End If

                Dim weightedSum As Scaler = b

                For i As Integer = 0 To data.Count - 1
                    Dim weightedInput = data(i) * w(i)
                    weightedSum += weightedInput

                    ' Calculate gradients for inputs, weights, and bias
                    data(i).grad += w(i).data ' Gradient for input
                    w(i).grad += data(i).data ' Gradient for weight
                    b.grad += 1 ' Gradient for bias (constant 1)

                    ' Chain rule: d(weightedInput)/d(x.data(i)) = w(i).data
                    data(i).grad += weightedInput.grad * w(i).data ' Gradient for input
                    w(i).grad += weightedInput.grad * data(i).data ' Gradient for weight
                    b.grad += weightedInput.grad ' Gradient for bias
                Next

                Return weightedSum
            End Function

            ''' <summary>
            ''' Returns the sum of squares of the Scalers in the vector.
            ''' </summary>
            ''' <param name="vect">The vector whose sum of squares is to be calculated.</param>
            ''' <returns>The sum of squares of the vector Scalers.</returns>
            Public Shared Function SumOfSquares(ByVal vect As Vector) As Scaler
                If vect Is Nothing Then
                    Throw New ArgumentNullException("Vector cannot be null.")
                End If
                Dim product = New Scaler(0, vect, "SumOfSquares")

                For Each iScaler As Scaler In vect.data
                    Dim square = iScaler * iScaler
                    product += square

                    ' Calculate the gradient for the squared Scaler
                    iScaler.grad += 2 * square.grad ' Using chain rule: d(x^2)/dx = 2x

                    ' Update the gradient of the sumOfSquares with respect to iScaler
                    vect.data(vect.data.IndexOf(iScaler)).grad += iScaler.grad
                Next

                Return product
            End Function
            Public Function SumOfSquares() As Scaler
                If Me Is Nothing Then
                    Throw New ArgumentNullException("Vector cannot be null.")
                End If
                Dim product = New Scaler(0, Me, "SumOfSquares")

                For Each iScaler As Scaler In data
                    Dim square = iScaler * iScaler
                    product += square

                    ' Calculate the gradient for the squared Scaler
                    iScaler.grad += 2 * square.grad ' Using chain rule: d(x^2)/dx = 2x

                    ' Update the gradient of the sumOfSquares with respect to iScaler
                    data(data.IndexOf(iScaler)).grad += iScaler.grad
                Next

                Return product
            End Function

            ''' <summary>
            ''' Square each Scaler of the vector.
            ''' </summary>
            ''' <param name="vect">The vector to be squared.</param>
            ''' <returns>A new vector containing squared Scalers.</returns>
            Public Shared Function SquareScalers(ByVal vect As Vector) As Vector
                If vect Is Nothing Then
                    Throw New ArgumentNullException("Vector cannot be null.")
                End If

                Dim squaredScalers As List(Of Scaler) = New List(Of Scaler)

                For Each Scaler As Scaler In vect.data
                    Dim squaredScaler = Scaler * Scaler
                    squaredScalers.Add(squaredScaler)

                    ' Calculate the gradient for the squared Scaler
                    Scaler.grad += 2 * Scaler.data * squaredScaler.grad ' Using chain rule: d(x^2)/dx = 2x

                    ' Update the gradient of the squared Scaler
                    squaredScaler.grad = 0 ' Reset the gradient of the squared Scaler
                Next
                Dim x As New List(Of Double)
                For Each item In squaredScalers
                    x.Add(item.data)
                Next
                Return New Vector(x, vect, "Square Series")
            End Function
            Public Function SquareScalers() As Vector
                If Me Is Nothing Then
                    Throw New ArgumentNullException("Vector cannot be null.")
                End If

                Dim squaredScalers As List(Of Scaler) = New List(Of Scaler)

                For Each Scaler As Scaler In Me.data
                    Dim squaredScaler = Scaler * Scaler
                    squaredScalers.Add(squaredScaler)

                    ' Calculate the gradient for the squared Scaler
                    Scaler.grad += 2 * Scaler.data * squaredScaler.grad ' Using chain rule: d(x^2)/dx = 2x

                    ' Update the gradient of the squared Scaler
                    squaredScaler.grad = 0 ' Reset the gradient of the squared Scaler
                Next
                Dim x As New List(Of Double)
                For Each item In squaredScalers
                    x.Add(item.data)
                Next
                Return New Vector(x, Me, "Square Series")
            End Function

            Public Shared Function SequenceCrossEntropy(InputSequence As Vector, targetSequence As Vector) As Scaler
                ' Custom sequence cross-entropy loss function
                Dim loss As Double = 0
                For i As Integer = 0 To InputSequence.data.Count - 1
                    loss -= targetSequence.data(i).data * Math.Log(InputSequence.data(i).data)
                Next
                Return New Scaler(loss, InputSequence, "SequenceCrossEntropy")
            End Function
            Public Function SequenceCrossEntropy(targetSequence As Vector) As Scaler
                ' Custom sequence cross-entropy loss function
                Dim loss As Double = 0
                For i As Integer = 0 To data.Count - 1
                    loss -= targetSequence.data(i).data * Math.Log(data(i).data)
                Next
                Return New Scaler(loss, Me, "SequenceCrossEntropy")
            End Function
            ''' <summary>
            ''' Returns the dot product of two vectors.
            ''' </summary>
            ''' <param name="Left">The first vector.</param>
            ''' <param name="Right">The second vector.</param>
            ''' <returns>The dot product of the input vectors.</returns>
            Public Shared Function DotProduct(ByVal Left As Vector, ByVal Right As Vector) As Scaler
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

                ' Calculate gradients for the input vectors using the chain rule
                For i = 0 To Left.data.Count - 1
                    Left.data(i).grad += Right.data(i).data * Product
                    Right.data(i).grad += Left.data(i).data * Product
                Next

                Return New Scaler(Product, , "DotProduct")
            End Function
            ''' <summary>
            ''' Returns the dot product of this vector with another vector.
            ''' </summary>
            ''' <param name="vect">The vector to calculate the dot product with.</param>
            ''' <returns>The dot product of the two vectors.</returns>
            Public Function DotProduct(ByVal vect As Vector) As Scaler
                If vect Is Nothing Then
                    Throw New ArgumentNullException("Vector cannot be null.")
                End If

                Dim product = 0

                For i = 0 To Math.Min(data.Count, vect.data.Count) - 1
                    product += data(i).data * vect.data(i).data
                Next

                ' Calculate the gradients
                Dim gradientsMe As New List(Of Double)()
                Dim gradientsVect As New List(Of Double)()

                For i = 0 To Math.Min(data.Count, vect.data.Count) - 1
                    gradientsMe.Add(vect.data(i).data) ' Gradient with respect to this vector
                    gradientsVect.Add(data(i).data) ' Gradient with respect to the other vector
                Next

                ' Create Scaler objects for the result and store the gradients
                Dim resultMe As New Scaler(product, Me, "DotProduct")
                Dim resultVect As New Scaler(product, vect, "DotProduct")

                ' Set the SGD for these operations
                resultMe.grad = 1
                resultVect.grad = 1

                ' Store the gradients for each input vector
                For i = 0 To Math.Min(data.Count, vect.data.Count) - 1
                    data(i).grad += gradientsMe(i)
                    vect.data(i).grad += gradientsVect(i)
                Next

                Return resultMe ' You can return either resultMe or resultVect, depending on which vector you want to consider as the result.
            End Function
            Public Shared Function CategoricalCrossEntropy(predicted As Vector, target As Vector) As Scaler
                Dim loss As Scaler = New Scaler(0)
                For i As Integer = 0 To target.data.Count - 1
                    loss -= target.data(i).data * Math.Log(predicted.data(i).data + 0.000000000000001) ' Add a small epsilon to prevent log(0)
                Next
                Return loss
            End Function
            Public Function CategoricalCrossEntropy(target As Vector) As Scaler
                Dim loss As Scaler = New Scaler(0)
                For i As Integer = 0 To target.data.Count - 1
                    loss -= target.data(i).data * Math.Log(data(i).data + 0.000000000000001) ' Add a small epsilon to prevent log(0)
                Next
                Return loss
            End Function
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
                        resultVector.data.Add(New Scaler(0))
                    End If
                Next

                Return resultVector
            End Function

            ''' <summary>
            ''' Applies the Softmax operation to this vector of Scalers.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the softmax Scalers.</returns>
            Public Function Softmax() As Vector
                ' Softmax operation
                ' Assuming this is used in a vector context, where data represents logits
                Dim expScalers As List(Of Double) = Me.ToDoubleLst
                Dim expSum = expScalers.Sum()

                ' Compute softmax Scalers for each element in the vector
                Dim softmaxScalers As List(Of Double) = expScalers.Select(Function(exp) exp / expSum).ToList()

                ' Construct a new Scaler object for the softmax result
                Dim result As New Vector(softmaxScalers, Me, "Softmax")

                ' Calculate and store the gradient
                For i As Integer = 0 To Me.data.Count - 1
                    Dim gradient = softmaxScalers(i) * (1 - softmaxScalers(i)) ' Derivative of softmax
                    Me.data(i).grad += gradient
                Next

                Return result
            End Function

            ''' <summary>
            ''' Applies the Softmax operation to this vector of Scalers.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the softmax Scalers.</returns>
            Public Shared Function Softmax(ByRef Lst As Vector) As Vector
                ' Softmax operation
                ' Assuming this is used in a vector context, where data represents logits
                Dim expScalers As List(Of Double) = Lst.ToDoubleLst
                Dim expSum = expScalers.Sum()

                ' Compute softmax Scalers for each element in the vector
                Dim softmaxScalers As List(Of Double) = expScalers.Select(Function(exp) exp / expSum).ToList()

                ' Construct a new Scaler object for the softmax result
                Dim result As New Vector(softmaxScalers, Lst, "Softmax")

                ' Calculate and store the gradient
                For i As Integer = 0 To Lst.data.Count - 1
                    Dim gradient = softmaxScalers(i) * (1 - softmaxScalers(i)) ' Derivative of softmax
                    Lst.data(i).grad += gradient
                Next

                Return result
            End Function
            ''' <summary>
            ''' Applies the Softmin activation function to this vector of Scalers.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the softmin Scalers.</returns>
            Public Function Softmin() As Vector
                ' Softmin activation function
                Dim expScalers As List(Of Double) = Me.data.Select(Function(scaler) Math.Exp(-scaler.data)).ToList()
                Dim expSum = expScalers.Sum()

                ' Compute softmin Scalers for each element in the vector
                Dim softminScalers As List(Of Double) = expScalers.Select(Function(exp) exp / expSum).ToList()

                ' Construct a new Scaler object for the softmin result
                Dim result As New Vector(softminScalers, Me, "Softmin")

                ' Calculate and store the gradient
                For i As Integer = 0 To Me.data.Count - 1
                    Dim gradient = -softminScalers(i) / expSum
                    Me.data(i).grad += gradient
                Next

                Return result
            End Function


            ''' <summary>
            ''' Applies the Softmin activation function to this vector of Scalers.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the softmin Scalers.</returns>
            Public Shared Function Softmin(lst As Vector) As Vector
                ' Softmin activation function
                Dim expScalers As List(Of Double) = lst.data.Select(Function(scaler) Math.Exp(-scaler.data)).ToList()
                Dim expSum = expScalers.Sum()

                ' Compute softmin Scalers for each element in the vector
                Dim softminScalers As List(Of Double) = expScalers.Select(Function(exp) exp / expSum).ToList()

                ' Construct a new Scaler object for the softmin result
                Dim result As New Vector(softminScalers, lst, "Softmin")

                ' Calculate and store the gradient
                For i As Integer = 0 To lst.data.Count - 1
                    Dim gradient = -softminScalers(i) / expSum
                    lst.data(i).grad += gradient
                Next

                Return result
            End Function

            ''' <summary>
            ''' Applies the Log-Softmax operation to this vector of Scalers.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the log-softmax Scalers.</returns>
            Public Function LogSoftmax() As Vector
                ' Log-Softmax operation
                ' Assuming this is used in a vector context, where data represents logits
                Dim expScalers As List(Of Double) = Me.ToDoubleLst
                Dim expSum = expScalers.Sum()

                ' Compute log-softmax Scalers for each element in the vector
                Dim logSoftmaxScalers As List(Of Double) = expScalers.Select(Function(exp) Math.Log(exp / expSum)).ToList()

                ' Construct a new Scaler object for the log-softmax result
                Dim result As New Vector(logSoftmaxScalers, Me, "LogSoftmax")

                ' Calculate and store the gradient
                For i As Integer = 0 To Me.data.Count - 1
                    Dim gradient = 1 - Math.Exp(logSoftmaxScalers(i)) ' Derivative of log-softmax
                    Me.data(i).grad += gradient
                Next

                Return result
            End Function

            ''' <summary>
            ''' Applies the Log-Softmax operation to this vector of Scalers.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the log-softmax Scalers.</returns>
            Public Shared Function LogSoftmax(ByRef lst As Vector) As Vector
                ' Log-Softmax operation
                ' Assuming this is used in a vector context, where data represents logits
                Dim expScalers As List(Of Double) = lst.ToDoubleLst
                Dim expSum = expScalers.Sum()

                ' Compute log-softmax Scalers for each element in the vector
                Dim logSoftmaxScalers As List(Of Double) = expScalers.Select(Function(exp) Math.Log(exp / expSum)).ToList()

                ' Construct a new Scaler object for the log-softmax result
                Dim result As New Vector(logSoftmaxScalers, lst, "LogSoftmax")

                ' Calculate and store the gradient
                For i As Integer = 0 To lst.data.Count - 1
                    Dim gradient = 1 - Math.Exp(logSoftmaxScalers(i)) ' Derivative of log-softmax
                    lst.data(i).grad += gradient
                Next

                Return result
            End Function
            ''' <summary>
            ''' Applies the Logarithmic Softmin activation function to this vector of Scalers.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the log-softmin Scalers.</returns>
            Public Function LogSoftmin() As Vector
                ' Logarithmic Softmin activation function
                Dim expScalers As List(Of Double) = Me.ToDoubleLst
                Dim expSum = expScalers.Sum()

                ' Compute log-softmin Scalers for each element in the vector
                Dim logSoftminScalers As List(Of Double) = expScalers.Select(Function(exp) -Math.Log(exp / expSum)).ToList()

                ' Construct a new Scaler object for the log-softmin result
                Dim result As New Vector(logSoftminScalers, Me, "LogSoftmin")

                ' Calculate and store the gradient
                For i As Integer = 0 To Me.data.Count - 1
                    Dim gradient = 1 - (expScalers(i) / expSum) ' Derivative of log-softmin
                    Me.data(i).grad += gradient
                Next

                Return result
            End Function

            ''' <summary>
            ''' Applies the Logarithmic Softmin activation function to this vector of Scalers.
            ''' </summary>
            ''' <returns>A new <see cref="Vector"/> containing the log-softmin Scalers.</returns>
            Public Shared Function LogSoftmin(ByRef lst As Vector) As Vector
                ' Logarithmic Softmin activation function
                Dim expScalers As List(Of Double) = lst.ToDoubleLst
                Dim expSum = expScalers.Sum()

                ' Compute log-softmin Scalers for each element in the vector
                Dim logSoftminScalers As List(Of Double) = expScalers.Select(Function(exp) -Math.Log(exp / expSum)).ToList()

                ' Construct a new Scaler object for the log-softmin result
                Dim result As New Vector(logSoftminScalers, lst, "LogSoftmin")

                ' Calculate and store the gradient
                For i As Integer = 0 To lst.data.Count - 1
                    Dim gradient = 1 - (expScalers(i) / expSum) ' Derivative of log-softmin
                    lst.data(i).grad += gradient
                Next

                Return result
            End Function
            Public Function Roll(positions As Integer, direction As Integer) As Vector
                Dim result As New Vector(data)

                For i As Integer = 0 To data.Count - 1
                    Dim newIndex As Integer = (i + (positions * direction)) Mod data.Count
                    If newIndex < 0 Then newIndex += data.Count
                    result.data(i) = data(newIndex)
                Next

                Return result
            End Function
            Public Shared Function Roll(input As Vector, positions As Integer, direction As Integer) As Vector
                Dim result As New Vector(input.data)

                For i As Integer = 0 To input.data.Count - 1
                    Dim newIndex As Integer = (i + (positions * direction)) Mod input.data.Count
                    If newIndex < 0 Then newIndex += input.data.Count
                    result.data(i) = input.data(newIndex)
                Next

                Return result
            End Function
        End Class
        Public Class AutoGradientParameters
            ''' <summary>
            ''' Returns a list of parameters in the module.
            ''' </summary>
            ''' <returns>A list of parameters.</returns>
            Public Overridable Function Parameters() As List(Of Scaler)
                Return New List(Of Scaler)()
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
        Namespace NN


            Public Class Neuron
                Inherits AutoGradientParameters
                Private actType As ActivationType
                Private b As Scaler
                Private nonlin As Boolean
                Private w As List(Of Scaler)
                Public Sub New(nin As Integer, Optional nonlin As Boolean = True, Optional MyActivationType As ActivationType = ActivationType.Relu)
                    Me.nonlin = nonlin
                    w = New List(Of Scaler)()
                    Me.actType = MyActivationType
                    For i As Integer = 1 To nin
                        w.Add(New Scaler(Rnd() * 2 - 1)) ' Random Scalers between -1 and 1
                    Next

                    b = New Scaler(0)
                End Sub

                Public Function Compute(x As List(Of Scaler)) As Scaler
                    Dim act As Scaler = SumWeightedInputs(x)

                    If nonlin Then
                        Select Case actType
                            Case ActivationType.Gelu
                                Return act.GELU
                            Case ActivationType.LogSigmoid
                                Return act.LOGSigmoid
                            Case ActivationType.Sigmoid
                                Return act.Sigmoid
                            Case ActivationType.Tanh
                                Return act.Tanh
                            Case ActivationType.Relu
                                Return act.Relu
                        End Select

                    Else
                        Return act
                    End If
                    Return act
                End Function

                Public Overrides Function Parameters() As List(Of Scaler)
                    Parameters = New List(Of Scaler)()
                    Parameters.AddRange(w)
                    Parameters.Add(b)
                    Return Parameters
                End Function

                Public Overrides Function ToString() As String
                    Dim activationType As String = If(nonlin, "ReLU", "Linear")
                    Return $"{activationType}Neuron({w.Count})"
                End Function

                Private Function SumWeightedInputs(x As List(Of Scaler)) As Scaler
                    ' Ensure that the number of inputs matches the number of weights
                    If x.Count <> w.Count Then
                        Throw New ArgumentException("Number of inputs must match the number of weights.")
                    End If

                    Dim weightedSum As Scaler = b

                    For i As Integer = 0 To x.Count - 1
                        weightedSum += x(i) * w(i)
                    Next

                    Return weightedSum
                End Function
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

                Public Function Compute(x As List(Of Scaler)) As Object
                    Dim out As New List(Of Scaler)

                    For Each n As Neuron In neurons
                        out.Add(n.Compute(x))
                    Next

                    If out.Count = 1 Then
                        Return out(0)
                    Else
                        Return out
                    End If
                End Function
                Public Sub Train(inputData As List(Of Scaler), targets As List(Of Scaler), learningRate As Double, epochs As Integer)
                    For epoch = 1 To epochs
                        Dim totalLoss As Scaler = New Scaler(0)

                        ' Loop through each training example
                        For exampleIndex = 0 To inputData.Count - 1
                            Dim inputExample As Scaler = inputData(exampleIndex)
                            Dim targetExample As Scaler = targets(exampleIndex)

                            ' Forward pass
                            Dim output As Object = Compute(New List(Of Scaler)() From {inputExample})

                            ' Calculate loss (mean squared error for regression)
                            Dim loss As Scaler
                            If TypeOf output Is Scaler Then
                                loss = CType(output, Scaler).MeanSquaredError(targetExample.data)
                            Else
                                ' Handle multiple outputs
                                Dim outputList As List(Of Scaler) = CType(output, List(Of Scaler))
                                loss = New Scaler(0)
                                For i = 0 To outputList.Count - 1
                                    loss += outputList(i).MeanSquaredError(targetExample.data)
                                Next
                                loss /= outputList.Count
                            End If

                            totalLoss += loss

                            ' Backpropagation
                            loss.backward()

                            ' Update model parameters using SGD
                            For Each parameter As Scaler In Parameters()
                                parameter.data -= learningRate * parameter.grad
                                parameter.grad = 0 ' Reset gradients for the next iteration
                            Next
                        Next

                        ' Calculate the average loss for this epoch
                        Dim averageLoss As Double = totalLoss.data / inputData.Count

                        ' Print loss for monitoring training progress
                        Console.WriteLine($"Epoch {epoch}, Average Loss: {averageLoss}")
                    Next
                End Sub


                Public Overrides Function Parameters() As List(Of Scaler)
                    Parameters = New List(Of Scaler)()

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
                Private customTrainingLayers As List(Of Layer) ' Layers with custom training logic

                Public Sub New(nin As Integer, nouts As List(Of Integer))
                    layers = New List(Of Layer)()
                    customTrainingLayers = New List(Of Layer)()
                    Dim sz As List(Of Integer) = New List(Of Integer)()
                    sz.Add(nin)
                    sz.AddRange(nouts)

                    For i As Integer = 0 To nouts.Count - 1
                        Dim isLastLayer As Boolean = (i = nouts.Count - 1)
                        Dim applyNonLinearActivation As Boolean = Not isLastLayer
                        Dim layer As New Layer(sz(i), sz(i + 1), applyNonLinearActivation)

                        ' Identify layers that need custom training
                        If i Mod 2 = 0 Then
                            customTrainingLayers.Add(layer)
                        End If

                        layers.Add(layer)
                    Next
                End Sub

                Public Function Compute(x As List(Of Scaler)) As List(Of Scaler)
                    For Each layer As Layer In layers
                        x = layer.Compute(x)
                    Next

                    Return x
                End Function

                Public Overrides Function Parameters() As List(Of Scaler)
                    Parameters = New List(Of Scaler)()

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
                Sub MainExample()
                    ' Define a simple neural network architecture
                    Dim inputSize As Integer = 2
                    Dim hiddenSize As Integer = 2
                    Dim outputSize As Integer = 1
                    Dim learningRate As Double = 0.1
                    Dim epochs As Integer = 1000

                    ' Create input data
                    Dim inputData As New Scaler(0.2)
                    Dim target As New Scaler(0.8)

                    ' Define model parameters (weights and biases)
                    Dim w1 As New Scaler(0.5)
                    Dim b1 As New Scaler(0.1)
                    Dim w2 As New Scaler(0.3)
                    Dim b2 As New Scaler(-0.2)

                    ' Training loop
                    For epoch = 1 To epochs
                        ' Forward pass
                        Dim z1 As Scaler = inputData * w1 + b1
                        Dim a1 As Scaler = z1.Relu() ' Apply ReLU activation
                        Dim z2 As Scaler = a1 * w2 + b2
                        Dim output As Scaler = z2.Sigmoid() ' Apply Sigmoid activation

                        ' Calculate loss (mean squared error)
                        Dim loss As Scaler = output.MeanSquaredError(target.data)

                        ' Backpropagation
                        loss.backward()

                        ' Update model parameters using SGD
                        w1.data = -learningRate * w1.grad
                        b1.data = -learningRate * b1.grad
                        w2.data = -learningRate * w2.grad
                        b2.data = -learningRate * b2.grad

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
                    Dim testInput As New Scaler(0.4)
                    Dim z1Test As Scaler = testInput * w1 + b1
                    Dim a1Test As Scaler = z1Test.Relu()
                    Dim z2Test As Scaler = a1Test * w2 + b2
                    Dim outputTest As Scaler = z2Test.Sigmoid()

                    Console.WriteLine($"Inference Result: {outputTest.data}")

                    Console.ReadLine()
                End Sub


                Public Sub Train(inputData As List(Of Scaler), targets As List(Of Scaler), learningRate As Double, epochs As Integer)
                    For epoch = 1 To epochs
                        Dim totalLoss As Scaler = New Scaler(0)

                        ' Loop through each training example
                        For exampleIndex = 0 To inputData.Count - 1
                            ' Forward pass
                            Dim x As List(Of Scaler) = New List(Of Scaler)()
                            x.Add(inputData(exampleIndex))
                            For Each layer As Layer In layers
                                x = layer.Compute(x)
                            Next

                            ' Calculate loss (mean squared error)
                            Dim loss As Scaler = x(0).MeanSquaredError(targets(exampleIndex).data)
                            totalLoss += loss

                            ' Backpropagation
                            loss.backward()

                            ' Update model parameters using SGD
                            For Each layer As Layer In layers
                                For Each parameter As Scaler In layer.Parameters()
                                    parameter.data -= learningRate * parameter.grad
                                    parameter.grad = 0 ' Reset gradients for the next iteration
                                Next
                            Next
                        Next

                        ' Calculate the average loss for this epoch
                        Dim averageLoss As Double = totalLoss.data / inputData.Count

                        ' Print loss for monitoring training progress
                        Console.WriteLine($"Epoch {epoch}, Average Loss: {averageLoss}")
                    Next
                End Sub

                ' Uniform Training Strategy
                Public Sub TrainUniform(inputData As List(Of Scaler), targets As List(Of Scaler), learningRate As Double, epochs As Integer)
                    For epoch = 1 To epochs
                        Dim totalLoss As Scaler = New Scaler(0)

                        ' Loop through each training example
                        For exampleIndex = 0 To inputData.Count - 1
                            ' Forward pass
                            Dim x As List(Of Scaler) = New List(Of Scaler)()
                            x.Add(inputData(exampleIndex))
                            For Each layer As Layer In layers
                                x = layer.Compute(x)
                            Next

                            ' Calculate loss (mean squared error)
                            Dim loss As Scaler = x(0).MeanSquaredError(targets(exampleIndex).data)
                            totalLoss += loss

                            ' Backpropagation
                            loss.backward()

                            ' Update model parameters using SGD for all layers
                            For Each layer As Layer In layers
                                For Each parameter As Scaler In layer.Parameters()
                                    parameter.data -= learningRate * parameter.grad
                                    parameter.grad = 0 ' Reset gradients for the next iteration
                                Next
                            Next
                        Next

                        ' Calculate the average loss for this epoch
                        Dim averageLoss As Double = totalLoss.data / inputData.Count

                        ' Print loss for monitoring training progress
                        Console.WriteLine($"Epoch {epoch}, Average Loss: {averageLoss}")
                    Next
                End Sub

                ' Layer-Specific Training
                Public Sub TrainCustom(inputData As List(Of Scaler), targets As List(Of Scaler), learningRate As Double, epochs As Integer)
                    For Each customLayer As Layer In customTrainingLayers
                        ' Call custom training logic for each custom layer
                        customLayer.Train(inputData, targets, learningRate, epochs)
                    Next
                End Sub

                ' Hybrid Approach
                Public Sub TrainHybrid(inputData As List(Of Scaler), targets As List(Of Scaler), learningRate As Double, epochs As Integer)
                    For epoch = 1 To epochs
                        Dim totalLoss As Scaler = New Scaler(0)

                        ' Loop through each training example
                        For exampleIndex = 0 To inputData.Count - 1
                            ' Forward pass
                            Dim x As List(Of Scaler) = New List(Of Scaler)()
                            x.Add(inputData(exampleIndex))
                            For Each layer As Layer In layers
                                x = layer.Compute(x)
                            Next

                            ' Calculate loss (mean squared error)
                            Dim loss As Scaler = x(0).MeanSquaredError(targets(exampleIndex).data)
                            totalLoss += loss

                            ' Backpropagation
                            loss.backward()

                            ' Update model parameters using SGD for uniform layers
                            For Each layer As Layer In layers
                                If Not customTrainingLayers.Contains(layer) Then
                                    For Each parameter As Scaler In layer.Parameters()
                                        parameter.data -= learningRate * parameter.grad
                                        parameter.grad = 0 ' Reset gradients for the next iteration
                                    Next
                                End If
                            Next
                        Next

                        ' Calculate the average loss for this epoch
                        Dim averageLoss As Double = totalLoss.data / inputData.Count

                        ' Print loss for monitoring training progress
                        Console.WriteLine($"Epoch {epoch}, Average Loss: {averageLoss}")
                    Next

                    ' Train custom layers separately
                    TrainCustom(inputData, targets, learningRate, epochs)
                End Sub
            End Class


            Public Enum ActivationType
                Tanh
                Relu
                Gelu
                Sigmoid
                LogSigmoid
            End Enum
            Public Enum LossType
                HuberLoss
                WeightedLoss
                MeanSquaredError
                Difference
                CatagoricalCrossEntropy
                SequenceCrossEntropy
                SumOfSquares
            End Enum
            Public Enum OuputType
                LogSoftMax
                LogSoftMin
                SoftMax
                SoftMin
                OuterProduct
                DotProduct
                Concatenate
            End Enum



            Namespace Models
                Namespace Blocks
                    Public Class LayerBlock
                        Inherits AutoGradientParameters ' Inherit from Layer instead of AutoGradientParameters

                        Private layers As List(Of AutoGradientParameters)



                        Public Sub AddLayer(layer As AutoGradientParameters)
                            layers.Add(layer)
                        End Sub
                        Public Function Compute(inputs As List(Of Scaler)) As List(Of Scaler)
                            Dim output As List(Of Scaler) = inputs

                            For Each layer In layers
                                If TypeOf layer Is Layer Then
                                    ' If the layer is of type Layer, call its Compute method
                                    output = DirectCast(layer, Layer).Compute(output)
                                Else
                                    ' If it's not a Layer, just call its Parameters method (no computation)
                                    layer.Parameters()
                                End If
                            Next

                            Return output
                        End Function



                        Public Overrides Function Parameters() As List(Of Scaler)
                            Dim iparameters As List(Of Scaler) = New List(Of Scaler)()

                            For Each layer In layers
                                iparameters.AddRange(layer.Parameters())
                            Next

                            Return iparameters
                        End Function
                    End Class

                    Public Class EncoderBlock
                        Inherits LayerBlock

                        Public Sub New(heads As Integer, inputSize As Integer, querySize As Integer, keySize As Integer, valueSize As Integer)
                            ' Create Multihead Attention Layer
                            Dim attentionLayer As New MultiheadAttentionLayer(inputSize, querySize, keySize, valueSize, heads)

                            ' Create a Residual Layer
                            Dim residualLayer As New ResidualLayer()

                            ' Add the attention layer followed by the residual layer to the encoder block
                            Me.AddLayer(attentionLayer)
                            Me.AddLayer(residualLayer)
                        End Sub
                    End Class

                    Public Class DecoderBlock
                        Inherits LayerBlock

                        Public Sub New(heads As Integer, inputSize As Integer, querySize As Integer, keySize As Integer, valueSize As Integer)
                            ' Create Multihead Attention Layer for self-attention
                            Dim selfAttentionLayer As New MultiheadAttentionLayer(inputSize, querySize, keySize, valueSize, heads)

                            ' Create a Multihead Attention Layer for encoder-decoder attention
                            Dim encoderDecoderAttentionLayer As New MultiheadAttentionLayer(inputSize, querySize, keySize, valueSize, heads)

                            ' Create a Residual Layer
                            Dim residualLayer As New ResidualLayer()

                            ' Add the self-attention layer, encoder-decoder attention layer, and residual layer to the decoder block
                            Me.AddLayer(selfAttentionLayer)
                            Me.AddLayer(encoderDecoderAttentionLayer)
                            Me.AddLayer(residualLayer)
                        End Sub
                    End Class
                    Public Class TransformerBlock
                        Inherits AutoGradientParameters

                        Private encoderBlock As EncoderBlock
                        Private decoderBlock As DecoderBlock

                        Public Sub New(encoderHeads As Integer, decoderHeads As Integer, inputSize As Integer, querySize As Integer, keySize As Integer, valueSize As Integer)
                            ' Create the encoder block with specified heads and dimensions
                            encoderBlock = New EncoderBlock(encoderHeads, inputSize, querySize, keySize, valueSize)

                            ' Create the decoder block with specified heads and dimensions
                            decoderBlock = New DecoderBlock(decoderHeads, inputSize, querySize, keySize, valueSize)
                        End Sub

                        Public Function Forward(encoderInput As List(Of Scaler), decoderInput As List(Of Scaler)) As List(Of Scaler)
                            ' Pass the encoder input through the encoder block
                            Dim encoderOutput As List(Of Scaler) = encoderBlock.Compute(encoderInput)

                            ' Pass the decoder input through the decoder block
                            Dim decoderOutput As List(Of Scaler) = decoderBlock.Compute(decoderInput)

                            ' Return the combined output of the encoder and decoder blocks
                            Return encoderOutput.Concat(decoderOutput).ToList()
                        End Function

                        Public Overrides Function Parameters() As List(Of Scaler)
                            Dim iparameters As List(Of Scaler) = New List(Of Scaler)()

                            ' Include parameters from both encoder and decoder blocks
                            iparameters.AddRange(encoderBlock.Parameters())
                            iparameters.AddRange(decoderBlock.Parameters())

                            Return iparameters
                        End Function
                    End Class
                End Namespace
                Namespace Layers
                    Public Class LinearLayer
                        Inherits AutoGradientParameters

                        Private weights As List(Of Scaler)
                        Private bias As Scaler

                        Public Sub New(inputSize As Integer, outputSize As Integer)
                            ' Initialize the weights with random values
                            weights = New List(Of Scaler)()

                            For i As Integer = 1 To outputSize
                                Dim weight As New Scaler(Rnd() * 2 - 1)
                                weights.Add(weight)
                            Next

                            ' Initialize the bias with zeros
                            bias = New Scaler(0)
                        End Sub

                        Public Function Forward(inputs As List(Of Scaler)) As List(Of Scaler)
                            ' Perform linear transformation (matrix multiplication + bias addition)
                            Dim output As New List(Of Scaler)()

                            For Each nInput In inputs
                                Dim weightedSum As Scaler = nInput.DotSum(weights)
                                Dim linearOutput As Scaler = weightedSum + bias
                                output.Add(linearOutput)
                            Next

                            Return output
                        End Function

                        Public Overrides Function Parameters() As List(Of Scaler)
                            Dim iparameters As List(Of Scaler) = New List(Of Scaler)()

                            ' Include weights and bias as parameters
                            iparameters.AddRange(weights)
                            iparameters.Add(bias)

                            Return iparameters
                        End Function
                    End Class

                    Public Class MaskedAttentionLayer
                        Inherits AutoGradientParameters

                        Private Wq As List(Of Scaler) ' Query weights
                        Private Wk As List(Of Scaler) ' Key weights
                        Private Wv As List(Of Scaler) ' Value weights

                        Public Sub New(inputSize As Integer, querySize As Integer, keySize As Integer, valueSize As Integer)
                            ' Initialize the masked attention layer with weights
                            Wq = New List(Of Scaler)()
                            Wk = New List(Of Scaler)()
                            Wv = New List(Of Scaler)()

                            ' Initialize weights with random values (you can use different initialization methods)
                            For i As Integer = 1 To querySize
                                Wq.Add(New Scaler(Rnd() * 2 - 1)) ' Query weights
                            Next

                            For i As Integer = 1 To keySize
                                Wk.Add(New Scaler(Rnd() * 2 - 1)) ' Key weights
                            Next

                            For i As Integer = 1 To valueSize
                                Wv.Add(New Scaler(Rnd() * 2 - 1)) ' Value weights
                            Next
                        End Sub

                        Public Function Attend(queries As List(Of Scaler), keys As List(Of Scaler), values As List(Of Scaler), mask As List(Of Double)) As List(Of Scaler)
                            ' Compute attention scores using dot-product attention with masking
                            Dim attentionScores As New List(Of Scaler)()

                            For i As Integer = 0 To queries.Count - 1
                                Dim query As Scaler = queries(i)
                                Dim scores As New List(Of Scaler)()

                                For j As Integer = 0 To keys.Count - 1
                                    Dim key As Scaler = keys(j)
                                    Dim score As Scaler = query.Dot(key) ' Dot product attention

                                    ' Apply mask to the score
                                    score *= mask(j)

                                    scores.Add(score)
                                Next

                                ' Apply softmax to the scores
                                Dim softmaxScores As List(Of Scaler) = Softmax(scores)

                                ' Calculate the weighted sum of values
                                Dim weightedSum As Scaler = GetWeightedSum(values, softmaxScores)

                                attentionScores.Add(weightedSum)
                            Next

                            ' Return the attended vectors
                            Return attentionScores
                        End Function

                        Private Function Softmax(scores As List(Of Scaler)) As List(Of Scaler)
                            ' Compute the softmax of a list of scores
                            Dim expScores As List(Of Scaler) = scores.Select(Function(score) score.Exp()).ToList()
                            Dim sumExpScores As Scaler = Scaler.Sum(expScores)

                            Dim softmaxValues As New List(Of Scaler)()

                            For Each expScore In expScores
                                Dim softmaxScore As Scaler = expScore / sumExpScores
                                softmaxValues.Add(softmaxScore)
                            Next

                            Return softmaxValues
                        End Function

                        Private Function GetWeightedSum(values As List(Of Scaler), weights As List(Of Scaler)) As Scaler
                            ' Calculate the weighted sum of values based on attention scores
                            Dim weightedSum As Scaler = New Scaler(0)

                            For i As Integer = 0 To values.Count - 1
                                weightedSum += values(i) * weights(i)
                            Next

                            Return weightedSum
                        End Function
                    End Class
                    Public Class MultiheadAttentionLayer
                        Inherits AutoGradientParameters

                        Private ReadOnly heads As Integer
                        Private attentionLayers As List(Of AttentionLayer)

                        Public Sub New(inputSize As Integer, querySize As Integer, keySize As Integer, valueSize As Integer, heads As Integer)
                            Me.heads = heads
                            attentionLayers = New List(Of AttentionLayer)()

                            ' Create multiple attention layers (heads)
                            For i As Integer = 1 To heads
                                Dim attentionLayer As New AttentionLayer(inputSize, querySize, keySize, valueSize)
                                attentionLayers.Add(attentionLayer)
                            Next
                        End Sub

                        Public Function Attend(queries As List(Of Scaler), keys As List(Of Scaler), values As List(Of Scaler)) As List(Of Scaler)
                            If queries.Count <> heads OrElse keys.Count <> heads OrElse values.Count <> heads Then
                                Throw New ArgumentException("Number of queries, keys, and values must match the number of heads.")
                            End If

                            Dim attendedOutputs As New List(Of Scaler)()

                            ' Attend using each attention head and concatenate the results
                            For i As Integer = 0 To heads - 1
                                Dim attendedOutput As List(Of Scaler) = attentionLayers(i).Attend(queries, keys, values)
                                attendedOutputs.AddRange(attendedOutput)
                            Next

                            Return attendedOutputs
                        End Function

                        Public Overrides Function Parameters() As List(Of Scaler)
                            Dim iparameters As List(Of Scaler) = New List(Of Scaler)()

                            ' Gather parameters from all attention heads
                            For Each attentionLayer As AttentionLayer In attentionLayers
                                iparameters.AddRange(attentionLayer.Parameters())
                            Next

                            Return iparameters
                        End Function

                        Public Overrides Function ToString() As String
                            Return $"MultiheadAttentionLayer with {heads} heads"
                        End Function
                    End Class
                    Public Class ResidualLayer
                        Inherits AutoGradientParameters

                        Public Function AddResidual(inputs As List(Of Scaler), residual As List(Of Scaler)) As List(Of Scaler)
                            ' Ensure that the input and residual lists have the same length
                            If inputs.Count <> residual.Count Then
                                Throw New ArgumentException("Input and residual lists must have the same length for the residual layer.")
                            End If

                            Dim result As New List(Of Scaler)()

                            ' Perform element-wise addition of inputs and residual
                            For i As Integer = 0 To inputs.Count - 1
                                Dim sum As Scaler = inputs(i) + residual(i)
                                result.Add(sum)
                            Next

                            Return result
                        End Function

                        Public Overrides Function Parameters() As List(Of Scaler)
                            ' This layer doesn't have any learnable parameters, so return an empty list
                            Return New List(Of Scaler)()
                        End Function
                    End Class
                    Public Class AttentionLayer
                        Inherits AutoGradientParameters

                        Private Wq As List(Of Scaler) ' Query weights
                        Private Wk As List(Of Scaler) ' Key weights
                        Private Wv As List(Of Scaler) ' Value weights

                        Public Sub New(inputSize As Integer, querySize As Integer, keySize As Integer, valueSize As Integer)
                            ' Initialize the attention layer with weights
                            Wq = New List(Of Scaler)()
                            Wk = New List(Of Scaler)()
                            Wv = New List(Of Scaler)()

                            ' Initialize weights with random values (you can use different initialization methods)
                            For i As Integer = 1 To querySize
                                Wq.Add(New Scaler(Rnd() * 2 - 1)) ' Query weights
                            Next

                            For i As Integer = 1 To keySize
                                Wk.Add(New Scaler(Rnd() * 2 - 1)) ' Key weights
                            Next

                            For i As Integer = 1 To valueSize
                                Wv.Add(New Scaler(Rnd() * 2 - 1)) ' Value weights
                            Next
                        End Sub

                        Public Function Attend(queries As List(Of Scaler), keys As List(Of Scaler), values As List(Of Scaler)) As List(Of Scaler)
                            ' Compute attention scores using dot-product attention
                            Dim attentionScores As New List(Of Scaler)()

                            For Each query In queries
                                Dim scores As New List(Of Scaler)()

                                For Each key In keys
                                    Dim score As Scaler = query.Dot(key) ' Dot product attention
                                    scores.Add(score)
                                Next

                                ' Apply softmax to the scores
                                Dim softmaxScores As List(Of Scaler) = Softmax(scores)

                                ' Calculate the weighted sum of values
                                Dim weightedSum As Scaler = GetWeightedSum(values, softmaxScores)

                                attentionScores.Add(weightedSum)
                            Next

                            ' Return the attended vectors
                            Return attentionScores
                        End Function

                        Private Function Softmax(scores As List(Of Scaler)) As List(Of Scaler)
                            ' Compute the softmax of a list of scores
                            Dim expScores As List(Of Scaler) = scores.Select(Function(score) score.Exp()).ToList()
                            Dim sumExpScores As Scaler = Scaler.Sum(expScores)

                            Dim softmaxValues As New List(Of Scaler)()

                            For Each expScore In expScores
                                Dim softmaxScore As Scaler = expScore / sumExpScores
                                softmaxValues.Add(softmaxScore)
                            Next

                            Return softmaxValues
                        End Function

                        Private Function GetWeightedSum(values As List(Of Scaler), weights As List(Of Scaler)) As Scaler
                            ' Calculate the weighted sum of values based on attention scores
                            Dim iweightedSum As Scaler = New Scaler(0)

                            For i As Integer = 0 To values.Count - 1
                                iweightedSum += values(i) * weights(i)
                            Next

                            Return iweightedSum
                        End Function
                    End Class
                    Public Class MathMulLayer
                        Inherits AutoGradientParameters

                        Public Function Multiply(inputs1 As List(Of Scaler), inputs2 As List(Of Scaler)) As List(Of Scaler)
                            ' Ensure that the two input lists have the same length
                            If inputs1.Count <> inputs2.Count Then
                                Throw New ArgumentException("Input lists must have the same length for element-wise multiplication.")
                            End If

                            Dim result As New List(Of Scaler)()

                            ' Perform element-wise multiplication
                            For i As Integer = 0 To inputs1.Count - 1
                                Dim product As Scaler = inputs1(i) * inputs2(i)
                                result.Add(product)
                            Next

                            Return result
                        End Function

                        Public Overrides Function Parameters() As List(Of Scaler)
                            ' This layer doesn't have any learnable parameters, so return an empty list
                            Return New List(Of Scaler)()
                        End Function
                    End Class

                    Public Class ReLULayer
                        Inherits Layer

                        Public Sub New(nin As Integer, nout As Integer)
                            MyBase.New(nin, nout)
                        End Sub

                        Public Sub New(nin As Integer, nout As Integer, applyNonLinearActivation As Boolean)
                            MyBase.New(nin, nout, applyNonLinearActivation)
                        End Sub

                        Public Function Forward(input As Vector) As Vector
                            ' Apply ReLU activation to each element of the input vector
                            Dim outputData As New List(Of Double)()
                            For Each element In input.data
                                outputData.Add(Math.Max(0, element.data))
                            Next
                            Return New Vector(outputData, input, "ReLU")
                        End Function
                    End Class
                    Public Class TanhLayer
                        Inherits Layer

                        Public Sub New(nin As Integer, nout As Integer)
                            MyBase.New(nin, nout)
                        End Sub

                        Public Sub New(nin As Integer, nout As Integer, applyNonLinearActivation As Boolean)
                            MyBase.New(nin, nout, applyNonLinearActivation)
                        End Sub

                        Public Function Forward(input As Vector) As Vector
                            ' Apply Tanh activation to each element of the input vector
                            Dim outputData As New List(Of Double)()
                            For Each element In input.data
                                outputData.Add(Math.Tanh(element.data))
                            Next
                            Return New Vector(outputData, input, "Tanh")
                        End Function
                    End Class
                    Public Class SigmoidLayer
                        Inherits Layer

                        Public Sub New(nin As Integer, nout As Integer)
                            MyBase.New(nin, nout)
                        End Sub

                        Public Sub New(nin As Integer, nout As Integer, applyNonLinearActivation As Boolean)
                            MyBase.New(nin, nout, applyNonLinearActivation)
                        End Sub

                        Public Function Forward(input As Vector) As Vector
                            ' Apply Sigmoid activation to each element of the input vector
                            Dim outputData As New List(Of Double)()
                            For Each element In input.data
                                outputData.Add(1 / (1 + Math.Exp(-element.data)))
                            Next
                            Return New Vector(outputData, input, "Sigmoid")
                        End Function
                    End Class
                End Namespace
                Public Class TransformerModel
                    Inherits LayerBlock ' Inherit from LayerBlock

                    Private transformerBlocks As List(Of TransformerBlock)

                    Public Sub New(numBlocks As Integer, encoderHeads As Integer, decoderHeads As Integer, inputSize As Integer, querySize As Integer, keySize As Integer, valueSize As Integer)
                        transformerBlocks = New List(Of TransformerBlock)()

                        ' Create and add multiple Transformer blocks to the model
                        For i As Integer = 1 To numBlocks
                            Dim transformerBlock As New TransformerBlock(encoderHeads, decoderHeads, inputSize, querySize, keySize, valueSize)
                            AddBlock(transformerBlock)
                        Next
                    End Sub

                    Public Sub AddBlock(block As TransformerBlock)
                        transformerBlocks.Add(block)
                    End Sub

                    Public Shadows Function Compute(inputs As List(Of Scaler)) As List(Of Scaler)
                        Dim output As List(Of Scaler) = inputs

                        ' Pass the inputs through each Transformer block in the model
                        For Each block In transformerBlocks
                            output = block.Forward(output, output) ' Use the same output as both encoder and decoder input for simplicity
                        Next

                        Return output
                    End Function

                    Public Overrides Function Parameters() As List(Of Scaler)
                        Dim iparameters As List(Of Scaler) = New List(Of Scaler)()

                        ' Include parameters from all Transformer blocks in the model
                        For Each block In transformerBlocks
                            iparameters.AddRange(block.Parameters())
                        Next

                        Return iparameters
                    End Function
                End Class
                Public Class MLPWithMultiheadAttention
                    Inherits AutoGradientParameters

                    Private layers As List(Of Layer)
                    Private multiheadAttentionLayer As MultiheadAttentionLayer ' Add a multihead attention layer

                    Public Sub New(nin As Integer, nouts As List(Of Integer), attentionQuerySize As Integer, attentionKeySize As Integer, attentionValueSize As Integer, attentionHeads As Integer)
                        layers = New List(Of Layer)()
                        Dim sz As List(Of Integer) = New List(Of Integer)()
                        sz.Add(nin)
                        sz.AddRange(nouts)

                        For i As Integer = 0 To nouts.Count - 1
                            Dim isLastLayer As Boolean = (i = nouts.Count - 1)
                            Dim applyNonLinearActivation As Boolean = Not isLastLayer
                            layers.Add(New Layer(sz(i), sz(i + 1), applyNonLinearActivation))
                        Next

                        ' Initialize the multihead attention layer
                        multiheadAttentionLayer = New MultiheadAttentionLayer(nin, attentionQuerySize, attentionKeySize, attentionValueSize, attentionHeads)
                    End Sub

                    Public Function ComputeWithMultiheadAttention(x As List(Of Scaler), queries As List(Of Scaler), keys As List(Of Scaler), values As List(Of Scaler)) As List(Of Scaler)
                        ' Compute the output of the MLP with multihead attention
                        Dim output As List(Of Scaler) = x

                        ' Forward pass through layers
                        For Each layer As Layer In layers
                            output = layer.Compute(output)
                        Next

                        ' Use the multihead attention layer to attend to the given queries, keys, and values
                        Dim attendedOutputs As List(Of Scaler) = multiheadAttentionLayer.Attend(queries, keys, values)

                        ' Concatenate the output of the MLP and the attended outputs from multihead attention
                        output.AddRange(attendedOutputs)

                        Return output
                    End Function

                    Public Overrides Function Parameters() As List(Of Scaler)
                        Dim iparameters As List(Of Scaler) = New List(Of Scaler)()

                        For Each layer As Layer In layers
                            iparameters.AddRange(layer.Parameters())
                        Next

                        ' Add multihead attention layer parameters
                        iparameters.AddRange(multiheadAttentionLayer.Parameters())

                        Return iparameters
                    End Function

                    Public Overrides Function ToString() As String
                        Dim layerDescriptions As New List(Of String)()

                        For Each layer As Layer In layers
                            layerDescriptions.Add(layer.ToString())
                        Next

                        Return $"MLP with Multihead Attention of [{String.Join(", ", layerDescriptions)}]"
                    End Function
                End Class
                Public Class MLPWithAttention
                    Inherits AutoGradientParameters

                    Private layers As List(Of Layer)
                    Private attentionLayer As AttentionLayer ' Add an attention layer

                    Public Sub New(nin As Integer, nouts As List(Of Integer), attentionQuerySize As Integer, attentionKeySize As Integer, attentionValueSize As Integer)
                        layers = New List(Of Layer)()
                        Dim sz As List(Of Integer) = New List(Of Integer)()
                        sz.Add(nin)
                        sz.AddRange(nouts)

                        For i As Integer = 0 To nouts.Count - 1
                            Dim isLastLayer As Boolean = (i = nouts.Count - 1)
                            Dim applyNonLinearActivation As Boolean = Not isLastLayer
                            layers.Add(New Layer(sz(i), sz(i + 1), applyNonLinearActivation))
                        Next

                        ' Initialize the attention layer
                        attentionLayer = New AttentionLayer(nin, attentionQuerySize, attentionKeySize, attentionValueSize)
                    End Sub

                    Public Function ComputeWithAttention(x As List(Of Scaler), queries As List(Of Scaler), keys As List(Of Scaler), values As List(Of Scaler)) As List(Of Scaler)
                        ' Compute the output of the MLP with attention
                        Dim output As List(Of Scaler) = x

                        ' Forward pass through layers
                        For Each layer As Layer In layers
                            output = layer.Compute(output)
                        Next

                        ' Use the attention layer to attend to the given queries, keys, and values
                        Dim attendedValues As List(Of Scaler) = attentionLayer.Attend(queries, keys, values)

                        ' Concatenate the output of the MLP and the attended values
                        output.AddRange(attendedValues)

                        Return output
                    End Function

                    Public Overrides Function Parameters() As List(Of Scaler)
                        Dim iparameters As List(Of Scaler) = New List(Of Scaler)()

                        For Each layer As Layer In layers
                            iparameters.AddRange(layer.Parameters())
                        Next

                        ' Add attention layer parameters
                        iparameters.AddRange(attentionLayer.Parameters())

                        Return iparameters
                    End Function

                    Public Overrides Function ToString() As String
                        Dim layerDescriptions As New List(Of String)()

                        For Each layer As Layer In layers
                            layerDescriptions.Add(layer.ToString())
                        Next

                        Return $"MLP with Attention of [{String.Join(", ", layerDescriptions)}]"
                    End Function
                End Class
            End Namespace
        End Namespace
    End Namespace
End Namespace

