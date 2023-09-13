       ''' <summary>
            ''' Add full stop to end of String
            ''' </summary>
            ''' <param name="MESSAGE"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function AddFullStop(ByRef MESSAGE As String) As String
                AddFullStop = MESSAGE
                If MESSAGE = "" Then Exit Function
                MESSAGE = Trim(MESSAGE)
                If MESSAGE Like "*." Then Exit Function
                AddFullStop = MESSAGE + "."
            End Function

            ''' <summary>
            ''' Adds string to end of string (no spaces)
            ''' </summary>
            ''' <param name="Str">base string</param>
            ''' <param name="Prefix">Add before (no spaces)</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function AddPrefix(ByRef Str As String, ByVal Prefix As String) As String
                Return Prefix & Str
            End Function



            ''' <summary>
            ''' GO THROUGH EACH CHARACTER AND ' IF PUNCTUATION IE .!?,:'"; REPLACE WITH A SPACE ' IF ,
            ''' OR . THEN CHECK IF BETWEEN TWO NUMBERS, IF IT IS ' THEN LEAVE IT, ELSE REPLACE IT WITH A
            ''' SPACE '
            ''' </summary>
            ''' <param name="STRINPUT">String to be formatted</param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function AlphaNumericalOnly(ByRef STRINPUT As String) As String

                Dim A As Short
                For A = 1 To Len(STRINPUT)
                    If Mid(STRINPUT, A, 1) = "." Or
                Mid(STRINPUT, A, 1) = "!" Or
                Mid(STRINPUT, A, 1) = "?" Or
                Mid(STRINPUT, A, 1) = "," Or
                Mid(STRINPUT, A, 1) = ":" Or
                Mid(STRINPUT, A, 1) = "'" Or
                Mid(STRINPUT, A, 1) = "[" Or
                Mid(STRINPUT, A, 1) = """" Or
                Mid(STRINPUT, A, 1) = ";" Then

                        ' BEGIN CHECKING PERIODS AND COMMAS THAT ARE IN BETWEEN NUMBERS '
                        If Mid(STRINPUT, A, 1) = "." Or Mid(STRINPUT, A, 1) = "," Then
                            If Not (A - 1 = 0 Or A = Len(STRINPUT)) Then
                                If Not (IsNumeric(Mid(STRINPUT, A - 1, 1)) Or IsNumeric(Mid(STRINPUT, A + 1, 1))) Then
                                    STRINPUT = Mid(STRINPUT, 1, A - 1) & " " & Mid(STRINPUT, A + 1, Len(STRINPUT) - A)
                                End If
                            Else
                                STRINPUT = Mid(STRINPUT, 1, A - 1) & " " & Mid(STRINPUT, A + 1, Len(STRINPUT) - A)
                            End If
                        Else
                            STRINPUT = Mid(STRINPUT, 1, A - 1) & " " & Mid(STRINPUT, A + 1, Len(STRINPUT) - A)
                        End If

                        ' END CHECKING PERIODS AND COMMAS IN BETWEEN NUMBERS '
                    End If
                Next A
                ' RETURN PUNCTUATION STRIPPED STRING '
                AlphaNumericalOnly = STRINPUT.Replace("  ", " ")
            End Function
                   'Text
            <Runtime.CompilerServices.Extension()>
            Public Function Capitalise(ByRef MESSAGE As String) As String
                Dim FirstLetter As String
                Capitalise = ""
                If MESSAGE = "" Then Exit Function
                FirstLetter = Left(MESSAGE, 1)
                FirstLetter = UCase(FirstLetter)
                MESSAGE = Right(MESSAGE, Len(MESSAGE) - 1)
                Capitalise = (FirstLetter + MESSAGE)
            End Function

            ''' <summary>
            ''' Capitalizes the text
            ''' </summary>
            ''' <param name="MESSAGE"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function CapitaliseTEXT(ByVal MESSAGE As String) As String
                Dim FirstLetter As String = ""
                CapitaliseTEXT = ""
                If MESSAGE = "" Then Exit Function
                FirstLetter = Left(MESSAGE, 1)
                FirstLetter = UCase(FirstLetter)
                MESSAGE = Right(MESSAGE, Len(MESSAGE) - 1)
                CapitaliseTEXT = (FirstLetter + MESSAGE)
            End Function
          ''' <summary>
            ''' Capitalise the first letter of each word / Tilte Case
            ''' </summary>
            ''' <param name="words">A string - paragraph or sentence</param>
            ''' <returns>String</returns>
            <Runtime.CompilerServices.Extension()>
            Public Function CapitalizeWords(ByVal words As String)
                Dim output As System.Text.StringBuilder = New System.Text.StringBuilder()
                Dim exploded = words.Split(" ")
                If (exploded IsNot Nothing) Then
                    For Each word As String In exploded
                        If word IsNot Nothing Then
                            output.Append(word.Substring(0, 1).ToUpper).Append(word.Substring(1, word.Length - 1)).Append(" ")
                        End If

                    Next
                End If

                Return output.ToString()

            End Function
        ''' <summary>
            ''' Checks if String Contains Letters
            ''' </summary>
            ''' <param name="str"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function ContainsLetters(ByVal str As String) As Boolean

                For i = 0 To str.Length - 1
                    If Char.IsLetter(str.Chars(i)) Then
                        Return True
                    End If
                Next

                Return False

            End Function

                      ''' <summary>
            ''' Counts the number of elements in the text, useful for declaring arrays when the element
            ''' length is unknown could be used to split sentence on full stop Find Sentences then again
            ''' on comma(conjunctions) "Find Clauses" NumberOfElements = CountElements(Userinput, delimiter)
            ''' </summary>
            ''' <param name="PHRASE"></param>
            ''' <param name="Delimiter"></param>
            ''' <returns>Integer : number of elements found</returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function CountElements(ByVal PHRASE As String, ByVal Delimiter As String) As Integer
                Dim elementcounter As Integer = 0
                Dim PhraseArray As String()
                PhraseArray = PHRASE.Split(Delimiter)
                elementcounter = UBound(PhraseArray)
                Return elementcounter
            End Function

            ''' <summary>
            ''' counts occurrences of a specific phoneme
            ''' </summary>
            ''' <param name="strIn"></param>
            ''' <param name="strFind"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            <Runtime.CompilerServices.Extension()>
            Public Function CountOccurrences(ByRef strIn As String, ByRef strFind As String) As Integer
                '**
                ' Returns: the number of times a string appears in a string
                '
                '@rem           Example code for CountOccurrences()
                '
                '  ' Counts the occurrences of "ow" in the supplied string.
                '
                '    strTmp = "How now, brown cow"
                '    Returns a value of 4
                '
                '
                'Debug.Print "CountOccurrences(): there are " &  CountOccurrences(strTmp, "ow") &
                '" occurrences of 'ow'" &    " in the string '" & strTmp & "'"
                '
                '@param        strIn Required. String.
                '@param        strFind Required. String.
                '@return       Long.

                Dim lngPos As Integer
                Dim lngWordCount As Integer

                On Error GoTo PROC_ERR

                lngWordCount = 1

                ' Find the first occurrence
                lngPos = InStr(strIn, strFind)

                Do While lngPos > 0
                    ' Find remaining occurrences
                    lngPos = InStr(lngPos + 1, strIn, strFind)
                    If lngPos > 0 Then
                        ' Increment the hit counter
                        lngWordCount = lngWordCount + 1
                    End If
                Loop

                ' Return the value
                CountOccurrences = lngWordCount

PROC_EXIT:
                Exit Function

PROC_ERR:
                MsgBox("Error: " & Err.Number & ". " & Err.Description, , NameOf(CountOccurrences))
                Resume PROC_EXIT

            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function CountVowels(ByVal InputString As String) As Integer
                Dim v(9) As String 'Declare an array  of 10 elements 0 to 9
                Dim vcount As Short 'This variable will contain number of vowels
                Dim flag As Integer
                Dim strLen As Integer
                Dim i As Integer
                v(0) = "a" 'First element of array is assigned small a
                v(1) = "i"
                v(2) = "o"
                v(3) = "u"
                v(4) = "e"
                v(5) = "A" 'Sixth element is assigned Capital A
                v(6) = "I"
                v(7) = "O"
                v(8) = "U"
                v(9) = "E"
                strLen = Len(InputString)

                For flag = 1 To strLen 'It will get every letter of entered string and loop
                    'will terminate when all letters have been examined

                    For i = 0 To 9 'Takes every element of v(9) one by one
                        'Check if current letter is a vowel
                        If Mid(InputString, flag, 1) = v(i) Then
                            vcount = vcount + 1 ' If letter is equal to vowel
                            'then increment vcount by 1
                        End If
                    Next i 'Consider next value of v(i)
                Next flag 'Consider next letter of the entered string

                CountVowels = vcount

            End Function

            ''' <summary>
            ''' Counts tokens in string
            ''' </summary>
            ''' <param name="Str">string to be searched</param>
            ''' <param name="Delimiter">delimiter such as space comma etc</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function CountTokensInString(ByRef Str As String, ByRef Delimiter As String) As Integer
                Dim Words() As String = Split(Str, Delimiter)
                Return Words.Count
            End Function

            ''' <summary>
            ''' Counts the words in a given text
            ''' </summary>
            ''' <param name="NewText"></param>
            ''' <returns>integer: number of words</returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function CountWords(NewText As String) As Integer
                Dim TempArray() As String = NewText.Split(" ")
                CountWords = UBound(TempArray)
                Return CountWords
            End Function

            ''' <summary>
            ''' checks Str contains keyword regardless of case
            ''' </summary>
            ''' <param name="Userinput"></param>
            ''' <param name="Keyword"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function DetectKeyWord(ByRef Userinput As String, ByRef Keyword As String) As Boolean
                Dim mfound As Boolean = False
                If UCase(Userinput).Contains(UCase(Keyword)) = True Or
                    InStr(Userinput, Keyword) > 1 Then
                    mfound = True
                End If

                Return mfound
            End Function

            ''' <summary>
            ''' DETECT IF STATMENT IS AN IF/THEN DETECT IF STATMENT IS AN IF/THEN -- -RETURNS PARTS DETIFTHEN
            ''' = DETECTLOGIC(USERINPUT, "IF", "THEN", IFPART, THENPART)
            ''' </summary>
            ''' <param name="userinput"></param>
            ''' <param name="LOGICA">"IF", can also be replace by "IT CAN BE SAID THAT</param>
            ''' <param name="LOGICB">"THEN" can also be replaced by "it must follow that"</param>
            ''' <param name="IFPART">supply empty string to be used to hold part</param>
            ''' <param name="THENPART">supply empty string to be used to hold part</param>
            ''' <returns>true/false</returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function DetectLOGIC(ByRef userinput As String, ByRef LOGICA As String, ByRef LOGICB As String, ByRef IFPART As String, ByRef THENPART As String) As Boolean
                If InStr(1, userinput, LOGICA, 1) > 0 And InStr(1, userinput, " " & LOGICB & " ", 1) > 0 Then
                    'SPLIT USER INPUT
                    Call SplitPhrase(userinput, " " & LOGICB & " ", IFPART, THENPART)

                    IFPART = Replace(IFPART, LOGICA, "", 1, -1, CompareMethod.Text)
                    THENPART = Replace(THENPART, " " & LOGICB & " ", "", 1, -1, CompareMethod.Text)
                    DetectLOGIC = True
                Else
                    DetectLOGIC = False
                End If
            End Function

            ''' <summary>
            ''' Expand a string such as a field name by inserting a space ahead of each capitalized
            ''' letter (where none exists).
            ''' </summary>
            ''' <param name="inputString"></param>
            ''' <returns>Expanded string</returns>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function ExpandToWords(ByVal inputString As String) As String
                If inputString Is Nothing Then Return Nothing
                Dim charArray = inputString.ToCharArray
                Dim outStringBuilder As New System.Text.StringBuilder(inputString.Length + 10)
                For index = 0 To charArray.GetUpperBound(0)
                    If Char.IsUpper(charArray(index)) Then
                        'If previous character is also uppercase, don't expand as this may be an acronym.
                        If (index > 0) AndAlso Char.IsUpper(charArray(index - 1)) Then
                            outStringBuilder.Append(charArray(index))
                        Else
                            outStringBuilder.Append(String.Concat(" ", charArray(index)))
                        End If
                    Else
                        outStringBuilder.Append(charArray(index))
                    End If
                Next

                Return outStringBuilder.ToString.Replace("_", " ").Trim

            End Function
            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractFirstChar(ByRef InputStr As String) As String

                ExtractFirstChar = Left(InputStr, 1)
            End Function

            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractFirstWord(ByRef Statement As String) As String
                Dim StrArr() As String = Split(Statement, " ")
                Return StrArr(0)
            End Function

            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractLastChar(ByRef InputStr As String) As String

                ExtractLastChar = Right(InputStr, 1)
            End Function

            ''' <summary>
            ''' Returns The last word in String
            ''' NOTE: String ois converted to Array then the last element is extracted Count-1
            ''' </summary>
            ''' <param name="InputStr"></param>
            ''' <returns>String</returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractLastWord(ByRef InputStr As String) As String
                Dim TempArr() As String = Split(InputStr, " ")
                Dim Count As Integer = TempArr.Count - 1
                Return TempArr(Count)
            End Function
         ''' <summary>
            ''' extracts string between defined strings
            ''' </summary>
            ''' <param name="value">base sgtring</param>
            ''' <param name="strStart">Start string</param>
            ''' <param name="strEnd">End string</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function ExtractStringBetween(ByVal value As String, ByVal strStart As String, ByVal strEnd As String) As String
                If Not String.IsNullOrEmpty(value) Then
                    Dim i As Integer = value.IndexOf(strStart)
                    Dim j As Integer = value.IndexOf(strEnd)
                    Return value.Substring(i, j - i)
                Else
                    Return value
                End If
            End Function

            ''' <summary>
            ''' Extract words Either side of Divider
            ''' </summary>
            ''' <param name="TextStr"></param>
            ''' <param name="Divider"></param>
            ''' <param name="Mode">Front = F Back =B</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function ExtractWordsEitherSide(ByRef TextStr As String, ByRef Divider As String, ByRef Mode As String) As String
                ExtractWordsEitherSide = ""
                Select Case Mode
                    Case "F"
                        Return ExtractWordsEitherSide(TextStr, Divider, "F")
                    Case "B"
                        Return ExtractWordsEitherSide(TextStr, Divider, "B")
                End Select

            End Function

            ' Generate a random number based on the upper and lower bounds of the array,
            'then use that to return the item.
            <Runtime.CompilerServices.Extension()>
            Public Function FetchRandomItem(Of t)(ByRef theArray() As t) As t

                Dim randNumberGenerator As New Random
                Randomize()
                Dim index As Integer = randNumberGenerator.Next(theArray.GetLowerBound(0),
                                                        theArray.GetUpperBound(0) + 1)

                Return theArray(index)

            End Function

            ''' <summary>
            ''' Define the search terms. This list could also be dynamically populated at runtime Find
            ''' sentences that contain all the terms in the wordsToMatch array Note that the number of
            ''' terms to match is not specified at compile time
            ''' </summary>
            ''' <param name="TextStr1">String to be searched</param>
            ''' <param name="Words">List of Words to be detected</param>
            ''' <returns>Sentences containing words</returns>
            <Runtime.CompilerServices.Extension()>
            Public Function FindSentencesContaining(ByRef TextStr1 As String, ByRef Words As List(Of String)) As List(Of String)
                ' Split the text block into an array of sentences.
                Dim sentences As String() = TextStr1.Split(New Char() {".", "?", "!"})

                Dim wordsToMatch(Words.Count) As String
                Dim I As Integer = 0
                For Each item In Words
                    wordsToMatch(I) = item
                    I += 1
                Next

                Dim sentenceQuery = From sentence In sentences
                                    Let w = sentence.Split(New Char() {" ", ",", ".", ";", ":"},
                                                   StringSplitOptions.RemoveEmptyEntries)
                                    Where w.Distinct().Intersect(wordsToMatch).Count = wordsToMatch.Count()
                                    Select sentence

                ' Execute the query

                Dim StrList As New List(Of String)
                For Each str As String In sentenceQuery
                    StrList.Add(str)
                Next
                Return StrList
            End Function

            <Runtime.CompilerServices.Extension()>
            Public Function FormatJsonOutput(ByVal jsonString As String) As String
                Dim stringBuilder = New StringBuilder()
                Dim escaping As Boolean = False
                Dim inQuotes As Boolean = False
                Dim indentation As Integer = 0

                For Each character As Char In jsonString

                    If escaping Then
                        escaping = False
                        stringBuilder.Append(character)
                    Else

                        If character = "\"c Then
                            escaping = True
                            stringBuilder.Append(character)
                        ElseIf character = """"c Then
                            inQuotes = Not inQuotes
                            stringBuilder.Append(character)
                        ElseIf Not inQuotes Then

                            If character = ","c Then
                                stringBuilder.Append(character)
                                stringBuilder.Append(vbCrLf)
                                stringBuilder.Append(vbTab, indentation)
                            ElseIf character = "["c OrElse character = "{"c Then
                                stringBuilder.Append(character)
                                stringBuilder.Append(vbCrLf)
                                stringBuilder.Append(vbTab, System.Threading.Interlocked.Increment(indentation))
                            ElseIf character = "]"c OrElse character = "}"c Then
                                stringBuilder.Append(vbCrLf)
                                stringBuilder.Append(vbTab, System.Threading.Interlocked.Decrement(indentation))
                                stringBuilder.Append(character)
                            ElseIf character = ":"c Then
                                stringBuilder.Append(character)
                                stringBuilder.Append(vbTab)
                            ElseIf Not Char.IsWhiteSpace(character) Then
                                stringBuilder.Append(character)
                            End If
                        Else
                            stringBuilder.Append(character)
                        End If
                    End If
                Next

                Return stringBuilder.ToString()
            End Function
            ''' <summary>
            ''' Gets the string after the given string parameter.
            ''' </summary>
            ''' <param name="value">The default value.</param>
            ''' <param name="x">The given string parameter.</param>
            ''' <returns></returns>
            ''' <remarks>Unlike GetBefore, this method trims the result</remarks>
            <System.Runtime.CompilerServices.Extension>
            Public Function GetAfter(value As String, x As String) As String
                Dim xPos = value.LastIndexOf(x, StringComparison.Ordinal)
                If xPos = -1 Then
                    Return [String].Empty
                End If
                Dim startIndex = xPos + x.Length
                Return If(startIndex >= value.Length, [String].Empty, value.Substring(startIndex).Trim())
            End Function

            ''' <summary>
            ''' Gets the string before the given string parameter.
            ''' </summary>
            ''' <param name="value">The default value.</param>
            ''' <param name="x">The given string parameter.</param>
            ''' <returns></returns>
            ''' <remarks>Unlike GetBetween and GetAfter, this does not Trim the result.</remarks>
            <System.Runtime.CompilerServices.Extension>
            Public Function GetBefore(value As String, x As String) As String
                Dim xPos = value.IndexOf(x, StringComparison.Ordinal)
                Return If(xPos = -1, [String].Empty, value.Substring(0, xPos))
            End Function

            ''' <summary>
            ''' Gets the string between the given string parameters.
            ''' </summary>
            ''' <param name="value">The source value.</param>
            ''' <param name="x">The left string sentinel.</param>
            ''' <param name="y">The right string sentinel</param>
            ''' <returns></returns>
            ''' <remarks>Unlike GetBefore, this method trims the result</remarks>
            <System.Runtime.CompilerServices.Extension>
            Public Function GetBetween(value As String, x As String, y As String) As String
                Dim xPos = value.IndexOf(x, StringComparison.Ordinal)
                Dim yPos = value.LastIndexOf(y, StringComparison.Ordinal)
                If xPos = -1 OrElse xPos = -1 Then
                    Return [String].Empty
                End If
                Dim startIndex = xPos + x.Length
                Return If(startIndex >= yPos, [String].Empty, value.Substring(startIndex, yPos - startIndex).Trim())
            End Function

            ''' <summary>
            ''' Returns the first Word
            ''' </summary>
            ''' <param name="Statement"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function GetPrefix(ByRef Statement As String) As String
                Dim StrArr() As String = Split(Statement, " ")
                Return StrArr(0)
            End Function
            ''' <summary>
            ''' Returns random character from string given length of the string to choose from
            ''' </summary>
            ''' <param name="Source"></param>
            ''' <param name="Length"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function GetRndChar(ByVal Source As String, ByVal Length As Integer) As String
                Dim rnd As New Random
                If Source Is Nothing Then Throw New ArgumentNullException(NameOf(Source), "Must contain a string,")
                If Length <= 0 Then Throw New ArgumentException("Length must be a least one.", NameOf(Length))
                Dim s As String = ""
                Dim builder As New System.Text.StringBuilder()
                builder.Append(s)
                For i = 1 To Length
                    builder.Append(Source(rnd.Next(0, Source.Length)))
                Next
                s = builder.ToString()
                Return s
            End Function

            ''' <summary>
            ''' Returns from index to end of file
            ''' </summary>
            ''' <param name="Str">String</param>
            ''' <param name="indx">Index</param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function GetSlice(ByRef Str As String, ByRef indx As Integer) As String
                If indx <= Str.Length Then
                    Str.Substring(indx, Str.Length)
                    Return Str(indx)
                Else
                End If
                Return Nothing
            End Function

            ''' <summary>
            ''' gets the last word
            ''' </summary>
            ''' <param name="InputStr"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function GetSuffix(ByRef InputStr As String) As String
                Dim TempArr() As String = Split(InputStr, " ")
                Dim Count As Integer = TempArr.Count - 1
                Return TempArr(Count)
            End Function

            <System.Runtime.CompilerServices.Extension>
            Public Function GetWordsBetween(ByRef InputStr As String, ByRef StartStr As String, ByRef StopStr As String)
                Return InputStr.ExtractStringBetween(StartStr, StopStr)
            End Function

            ''' <summary>
            '''     A string extension method that query if '@this' satisfy the specified pattern.
            ''' </summary>
            ''' <param name="this">The @this to act on.</param>
            ''' <param name="pattern">The pattern to use. Use '*' as wildcard string.</param>
            ''' <returns>true if '@this' satisfy the specified pattern, false if not.</returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function IsLike(this As String, pattern As String) As Boolean
                ' Turn the pattern into regex pattern, and match the whole string with ^$
                Dim regexPattern As String = "^" + Regex.Escape(pattern) + "$"

                ' Escape special character ?, #, *, [], and [!]
                regexPattern = regexPattern.Replace("\[!", "[^").Replace("\[", "[").Replace("\]", "]").Replace("\?", ".").Replace("\*", ".*").Replace("\#", "\d")

                Return Regex.IsMatch(this, regexPattern)
            End Function

            ''' <summary>
            ''' Checks if string is a reserved VBscipt Keyword
            ''' </summary>
            ''' <param name="keyword"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Function IsReservedWord(ByVal keyword As String) As Boolean
                Dim IsReserved = False
                Select Case LCase(keyword)
                    Case "and" : IsReserved = True
                    Case "as" : IsReserved = True
                    Case "boolean" : IsReserved = True
                    Case "byref" : IsReserved = True
                    Case "byte" : IsReserved = True
                    Case "byval" : IsReserved = True
                    Case "call" : IsReserved = True
                    Case "case" : IsReserved = True
                    Case "class" : IsReserved = True
                    Case "const" : IsReserved = True
                    Case "currency" : IsReserved = True
                    Case "debug" : IsReserved = True
                    Case "dim" : IsReserved = True
                    Case "do" : IsReserved = True
                    Case "double" : IsReserved = True
                    Case "each" : IsReserved = True
                    Case "else" : IsReserved = True
                    Case "elseif" : IsReserved = True
                    Case "empty" : IsReserved = True
                    Case "end" : IsReserved = True
                    Case "endif" : IsReserved = True
                    Case "enum" : IsReserved = True
                    Case "eqv" : IsReserved = True
                    Case "event" : IsReserved = True
                    Case "exit" : IsReserved = True
                    Case "false" : IsReserved = True
                    Case "for" : IsReserved = True
                    Case "function" : IsReserved = True
                    Case "get" : IsReserved = True
                    Case "goto" : IsReserved = True
                    Case "if" : IsReserved = True
                    Case "imp" : IsReserved = True
                    Case "implements" : IsReserved = True
                    Case "in" : IsReserved = True
                    Case "integer" : IsReserved = True
                    Case "is" : IsReserved = True
                    Case "let" : IsReserved = True
                    Case "like" : IsReserved = True
                    Case "long" : IsReserved = True
                    Case "loop" : IsReserved = True
                    Case "lset" : IsReserved = True
                    Case "me" : IsReserved = True
                    Case "mod" : IsReserved = True
                    Case "new" : IsReserved = True
                    Case "next" : IsReserved = True
                    Case "not" : IsReserved = True
                    Case "nothing" : IsReserved = True
                    Case "null" : IsReserved = True
                    Case "on" : IsReserved = True
                    Case "option" : IsReserved = True
                    Case "optional" : IsReserved = True
                    Case "or" : IsReserved = True
                    Case "paramarray" : IsReserved = True
                    Case "preserve" : IsReserved = True
                    Case "private" : IsReserved = True
                    Case "public" : IsReserved = True
                    Case "raiseevent" : IsReserved = True
                    Case "redim" : IsReserved = True
                    Case "rem" : IsReserved = True
                    Case "resume" : IsReserved = True
                    Case "rset" : IsReserved = True
                    Case "select" : IsReserved = True
                    Case "set" : IsReserved = True
                    Case "shared" : IsReserved = True
                    Case "single" : IsReserved = True
                    Case "static" : IsReserved = True
                    Case "stop" : IsReserved = True
                    Case "sub" : IsReserved = True
                    Case "then" : IsReserved = True
                    Case "to" : IsReserved = True
                    Case "true" : IsReserved = True
                    Case "type" : IsReserved = True
                    Case "typeof" : IsReserved = True
                    Case "until" : IsReserved = True
                    Case "variant" : IsReserved = True
                    Case "wend" : IsReserved = True
                    Case "while" : IsReserved = True
                    Case "with" : IsReserved = True
                    Case "xor" : IsReserved = True
                End Select
                Return IsReserved
            End Function

            ''' <summary>
            ''' Returns Propercase Sentence
            ''' </summary>
            ''' <param name="TheString">String to be formatted</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function ProperCase(ByRef TheString As String) As String
                ProperCase = UCase(Left(TheString, 1))

                For i = 2 To Len(TheString)

                    ProperCase = If(Mid(TheString, i - 1, 1) = " ", ProperCase & UCase(Mid(TheString, i, 1)), ProperCase & LCase(Mid(TheString, i, 1)))
                Next i
            End Function
            <Runtime.CompilerServices.Extension()>
            Public Function RemoveFullStop(ByRef MESSAGE As String) As String
Loop1:
                If Right(MESSAGE, 1) = "." Then MESSAGE = Left(MESSAGE, Len(MESSAGE) - 1) : GoTo Loop1
                Return MESSAGE
            End Function
            ''' <summary>
            ''' Advanced search String pattern Wildcard denotes which position 1st =1 or 2nd =2 Send
            ''' Original input &gt; Search pattern to be used &gt; Wildcard requred SPattern = "WHAT
            ''' COLOUR DO YOU LIKE * OR *" Textstr = "WHAT COLOUR DO YOU LIKE red OR black" ITEM_FOUND =
            ''' = SearchPattern(USERINPUT, SPattern, 1) ---- RETURNS RED ITEM_FOUND = =
            ''' SearchPattern(USERINPUT, SPattern, 1) ---- RETURNS black
            ''' </summary>
            ''' <param name="TextSTR">
            ''' TextStr Required. String.EG: "WHAT COLOUR DO YOU LIKE red OR black"
            ''' </param>
            ''' <param name="SPattern">
            ''' SPattern Required. String.EG: "WHAT COLOUR DO YOU LIKE * OR *"
            ''' </param>
            ''' <param name="Wildcard">Wildcard Required. Integer.EG: 1st =1 or 2nd =2</param>
            ''' <returns></returns>
            ''' <remarks>* in search pattern</remarks>
            <Runtime.CompilerServices.Extension()>
            Public Function SearchPattern(ByRef TextSTR As String, ByRef SPattern As String, ByRef Wildcard As Short) As String
                Dim SearchP2 As String
                Dim SearchP1 As String
                Dim TextStrp3 As String
                Dim TextStrp4 As String
                SearchPattern = ""
                SearchP2 = ""
                SearchP1 = ""
                TextStrp3 = ""
                TextStrp4 = ""
                If TextSTR Like SPattern = True Then
                    Select Case Wildcard
                        Case 1
                            Call SplitPhrase(SPattern, "*", SearchP1, SearchP2)
                            TextSTR = Replace(TextSTR, SearchP1, "", 1, -1, CompareMethod.Text)

                            SearchP2 = Replace(SearchP2, "*", "", 1, -1, CompareMethod.Text)
                            Call SplitPhrase(TextSTR, SearchP2, TextStrp3, TextStrp4)

                            TextSTR = TextStrp3

                        Case 2
                            Call SplitPhrase(SPattern, "*", SearchP1, SearchP2)
                            SPattern = Replace(SPattern, SearchP1, " ", 1, -1, CompareMethod.Text)
                            TextSTR = Replace(TextSTR, SearchP1, " ", 1, -1, CompareMethod.Text)

                            Call SplitPhrase(SearchP2, "*", SearchP1, SearchP2)
                            Call SplitPhrase(TextSTR, SearchP1, TextStrp3, TextStrp4)

                            TextSTR = TextStrp4

                    End Select

                    SearchPattern = TextSTR
                    LTrim(SearchPattern)
                    RTrim(SearchPattern)
                Else
                End If

            End Function

            ''' <summary>
            ''' Advanced search String pattern Wildcard denotes which position 1st =1 or 2nd =2 Send
            ''' Original input &gt; Search pattern to be used &gt; Wildcard requred SPattern = "WHAT
            ''' COLOUR DO YOU LIKE * OR *" Textstr = "WHAT COLOUR DO YOU LIKE red OR black" ITEM_FOUND =
            ''' = SearchPattern(USERINPUT, SPattern, 1) ---- RETURNS RED ITEM_FOUND = =
            ''' SearchPattern(USERINPUT, SPattern, 2) ---- RETURNS black
            ''' </summary>
            ''' <param name="TextSTR">TextStr = "Pick Red OR Blue" . String.</param>
            ''' <param name="SPattern">Search String = ("Pick * OR *") String.</param>
            ''' <param name="Wildcard">Wildcard Required. Integer. = 1= Red / 2= Blue</param>
            ''' <returns></returns>
            ''' <remarks>finds the * in search pattern</remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Function SearchStringbyPattern(ByRef TextSTR As String, ByRef SPattern As String, ByRef Wildcard As Short) As String
                Dim SearchP2 As String
                Dim SearchP1 As String
                Dim TextStrp3 As String
                Dim TextStrp4 As String
                SearchStringbyPattern = ""
                SearchP2 = ""
                SearchP1 = ""
                TextStrp3 = ""
                TextStrp4 = ""
                If TextSTR Like SPattern = True Then
                    Select Case Wildcard
                        Case 1
                            Call SplitString(SPattern, "*", SearchP1, SearchP2)
                            TextSTR = Replace(TextSTR, SearchP1, "", 1, -1, CompareMethod.Text)

                            SearchP2 = Replace(SearchP2, "*", "", 1, -1, CompareMethod.Text)
                            Call SplitString(TextSTR, SearchP2, TextStrp3, TextStrp4)

                            TextSTR = TextStrp3

                        Case 2
                            Call SplitString(SPattern, "*", SearchP1, SearchP2)
                            SPattern = Replace(SPattern, SearchP1, " ", 1, -1, CompareMethod.Text)
                            TextSTR = Replace(TextSTR, SearchP1, " ", 1, -1, CompareMethod.Text)

                            Call SplitString(SearchP2, "*", SearchP1, SearchP2)
                            Call SplitString(TextSTR, SearchP1, TextStrp3, TextStrp4)

                            TextSTR = TextStrp4

                    End Select

                    SearchStringbyPattern = TextSTR
                    LTrim(SearchStringbyPattern)
                    RTrim(SearchStringbyPattern)
                Else
                End If

            End Function
           ''' <summary>
            ''' SPLITS THE GIVEN PHRASE UP INTO TWO PARTS by dividing word SplitPhrase(Userinput, "and",
            ''' Firstp, SecondP)
            ''' </summary>
            ''' <param name="PHRASE">Sentence to be divided</param>
            ''' <param name="DIVIDINGWORD">String: Word to divide sentence by</param>
            ''' <param name="FIRSTPART">String: firstpart of sentence to be populated</param>
            ''' <param name="SECONDPART">String: Secondpart of sentence to be populated</param>
            ''' <remarks></remarks>
            <Runtime.CompilerServices.Extension()>
            Public Sub SplitPhrase(ByVal PHRASE As String, ByRef DIVIDINGWORD As String, ByRef FIRSTPART As String, ByRef SECONDPART As String)
                Dim POS As Short
                POS = InStr(PHRASE, DIVIDINGWORD)
                If (POS > 0) Then
                    FIRSTPART = Trim(Left(PHRASE, POS - 1))
                    SECONDPART = Trim(Right(PHRASE, Len(PHRASE) - POS - Len(DIVIDINGWORD) + 1))
                Else
                    FIRSTPART = ""
                    SECONDPART = PHRASE
                End If
            End Sub

            ''' <summary>
            ''' SPLITS THE GIVEN PHRASE UP INTO TWO PARTS by dividing word SplitPhrase(Userinput, "and",
            ''' Firstp, SecondP)
            ''' </summary>
            ''' <param name="PHRASE">String: Sentence to be divided</param>
            ''' <param name="DIVIDINGWORD">String: Word to divide sentence by</param>
            ''' <param name="FIRSTPART">String-Returned : firstpart of sentence to be populated</param>
            ''' <param name="SECONDPART">String-Returned : Secondpart of sentence to be populated</param>
            ''' <remarks></remarks>
            <System.Runtime.CompilerServices.Extension()>
            Public Sub SplitString(ByVal PHRASE As String, ByRef DIVIDINGWORD As String, ByRef FIRSTPART As String, ByRef SECONDPART As String)
                Dim POS As Short
                'Check Error
                If DIVIDINGWORD IsNot Nothing And PHRASE IsNot Nothing Then

                    POS = InStr(PHRASE, DIVIDINGWORD)
                    If (POS > 0) Then
                        FIRSTPART = Trim(Left(PHRASE, POS - 1))
                        SECONDPART = Trim(Right(PHRASE, Len(PHRASE) - POS - Len(DIVIDINGWORD) + 1))
                    Else
                        FIRSTPART = ""
                        SECONDPART = PHRASE
                    End If
                Else

                End If
            End Sub

            ''' <summary>
            ''' Split string to List of strings
            ''' </summary>
            ''' <param name="Str">base string</param>
            ''' <param name="Seperator">to be seperated by</param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension()>
            Public Function SplitToList(ByRef Str As String, ByVal Seperator As String) As List(Of String)
                Dim lst As New List(Of String)
                If Str <> "" = True And Seperator <> "" Then

                    Dim Found() As String = Str.Split(Seperator)
                    For Each item In Found
                        lst.Add(item)
                    Next
                Else

                End If
                Return lst
            End Function

            ''' <summary>
            ''' Returns a delimited string from the list.
            ''' </summary>
            ''' <param name="ls"></param>
            ''' <param name="delimiter"></param>
            ''' <returns></returns>
            <System.Runtime.CompilerServices.Extension>
            Public Function ToDelimitedString(ls As List(Of String), delimiter As String) As String
                Dim sb As New StringBuilder
                For Each buf As String In ls
                    sb.Append(buf)
                    sb.Append(delimiter)
                Next
                Return sb.ToString.Trim(CChar(delimiter))
            End Function

            ''' <summary>
            ''' Convert object to Json String
            ''' </summary>
            ''' <param name="Item"></param>
            ''' <returns></returns>
            <Runtime.CompilerServices.Extension()>
            Public Function ToJson(ByRef Item As Object) As String
                Dim Converter As New JavaScriptSerializer
                Return Converter.Serialize(Item)

            End Function

            ''' <summary>
            ''' Counts the vowels used (AEIOU)
            ''' </summary>
            ''' <param name="InputString"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            <Runtime.CompilerServices.Extension()>
            Public Function VowelCount(ByVal InputString As String) As Integer
                Dim v(9) As String 'Declare an array  of 10 elements 0 to 9
                Dim vcount As Short 'This variable will contain number of vowels
                Dim flag As Integer
                Dim strLen As Integer
                Dim i As Integer
                v(0) = "a" 'First element of array is assigned small a
                v(1) = "i"
                v(2) = "o"
                v(3) = "u"
                v(4) = "e"
                v(5) = "A" 'Sixth element is assigned Capital A
                v(6) = "I"
                v(7) = "O"
                v(8) = "U"
                v(9) = "E"
                strLen = Len(InputString)

                For flag = 1 To strLen 'It will get every letter of entered string and loop
                    'will terminate when all letters have been examined

                    For i = 0 To 9 'Takes every element of v(9) one by one
                        'Check if current letter is a vowel
                        If Mid(InputString, flag, 1) = v(i) Then
                            vcount = vcount + 1 ' If letter is equal to vowel
                            'then increment vcount by 1
                        End If
                    Next i 'Consider next value of v(i)
                Next flag 'Consider next letter of the enterd string

                VowelCount = vcount

            End Function

