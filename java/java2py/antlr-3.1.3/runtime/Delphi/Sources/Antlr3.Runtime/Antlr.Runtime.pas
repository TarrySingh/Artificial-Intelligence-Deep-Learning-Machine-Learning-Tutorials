unit Antlr.Runtime;
(*
[The "BSD licence"]
Copyright (c) 2008 Erik van Bilsen
Copyright (c) 2005-2007 Kunle Odutola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code MUST RETAIN the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form MUST REPRODUCE the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior WRITTEN permission.
4. Unless explicitly state otherwise, any contribution intentionally 
   submitted for inclusion in this work to the copyright owner or licensor
   shall be under the terms and conditions of this license, without any 
   additional terms or conditions.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*)

interface

{$IF CompilerVersion < 20}
{$MESSAGE ERROR 'You need Delphi 2009 or higher to use the Antlr runtime'}
{$IFEND}

uses
  SysUtils,
  Classes,
  Generics.Defaults,
  Generics.Collections,
  Antlr.Runtime.Tools,
  Antlr.Runtime.Collections;

type
  TCharStreamConstants = (cscEOF = -1);

type
  ERecognitionException = class;
  ENoViableAltException = class;

  /// <summary>
  /// A simple stream of integers. This is useful when all we care about is the char
  /// or token type sequence (such as for interpretation).
  /// </summary>
  IIntStream = interface(IANTLRInterface)
  ['{6B851BDB-DD9C-422B-AD1E-567E52D2654F}']
    { Property accessors }
    function GetSourceName: String;

    { Methods }
    /// <summary>
    /// Advances the read position of the stream. Updates line and column state
    /// </summary>
    procedure Consume;

    /// <summary>
    /// Get int at current input pointer + I ahead (where I=1 is next int)
    /// Negative indexes are allowed.  LA(-1) is previous token (token just matched).
    /// LA(-i) where i is before first token should yield -1, invalid char or EOF.
    /// </summary>
    function LA(I: Integer): Integer;
    function LAChar(I: Integer): Char;

    /// <summary>Tell the stream to start buffering if it hasn't already.</summary>
    /// <remarks>
    /// Executing Rewind(Mark()) on a stream should not affect the input position.
    /// The Lexer tracks line/col info as well as input index so its markers are
    /// not pure input indexes.  Same for tree node streams.                          */
    /// </remarks>
    /// <returns>Return a marker that can be passed to
    /// <see cref="IIntStream.Rewind(Integer)"/> to return to the current position.
    /// This could be the current input position, a value return from
    /// <see cref="IIntStream.Index"/>, or some other marker.</returns>
    function Mark: Integer;

    /// <summary>
    /// Return the current input symbol index 0..N where N indicates the
    /// last symbol has been read. The index is the symbol about to be
    /// read not the most recently read symbol.
    /// </summary>
    function Index: Integer;

    /// <summary>
    /// Resets the stream so that the next call to
    /// <see cref="IIntStream.Index"/> would  return marker.
    /// </summary>
    /// <remarks>
    /// The marker will usually be <see cref="IIntStream.Index"/> but
    /// it doesn't have to be.  It's just a marker to indicate what
    /// state the stream was in.  This is essentially calling
    /// <see cref="IIntStream.Release"/> and <see cref="IIntStream.Seek"/>.
    /// If there are other markers created after the specified marker,
    /// this routine must unroll them like a stack.  Assumes the state the
    /// stream was in when this marker was created.
    /// </remarks>
    procedure Rewind(const Marker: Integer); overload;

    /// <summary>
    /// Rewind to the input position of the last marker.
    /// </summary>
    /// <remarks>
    /// Used currently only after a cyclic DFA and just before starting
    /// a sem/syn predicate to get the input position back to the start
    /// of the decision. Do not "pop" the marker off the state.  Mark(I)
    /// and Rewind(I) should balance still. It is like invoking
    /// Rewind(last marker) but it should not "pop" the marker off.
    /// It's like Seek(last marker's input position).
    /// </remarks>
    procedure Rewind; overload;

    /// <summary>
    /// You may want to commit to a backtrack but don't want to force the
    /// stream to keep bookkeeping objects around for a marker that is
    /// no longer necessary.  This will have the same behavior as
    /// <see cref="IIntStream.Rewind(Integer)"/> except it releases resources without
    /// the backward seek.
    /// </summary>
    /// <remarks>
    /// This must throw away resources for all markers back to the marker
    /// argument. So if you're nested 5 levels of Mark(), and then Release(2)
    /// you have to release resources for depths 2..5.
    /// </remarks>
    procedure Release(const Marker: Integer);

    /// <summary>
    /// Set the input cursor to the position indicated by index.  This is
    /// normally used to seek ahead in the input stream.
    /// </summary>
    /// <remarks>
    /// No buffering is required to do this unless you know your stream
    /// will use seek to move backwards such as when backtracking.
    ///
    /// This is different from rewind in its multi-directional requirement
    /// and in that its argument is strictly an input cursor (index).
    ///
    /// For char streams, seeking forward must update the stream state such
    /// as line number.  For seeking backwards, you will be presumably
    /// backtracking using the
    /// <see cref="IIntStream.Mark"/>/<see cref="IIntStream.Rewind(Integer)"/>
    /// mechanism that restores state and so this method does not need to
    /// update state when seeking backwards.
    ///
    /// Currently, this method is only used for efficient backtracking using
    /// memoization, but in the future it may be used for incremental parsing.
    ///
    /// The index is 0..N-1. A seek to position i means that LA(1) will return
    /// the ith symbol.  So, seeking to 0 means LA(1) will return the first
    /// element in the stream.
    /// </remarks>
    procedure Seek(const Index: Integer);

    /// <summary>Returns the size of the entire stream.</summary>
    /// <remarks>
    /// Only makes sense for streams that buffer everything up probably,
    /// but might be useful to display the entire stream or for testing.
    /// This value includes a single EOF.
    /// </remarks>
    function Size: Integer;

    { Properties }

    /// <summary>
    /// Where are you getting symbols from?  Normally, implementations will
    /// pass the buck all the way to the lexer who can ask its input stream
    /// for the file name or whatever.
    /// </summary>
    property SourceName: String read GetSourceName;
  end;

  /// <summary>A source of characters for an ANTLR lexer </summary>
  ICharStream = interface(IIntStream)
  ['{C30EF0DB-F4BD-4CBC-8C8F-828DABB6FF36}']
    { Property accessors }
    function GetLine: Integer;
    procedure SetLine(const Value: Integer);
    function GetCharPositionInLine: Integer;
    procedure SetCharPositionInLine(const Value: Integer);

    { Methods }

    /// <summary>
    /// Get the ith character of lookahead.  This is usually the same as
    /// LA(I).  This will be used for labels in the generated lexer code.
    /// I'd prefer to return a char here type-wise, but it's probably
    /// better to be 32-bit clean and be consistent with LA.
    /// </summary>
    function LT(const I: Integer): Integer;

    /// <summary>
    /// This primarily a useful interface for action code (just make sure
    /// actions don't use this on streams that don't support it).
    /// For infinite streams, you don't need this.
    /// </summary>
    function Substring(const Start, Stop: Integer): String;

    { Properties }

    /// <summary>
    /// The current line in the character stream (ANTLR tracks the
    /// line information automatically. To support rewinding character
    /// streams, we are able to [re-]set the line.
    /// </summary>
    property Line: Integer read GetLine write SetLine;

    /// <summary>
    /// The index of the character relative to the beginning of the
    /// line (0..N-1). To support rewinding character streams, we are
    /// able to [re-]set the character position.
    /// </summary>
    property CharPositionInLine: Integer read GetCharPositionInLine write SetCharPositionInLine;
  end;

  IToken = interface(IANTLRInterface)
  ['{73BF129C-2F45-4C68-838E-BF5D3536AC6D}']
    { Property accessors }
    function GetTokenType: Integer;
    procedure SetTokenType(const Value: Integer);
    function GetLine: Integer;
    procedure SetLine(const Value: Integer);
    function GetCharPositionInLine: Integer;
    procedure SetCharPositionInLine(const Value: Integer);
    function GetChannel: Integer;
    procedure SetChannel(const Value: Integer);
    function GetTokenIndex: Integer;
    procedure SetTokenIndex(const Value: Integer);
    function GetText: String;
    procedure SetText(const Value: String);

    { Properties }
    property TokenType: Integer read GetTokenType write SetTokenType;

    /// <summary>The line number on which this token was matched; line=1..N</summary>
    property Line: Integer read GetLine write SetLine;

    /// <summary>
    /// The index of the first character relative to the beginning of the line 0..N-1
    /// </summary>
    property CharPositionInLine: Integer read GetCharPositionInLine write SetCharPositionInLine;

    /// <summary>The line number on which this token was matched; line=1..N</summary>
    property Channel: Integer read GetChannel write SetChannel;

    /// <summary>
    /// An index from 0..N-1 of the token object in the input stream
    /// </summary>
    /// <remarks>
    /// This must be valid in order to use the ANTLRWorks debugger.
    /// </remarks>
    property TokenIndex: Integer read GetTokenIndex write SetTokenIndex;

    /// <summary>The text of the token</summary>
    /// <remarks>
    /// When setting the text, it might be a NOP such as for the CommonToken,
    /// which doesn't have string pointers, just indexes into a char buffer.
    /// </remarks>
    property Text: String read GetText write SetText;
  end;

  /// <summary>
  /// A source of tokens must provide a sequence of tokens via NextToken()
  /// and also must reveal it's source of characters; CommonToken's text is
  /// computed from a CharStream; it only store indices into the char stream.
  ///
  /// Errors from the lexer are never passed to the parser.  Either you want
  /// to keep going or you do not upon token recognition error.  If you do not
  /// want to continue lexing then you do not want to continue parsing.  Just
  /// throw an exception not under RecognitionException and Delphi will naturally
  /// toss you all the way out of the recognizers.  If you want to continue
  /// lexing then you should not throw an exception to the parser--it has already
  /// requested a token.  Keep lexing until you get a valid one.  Just report
  /// errors and keep going, looking for a valid token.
  /// </summary>
  ITokenSource = interface(IANTLRInterface)
  ['{2C71FAD0-AEEE-417D-B576-4059F7C4CEB4}']
    { Property accessors }
    function GetSourceName: String;

    { Methods }

    /// <summary>
    /// Returns a Token object from the input stream (usually a CharStream).
    /// Does not fail/return upon lexing error; just keeps chewing on the
    /// characters until it gets a good one; errors are not passed through
    /// to the parser.
    /// </summary>
    function NextToken: IToken;

    { Properties }

    /// <summary>
    /// Where are you getting tokens from? normally the implication will simply
    /// ask lexers input stream.
    /// </summary>
    property SourceName: String read GetSourceName;
  end;

  /// <summary>A stream of tokens accessing tokens from a TokenSource </summary>
  ITokenStream = interface(IIntStream)
  ['{59E5B39D-31A6-496D-9FA9-AC75CC584B68}']
    { Property accessors }
    function GetTokenSource: ITokenSource;
    procedure SetTokenSource(const Value: ITokenSource);

    { Methods }

    /// <summary>
    /// Get Token at current input pointer + I ahead (where I=1 is next
    /// Token).
    /// I &lt; 0 indicates tokens in the past.  So -1 is previous token and -2 is
    /// two tokens ago. LT(0) is undefined.  For I>=N, return Token.EOFToken.
    /// Return null for LT(0) and any index that results in an absolute address
    /// that is negative.
    /// </summary>
    function LT(const K: Integer): IToken;

    /// <summary>
    /// Get a token at an absolute index I; 0..N-1.  This is really only
    /// needed for profiling and debugging and token stream rewriting.
    /// If you don't want to buffer up tokens, then this method makes no
    /// sense for you.  Naturally you can't use the rewrite stream feature.
    /// I believe DebugTokenStream can easily be altered to not use
    /// this method, removing the dependency.
    /// </summary>
    function Get(const I: Integer): IToken;

    /// <summary>Return the text of all tokens from start to stop, inclusive.
    /// If the stream does not buffer all the tokens then it can just
    /// return '';  Users should not access $ruleLabel.text in
    /// an action of course in that case.
    /// </summary>
    function ToString(const Start, Stop: Integer): String; overload;

    /// <summary>Because the user is not required to use a token with an index stored
    /// in it, we must provide a means for two token objects themselves to
    /// indicate the start/end location.  Most often this will just delegate
    /// to the other ToString(Integer,Integer).  This is also parallel with
    /// the TreeNodeStream.ToString(Object,Object).
    /// </summary>
    function ToString(const Start, Stop: IToken): String; overload;

    { Properties }
    property TokenSource: ITokenSource read GetTokenSource write SetTokenSource;
  end;

  /// <summary>
  /// This is the complete state of a stream.
  ///
  /// When walking ahead with cyclic DFA for syntactic predicates, we
  /// need to record the state of the input stream (char index, line,
  /// etc...) so that we can rewind the state after scanning ahead.
  /// </summary>
  ICharStreamState = interface(IANTLRInterface)
  ['{62D2A1CD-ED3A-4C95-A366-AB8F2E54060B}']
    { Property accessors }
    function GetP: Integer;
    procedure SetP(const Value: Integer);
    function GetLine: Integer;
    procedure SetLine(const Value: Integer);
    function GetCharPositionInLine: Integer;
    procedure SetCharPositionInLine(const Value: Integer);

    { Properties }
    /// <summary>Index into the char stream of next lookahead char </summary>
    property P: Integer read GetP write SetP;

    /// <summary>What line number is the scanner at before processing buffer[P]? </summary>
    property Line: Integer read GetLine write SetLine;

    /// <summary>What char position 0..N-1 in line is scanner before processing buffer[P]? </summary>
    property CharPositionInLine: Integer read GetCharPositionInLine write SetCharPositionInLine;
  end;

  /// <summary>
  /// A pretty quick <see cref="ICharStream"/> that uses a character array
  /// directly as it's underlying source.
  /// </summary>
  IANTLRStringStream = interface(ICharStream)
  ['{2FA24299-FF97-4AB6-8CA6-5D3DA13C4AB2}']
    { Methods }

    /// <summary>
    /// Resets the stream so that it is in the same state it was
    /// when the object was created *except* the data array is not
    /// touched.
    /// </summary>
    procedure Reset;

  end;

  /// <summary>
  /// A character stream - an <see cref="ICharStream"/> - that loads
  /// and caches the contents of it's underlying file fully during
  /// object construction
  /// </summary>
  /// <remarks>
  /// This looks very much like an ANTLReaderStream or an ANTLRInputStream
  /// but, it is a special case. Since we know the exact size of the file to
  /// load, we can avoid lots of data copying and buffer resizing.
  /// </remarks>
  IANTLRFileStream = interface(IANTLRStringStream)
  ['{2B0145DB-2DAA-48A0-8316-B47A69EDDD1A}']
    { Methods }

    /// <summary>
    /// Loads and buffers the specified file to be used as this
    /// ANTLRFileStream's source
    /// </summary>
    /// <param name="FileName">File to load</param>
    /// <param name="Encoding">Encoding to apply to file</param>
    procedure Load(const FileName: String; const Encoding: TEncoding);
  end;

  /// <summary>
  /// A stripped-down version of org.antlr.misc.BitSet that is just
  /// good enough to handle runtime requirements such as FOLLOW sets
  /// for automatic error recovery.
  /// </summary>
  IBitSet = interface(IANTLRInterface)
  ['{F2045045-FC46-4779-A65D-56C65D257A8E}']
    { Property accessors }
    function GetIsNil: Boolean;

    { Methods }

    /// <summary>return "this or a" in a new set </summary>
    function BitSetOr(const A: IBitSet): IBitSet;

    /// <summary>Or this element into this set (grow as necessary to accommodate)</summary>
    procedure Add(const El: Integer);

    /// <summary> Grows the set to a larger number of bits.</summary>
    /// <param name="bit">element that must fit in set
    /// </param>
    procedure GrowToInclude(const Bit: Integer);

    procedure OrInPlace(const A: IBitSet);
    function Size: Integer;
    function Member(const El: Integer): Boolean;

    // remove this element from this set
    procedure Remove(const El: Integer);

    function NumBits: Integer;

    /// <summary>return how much space is being used by the bits array not
    /// how many actually have member bits on.
    /// </summary>
    function LengthInLongWords: Integer;

    function ToArray: TIntegerArray;
    function ToPackedArray: TUInt64Array;

    function ToString: String; overload;
    function ToString(const TokenNames: TStringArray): String; overload;
    function Equals(Obj: TObject): Boolean;

    { Properties }
    property IsNil: Boolean read GetIsNil;
  end;
  TBitSetArray = array of IBitSet;

  /// <summary>
  /// The set of fields needed by an abstract recognizer to recognize input
  /// and recover from errors
  /// </summary>
  /// <remarks>
  /// As a separate state object, it can be shared among multiple grammars;
  /// e.g., when one grammar imports another.
  /// These fields are publicly visible but the actual state pointer per
  /// parser is protected.
  /// </remarks>
  IRecognizerSharedState = interface(IANTLRInterface)
  ['{6CB6E17A-0B01-4AA7-8D49-5742A3CB8901}']
    { Property accessors }
    function GetFollowing: TBitSetArray;
    procedure SetFollowing(const Value: TBitSetArray);
    function GetFollowingStackPointer: Integer;
    procedure SetFollowingStackPointer(const Value: Integer);
    function GetErrorRecovery: Boolean;
    procedure SetErrorRecovery(const Value: Boolean);
    function GetLastErrorIndex: Integer;
    procedure SetLastErrorIndex(const Value: Integer);
    function GetFailed: Boolean;
    procedure SetFailed(const Value: Boolean);
    function GetSyntaxErrors: Integer;
    procedure SetSyntaxErrors(const Value: Integer);
    function GetBacktracking: Integer;
    procedure SetBacktracking(const Value: Integer);
    function GetRuleMemo: TDictionaryArray<Integer, Integer>;
    function GetRuleMemoCount: Integer;
    procedure SetRuleMemoCount(const Value: Integer);
    function GetToken: IToken;
    procedure SetToken(const Value: IToken);
    function GetTokenStartCharIndex: Integer;
    procedure SetTokenStartCharIndex(const Value: Integer);
    function GetTokenStartLine: Integer;
    procedure SetTokenStartLine(const Value: Integer);
    function GetTokenStartCharPositionInLine: Integer;
    procedure SetTokenStartCharPositionInLine(const Value: Integer);
    function GetChannel: Integer;
    procedure SetChannel(const Value: Integer);
    function GetTokenType: Integer;
    procedure SetTokenType(const Value: Integer);
    function GetText: String;
    procedure SetText(const Value: String);

    { Properties }

    /// <summary>
    /// Tracks the set of token types that can follow any rule invocation.
    /// Stack grows upwards.  When it hits the max, it grows 2x in size
    /// and keeps going.
    /// </summary>
    property Following: TBitSetArray read GetFollowing write SetFollowing;
    property FollowingStackPointer: Integer read GetFollowingStackPointer write SetFollowingStackPointer;

    /// <summary>
    /// This is true when we see an error and before having successfully
    /// matched a token.  Prevents generation of more than one error message
    /// per error.
    /// </summary>
    property ErrorRecovery: Boolean read GetErrorRecovery write SetErrorRecovery;

    /// <summary>
    /// The index into the input stream where the last error occurred.
    /// </summary>
    /// <remarks>
    /// This is used to prevent infinite loops where an error is found
    /// but no token is consumed during recovery...another error is found,
    /// ad naseum.  This is a failsafe mechanism to guarantee that at least
    /// one token/tree node is consumed for two errors.
    /// </remarks>
    property LastErrorIndex: Integer read GetLastErrorIndex write SetLastErrorIndex;

    /// <summary>
    /// In lieu of a return value, this indicates that a rule or token
    /// has failed to match.  Reset to false upon valid token match.
    /// </summary>
    property Failed: Boolean read GetFailed write SetFailed;

    /// <summary>
    /// Did the recognizer encounter a syntax error?  Track how many.
    /// </summary>
    property SyntaxErrors: Integer read GetSyntaxErrors write SetSyntaxErrors;

    /// <summary>
    /// If 0, no backtracking is going on.  Safe to exec actions etc...
    /// If >0 then it's the level of backtracking.
    /// </summary>
    property Backtracking: Integer read GetBacktracking write SetBacktracking;

    /// <summary>
    /// An array[size num rules] of Map&lt;Integer,Integer&gt; that tracks
    /// the stop token index for each rule.
    /// </summary>
    /// <remarks>
    ///  RuleMemo[RuleIndex] is the memoization table for RuleIndex.
    ///  For key RuleStartIndex, you get back the stop token for
    ///  associated rule or MEMO_RULE_FAILED.
    ///
    ///  This is only used if rule memoization is on (which it is by default).
    ///  </remarks>
    property RuleMemo: TDictionaryArray<Integer, Integer> read GetRuleMemo;
    property RuleMemoCount: Integer read GetRuleMemoCount write SetRuleMemoCount;

    // Lexer Specific Members
    // LEXER FIELDS (must be in same state object to avoid casting
    //               constantly in generated code and Lexer object) :(

    /// <summary>
    /// Token object normally returned by NextToken() after matching lexer rules.
    /// </summary>
    /// <remarks>
    /// The goal of all lexer rules/methods is to create a token object.
    /// This is an instance variable as multiple rules may collaborate to
    /// create a single token.  NextToken will return this object after
    /// matching lexer rule(s).  If you subclass to allow multiple token
    /// emissions, then set this to the last token to be matched or
    /// something nonnull so that the auto token emit mechanism will not
    /// emit another token.
    /// </remarks>
    property Token: IToken read GetToken write SetToken;

    /// <summary>
    /// What character index in the stream did the current token start at?
    /// </summary>
    /// <remarks>
    /// Needed, for example, to get the text for current token.  Set at
    /// the start of nextToken.
    /// </remarks>
    property TokenStartCharIndex: Integer read GetTokenStartCharIndex write SetTokenStartCharIndex;

    /// <summary>
    /// The line on which the first character of the token resides
    /// </summary>
    property TokenStartLine: Integer read GetTokenStartLine write SetTokenStartLine;

    /// <summary>The character position of first character within the line</summary>
    property TokenStartCharPositionInLine: Integer read GetTokenStartCharPositionInLine write SetTokenStartCharPositionInLine;

    /// <summary>The channel number for the current token</summary>
    property Channel: Integer read GetChannel write SetChannel;

    /// <summary>The token type for the current token</summary>
    property TokenType: Integer read GetTokenType write SetTokenType;

    /// <summary>
    /// You can set the text for the current token to override what is in
    /// the input char buffer.  Use setText() or can set this instance var.
    /// </summary>
    property Text: String read GetText write SetText;
  end;

  ICommonToken = interface(IToken)
  ['{06B1B0C3-2A0D-477A-AE30-414F51ACE8A0}']
    { Property accessors }
    function GetStartIndex: Integer;
    procedure SetStartIndex(const Value: Integer);
    function GetStopIndex: Integer;
    procedure SetStopIndex(const Value: Integer);
    function GetInputStream: ICharStream;
    procedure SetInputStream(const Value: ICharStream);

    { Methods }
    function ToString: String;

    { Properties }
    property StartIndex: Integer read GetStartIndex write SetStartIndex;
    property StopIndex: Integer read GetStopIndex write SetStopIndex;
    property InputStream: ICharStream read GetInputStream write SetInputStream;
  end;

  /// <summary>
  /// A Token object like we'd use in ANTLR 2.x; has an actual string created
  /// and associated with this object.  These objects are needed for imaginary
  /// tree nodes that have payload objects.  We need to create a Token object
  /// that has a string; the tree node will point at this token.  CommonToken
  /// has indexes into a char stream and hence cannot be used to introduce
  /// new strings.
  /// </summary>
  IClassicToken = interface(IToken)
    { Property accessors }
    function GetTokenType: Integer;
    procedure SetTokenType(const Value: Integer);
    function GetLine: Integer;
    procedure SetLine(const Value: Integer);
    function GetCharPositionInLine: Integer;
    procedure SetCharPositionInLine(const Value: Integer);
    function GetChannel: Integer;
    procedure SetChannel(const Value: Integer);
    function GetTokenIndex: Integer;
    procedure SetTokenIndex(const Value: Integer);
    function GetText: String;
    procedure SetText(const Value: String);
    function GetInputStream: ICharStream;
    procedure SetInputStream(const Value: ICharStream);

    { Properties }
    property TokenType: Integer read GetTokenType write SetTokenType;
    property Line: Integer read GetLine write SetLine;
    property CharPositionInLine: Integer read GetCharPositionInLine write SetCharPositionInLine;
    property Channel: Integer read GetChannel write SetChannel;
    property TokenIndex: Integer read GetTokenIndex write SetTokenIndex;
    property Text: String read GetText write SetText;
    property InputStream: ICharStream read GetInputStream write SetInputStream;
  end;

  /// <summary>
  /// A generic recognizer that can handle recognizers generated from
  /// lexer, parser, and tree grammars.  This is all the parsing
  /// support code essentially; most of it is error recovery stuff and
  /// backtracking.
  /// </summary>
  IBaseRecognizer = interface(IANTLRObject)
  ['{90813CE2-614B-4773-A26E-936E7DE7E9E9}']
    { Property accessors }
    function GetInput: IIntStream;
    function GetBacktrackingLevel: Integer;
    function GetState: IRecognizerSharedState;
    function GetNumberOfSyntaxErrors: Integer;
    function GetGrammarFileName: String;
    function GetSourceName: String;
    function GetTokenNames: TStringArray;

    { Methods }
    procedure BeginBacktrack(const Level: Integer);
    procedure EndBacktrack(const Level: Integer; const Successful: Boolean);

    /// <summary>Reset the parser's state. Subclasses must rewind the input stream.</summary>
    procedure Reset;

    /// <summary>
    /// Match current input symbol against ttype.  Attempt
    /// single token insertion or deletion error recovery.  If
    /// that fails, throw EMismatchedTokenException.
    /// </summary>
    /// <remarks>
    /// To turn off single token insertion or deletion error
    /// recovery, override MismatchRecover() and have it call
    /// plain Mismatch(), which does not recover. Then any error
    /// in a rule will cause an exception and immediate exit from
    /// rule. Rule would recover by resynchronizing to the set of
    /// symbols that can follow rule ref.
    /// </remarks>
    function Match(const Input: IIntStream; const TokenType: Integer;
      const Follow: IBitSet): IANTLRInterface;

    function MismatchIsUnwantedToken(const Input: IIntStream;
      const TokenType: Integer): Boolean;

    function MismatchIsMissingToken(const Input: IIntStream;
      const Follow: IBitSet): Boolean;

    /// <summary>A hook to listen in on the token consumption during error recovery.
    /// The DebugParser subclasses this to fire events to the listenter.
    /// </summary>
    procedure BeginResync;
    procedure EndResync;

    /// <summary>
    /// Report a recognition problem.
    /// </summary>
    /// <remarks>
    /// This method sets errorRecovery to indicate the parser is recovering
    /// not parsing.  Once in recovery mode, no errors are generated.
    /// To get out of recovery mode, the parser must successfully Match
    /// a token (after a resync).  So it will go:
    ///
    /// 1. error occurs
    /// 2. enter recovery mode, report error
    /// 3. consume until token found in resynch set
    /// 4. try to resume parsing
    /// 5. next Match() will reset errorRecovery mode
    ///
    /// If you override, make sure to update syntaxErrors if you care about that.
    /// </remarks>
    procedure ReportError(const E: ERecognitionException);

    /// <summary> Match the wildcard: in a symbol</summary>
    procedure MatchAny(const Input: IIntStream);

    procedure DisplayRecognitionError(const TokenNames: TStringArray;
      const E: ERecognitionException);

    /// <summary>
    /// What error message should be generated for the various exception types?
    ///
    /// Not very object-oriented code, but I like having all error message generation
    /// within one method rather than spread among all of the exception classes. This
    /// also makes it much easier for the exception handling because the exception
    /// classes do not have to have pointers back to this object to access utility
    /// routines and so on. Also, changing the message for an exception type would be
    /// difficult because you would have to subclassing exception, but then somehow get
    /// ANTLR to make those kinds of exception objects instead of the default.
    ///
    /// This looks weird, but trust me--it makes the most sense in terms of flexibility.
    ///
    /// For grammar debugging, you will want to override this to add more information
    /// such as the stack frame with GetRuleInvocationStack(e, this.GetType().Fullname)
    /// and, for no viable alts, the decision description and state etc...
    ///
    /// Override this to change the message generated for one or more exception types.
    /// </summary>
    function GetErrorMessage(const E: ERecognitionException;
      const TokenNames: TStringArray): String;

    /// <summary>
    /// What is the error header, normally line/character position information?
    /// </summary>
    function GetErrorHeader(const E: ERecognitionException): String;

    /// <summary>
    /// How should a token be displayed in an error message? The default
    /// is to display just the text, but during development you might
    /// want to have a lot of information spit out.  Override in that case
    /// to use t.ToString() (which, for CommonToken, dumps everything about
    /// the token). This is better than forcing you to override a method in
    /// your token objects because you don't have to go modify your lexer
    /// so that it creates a new type.
    /// </summary>
    function GetTokenErrorDisplay(const T: IToken): String;

    /// <summary>
    /// Override this method to change where error messages go
    /// </summary>
    procedure EmitErrorMessage(const Msg: String);

    /// <summary>
    /// Recover from an error found on the input stream.  This is
    /// for NoViableAlt and mismatched symbol exceptions.  If you enable
    /// single token insertion and deletion, this will usually not
    /// handle mismatched symbol exceptions but there could be a mismatched
    /// token that the Match() routine could not recover from.
    /// </summary>
    procedure Recover(const Input: IIntStream; const RE: ERecognitionException);

    // Not currently used
    function RecoverFromMismatchedSet(const Input: IIntStream;
      const E: ERecognitionException; const Follow: IBitSet): IANTLRInterface;

    procedure ConsumeUntil(const Input: IIntStream; const TokenType: Integer); overload;

    /// <summary>Consume tokens until one matches the given token set </summary>
    procedure ConsumeUntil(const Input: IIntStream; const BitSet: IBitSet); overload;

    /// <summary>
    /// Returns List &lt;String&gt; of the rules in your parser instance
    /// leading up to a call to this method.  You could override if
    /// you want more details such as the file/line info of where
    /// in the parser source code a rule is invoked.
    /// </summary>
    /// <remarks>
    /// NOT IMPLEMENTED IN THE DELPHI VERSION YET
    /// This is very useful for error messages and for context-sensitive
    /// error recovery.
    /// </remarks>
    //function GetRuleInvocationStack: IList<IANTLRInterface>; overload;

    /// <summary>
    /// A more general version of GetRuleInvocationStack where you can
    /// pass in, for example, a RecognitionException to get it's rule
    /// stack trace.  This routine is shared with all recognizers, hence,
    /// static.
    ///
    /// TODO: move to a utility class or something; weird having lexer call this
    /// </summary>
    /// <remarks>
    /// NOT IMPLEMENTED IN THE DELPHI VERSION YET
    /// </remarks>
    //function GetRuleInvocationStack(const E: Exception;
    //  const RecognizerClassName: String): IList<IANTLRInterface>; overload;

    /// <summary>A convenience method for use most often with template rewrites.
    /// Convert a List&lt;Token&gt; to List&lt;String&gt;
    /// </summary>
    function ToStrings(const Tokens: IList<IToken>): IList<String>;

    /// <summary>
    /// Given a rule number and a start token index number, return
    /// MEMO_RULE_UNKNOWN if the rule has not parsed input starting from
    /// start index.  If this rule has parsed input starting from the
    /// start index before, then return where the rule stopped parsing.
    /// It returns the index of the last token matched by the rule.
    /// </summary>
    /// <remarks>
    /// For now we use a hashtable and just the slow Object-based one.
    /// Later, we can make a special one for ints and also one that
    /// tosses out data after we commit past input position i.
    /// </remarks>
    function GetRuleMemoization(const RuleIndex, RuleStartIndex: Integer): Integer;

    /// <summary>
    /// Has this rule already parsed input at the current index in the
    /// input stream?  Return the stop token index or MEMO_RULE_UNKNOWN.
    /// If we attempted but failed to parse properly before, return
    /// MEMO_RULE_FAILED.
    ///
    /// This method has a side-effect: if we have seen this input for
    /// this rule and successfully parsed before, then seek ahead to
    /// 1 past the stop token matched for this rule last time.
    /// </summary>
    function AlreadyParsedRule(const Input: IIntStream;
      const RuleIndex: Integer): Boolean;

    /// <summary>
    /// Record whether or not this rule parsed the input at this position
    /// successfully.  Use a standard hashtable for now.
    /// </summary>
    procedure Memoize(const Input: IIntStream; const RuleIndex,
      RuleStartIndex: Integer);

    /// <summary>
    /// Return how many rule/input-index pairs there are in total.
    ///  TODO: this includes synpreds. :(
    /// </summary>
    /// <returns></returns>
    function GetRuleMemoizationChaceSize: Integer;

    procedure TraceIn(const RuleName: String; const RuleIndex: Integer;
      const InputSymbol: String);
    procedure TraceOut(const RuleName: String; const RuleIndex: Integer;
      const InputSymbol: String);

    { Properties }
    property Input: IIntStream read GetInput;
    property BacktrackingLevel: Integer read GetBacktrackingLevel;
    property State: IRecognizerSharedState read GetState;

    /// <summary>
    /// Get number of recognition errors (lexer, parser, tree parser).  Each
    /// recognizer tracks its own number.  So parser and lexer each have
    /// separate count.  Does not count the spurious errors found between
    /// an error and next valid token match
    ///
    /// See also ReportError()
    /// </summary>
    property NumberOfSyntaxErrors: Integer read GetNumberOfSyntaxErrors;

    /// <summary>
    /// For debugging and other purposes, might want the grammar name.
    /// Have ANTLR generate an implementation for this property.
    /// </summary>
    /// <returns></returns>
    property GrammarFileName: String read GetGrammarFileName;

    /// <summary>
    /// For debugging and other purposes, might want the source name.
    /// Have ANTLR provide a hook for this property.
    /// </summary>
    /// <returns>The source name</returns>
    property SourceName: String read GetSourceName;

    /// <summary>
    /// Used to print out token names like ID during debugging and
    /// error reporting.  The generated parsers implement a method
    /// that overrides this to point to their string[] tokenNames.
    /// </summary>
    property TokenNames: TStringArray read GetTokenNames;
  end;

  /// <summary>
  /// The most common stream of tokens is one where every token is buffered up
  /// and tokens are prefiltered for a certain channel (the parser will only
  /// see these tokens and cannot change the filter channel number during the
  /// parse).
  ///
  /// TODO: how to access the full token stream?  How to track all tokens matched per rule?
  /// </summary>
  ICommonTokenStream = interface(ITokenStream)
    { Methods }

    /// <summary>
    /// A simple filter mechanism whereby you can tell this token stream
    /// to force all tokens of type TType to be on Channel.
    /// </summary>
    ///
    /// <remarks>
    /// For example,
    /// when interpreting, we cannot exec actions so we need to tell
    /// the stream to force all WS and NEWLINE to be a different, ignored
    /// channel.
    /// </remarks>
    procedure SetTokenTypeChannel(const TType, Channel: Integer);

    procedure DiscardTokenType(const TType: Integer);

    procedure DiscardOffChannelTokens(const Discard: Boolean);

    function GetTokens: IList<IToken>; overload;
    function GetTokens(const Start, Stop: Integer): IList<IToken>; overload;

    /// <summary>Given a start and stop index, return a List of all tokens in
    /// the token type BitSet.  Return null if no tokens were found.  This
    /// method looks at both on and off channel tokens.
    /// </summary>
    function GetTokens(const Start, Stop: Integer;
      const Types: IBitSet): IList<IToken>; overload;

    function GetTokens(const Start, Stop: Integer;
      const Types: IList<Integer>): IList<IToken>; overload;

    function GetTokens(const Start, Stop,
      TokenType: Integer): IList<IToken>; overload;

    procedure Reset;
  end;

  IDFA = interface;

  TSpecialStateTransitionHandler = function(const DFA: IDFA; S: Integer;
    const Input: IIntStream): Integer of Object;

  /// <summary>
  ///  A DFA implemented as a set of transition tables.
  /// </summary>
  /// <remarks>
  /// <para>
  /// Any state that has a semantic predicate edge is special; those states are
  /// generated with if-then-else structures in a SpecialStateTransition()
  /// which is generated by cyclicDFA template.
  /// </para>
  /// <para>
  /// There are at most 32767 states (16-bit signed short). Could get away with byte
  /// sometimes but would have to generate different types and the simulation code too.
  /// </para>
  /// <para>
  /// As a point of reference, the Tokens rule DFA for the lexer in the Java grammar
  /// sample has approximately 326 states.
  /// </para>
  /// </remarks>
  IDFA = interface(IANTLRInterface)
  ['{36312B59-B718-48EF-A0EC-4529DE70F4C2}']
    { Property accessors }
    function GetSpecialStateTransitionHandler: TSpecialStateTransitionHandler;
    procedure SetSpecialStateTransitionHandler(const Value: TSpecialStateTransitionHandler);

    { Methods }

    /// <summary>
    /// From the input stream, predict what alternative will succeed using this
    /// DFA (representing the covering regular approximation to the underlying CFL).
    /// </summary>
    /// <param name="Input">Input stream</param>
    /// <returns>Return an alternative number 1..N.  Throw an exception upon error.</returns>
    function Predict(const Input: IIntStream): Integer;

    /// <summary>
    /// A hook for debugging interface
    /// </summary>
    /// <param name="NVAE"></param>
    procedure Error(const NVAE: ENoViableAltException);

    function SpecialStateTransition(const S: Integer; const Input: IIntStream): Integer;

    function Description: String;

    function SpecialTransition(const State, Symbol: Integer): Integer;

    { Properties }
    property SpecialStateTransitionHandler: TSpecialStateTransitionHandler read GetSpecialStateTransitionHandler write SetSpecialStateTransitionHandler;
  end;

  /// <summary>
  /// A lexer is recognizer that draws input symbols from a character stream.
  /// lexer grammars result in a subclass of this object. A Lexer object
  /// uses simplified Match() and error recovery mechanisms in the interest
  /// of speed.
  /// </summary>
  ILexer = interface(IBaseRecognizer)
  ['{331AAB49-E7CD-40E7-AEF5-427F7D6577AD}']
    { Property accessors }
    function GetCharStream: ICharStream;
    procedure SetCharStream(const Value: ICharStream);
    function GetLine: Integer;
    function GetCharPositionInLine: Integer;
    function GetCharIndex: Integer;
    function GetText: String;
    procedure SetText(const Value: String);

    { Methods }

    /// <summary>
    /// Return a token from this source; i.e., Match a token on the char stream.
    /// </summary>
    function NextToken: IToken;

    /// <summary>
    /// Instruct the lexer to skip creating a token for current lexer rule and
    /// look for another token.  NextToken() knows to keep looking when a lexer
    /// rule finishes with token set to SKIP_TOKEN.  Recall that if token==null
    /// at end of any token rule, it creates one for you and emits it.
    /// </summary>
    procedure Skip;

    /// <summary>This is the lexer entry point that sets instance var 'token' </summary>
    procedure DoTokens;

    /// <summary>
    /// Currently does not support multiple emits per nextToken invocation
    /// for efficiency reasons.  Subclass and override this method and
    /// NextToken (to push tokens into a list and pull from that list rather
    /// than a single variable as this implementation does).
    /// </summary>
    procedure Emit(const Token: IToken); overload;

    /// <summary>
    /// The standard method called to automatically emit a token at the
    /// outermost lexical rule.  The token object should point into the
    /// char buffer start..stop.  If there is a text override in 'text',
    /// use that to set the token's text.
    /// </summary>
    /// <remarks><para>Override this method to emit custom Token objects.</para>
    /// <para>If you are building trees, then you should also override
    /// Parser or TreeParser.GetMissingSymbol().</para>
    ///</remarks>
    function Emit: IToken; overload;

    procedure Match(const S: String); overload;
    procedure Match(const C: Integer); overload;
    procedure MatchAny;
    procedure MatchRange(const A, B: Integer);

    /// <summary>
    /// Lexers can normally Match any char in it's vocabulary after matching
    /// a token, so do the easy thing and just kill a character and hope
    /// it all works out.  You can instead use the rule invocation stack
    /// to do sophisticated error recovery if you are in a Fragment rule.
    /// </summary>
    procedure Recover(const RE: ERecognitionException);

    function GetCharErrorDisplay(const C: Integer): String;

    procedure TraceIn(const RuleName: String; const RuleIndex: Integer);
    procedure TraceOut(const RuleName: String; const RuleIndex: Integer);

    { Properties }

    /// <summary>Set the char stream and reset the lexer </summary>
    property CharStream: ICharStream read GetCharStream write SetCharStream;
    property Line: Integer read GetLine;
    property CharPositionInLine: Integer read GetCharPositionInLine;

    /// <summary>What is the index of the current character of lookahead? </summary>
    property CharIndex: Integer read GetCharIndex;

    /// <summary>
    /// Gets or sets the 'lexeme' for the current token.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The getter returns the text matched so far for the current token or any
    /// text override.
    /// </para>
    /// <para>
    /// The setter sets the complete text of this token. It overrides/wipes any
    /// previous changes to the text.
    /// </para>
    /// </remarks>
    property Text: String read GetText write SetText;
  end;

  /// <summary>A parser for TokenStreams.  Parser grammars result in a subclass
  /// of this.
  /// </summary>
  IParser = interface(IBaseRecognizer)
  ['{7420879A-5D1F-43CA-BD49-2264D7514501}']
    { Property accessors }
    function GetTokenStream: ITokenStream;
    procedure SetTokenStream(const Value: ITokenStream);

    { Methods }
    procedure TraceIn(const RuleName: String; const RuleIndex: Integer);
    procedure TraceOut(const RuleName: String; const RuleIndex: Integer);

    { Properties }

    /// <summary>Set the token stream and reset the parser </summary>
    property TokenStream: ITokenStream read GetTokenStream write SetTokenStream;
  end;

  /// <summary>
  /// Rules can return start/stop info as well as possible trees and templates
  /// </summary>
  IRuleReturnScope = interface(IANTLRInterface)
  ['{E9870056-BF6D-4CB2-B71C-10B80797C0B4}']
    { Property accessors }
    function GetStart: IANTLRInterface;
    procedure SetStart(const Value: IANTLRInterface);
    function GetStop: IANTLRInterface;
    procedure SetStop(const Value: IANTLRInterface);
    function GetTree: IANTLRInterface;
    procedure SetTree(const Value: IANTLRInterface);
    function GetTemplate: IANTLRInterface;

    { Properties }

    /// <summary>Return the start token or tree </summary>
    property Start: IANTLRInterface read GetStart write SetStart;

    /// <summary>Return the stop token or tree </summary>
    property Stop: IANTLRInterface read GetStop write SetStop;

    /// <summary>Has a value potentially if output=AST; </summary>
    property Tree: IANTLRInterface read GetTree write SetTree;

    /// <summary>
    /// Has a value potentially if output=template;
    /// Don't use StringTemplate type to avoid dependency on ST assembly
    /// </summary>
    property Template: IANTLRInterface read GetTemplate;
  end;

  /// <summary>
  /// Rules that return more than a single value must return an object
  /// containing all the values.  Besides the properties defined in
  /// RuleLabelScope.PredefinedRulePropertiesScope there may be user-defined
  /// return values.  This class simply defines the minimum properties that
  /// are always defined and methods to access the others that might be
  /// available depending on output option such as template and tree.
  ///
  /// Note text is not an actual property of the return value, it is computed
  /// from start and stop using the input stream's ToString() method.  I
  /// could add a ctor to this so that we can pass in and store the input
  /// stream, but I'm not sure we want to do that.  It would seem to be undefined
  /// to get the .text property anyway if the rule matches tokens from multiple
  /// input streams.
  ///
  /// I do not use getters for fields of objects that are used simply to
  /// group values such as this aggregate.
  /// </summary>
  IParserRuleReturnScope = interface(IRuleReturnScope)
  ['{9FB62050-E23B-4FE4-87D5-2C1EE67AEC3E}']
  end;

  /// <summary>Useful for dumping out the input stream after doing some
  /// augmentation or other manipulations.
  /// </summary>
  ///
  /// <remarks>
  /// You can insert stuff, Replace, and delete chunks.  Note that the
  /// operations are done lazily--only if you convert the buffer to a
  /// String.  This is very efficient because you are not moving data around
  /// all the time.  As the buffer of tokens is converted to strings, the
  /// ToString() method(s) check to see if there is an operation at the
  /// current index.  If so, the operation is done and then normal String
  /// rendering continues on the buffer.  This is like having multiple Turing
  /// machine instruction streams (programs) operating on a single input tape. :)
  ///
  /// Since the operations are done lazily at ToString-time, operations do not
  /// screw up the token index values.  That is, an insert operation at token
  /// index I does not change the index values for tokens I+1..N-1.
  ///
  /// Because operations never actually alter the buffer, you may always get
  /// the original token stream back without undoing anything.  Since
  /// the instructions are queued up, you can easily simulate transactions and
  /// roll back any changes if there is an error just by removing instructions.
  /// For example,
  ///
  /// var
  ///   Input: ICharStream;
  ///   Lex: ILexer;
  ///   Tokens: ITokenRewriteStream;
  ///   Parser: IParser;
  /// Input := TANTLRFileStream.Create('input');
  /// Lex := TLexer.Create(Input);
  /// Tokens := TTokenRewriteStream.Create(Lex);
  /// Parser := TParser.Create(tokens);
  /// Parser.startRule();
  ///
  /// Then in the rules, you can execute
  /// var
  ///   t,u: IToken;
  /// ...
  /// Input.InsertAfter(t, 'text to put after t');
  /// Input.InsertAfter(u, 'text after u');
  /// WriteLn(Tokens.ToString());
  ///
  /// Actually, you have to cast the 'input' to a TokenRewriteStream. :(
  ///
  /// You can also have multiple "instruction streams" and get multiple
  /// rewrites from a single pass over the input.  Just name the instruction
  /// streams and use that name again when printing the buffer.  This could be
  /// useful for generating a C file and also its header file--all from the
  /// same buffer:
  ///
  /// Tokens.InsertAfter('pass1', t, 'text to put after t');
  /// Tokens.InsertAfter('pass2', u, 'text after u');
  /// WriteLn(Tokens.ToString('pass1'));
  /// WriteLn(Tokens.ToString('pass2'));
  ///
  /// If you don't use named rewrite streams, a "default" stream is used as
  /// the first example shows.
  /// </remarks>
  ITokenRewriteStream = interface(ICommonTokenStream)
  ['{7B49CBB6-9395-4781-B616-F201889EEA13}']
    { Methods }
    procedure Rollback(const InstructionIndex: Integer); overload;

    /// <summary>Rollback the instruction stream for a program so that
    /// the indicated instruction (via instructionIndex) is no
    /// longer in the stream.  UNTESTED!
    /// </summary>
    procedure Rollback(const ProgramName: String;
      const InstructionIndex: Integer); overload;

    procedure DeleteProgram; overload;

    /// <summary>Reset the program so that no instructions exist </summary>
    procedure DeleteProgram(const ProgramName: String); overload;

    procedure InsertAfter(const T: IToken; const Text: IANTLRInterface); overload;
    procedure InsertAfter(const Index: Integer; const Text: IANTLRInterface); overload;
    procedure InsertAfter(const ProgramName: String; const T: IToken;
      const Text: IANTLRInterface); overload;
    procedure InsertAfter(const ProgramName: String; const Index: Integer;
      const Text: IANTLRInterface); overload;
    procedure InsertAfter(const T: IToken; const Text: String); overload;
    procedure InsertAfter(const Index: Integer; const Text: String); overload;
    procedure InsertAfter(const ProgramName: String; const T: IToken;
      const Text: String); overload;
    procedure InsertAfter(const ProgramName: String; const Index: Integer;
      const Text: String); overload;

    procedure InsertBefore(const T: IToken; const Text: IANTLRInterface); overload;
    procedure InsertBefore(const Index: Integer; const Text: IANTLRInterface); overload;
    procedure InsertBefore(const ProgramName: String; const T: IToken;
      const Text: IANTLRInterface); overload;
    procedure InsertBefore(const ProgramName: String; const Index: Integer;
      const Text: IANTLRInterface); overload;
    procedure InsertBefore(const T: IToken; const Text: String); overload;
    procedure InsertBefore(const Index: Integer; const Text: String); overload;
    procedure InsertBefore(const ProgramName: String; const T: IToken;
      const Text: String); overload;
    procedure InsertBefore(const ProgramName: String; const Index: Integer;
      const Text: String); overload;

    procedure Replace(const Index: Integer; const Text: IANTLRInterface); overload;
    procedure Replace(const Start, Stop: Integer; const Text: IANTLRInterface); overload;
    procedure Replace(const IndexT: IToken; const Text: IANTLRInterface); overload;
    procedure Replace(const Start, Stop: IToken; const Text: IANTLRInterface); overload;
    procedure Replace(const ProgramName: String; const Start, Stop: Integer;
      const Text: IANTLRInterface); overload;
    procedure Replace(const ProgramName: String; const Start, Stop: IToken;
      const Text: IANTLRInterface); overload;
    procedure Replace(const Index: Integer; const Text: String); overload;
    procedure Replace(const Start, Stop: Integer; const Text: String); overload;
    procedure Replace(const IndexT: IToken; const Text: String); overload;
    procedure Replace(const Start, Stop: IToken; const Text: String); overload;
    procedure Replace(const ProgramName: String; const Start, Stop: Integer;
      const Text: String); overload;
    procedure Replace(const ProgramName: String; const Start, Stop: IToken;
      const Text: String); overload;

    procedure Delete(const Index: Integer); overload;
    procedure Delete(const Start, Stop: Integer); overload;
    procedure Delete(const IndexT: IToken); overload;
    procedure Delete(const Start, Stop: IToken); overload;
    procedure Delete(const ProgramName: String; const Start, Stop: Integer); overload;
    procedure Delete(const ProgramName: String; const Start, Stop: IToken); overload;

    function GetLastRewriteTokenIndex: Integer;

    function ToOriginalString: String; overload;
    function ToOriginalString(const Start, Stop: Integer): String; overload;

    function ToString(const ProgramName: String): String; overload;
    function ToString(const ProgramName: String;
      const Start, Stop: Integer): String; overload;

    function ToDebugString: String; overload;
    function ToDebugString(const Start, Stop: Integer): String; overload;
  end;

  /// <summary>The root of the ANTLR exception hierarchy.</summary>
  /// <remarks>
  /// To avoid English-only error messages and to generally make things
  /// as flexible as possible, these exceptions are not created with strings,
  /// but rather the information necessary to generate an error.  Then
  /// the various reporting methods in Parser and Lexer can be overridden
  /// to generate a localized error message.  For example, MismatchedToken
  /// exceptions are built with the expected token type.
  /// So, don't expect getMessage() to return anything.
  ///
  /// You can access the stack trace, which means that you can compute the
  /// complete trace of rules from the start symbol. This gives you considerable
  /// context information with which to generate useful error messages.
  ///
  /// ANTLR generates code that throws exceptions upon recognition error and
  /// also generates code to catch these exceptions in each rule.  If you
  /// want to quit upon first error, you can turn off the automatic error
  /// handling mechanism using rulecatch action, but you still need to
  /// override methods mismatch and recoverFromMismatchSet.
  ///
  /// In general, the recognition exceptions can track where in a grammar a
  /// problem occurred and/or what was the expected input.  While the parser
  /// knows its state (such as current input symbol and line info) that
  /// state can change before the exception is reported so current token index
  /// is computed and stored at exception time.  From this info, you can
  /// perhaps print an entire line of input not just a single token, for example.
  /// Better to just say the recognizer had a problem and then let the parser
  /// figure out a fancy report.
  /// </remarks>
  ERecognitionException = class(Exception)
  strict private
    FApproximateLineInfo: Boolean;
  strict protected
    /// <summary>What input stream did the error occur in? </summary>
    FInput: IIntStream;

    /// <summary>
    /// What is index of token/char were we looking at when the error occurred?
    /// </summary>
    FIndex: Integer;

    /// <summary>
    /// The current Token when an error occurred.  Since not all streams
    /// can retrieve the ith Token, we have to track the Token object.
    /// </summary>
    FToken: IToken;

    /// <summary>[Tree parser] Node with the problem.</summary>
    FNode: IANTLRInterface;

    /// <summary>The current char when an error occurred. For lexers. </summary>
    FC: Integer;

    /// <summary>Track the line at which the error occurred in case this is
    /// generated from a lexer.  We need to track this since the
    /// unexpected char doesn't carry the line info.
    /// </summary>
    FLine: Integer;
    FCharPositionInLine: Integer;
  strict protected
    procedure ExtractInformationFromTreeNodeStream(const Input: IIntStream);
    function GetUnexpectedType: Integer; virtual;
  public
    /// <summary>Used for remote debugger deserialization </summary>
    constructor Create; overload;
    constructor Create(const AMessage: String); overload;
    constructor Create(const AInput: IIntStream); overload;
    constructor Create(const AMessage: String; const AInput: IIntStream); overload;

    /// <summary>
    /// If you are parsing a tree node stream, you will encounter some
    /// imaginary nodes w/o line/col info.  We now search backwards looking
    /// for most recent token with line/col info, but notify getErrorHeader()
    /// that info is approximate.
    /// </summary>
    property ApproximateLineInfo: Boolean read FApproximateLineInfo write FApproximateLineInfo;

    /// <summary>
    /// Returns the current Token when the error occurred (for parsers
    /// although a tree parser might also set the token)
    /// </summary>
    property Token: IToken read FToken write FToken;

    /// <summary>
    /// Returns the [tree parser] node where the error occured (for tree parsers).
    /// </summary>
    property Node: IANTLRInterface read FNode write FNode;

    /// <summary>
    /// Returns the line at which the error occurred (for lexers)
    /// </summary>
    property Line: Integer read FLine write FLine;

    /// <summary>
    /// Returns the character position in the line when the error
    /// occurred (for lexers)
    /// </summary>
    property CharPositionInLine: Integer read FCharPositionInLine write FCharPositionInLine;

    /// <summary>Returns the input stream in which the error occurred</summary>
    property Input: IIntStream read FInput write FInput;

    /// <summary>
    /// Returns the token type or char of the unexpected input element
    /// </summary>
    property UnexpectedType: Integer read GetUnexpectedType;

    /// <summary>
    /// Returns the current char when the error occurred (for lexers)
    /// </summary>
    property Character: Integer read FC write FC;

    /// <summary>
    /// Returns the token/char index in the stream when the error occurred
    /// </summary>
    property Index: Integer read FIndex write FIndex;
  end;

  /// <summary>
  /// A mismatched char or Token or tree node.
  /// </summary>
  EMismatchedTokenException = class(ERecognitionException)
  strict private
    FExpecting: Integer;
  public
    constructor Create(const AExpecting: Integer; const AInput: IIntStream);

    function ToString: String; override;

    property Expecting: Integer read FExpecting write FExpecting;
  end;

  EUnwantedTokenException = class(EMismatchedTokenException)
  strict private
    function GetUnexpectedToken: IToken;
  public
    property UnexpectedToken: IToken read GetUnexpectedToken;

    function ToString: String; override;
  end;

  /// <summary>
  /// We were expecting a token but it's not found. The current token
  /// is actually what we wanted next. Used for tree node errors too.
  /// </summary>
  EMissingTokenException = class(EMismatchedTokenException)
  strict private
    FInserted: IANTLRInterface;
    function GetMissingType: Integer;
  public
    constructor Create(const AExpecting: Integer; const AInput: IIntStream;
      const AInserted: IANTLRInterface);

    function ToString: String; override;

    property MissingType: Integer read GetMissingType;
    property Inserted: IANTLRInterface read FInserted write FInserted;
  end;

  EMismatchedTreeNodeException = class(ERecognitionException)
  strict private
    FExpecting: Integer;
  public
    constructor Create(const AExpecting: Integer; const AInput: IIntStream);

    function ToString: String; override;

    property Expecting: Integer read FExpecting write FExpecting;
  end;

  ENoViableAltException = class(ERecognitionException)
  strict private
    FGrammarDecisionDescription: String;
    FDecisionNumber: Integer;
    FStateNumber: Integer;
  public
    constructor Create(const AGrammarDecisionDescription: String;
      const ADecisionNumber, AStateNumber: Integer; const AInput: IIntStream);

    function ToString: String; override;

    property GrammarDecisionDescription: String read FGrammarDecisionDescription;
    property DecisionNumber: Integer read FDecisionNumber;
    property StateNumber: Integer read FStateNumber;
  end;

  EEarlyExitException = class(ERecognitionException)
  strict private
    FDecisionNumber: Integer;
  public
    constructor Create(const ADecisionNumber: Integer; const AInput: IIntStream);

    property DecisionNumber: Integer read FDecisionNumber;
  end;

  EMismatchedSetException = class(ERecognitionException)
  strict private
    FExpecting: IBitSet;
  public
    constructor Create(const AExpecting: IBitSet; const AInput: IIntStream);

    function ToString: String; override;

    property Expecting: IBitSet read FExpecting write FExpecting;
  end;

  EMismatchedNotSetException = class(EMismatchedSetException)

  public
    function ToString: String; override;
  end;

  EFailedPredicateException = class(ERecognitionException)
  strict private
    FRuleName: String;
    FPredicateText: String;
  public
    constructor Create(const AInput: IIntStream; const ARuleName,
      APredicateText: String);

    function ToString: String; override;

    property RuleName: String read FRuleName write FRuleName;
    property PredicateText: String read FPredicateText write FPredicateText;
  end;

  EMismatchedRangeException = class(ERecognitionException)
  strict private
    FA: Integer;
    FB: Integer;
  public
    constructor Create(const AA, AB: Integer; const AInput: IIntStream);

    function ToString: String; override;

    property A: Integer read FA write FA;
    property B: Integer read FB write FB;
  end;

type
  TCharStreamState = class(TANTLRObject, ICharStreamState)
  strict private
    FP: Integer;
    FLine: Integer;
    FCharPositionInLine: Integer;
  protected
    { ICharStreamState }
    function GetP: Integer;
    procedure SetP(const Value: Integer);
    function GetLine: Integer;
    procedure SetLine(const Value: Integer);
    function GetCharPositionInLine: Integer;
    procedure SetCharPositionInLine(const Value: Integer);
  end;

type
  TANTLRStringStream = class(TANTLRObject, IANTLRStringStream, ICharStream)
  private
    FData: PChar;
    FOwnsData: Boolean;

    /// <summary>How many characters are actually in the buffer?</summary>
    FN: Integer;

    /// <summary>Current line number within the input (1..n )</summary>
    FLine: Integer;

    /// <summary>Index in our array for the next char (0..n-1)</summary>
    FP: Integer;

    /// <summary>
    /// The index of the character relative to the beginning of the
    /// line (0..n-1)
    /// </summary>
    FCharPositionInLine: Integer;

    /// <summary>
    /// Tracks the depth of nested <see cref="IIntStream.Mark"/> calls
    /// </summary>
    FMarkDepth: Integer;

    /// <summary>
    /// A list of CharStreamState objects that tracks the stream state
    /// (i.e. line, charPositionInLine, and p) that can change as you
    /// move through the input stream.  Indexed from 1..markDepth.
    /// A null is kept @ index 0.  Create upon first call to Mark().
    /// </summary>
    FMarkers: IList<ICharStreamState>;

    /// <summary>
    /// Track the last Mark() call result value for use in Rewind().
    /// </summary>
    FLastMarker: Integer;
    /// <summary>
    /// What is name or source of this char stream?
    /// </summary>
    FName: String;
  protected
    { IIntStream }
    function GetSourceName: String; virtual;

    procedure Consume; virtual;
    function LA(I: Integer): Integer; virtual;
    function LAChar(I: Integer): Char;
    function Index: Integer;
    function Size: Integer;
    function Mark: Integer; virtual;
    procedure Rewind(const Marker: Integer); overload; virtual;
    procedure Rewind; overload; virtual;
    procedure Release(const Marker: Integer); virtual;
    procedure Seek(const Index: Integer); virtual;

    property SourceName: String read GetSourceName write FName;
  protected
    { ICharStream }
    function GetLine: Integer; virtual;
    procedure SetLine(const Value: Integer); virtual;
    function GetCharPositionInLine: Integer; virtual;
    procedure SetCharPositionInLine(const Value: Integer); virtual;
    function LT(const I: Integer): Integer; virtual;
    function Substring(const Start, Stop: Integer): String; virtual;
  protected
    { IANTLRStringStream }
    procedure Reset; virtual;
  public
    constructor Create; overload;

    /// <summary>
    /// Initializes a new instance of the ANTLRStringStream class for the
    /// specified string. This copies data from the string to a local
    /// character array
    /// </summary>
    constructor Create(const AInput: String); overload;

    /// <summary>
    /// Initializes a new instance of the ANTLRStringStream class for the
    /// specified character array. This is the preferred constructor as
    /// no data is copied
    /// </summary>
    constructor Create(const AData: PChar;
      const ANumberOfActualCharsInArray: Integer); overload;

    destructor Destroy; override;
  end;

  TANTLRFileStream = class(TANTLRStringStream, IANTLRFileStream)
  strict private
    /// <summary>Fully qualified name of the stream's underlying file</summary>
    FFileName: String;
  protected
    { IIntStream }
    function GetSourceName: String; override;
  protected
    { IANTLRFileStream }

    procedure Load(const FileName: String; const Encoding: TEncoding); virtual;
  public
    /// <summary>
    /// Initializes a new instance of the ANTLRFileStream class for the
    /// specified file name
    /// </summary>
    constructor Create(const AFileName: String); overload;

    /// <summary>
    /// Initializes a new instance of the ANTLRFileStream class for the
    /// specified file name and encoding
    /// </summary>
    constructor Create(const AFileName: String; const AEncoding: TEncoding); overload;
  end;

  TBitSet = class(TANTLRObject, IBitSet, ICloneable)
  strict private
    const
      BITS = 64; // number of bits / ulong
      LOG_BITS = 6; // 2 shl 6 = 64

      ///<summary> We will often need to do a mod operator (i mod nbits).
      /// Its turns out that, for powers of two, this mod operation is
      ///  same as <![CDATA[(I and (nbits-1))]]>.  Since mod is slow, we use a precomputed
      /// mod mask to do the mod instead.
      /// </summary>
      MOD_MASK = BITS - 1;
  strict private
    /// <summary>The actual data bits </summary>
    FBits: TUInt64Array;
  strict private
    class function WordNumber(const Bit: Integer): Integer; static;
    class function BitMask(const BitNumber: Integer): UInt64; static;
    class function NumWordsToHold(const El: Integer): Integer; static;
  protected
    { ICloneable }
    function Clone: IANTLRInterface; virtual;
  protected
    { IBitSet }
    function GetIsNil: Boolean; virtual;
    function BitSetOr(const A: IBitSet): IBitSet; virtual;
    procedure Add(const El: Integer); virtual;
    procedure GrowToInclude(const Bit: Integer); virtual;
    procedure OrInPlace(const A: IBitSet); virtual;
    function Size: Integer; virtual;
    function Member(const El: Integer): Boolean; virtual;
    procedure Remove(const El: Integer); virtual;
    function NumBits: Integer; virtual;
    function LengthInLongWords: Integer; virtual;
    function ToArray: TIntegerArray; virtual;
    function ToPackedArray: TUInt64Array; virtual;
    function ToString(const TokenNames: TStringArray): String; reintroduce; overload; virtual;
  public
    /// <summary>Construct a bitset of size one word (64 bits) </summary>
    constructor Create; overload;

    /// <summary>Construction from a static array of ulongs </summary>
    constructor Create(const ABits: array of UInt64); overload;

    /// <summary>Construction from a list of integers </summary>
    constructor Create(const AItems: IList<Integer>); overload;

    /// <summary>Construct a bitset given the size</summary>
    /// <param name="nbits">The size of the bitset in bits</param>
    constructor Create(const ANBits: Integer); overload;

    class function BitSetOf(const El: Integer): IBitSet; overload; static;
    class function BitSetOf(const A, B: Integer): IBitSet; overload; static;
    class function BitSetOf(const A, B, C: Integer): IBitSet; overload; static;
    class function BitSetOf(const A, B, C, D: Integer): IBitSet; overload; static;

    function ToString: String; overload; override;
    function Equals(Obj: TObject): Boolean; override;
  end;

  TRecognizerSharedState = class(TANTLRObject, IRecognizerSharedState)
  strict private
    FFollowing: TBitSetArray;
    FFollowingStackPointer: Integer;
    FErrorRecovery: Boolean;
    FLastErrorIndex: Integer;
    FFailed: Boolean;
    FSyntaxErrors: Integer;
    FBacktracking: Integer;
    FRuleMemo: TDictionaryArray<Integer, Integer>;
    FToken: IToken;
    FTokenStartCharIndex: Integer;
    FTokenStartLine: Integer;
    FTokenStartCharPositionInLine: Integer;
    FChannel: Integer;
    FTokenType: Integer;
    FText: String;
  protected
    { IRecognizerSharedState }
    function GetFollowing: TBitSetArray;
    procedure SetFollowing(const Value: TBitSetArray);
    function GetFollowingStackPointer: Integer;
    procedure SetFollowingStackPointer(const Value: Integer);
    function GetErrorRecovery: Boolean;
    procedure SetErrorRecovery(const Value: Boolean);
    function GetLastErrorIndex: Integer;
    procedure SetLastErrorIndex(const Value: Integer);
    function GetFailed: Boolean;
    procedure SetFailed(const Value: Boolean);
    function GetSyntaxErrors: Integer;
    procedure SetSyntaxErrors(const Value: Integer);
    function GetBacktracking: Integer;
    procedure SetBacktracking(const Value: Integer);
    function GetRuleMemo: TDictionaryArray<Integer, Integer>;
    function GetRuleMemoCount: Integer;
    procedure SetRuleMemoCount(const Value: Integer);
    function GetToken: IToken;
    procedure SetToken(const Value: IToken);
    function GetTokenStartCharIndex: Integer;
    procedure SetTokenStartCharIndex(const Value: Integer);
    function GetTokenStartLine: Integer;
    procedure SetTokenStartLine(const Value: Integer);
    function GetTokenStartCharPositionInLine: Integer;
    procedure SetTokenStartCharPositionInLine(const Value: Integer);
    function GetChannel: Integer;
    procedure SetChannel(const Value: Integer);
    function GetTokenType: Integer;
    procedure SetTokenType(const Value: Integer);
    function GetText: String;
    procedure SetText(const Value: String);
  public
    constructor Create;
  end;

  TCommonToken = class(TANTLRObject, ICommonToken, IToken)
  strict protected
    FTokenType: Integer;
    FLine: Integer;
    FCharPositionInLine: Integer;
    FChannel: Integer;
    FInput: ICharStream;

    /// <summary>We need to be able to change the text once in a while.  If
    /// this is non-null, then getText should return this.  Note that
    /// start/stop are not affected by changing this.
    /// </summary>
    FText: String;

    /// <summary>What token number is this from 0..n-1 tokens; &lt; 0 implies invalid index </summary>
    FIndex: Integer;

    /// <summary>The char position into the input buffer where this token starts </summary>
    FStart: Integer;

    /// <summary>The char position into the input buffer where this token stops </summary>
    FStop: Integer;
  protected
    { IToken }
    function GetTokenType: Integer; virtual;
    procedure SetTokenType(const Value: Integer); virtual;
    function GetLine: Integer; virtual;
    procedure SetLine(const Value: Integer); virtual;
    function GetCharPositionInLine: Integer; virtual;
    procedure SetCharPositionInLine(const Value: Integer); virtual;
    function GetChannel: Integer; virtual;
    procedure SetChannel(const Value: Integer); virtual;
    function GetTokenIndex: Integer; virtual;
    procedure SetTokenIndex(const Value: Integer); virtual;
    function GetText: String; virtual;
    procedure SetText(const Value: String); virtual;
  protected
    { ICommonToken }
    function GetStartIndex: Integer;
    procedure SetStartIndex(const Value: Integer);
    function GetStopIndex: Integer;
    procedure SetStopIndex(const Value: Integer);
    function GetInputStream: ICharStream;
    procedure SetInputStream(const Value: ICharStream);
  protected
    constructor Create; overload;
  public
    constructor Create(const ATokenType: Integer); overload;
    constructor Create(const AInput: ICharStream; const ATokenType, AChannel,
      AStart, AStop: Integer); overload;
    constructor Create(const ATokenType: Integer; const AText: String); overload;
    constructor Create(const AOldToken: IToken); overload;

    function ToString: String; override;
  end;

  TClassicToken = class(TANTLRObject, IClassicToken, IToken)
  strict private
    FText: String;
    FTokenType: Integer;
    FLine: Integer;
    FCharPositionInLine: Integer;
    FChannel: Integer;

    /// <summary>What token number is this from 0..n-1 tokens </summary>
    FIndex: Integer;
  protected
    { IClassicToken }
    function GetTokenType: Integer; virtual;
    procedure SetTokenType(const Value: Integer); virtual;
    function GetLine: Integer; virtual;
    procedure SetLine(const Value: Integer); virtual;
    function GetCharPositionInLine: Integer; virtual;
    procedure SetCharPositionInLine(const Value: Integer); virtual;
    function GetChannel: Integer; virtual;
    procedure SetChannel(const Value: Integer); virtual;
    function GetTokenIndex: Integer; virtual;
    procedure SetTokenIndex(const Value: Integer); virtual;
    function GetText: String; virtual;
    procedure SetText(const Value: String); virtual;
    function GetInputStream: ICharStream; virtual;
    procedure SetInputStream(const Value: ICharStream); virtual;
  public
    constructor Create(const ATokenType: Integer); overload;
    constructor Create(const AOldToken: IToken); overload;
    constructor Create(const ATokenType: Integer; const AText: String); overload;
    constructor Create(const ATokenType: Integer; const AText: String;
      const AChannel: Integer); overload;

    function ToString: String; override;
  end;

  TToken = class sealed
  public
    const
      EOR_TOKEN_TYPE = 1;

      /// <summary>imaginary tree navigation type; traverse "get child" link </summary>
      DOWN = 2;

      /// <summary>imaginary tree navigation type; finish with a child list </summary>
      UP = 3;

      MIN_TOKEN_TYPE = UP + 1;
      EOF = Integer(cscEOF);
      INVALID_TOKEN_TYPE = 0;

      /// <summary>
      /// All tokens go to the parser (unless skip() is called in that rule)
      /// on a particular "channel".  The parser tunes to a particular channel
      /// so that whitespace etc... can go to the parser on a "hidden" channel.
      /// </summary>
      DEFAULT_CHANNEL = 0;

      /// <summary>
      /// Anything on different channel than DEFAULT_CHANNEL is not parsed by parser.
      /// </summary>
      HIDDEN_CHANNEL = 99;
  public
    class var
      EOF_TOKEN: IToken;
      INVALID_TOKEN: IToken;
      /// <summary>
      /// In an action, a lexer rule can set token to this SKIP_TOKEN and ANTLR
      /// will avoid creating a token for this symbol and try to fetch another.
      /// </summary>
      SKIP_TOKEN: IToken;
  private
    class procedure Initialize; static;
  end;

  /// <summary>
    /// Global constants
  /// </summary>
  TConstants = class sealed
  public
    const
      VERSION = '3.1b1';

      // Moved to version 2 for v3.1: added grammar name to enter/exit Rule
      DEBUG_PROTOCOL_VERSION = '2';

      ANTLRWORKS_DIR = 'antlrworks';
  end;

  TBaseRecognizer = class abstract(TANTLRObject, IBaseRecognizer)
  public
    const
      MEMO_RULE_FAILED = -2;
      MEMO_RULE_UNKNOWN = -1;
      INITIAL_FOLLOW_STACK_SIZE = 100;
      NEXT_TOKEN_RULE_NAME = 'nextToken';
      // copies from Token object for convenience in actions
      DEFAULT_TOKEN_CHANNEL = TToken.DEFAULT_CHANNEL;
      HIDDEN = TToken.HIDDEN_CHANNEL;
  strict protected
    /// <summary>
    /// An externalized representation of the - shareable - internal state of
    /// this lexer, parser or tree parser.
    /// </summary>
    /// <remarks>
    /// The state of a lexer, parser, or tree parser are collected into
    /// external state objects so that the state can be shared. This sharing
    /// is needed to have one grammar import others and share same error
    /// variables and other state variables.  It's a kind of explicit multiple
    /// inheritance via delegation of methods and shared state.
    /// </remarks>
    FState: IRecognizerSharedState;

    property State: IRecognizerSharedState read FState;
  strict protected
    /// <summary>
    /// Match needs to return the current input symbol, which gets put
    /// into the label for the associated token ref; e.g., x=ID.  Token
    /// and tree parsers need to return different objects. Rather than test
    /// for input stream type or change the IntStream interface, I use
    /// a simple method to ask the recognizer to tell me what the current
    /// input symbol is.
    /// </summary>
    /// <remarks>This is ignored for lexers.</remarks>
    function GetCurrentInputSymbol(const Input: IIntStream): IANTLRInterface; virtual;

    /// <summary>
    /// Factor out what to do upon token mismatch so tree parsers can behave
    /// differently.  Override and call MismatchRecover(input, ttype, follow)
    /// to get single token insertion and deletion. Use this to turn off
    /// single token insertion and deletion. Override mismatchRecover
    /// to call this instead.
    /// </summary>
    procedure Mismatch(const Input: IIntStream; const TokenType: Integer;
      const Follow: IBitSet); virtual;

    /// <summary>
    /// Attempt to Recover from a single missing or extra token.
    /// </summary>
    /// <remarks>
    /// EXTRA TOKEN
    ///
    /// LA(1) is not what we are looking for.  If LA(2) has the right token,
    /// however, then assume LA(1) is some extra spurious token.  Delete it
    /// and LA(2) as if we were doing a normal Match(), which advances the
    /// input.
    ///
    /// MISSING TOKEN
    ///
    /// If current token is consistent with what could come after
    /// ttype then it is ok to "insert" the missing token, else throw
    /// exception For example, Input "i=(3;" is clearly missing the
    /// ')'.  When the parser returns from the nested call to expr, it
    /// will have call chain:
    ///
    /// stat -> expr -> atom
    ///
    /// and it will be trying to Match the ')' at this point in the
    /// derivation:
    ///
    /// => ID '=' '(' INT ')' ('+' atom)* ';'
    /// ^
    /// Match() will see that ';' doesn't Match ')' and report a
    /// mismatched token error.  To Recover, it sees that LA(1)==';'
    /// is in the set of tokens that can follow the ')' token
    /// reference in rule atom.  It can assume that you forgot the ')'.
    /// </remarks>
    function RecoverFromMismatchedToken(const Input: IIntStream;
      const TokenType: Integer; const Follow: IBitSet): IANTLRInterface; virtual;

    /// <summary>
    /// Conjure up a missing token during error recovery.
    /// </summary>
    /// <remarks>
    /// The recognizer attempts to recover from single missing
    /// symbols. But, actions might refer to that missing symbol.
    /// For example, x=ID {f($x);}. The action clearly assumes
    /// that there has been an identifier matched previously and that
    /// $x points at that token. If that token is missing, but
    /// the next token in the stream is what we want we assume that
    /// this token is missing and we keep going. Because we
    /// have to return some token to replace the missing token,
    /// we have to conjure one up. This method gives the user control
    /// over the tokens returned for missing tokens. Mostly,
    /// you will want to create something special for identifier
    /// tokens. For literals such as '{' and ',', the default
    /// action in the parser or tree parser works. It simply creates
    /// a CommonToken of the appropriate type. The text will be the token.
    /// If you change what tokens must be created by the lexer,
    /// override this method to create the appropriate tokens.
    /// </remarks>
    function GetMissingSymbol(const Input: IIntStream;
      const E: ERecognitionException; const ExpectedTokenType: Integer;
      const Follow: IBitSet): IANTLRInterface; virtual;

    /// <summary>
    /// Push a rule's follow set using our own hardcoded stack
    /// </summary>
    /// <param name="fset"></param>
    procedure PushFollow(const FSet: IBitSet);

    /// <summary>Compute the context-sensitive FOLLOW set for current rule.
    /// This is set of token types that can follow a specific rule
    /// reference given a specific call chain.  You get the set of
    /// viable tokens that can possibly come next (lookahead depth 1)
    /// given the current call chain.  Contrast this with the
    /// definition of plain FOLLOW for rule r:
    ///
    /// FOLLOW(r)={x | S=>*alpha r beta in G and x in FIRST(beta)}
    ///
    /// where x in T* and alpha, beta in V*; T is set of terminals and
    /// V is the set of terminals and nonterminals.  In other words,
    /// FOLLOW(r) is the set of all tokens that can possibly follow
    /// references to r in *any* sentential form (context).  At
    /// runtime, however, we know precisely which context applies as
    /// we have the call chain.  We may compute the exact (rather
    /// than covering superset) set of following tokens.
    ///
    /// For example, consider grammar:
    ///
    /// stat : ID '=' expr ';'      // FOLLOW(stat)=={EOF}
    /// | "return" expr '.'
    /// ;
    /// expr : atom ('+' atom)* ;   // FOLLOW(expr)=={';','.',')'}
    /// atom : INT                  // FOLLOW(atom)=={'+',')',';','.'}
    /// | '(' expr ')'
    /// ;
    ///
    /// The FOLLOW sets are all inclusive whereas context-sensitive
    /// FOLLOW sets are precisely what could follow a rule reference.
    /// For input input "i=(3);", here is the derivation:
    ///
    /// stat => ID '=' expr ';'
    /// => ID '=' atom ('+' atom)* ';'
    /// => ID '=' '(' expr ')' ('+' atom)* ';'
    /// => ID '=' '(' atom ')' ('+' atom)* ';'
    /// => ID '=' '(' INT ')' ('+' atom)* ';'
    /// => ID '=' '(' INT ')' ';'
    ///
    /// At the "3" token, you'd have a call chain of
    ///
    /// stat -> expr -> atom -> expr -> atom
    ///
    /// What can follow that specific nested ref to atom?  Exactly ')'
    /// as you can see by looking at the derivation of this specific
    /// input.  Contrast this with the FOLLOW(atom)={'+',')',';','.'}.
    ///
    /// You want the exact viable token set when recovering from a
    /// token mismatch.  Upon token mismatch, if LA(1) is member of
    /// the viable next token set, then you know there is most likely
    /// a missing token in the input stream.  "Insert" one by just not
    /// throwing an exception.
    /// </summary>
    function ComputeContextSensitiveRuleFOLLOW: IBitSet; virtual;

    (*  Compute the error recovery set for the current rule.  During
    *  rule invocation, the parser pushes the set of tokens that can
    *  follow that rule reference on the stack; this amounts to
    *  computing FIRST of what follows the rule reference in the
    *  enclosing rule. This local follow set only includes tokens
    *  from within the rule; i.e., the FIRST computation done by
    *  ANTLR stops at the end of a rule.
    *
    *  EXAMPLE
    *
    *  When you find a "no viable alt exception", the input is not
    *  consistent with any of the alternatives for rule r.  The best
    *  thing to do is to consume tokens until you see something that
    *  can legally follow a call to r *or* any rule that called r.
    *  You don't want the exact set of viable next tokens because the
    *  input might just be missing a token--you might consume the
    *  rest of the input looking for one of the missing tokens.
    *
    *  Consider grammar:
    *
    *  a : '[' b ']'
    *    | '(' b ')'
    *    ;
    *  b : c '^' INT ;
    *  c : ID
    *    | INT
    *    ;
    *
    *  At each rule invocation, the set of tokens that could follow
    *  that rule is pushed on a stack.  Here are the various "local"
    *  follow sets:
    *
    *  FOLLOW(b1_in_a) = FIRST(']') = ']'
    *  FOLLOW(b2_in_a) = FIRST(')') = ')'
    *  FOLLOW(c_in_b) = FIRST('^') = '^'
    *
    *  Upon erroneous input "[]", the call chain is
    *
    *  a -> b -> c
    *
    *  and, hence, the follow context stack is:
    *
    *  depth  local follow set     after call to rule
    *    0         <EOF>                    a (from main())
    *    1          ']'                     b
    *    3          '^'                     c
    *
    *  Notice that ')' is not included, because b would have to have
    *  been called from a different context in rule a for ')' to be
    *  included.
    *
    *  For error recovery, we cannot consider FOLLOW(c)
    *  (context-sensitive or otherwise).  We need the combined set of
    *  all context-sensitive FOLLOW sets--the set of all tokens that
    *  could follow any reference in the call chain.  We need to
    *  resync to one of those tokens.  Note that FOLLOW(c)='^' and if
    *  we resync'd to that token, we'd consume until EOF.  We need to
    *  sync to context-sensitive FOLLOWs for a, b, and c: {']','^'}.
    *  In this case, for input "[]", LA(1) is in this set so we would
    *  not consume anything and after printing an error rule c would
    *  return normally.  It would not find the required '^' though.
    *  At this point, it gets a mismatched token error and throws an
    *  exception (since LA(1) is not in the viable following token
    *  set).  The rule exception handler tries to Recover, but finds
    *  the same recovery set and doesn't consume anything.  Rule b
    *  exits normally returning to rule a.  Now it finds the ']' (and
    *  with the successful Match exits errorRecovery mode).
    *
    *  So, you cna see that the parser walks up call chain looking
    *  for the token that was a member of the recovery set.
    *
    *  Errors are not generated in errorRecovery mode.
    *
    *  ANTLR's error recovery mechanism is based upon original ideas:
    *
    *  "Algorithms + Data Structures = Programs" by Niklaus Wirth
    *
    *  and
    *
    *  "A note on error recovery in recursive descent parsers":
    *  http://portal.acm.org/citation.cfm?id=947902.947905
    *
    *  Later, Josef Grosch had some good ideas:
    *
    *  "Efficient and Comfortable Error Recovery in Recursive Descent
    *  Parsers":
    *  ftp://www.cocolab.com/products/cocktail/doca4.ps/ell.ps.zip
    *
    *  Like Grosch I implemented local FOLLOW sets that are combined
    *  at run-time upon error to avoid overhead during parsing.
    *)
    function ComputeErrorRecoverySet: IBitSet; virtual;

    function CombineFollows(const Exact: Boolean): IBitSet;
  protected
    { IBaseRecognizer }
    function GetInput: IIntStream; virtual; abstract;
    function GetBacktrackingLevel: Integer;
    function GetState: IRecognizerSharedState;
    function GetNumberOfSyntaxErrors: Integer;
    function GetGrammarFileName: String; virtual;
    function GetSourceName: String; virtual; abstract;
    function GetTokenNames: TStringArray; virtual;

    procedure BeginBacktrack(const Level: Integer); virtual;
    procedure EndBacktrack(const Level: Integer; const Successful: Boolean); virtual;
    procedure Reset; virtual;
    function Match(const Input: IIntStream; const TokenType: Integer;
      const Follow: IBitSet): IANTLRInterface; virtual;
    function MismatchIsUnwantedToken(const Input: IIntStream;
      const TokenType: Integer): Boolean;
    function MismatchIsMissingToken(const Input: IIntStream;
      const Follow: IBitSet): Boolean;
    procedure BeginResync; virtual;
    procedure EndResync; virtual;
    procedure ReportError(const E: ERecognitionException); virtual;
    procedure MatchAny(const Input: IIntStream); virtual;
    procedure DisplayRecognitionError(const TokenNames: TStringArray;
      const E: ERecognitionException); virtual;
    function GetErrorMessage(const E: ERecognitionException;
      const TokenNames: TStringArray): String; virtual;
    function GetErrorHeader(const E: ERecognitionException): String; virtual;
    function GetTokenErrorDisplay(const T: IToken): String; virtual;
    procedure EmitErrorMessage(const Msg: String); virtual;
    procedure Recover(const Input: IIntStream; const RE: ERecognitionException); virtual;
    function RecoverFromMismatchedSet(const Input: IIntStream;
      const E: ERecognitionException; const Follow: IBitSet): IANTLRInterface; virtual;
    procedure ConsumeUntil(const Input: IIntStream; const TokenType: Integer); overload; virtual;
    procedure ConsumeUntil(const Input: IIntStream; const BitSet: IBitSet); overload; virtual;
    //function GetRuleInvocationStack: IList<IANTLRInterface>; overload; virtual;
    //function GetRuleInvocationStack(const E: Exception;
    //  const RecognizerClassName: String): IList<IANTLRInterface>; overload;
    function ToStrings(const Tokens: IList<IToken>): IList<String>; virtual;
    function GetRuleMemoization(const RuleIndex, RuleStartIndex: Integer): Integer; virtual;
    function AlreadyParsedRule(const Input: IIntStream;
      const RuleIndex: Integer): Boolean; virtual;
    procedure Memoize(const Input: IIntStream; const RuleIndex,
      RuleStartIndex: Integer); virtual;
    function GetRuleMemoizationChaceSize: Integer;

    procedure TraceIn(const RuleName: String; const RuleIndex: Integer;
      const InputSymbol: String); virtual;
    procedure TraceOut(const RuleName: String; const RuleIndex: Integer;
      const InputSymbol: String); virtual;

    property Input: IIntStream read GetInput;
  public
    constructor Create; overload;
    constructor Create(const AState: IRecognizerSharedState); overload;
  end;

  TCommonTokenStream = class(TANTLRObject, ICommonTokenStream, ITokenStream)
  strict private
    FTokenSource: ITokenSource;

    /// <summary>Record every single token pulled from the source so we can reproduce
    /// chunks of it later.
    /// </summary>
    FTokens: IList<IToken>;

    /// <summary><![CDATA[Map<tokentype, channel>]]> to override some Tokens' channel numbers </summary>
    FChannelOverrideMap: IDictionary<Integer, Integer>;

    /// <summary><![CDATA[Set<tokentype>;]]> discard any tokens with this type </summary>
    FDiscardSet: IHashList<Integer, Integer>;

    /// <summary>Skip tokens on any channel but this one; this is how we skip whitespace... </summary>
    FChannel: Integer;

    /// <summary>By default, track all incoming tokens </summary>
    FDiscardOffChannelTokens: Boolean;

    /// <summary>Track the last Mark() call result value for use in Rewind().</summary>
    FLastMarker: Integer;

    /// <summary>
    /// The index into the tokens list of the current token (next token
    /// to consume).  p==-1 indicates that the tokens list is empty
    /// </summary>
    FP: Integer;
  strict protected
    /// <summary>Load all tokens from the token source and put in tokens.
    /// This is done upon first LT request because you might want to
    /// set some token type / channel overrides before filling buffer.
    /// </summary>
    procedure FillBuffer; virtual;

    /// <summary>Look backwards k tokens on-channel tokens </summary>
    function LB(const K: Integer): IToken; virtual;

    /// <summary>Given a starting index, return the index of the first on-channel
    /// token.
    /// </summary>
    function SkipOffTokenChannels(const I: Integer): Integer; virtual;
    function SkipOffTokenChannelsReverse(const I: Integer): Integer; virtual;
  protected
    { IIntStream }
    function GetSourceName: String; virtual;

    procedure Consume; virtual;
    function LA(I: Integer): Integer; virtual;
    function LAChar(I: Integer): Char;
    function Mark: Integer; virtual;
    function Index: Integer; virtual;
    procedure Rewind(const Marker: Integer); overload; virtual;
    procedure Rewind; overload; virtual;
    procedure Release(const Marker: Integer); virtual;
    procedure Seek(const Index: Integer); virtual;
    function Size: Integer; virtual;
  protected
    { ITokenStream }
    function GetTokenSource: ITokenSource; virtual;
    procedure SetTokenSource(const Value: ITokenSource); virtual;

    function LT(const K: Integer): IToken; virtual;
    function Get(const I: Integer): IToken; virtual;
    function ToString(const Start, Stop: Integer): String; reintroduce; overload; virtual;
    function ToString(const Start, Stop: IToken): String; reintroduce; overload; virtual;
  protected
    { ICommonTokenStream }
    procedure SetTokenTypeChannel(const TType, Channel: Integer);
    procedure DiscardTokenType(const TType: Integer);
    procedure DiscardOffChannelTokens(const Discard: Boolean);
    function GetTokens: IList<IToken>; overload;
    function GetTokens(const Start, Stop: Integer): IList<IToken>; overload;
    function GetTokens(const Start, Stop: Integer;
      const Types: IBitSet): IList<IToken>; overload;
    function GetTokens(const Start, Stop: Integer;
      const Types: IList<Integer>): IList<IToken>; overload;
    function GetTokens(const Start, Stop,
      TokenType: Integer): IList<IToken>; overload;
    procedure Reset; virtual;
  public
    constructor Create; overload;
    constructor Create(const ATokenSource: ITokenSource); overload;
    constructor Create(const ATokenSource: ITokenSource;
      const AChannel: Integer); overload;
    constructor Create(const ALexer: ILexer); overload;
    constructor Create(const ALexer: ILexer;
      const AChannel: Integer); overload;

    function ToString: String; overload; override;
  end;

  TDFA = class abstract(TANTLRObject, IDFA)
  strict private
    FSpecialStateTransitionHandler: TSpecialStateTransitionHandler;
    FEOT: TSmallintArray;
    FEOF: TSmallintArray;
    FMin: TCharArray;
    FMax: TCharArray;
    FAccept: TSmallintArray;
    FSpecial: TSmallintArray;
    FTransition: TSmallintMatrix;
    FDecisionNumber: Integer;
    FRecognizer: Pointer; { IBaseRecognizer }
    function GetRecognizer: IBaseRecognizer;
    procedure SetRecognizer(const Value: IBaseRecognizer);
  strict protected
    procedure NoViableAlt(const S: Integer; const Input: IIntStream);

    property Recognizer: IBaseRecognizer read GetRecognizer write SetRecognizer;
    property DecisionNumber: Integer read FDecisionNumber write FDecisionNumber;
    property EOT: TSmallintArray read FEOT write FEOT;
    property EOF: TSmallintArray read FEOF write FEOF;
    property Min: TCharArray read FMin write FMin;
    property Max: TCharArray read FMax write FMax;
    property Accept: TSmallintArray read FAccept write FAccept;
    property Special: TSmallintArray read FSpecial write FSpecial;
    property Transition: TSmallintMatrix read FTransition write FTransition;
  protected
    { IDFA }
    function GetSpecialStateTransitionHandler: TSpecialStateTransitionHandler;
    procedure SetSpecialStateTransitionHandler(const Value: TSpecialStateTransitionHandler);

    function Predict(const Input: IIntStream): Integer;
    procedure Error(const NVAE: ENoViableAltException); virtual;
    function SpecialStateTransition(const S: Integer;
      const Input: IIntStream): Integer; virtual;
    function Description: String; virtual;
    function SpecialTransition(const State, Symbol: Integer): Integer;
  public
    class function UnpackEncodedString(const EncodedString: String): TSmallintArray; static;
    class function UnpackEncodedStringArray(const EncodedStrings: TStringArray): TSmallintMatrix; overload; static;
    class function UnpackEncodedStringArray(const EncodedStrings: array of String): TSmallintMatrix; overload; static;
    class function UnpackEncodedStringToUnsignedChars(const EncodedString: String): TCharArray; static;
  end;

  TLexer = class abstract(TBaseRecognizer, ILexer, ITokenSource)
  strict private
    const
      TOKEN_dot_EOF = Ord(cscEOF);
  strict private
    /// <summary>Where is the lexer drawing characters from? </summary>
    FInput: ICharStream;
  protected
    { IBaseRecognizer }
    function GetSourceName: String; override;
    function GetInput: IIntStream; override;
    procedure Reset; override;
    procedure ReportError(const E: ERecognitionException); override;
    function GetErrorMessage(const E: ERecognitionException;
      const TokenNames: TStringArray): String; override;
  protected
    { ILexer }
    function GetCharStream: ICharStream; virtual;
    procedure SetCharStream(const Value: ICharStream); virtual;
    function GetLine: Integer; virtual;
    function GetCharPositionInLine: Integer; virtual;
    function GetCharIndex: Integer; virtual;
    function GetText: String; virtual;
    procedure SetText(const Value: String); virtual;

    function NextToken: IToken; virtual;
    procedure Skip;
    procedure DoTokens; virtual; abstract;
    procedure Emit(const Token: IToken); overload; virtual;
    function Emit: IToken; overload; virtual;
    procedure Match(const S: String); reintroduce; overload; virtual;
    procedure Match(const C: Integer); reintroduce; overload; virtual;
    procedure MatchAny; reintroduce; overload; virtual;
    procedure MatchRange(const A, B: Integer); virtual;
    procedure Recover(const RE: ERecognitionException); reintroduce; overload; virtual;
    function GetCharErrorDisplay(const C: Integer): String;
    procedure TraceIn(const RuleName: String; const RuleIndex: Integer); reintroduce; overload; virtual;
    procedure TraceOut(const RuleName: String; const RuleIndex: Integer); reintroduce; overload; virtual;
  strict protected
    property Input: ICharStream read FInput;
    property CharIndex: Integer read GetCharIndex;
    property Text: String read GetText write SetText;
  public
    constructor Create; overload;
    constructor Create(const AInput: ICharStream); overload;
    constructor Create(const AInput: ICharStream;
      const AState: IRecognizerSharedState); overload;
  end;

  TParser = class(TBaseRecognizer, IParser)
  strict private
    FInput: ITokenStream;
  protected
    property Input: ITokenStream read FInput;
  protected
    { IBaseRecognizer }
    procedure Reset; override;
    function GetCurrentInputSymbol(const Input: IIntStream): IANTLRInterface; override;
    function GetMissingSymbol(const Input: IIntStream;
      const E: ERecognitionException; const ExpectedTokenType: Integer;
      const Follow: IBitSet): IANTLRInterface; override;
    function GetSourceName: String; override;
    function GetInput: IIntStream; override;
  protected
    { IParser }
    function GetTokenStream: ITokenStream; virtual;
    procedure SetTokenStream(const Value: ITokenStream); virtual;

    procedure TraceIn(const RuleName: String; const RuleIndex: Integer); reintroduce; overload;
    procedure TraceOut(const RuleName: String; const RuleIndex: Integer); reintroduce; overload;
  public
    constructor Create(const AInput: ITokenStream); overload;
    constructor Create(const AInput: ITokenStream;
      const AState: IRecognizerSharedState); overload;
  end;

  TRuleReturnScope = class(TANTLRObject, IRuleReturnScope)
  protected
    { IRuleReturnScope }
    function GetStart: IANTLRInterface; virtual;
    procedure SetStart(const Value: IANTLRInterface); virtual;
    function GetStop: IANTLRInterface; virtual;
    procedure SetStop(const Value: IANTLRInterface); virtual;
    function GetTree: IANTLRInterface; virtual;
    procedure SetTree(const Value: IANTLRInterface); virtual;
    function GetTemplate: IANTLRInterface; virtual;
  end;

  TParserRuleReturnScope = class(TRuleReturnScope, IParserRuleReturnScope)
  strict private
    FStart: IToken;
    FStop: IToken;
  protected
    { IRuleReturnScope }
    function GetStart: IANTLRInterface; override;
    procedure SetStart(const Value: IANTLRInterface); override;
    function GetStop: IANTLRInterface; override;
    procedure SetStop(const Value: IANTLRInterface); override;
  end;

  TTokenRewriteStream = class(TCommonTokenStream, ITokenRewriteStream)
  public
    const
      DEFAULT_PROGRAM_NAME = 'default';
      PROGRAM_INIT_SIZE = 100;
      MIN_TOKEN_INDEX = 0;
  strict protected
    // Define the rewrite operation hierarchy
    type
      IRewriteOperation = interface(IANTLRInterface)
      ['{285A54ED-58FF-44B1-A268-2686476D4419}']
        { Property accessors }
        function GetInstructionIndex: Integer;
        procedure SetInstructionIndex(const Value: Integer);
        function GetIndex: Integer;
        procedure SetIndex(const Value: Integer);
        function GetText: IANTLRInterface;
        procedure SetText(const Value: IANTLRInterface);
        function GetParent: ITokenRewriteStream;
        procedure SetParent(const Value: ITokenRewriteStream);

        { Methods }

        /// <summary>Execute the rewrite operation by possibly adding to the buffer.
        /// Return the index of the next token to operate on.
        /// </summary>
        function Execute(const Buf: TStringBuilder): Integer;

        { Properties }
        property InstructionIndex: Integer read GetInstructionIndex write SetInstructionIndex;
        property Index: Integer read GetIndex write SetIndex;
        property Text: IANTLRInterface read GetText write SetText;
        property Parent: ITokenRewriteStream read GetParent write SetParent;
      end;

      TRewriteOperation = class(TANTLRObject, IRewriteOperation)
      strict private
        // What index into rewrites List are we?
        FInstructionIndex: Integer;
        // Token buffer index
        FIndex: Integer;
        FText: IANTLRInterface;
        FParent: Pointer; {ITokenRewriteStream;}
      protected
        { IRewriteOperation }
        function GetInstructionIndex: Integer;
        procedure SetInstructionIndex(const Value: Integer);
        function GetIndex: Integer;
        procedure SetIndex(const Value: Integer);
        function GetText: IANTLRInterface;
        procedure SetText(const Value: IANTLRInterface);
        function GetParent: ITokenRewriteStream;
        procedure SetParent(const Value: ITokenRewriteStream);

        function Execute(const Buf: TStringBuilder): Integer; virtual;
      protected
        constructor Create(const AIndex: Integer; const AText: IANTLRInterface;
          const AParent: ITokenRewriteStream);

        property Index: Integer read FIndex write FIndex;
        property Text: IANTLRInterface read FText write FText;
        property Parent: ITokenRewriteStream read GetParent write SetParent;
      public
        function ToString: String; override;
      end;

      IInsertBeforeOp = interface(IRewriteOperation)
      ['{BFB732E2-BE6A-4691-AE3B-5C8013DE924E}']
      end;

      TInsertBeforeOp = class(TRewriteOperation, IInsertBeforeOp)
      protected
        { IRewriteOperation }
        function Execute(const Buf: TStringBuilder): Integer; override;
      end;

      /// <summary>I'm going to try replacing range from x..y with (y-x)+1 ReplaceOp
      /// instructions.
      /// </summary>
      IReplaceOp = interface(IRewriteOperation)
      ['{630C434A-99EA-4589-A65D-64A7B3DAC407}']
        { Property accessors }
        function GetLastIndex: Integer;
        procedure SetLastIndex(const Value: Integer);

        { Properties }
        property LastIndex: Integer read GetLastIndex write SetLastIndex;
      end;

      TReplaceOp = class(TRewriteOperation, IReplaceOp)
      private
        FLastIndex: Integer;
      protected
        { IRewriteOperation }
        function Execute(const Buf: TStringBuilder): Integer; override;
      protected
        { IReplaceOp }
        function GetLastIndex: Integer;
        procedure SetLastIndex(const Value: Integer);
      public
        constructor Create(const AStart, AStop: Integer;
          const AText: IANTLRInterface; const AParent: ITokenRewriteStream);

        function ToString: String; override;
      end;

      IDeleteOp = interface(IRewriteOperation)
      ['{C39345BC-F170-4C3A-A989-65E6B9F0712B}']
      end;

      TDeleteOp = class(TReplaceOp)
      public
        function ToString: String; override;
      end;
  strict private
    type
      TRewriteOpComparer<T: IRewriteOperation> = class(TComparer<T>)
      public
        function Compare(const Left, Right: T): Integer; override;
      end;
  strict private
    /// <summary>You may have multiple, named streams of rewrite operations.
    /// I'm calling these things "programs."
    /// Maps String (name) -> rewrite (IList)
    /// </summary>
    FPrograms: IDictionary<String, IList<IRewriteOperation>>;

    /// <summary>Map String (program name) -> Integer index </summary>
    FLastRewriteTokenIndexes: IDictionary<String, Integer>;
  strict private
    function InitializeProgram(const Name: String): IList<IRewriteOperation>;
  protected
    { ITokenRewriteStream }
    procedure Rollback(const InstructionIndex: Integer); overload; virtual;
    procedure Rollback(const ProgramName: String;
      const InstructionIndex: Integer); overload; virtual;

    procedure DeleteProgram; overload; virtual;
    procedure DeleteProgram(const ProgramName: String); overload; virtual;

    procedure InsertAfter(const T: IToken; const Text: IANTLRInterface); overload; virtual;
    procedure InsertAfter(const Index: Integer; const Text: IANTLRInterface); overload; virtual;
    procedure InsertAfter(const ProgramName: String; const T: IToken;
      const Text: IANTLRInterface); overload; virtual;
    procedure InsertAfter(const ProgramName: String; const Index: Integer;
      const Text: IANTLRInterface); overload; virtual;
    procedure InsertAfter(const T: IToken; const Text: String); overload;
    procedure InsertAfter(const Index: Integer; const Text: String); overload;
    procedure InsertAfter(const ProgramName: String; const T: IToken;
      const Text: String); overload;
    procedure InsertAfter(const ProgramName: String; const Index: Integer;
      const Text: String); overload;

    procedure InsertBefore(const T: IToken; const Text: IANTLRInterface); overload; virtual;
    procedure InsertBefore(const Index: Integer; const Text: IANTLRInterface); overload; virtual;
    procedure InsertBefore(const ProgramName: String; const T: IToken;
      const Text: IANTLRInterface); overload; virtual;
    procedure InsertBefore(const ProgramName: String; const Index: Integer;
      const Text: IANTLRInterface); overload; virtual;
    procedure InsertBefore(const T: IToken; const Text: String); overload;
    procedure InsertBefore(const Index: Integer; const Text: String); overload;
    procedure InsertBefore(const ProgramName: String; const T: IToken;
      const Text: String); overload;
    procedure InsertBefore(const ProgramName: String; const Index: Integer;
      const Text: String); overload;

    procedure Replace(const Index: Integer; const Text: IANTLRInterface); overload; virtual;
    procedure Replace(const Start, Stop: Integer; const Text: IANTLRInterface); overload; virtual;
    procedure Replace(const IndexT: IToken; const Text: IANTLRInterface); overload; virtual;
    procedure Replace(const Start, Stop: IToken; const Text: IANTLRInterface); overload; virtual;
    procedure Replace(const ProgramName: String; const Start, Stop: Integer;
      const Text: IANTLRInterface); overload; virtual;
    procedure Replace(const ProgramName: String; const Start, Stop: IToken;
      const Text: IANTLRInterface); overload; virtual;
    procedure Replace(const Index: Integer; const Text: String); overload;
    procedure Replace(const Start, Stop: Integer; const Text: String); overload;
    procedure Replace(const IndexT: IToken; const Text: String); overload;
    procedure Replace(const Start, Stop: IToken; const Text: String); overload;
    procedure Replace(const ProgramName: String; const Start, Stop: Integer;
      const Text: String); overload;
    procedure Replace(const ProgramName: String; const Start, Stop: IToken;
      const Text: String); overload;

    procedure Delete(const Index: Integer); overload; virtual;
    procedure Delete(const Start, Stop: Integer); overload; virtual;
    procedure Delete(const IndexT: IToken); overload; virtual;
    procedure Delete(const Start, Stop: IToken); overload; virtual;
    procedure Delete(const ProgramName: String; const Start, Stop: Integer); overload; virtual;
    procedure Delete(const ProgramName: String; const Start, Stop: IToken); overload; virtual;

    function GetLastRewriteTokenIndex: Integer; overload; virtual;

    function ToOriginalString: String; overload; virtual;
    function ToOriginalString(const Start, Stop: Integer): String; overload; virtual;

    function ToString(const ProgramName: String): String; overload; virtual;
    function ToString(const ProgramName: String;
      const Start, Stop: Integer): String; overload; virtual;

    function ToDebugString: String; overload; virtual;
    function ToDebugString(const Start, Stop: Integer): String; overload; virtual;
  protected
    { ITokenStream }
    function ToString(const Start, Stop: Integer): String; overload; override;
  strict protected
    procedure Init; virtual;
    function GetProgram(const Name: String): IList<IRewriteOperation>; virtual;
    function GetLastRewriteTokenIndex(const ProgramName: String): Integer; overload; virtual;
    procedure SetLastRewriteTokenIndex(const ProgramName: String; const I: Integer); overload; virtual;

    /// <summary>
    /// Return a map from token index to operation.
    /// </summary>
    /// <remarks>We need to combine operations and report invalid operations (like
    /// overlapping replaces that are not completed nested).  Inserts to
    /// same index need to be combined etc...   Here are the cases:
    ///
    /// I.i.u I.j.v               leave alone, nonoverlapping
    /// I.i.u I.i.v               combine: Iivu
    ///
    /// R.i-j.u R.x-y.v | i-j in x-y      delete first R
    /// R.i-j.u R.i-j.v             delete first R
    /// R.i-j.u R.x-y.v | x-y in i-j      ERROR
    /// R.i-j.u R.x-y.v | boundaries overlap  ERROR
    ///
    /// I.i.u R.x-y.v | i in x-y        delete I
    /// I.i.u R.x-y.v | i not in x-y      leave alone, nonoverlapping
    /// R.x-y.v I.i.u | i in x-y        ERROR
    /// R.x-y.v I.x.u               R.x-y.uv (combine, delete I)
    /// R.x-y.v I.i.u | i not in x-y      leave alone, nonoverlapping
    ///
    /// I.i.u = insert u before op @ index i
    /// R.x-y.u = replace x-y indexed tokens with u
    ///
    /// First we need to examine replaces.  For any replace op:
    ///
    ///   1. wipe out any insertions before op within that range.
    ///   2. Drop any replace op before that is contained completely within
    ///        that range.
    ///   3. Throw exception upon boundary overlap with any previous replace.
    ///
    /// Then we can deal with inserts:
    ///
    ///   1. for any inserts to same index, combine even if not adjacent.
    ///   2. for any prior replace with same left boundary, combine this
    ///        insert with replace and delete this replace.
    ///   3. throw exception if index in same range as previous replace
    ///
    /// Don't actually delete; make op null in list. Easier to walk list.
    /// Later we can throw as we add to index -> op map.
    ///
    /// Note that I.2 R.2-2 will wipe out I.2 even though, technically, the
    /// inserted stuff would be before the replace range.  But, if you
    /// add tokens in front of a method body '{' and then delete the method
    /// body, I think the stuff before the '{' you added should disappear too.
    /// </remarks>
    function ReduceToSingleOperationPerIndex(
      const Rewrites: IList<IRewriteOperation>): IDictionary<Integer, IRewriteOperation>;

    function GetKindOfOps(const Rewrites: IList<IRewriteOperation>;
      const Kind: TGUID): IList<IRewriteOperation>; overload;
    /// <summary>
    /// Get all operations before an index of a particular kind
    /// </summary>
    function GetKindOfOps(const Rewrites: IList<IRewriteOperation>;
      const Kind: TGUID; const Before: Integer): IList<IRewriteOperation>; overload;

    function CatOpText(const A, B: IANTLRInterface): IANTLRInterface;
  public
    constructor Create; overload;
    constructor Create(const ATokenSource: ITokenSource); overload;
    constructor Create(const ATokenSource: ITokenSource;
      const AChannel: Integer); overload;
    constructor Create(const ALexer: ILexer); overload;
    constructor Create(const ALexer: ILexer;
      const AChannel: Integer); overload;

    function ToString: String; overload; override;
  end;

{ These functions return X or, if X = nil, an empty default instance }
function Def(const X: IToken): IToken; overload;
function Def(const X: IRuleReturnScope): IRuleReturnScope; overload;

implementation

uses
  StrUtils,
  Math,
  Antlr.Runtime.Tree;

{ ERecognitionException }

constructor ERecognitionException.Create;
begin
  Create('', nil);
end;

constructor ERecognitionException.Create(const AMessage: String);
begin
  Create(AMessage, nil);
end;

constructor ERecognitionException.Create(const AInput: IIntStream);
begin
  Create('', AInput);
end;

constructor ERecognitionException.Create(const AMessage: String;
  const AInput: IIntStream);
var
  TokenStream: ITokenStream;
  CharStream: ICharStream;
begin
  inherited Create(AMessage);
  FInput := AInput;
  FIndex := AInput.Index;

  if Supports(AInput, ITokenStream, TokenStream) then
  begin
    FToken := TokenStream.LT(1);
    FLine := FToken.Line;
    FCharPositionInLine := FToken.CharPositionInLine;
  end;

  if Supports(AInput, ITreeNodeStream) then
    ExtractInformationFromTreeNodeStream(AInput)
  else
  begin
    if Supports(AInput, ICharStream, CharStream) then
    begin
      FC := AInput.LA(1);
      FLine := CharStream.Line;
      FCharPositionInLine := CharStream.CharPositionInLine;
    end
    else
      FC := AInput.LA(1);
  end;
end;

procedure ERecognitionException.ExtractInformationFromTreeNodeStream(
  const Input: IIntStream);
var
  Nodes: ITreeNodeStream;
  Adaptor: ITreeAdaptor;
  Payload, PriorPayload: IToken;
  I, NodeType: Integer;
  PriorNode: IANTLRInterface;
  Tree: ITree;
  Text: String;
  CommonTree: ICommonTree;
begin
  Nodes := Input as ITreeNodeStream;
  FNode := Nodes.LT(1);
  Adaptor := Nodes.TreeAdaptor;
  Payload := Adaptor.GetToken(FNode);

  if Assigned(Payload) then
  begin
    FToken := Payload;
    if (Payload.Line <= 0) then
    begin
      // imaginary node; no line/pos info; scan backwards
      I := -1;
      PriorNode := Nodes.LT(I);
      while Assigned(PriorNode) do
      begin
        PriorPayload := Adaptor.GetToken(PriorNode);
        if Assigned(PriorPayload) and (PriorPayload.Line > 0) then
        begin
          // we found the most recent real line / pos info
          FLine := PriorPayload.Line;
          FCharPositionInLine := PriorPayload.CharPositionInLine;
          FApproximateLineInfo := True;
          Break;
        end;
        Dec(I);
        PriorNode := Nodes.LT(I)
      end;
    end
    else
    begin
      // node created from real token
      FLine := Payload.Line;
      FCharPositionInLine := Payload.CharPositionInLine;
    end;
  end else
    if Supports(FNode, ITree, Tree) then
    begin
      FLine := Tree.Line;
      FCharPositionInLine := Tree.CharPositionInLine;
      if Supports(FNode, ICommonTree, CommonTree) then
        FToken := CommonTree.Token;
    end
    else
    begin
      NodeType := Adaptor.GetNodeType(FNode);
      Text := Adaptor.GetNodeText(FNode);
      FToken := TCommonToken.Create(NodeType, Text);
    end;
end;

function ERecognitionException.GetUnexpectedType: Integer;
var
  Nodes: ITreeNodeStream;
  Adaptor: ITreeAdaptor;
begin
  if Supports(FInput, ITokenStream) then
    Result := FToken.TokenType
  else
    if Supports(FInput, ITreeNodeStream, Nodes) then
    begin
      Adaptor := Nodes.TreeAdaptor;
      Result := Adaptor.GetNodeType(FNode);
    end else
      Result := FC;
end;

{ EMismatchedTokenException }

constructor EMismatchedTokenException.Create(const AExpecting: Integer;
  const AInput: IIntStream);
begin
  inherited Create(AInput);
  FExpecting := AExpecting;
end;

function EMismatchedTokenException.ToString: String;
begin
  Result := 'MismatchedTokenException(' + IntToStr(UnexpectedType)
    + '!=' + IntToStr(Expecting) + ')';

end;

{ EUnwantedTokenException }

function EUnwantedTokenException.GetUnexpectedToken: IToken;
begin
  Result := FToken;
end;

function EUnwantedTokenException.ToString: String;
var
  Exp: String;
begin
  if (Expecting = TToken.INVALID_TOKEN_TYPE) then
    Exp := ''
  else
    Exp := ', expected ' + IntToStr(Expecting);
  if (Token = nil) then
    Result := 'UnwantedTokenException(found=nil' + Exp + ')'
  else
    Result := 'UnwantedTokenException(found=' + Token.Text + Exp + ')'
end;

{ EMissingTokenException }

constructor EMissingTokenException.Create(const AExpecting: Integer;
  const AInput: IIntStream; const AInserted: IANTLRInterface);
begin
  inherited Create(AExpecting, AInput);
  FInserted := AInserted;
end;

function EMissingTokenException.GetMissingType: Integer;
begin
  Result := Expecting;
end;

function EMissingTokenException.ToString: String;
begin
  if Assigned(FInserted) and Assigned(FToken) then
    Result := 'MissingTokenException(inserted ' + FInserted.ToString
      + ' at ' + FToken.Text + ')'
  else
    if Assigned(FToken) then
      Result := 'MissingTokenException(at ' + FToken.Text + ')'
    else
      Result := 'MissingTokenException';
end;

{ EMismatchedTreeNodeException }

constructor EMismatchedTreeNodeException.Create(const AExpecting: Integer;
  const AInput: IIntStream);
begin
  inherited Create(AInput);
  FExpecting := AExpecting;
end;

function EMismatchedTreeNodeException.ToString: String;
begin
  Result := 'MismatchedTreeNodeException(' + IntToStr(UnexpectedType)
    + '!=' + IntToStr(Expecting) + ')';
end;

{ ENoViableAltException }

constructor ENoViableAltException.Create(
  const AGrammarDecisionDescription: String; const ADecisionNumber,
  AStateNumber: Integer; const AInput: IIntStream);
begin
  inherited Create(AInput);
  FGrammarDecisionDescription := AGrammarDecisionDescription;
  FDecisionNumber := ADecisionNumber;
  FStateNumber := AStateNumber;
end;

function ENoViableAltException.ToString: String;
begin
  if Supports(Input, ICharStream) then
    Result := 'NoViableAltException(''' + Char(UnexpectedType) + '''@['
      + FGrammarDecisionDescription + '])'
  else
    Result := 'NoViableAltException(''' + IntToStr(UnexpectedType) + '''@['
      + FGrammarDecisionDescription + '])'
end;

{ EEarlyExitException }

constructor EEarlyExitException.Create(const ADecisionNumber: Integer;
  const AInput: IIntStream);
begin
  inherited Create(AInput);
  FDecisionNumber := ADecisionNumber;
end;

{ EMismatchedSetException }

constructor EMismatchedSetException.Create(const AExpecting: IBitSet;
  const AInput: IIntStream);
begin
  inherited Create(AInput);
  FExpecting := AExpecting;
end;

function EMismatchedSetException.ToString: String;
begin
  Result := 'MismatchedSetException(' + IntToStr(UnexpectedType)
    + '!=' + Expecting.ToString + ')';
end;

{ EMismatchedNotSetException }

function EMismatchedNotSetException.ToString: String;
begin
  Result := 'MismatchedNotSetException(' + IntToStr(UnexpectedType)
    + '!=' + Expecting.ToString + ')';
end;

{ EFailedPredicateException }

constructor EFailedPredicateException.Create(const AInput: IIntStream;
  const ARuleName, APredicateText: String);
begin
  inherited Create(AInput);
  FRuleName := ARuleName;
  FPredicateText := APredicateText;
end;

function EFailedPredicateException.ToString: String;
begin
  Result := 'FailedPredicateException(' + FRuleName + ',{' + FPredicateText + '}?)';
end;

{ EMismatchedRangeException }

constructor EMismatchedRangeException.Create(const AA, AB: Integer;
  const AInput: IIntStream);
begin
  inherited Create(FInput);
  FA := AA;
  FB := AB;
end;

function EMismatchedRangeException.ToString: String;
begin
  Result := 'MismatchedNotSetException(' + IntToStr(UnexpectedType)
    + ' not in [' + IntToStr(FA)+ ',' + IntToStr(FB) + '])';
end;

{ TCharStreamState }

function TCharStreamState.GetCharPositionInLine: Integer;
begin
  Result := FCharPositionInLine;
end;

function TCharStreamState.GetLine: Integer;
begin
  Result := FLine;
end;

function TCharStreamState.GetP: Integer;
begin
  Result := FP;
end;

procedure TCharStreamState.SetCharPositionInLine(const Value: Integer);
begin
  FCharPositionInLine := Value;
end;

procedure TCharStreamState.SetLine(const Value: Integer);
begin
  FLine := Value;
end;

procedure TCharStreamState.SetP(const Value: Integer);
begin
  FP := Value;
end;

{ TANTLRStringStream }

constructor TANTLRStringStream.Create(const AInput: String);
begin
  inherited Create;
  FLine := 1;
  FOwnsData := True;
  FN := Length(AInput);
  if (FN > 0) then
  begin
    GetMem(FData,FN * SizeOf(Char));
    Move(AInput[1],FData^,FN * SizeOf(Char));
  end;
end;

procedure TANTLRStringStream.Consume;
begin
  if (FP < FN) then
  begin
    Inc(FCharPositionInLine);
    if (FData[FP] = #10) then
    begin
      Inc(FLine);
      FCharPositionInLine := 0;
    end;
    Inc(FP);
  end;
end;

constructor TANTLRStringStream.Create(const AData: PChar;
  const ANumberOfActualCharsInArray: Integer);
begin
  inherited Create;
  FLine := 1;
  FOwnsData := False;
  FData := AData;
  FN := ANumberOfActualCharsInArray;
end;

constructor TANTLRStringStream.Create;
begin
  inherited Create;
  FLine := 1;
end;

destructor TANTLRStringStream.Destroy;
begin
  if (FOwnsData) then
    FreeMem(FData);
  inherited;
end;

function TANTLRStringStream.GetCharPositionInLine: Integer;
begin
  Result := FCharPositionInLine;
end;

function TANTLRStringStream.GetLine: Integer;
begin
  Result := FLine;
end;

function TANTLRStringStream.GetSourceName: String;
begin
  Result := FName;
end;

function TANTLRStringStream.Index: Integer;
begin
  Result := FP;
end;

function TANTLRStringStream.LA(I: Integer): Integer;
begin
  if (I = 0) then
    Result := 0 // undefined
  else begin
    if (I < 0) then
    begin
      Inc(I); // e.g., translate LA(-1) to use offset i=0; then data[p+0-1]
      if ((FP + I - 1) < 0) then
      begin
        Result := Integer(cscEOF);
        Exit;
      end;
    end;

    if ((FP + I - 1) >= FN) then
      Result := Integer(cscEOF)
    else
      Result := Integer(FData[FP + I - 1]);
  end;
end;

function TANTLRStringStream.LAChar(I: Integer): Char;
begin
  Result := Char(LA(I));
end;

function TANTLRStringStream.LT(const I: Integer): Integer;
begin
  Result := LA(I);
end;

function TANTLRStringStream.Mark: Integer;
var
  State: ICharStreamState;
begin
  if (FMarkers = nil) then
  begin
    FMarkers := TList<ICharStreamState>.Create;
    FMarkers.Add(nil);  // depth 0 means no backtracking, leave blank
  end;

  Inc(FMarkDepth);
  if (FMarkDepth >= FMarkers.Count) then
  begin
    State := TCharStreamState.Create;
    FMarkers.Add(State);
  end
  else
    State := FMarkers[FMarkDepth];

  State.P := FP;
  State.Line := FLine;
  State.CharPositionInLine := FCharPositionInLine;
  FLastMarker := FMarkDepth;
  Result := FMarkDepth;
end;

procedure TANTLRStringStream.Release(const Marker: Integer);
begin
  // unwind any other markers made after m and release m
  FMarkDepth := Marker;
  // release this marker
  Dec(FMarkDepth);
end;

procedure TANTLRStringStream.Reset;
begin
  FP := 0;
  FLine := 1;
  FCharPositionInLine := 0;
  FMarkDepth := 0;
end;

procedure TANTLRStringStream.Rewind(const Marker: Integer);
var
  State: ICharStreamState;
begin
  State := FMarkers[Marker];
  // restore stream state
  Seek(State.P);
  FLine := State.Line;
  FCharPositionInLine := State.CharPositionInLine;
  Release(Marker);
end;

procedure TANTLRStringStream.Rewind;
begin
  Rewind(FLastMarker);
end;

procedure TANTLRStringStream.Seek(const Index: Integer);
begin
  if (Index <= FP) then
    FP := Index // just jump; don't update stream state (line, ...)
  else begin
    // seek forward, consume until p hits index
    while (FP < Index) do
      Consume;
  end;
end;

procedure TANTLRStringStream.SetCharPositionInLine(const Value: Integer);
begin
  FCharPositionInLine := Value;
end;

procedure TANTLRStringStream.SetLine(const Value: Integer);
begin
  FLine := Value;
end;

function TANTLRStringStream.Size: Integer;
begin
  Result := FN;
end;

function TANTLRStringStream.Substring(const Start, Stop: Integer): String;
begin
  Result := Copy(FData, Start + 1, Stop - Start + 1);
end;

{ TANTLRFileStream }

constructor TANTLRFileStream.Create(const AFileName: String);
begin
  Create(AFilename,TEncoding.Default);
end;

constructor TANTLRFileStream.Create(const AFileName: String;
  const AEncoding: TEncoding);
begin
  inherited Create;
  FFileName := AFileName;
  Load(FFileName, AEncoding);
end;

function TANTLRFileStream.GetSourceName: String;
begin
  Result := FFileName;
end;

procedure TANTLRFileStream.Load(const FileName: String;
  const Encoding: TEncoding);
var
  FR: TStreamReader;
  S: String;
begin
  if (FFileName <> '') then
  begin
    if (Encoding = nil) then
      FR := TStreamReader.Create(FileName,TEncoding.Default)
    else
      FR := TStreamReader.Create(FileName,Encoding);

    try
      if (FOwnsData) then
      begin
        FreeMem(FData);
        FData := nil;
      end;

      FOwnsData := True;
      S := FR.ReadToEnd;
      FN := Length(S);
      if (FN > 0) then
      begin
        GetMem(FData,FN * SizeOf(Char));
        Move(S[1],FData^,FN * SizeOf(Char));
      end;
    finally
      FR.Free;
    end;
  end;
end;

{ TBitSet }

class function TBitSet.BitSetOf(const El: Integer): IBitSet;
begin
  Result := TBitSet.Create(El + 1);
  Result.Add(El);
end;

class function TBitSet.BitSetOf(const A, B: Integer): IBitSet;
begin
  Result := TBitSet.Create(Max(A,B) + 1);
  Result.Add(A);
  Result.Add(B);
end;

class function TBitSet.BitSetOf(const A, B, C: Integer): IBitSet;
begin
  Result := TBitSet.Create;
  Result.Add(A);
  Result.Add(B);
  Result.Add(C);
end;

class function TBitSet.BitSetOf(const A, B, C, D: Integer): IBitSet;
begin
  Result := TBitSet.Create;
  Result.Add(A);
  Result.Add(B);
  Result.Add(C);
  Result.Add(D);
end;

procedure TBitSet.Add(const El: Integer);
var
  N: Integer;
begin
  N := WordNumber(El);
  if (N >= Length(FBits)) then
    GrowToInclude(El);
  FBits[N] := FBits[N] or BitMask(El);
end;

class function TBitSet.BitMask(const BitNumber: Integer): UInt64;
var
  BitPosition: Integer;
begin
  BitPosition := BitNumber and MOD_MASK;
  Result := UInt64(1) shl BitPosition;
end;

function TBitSet.BitSetOr(const A: IBitSet): IBitSet;
begin
  Result := Clone as IBitSet;
  Result.OrInPlace(A);
end;

function TBitSet.Clone: IANTLRInterface;
var
  BS: TBitSet;
begin
  BS := TBitSet.Create;
  Result := BS;
  SetLength(BS.FBits,Length(FBits));
  if (Length(FBits) > 0) then
    Move(FBits[0],BS.FBits[0],Length(FBits) * SizeOf(UInt64));
end;

constructor TBitSet.Create;
begin
  Create(BITS);
end;

constructor TBitSet.Create(const ABits: array of UInt64);
begin
  inherited Create;
  SetLength(FBits, Length(ABits));
  if (Length(ABits) > 0) then
    Move(ABits[0], FBits[0], Length(ABits) * SizeOf(UInt64));
end;

constructor TBitSet.Create(const AItems: IList<Integer>);
var
  V: Integer;
begin
  Create(BITS);
  for V in AItems do
    Add(V);
end;

constructor TBitSet.Create(const ANBits: Integer);
begin
  inherited Create;
  SetLength(FBits,((ANBits - 1) shr LOG_BITS) + 1);
end;

function TBitSet.Equals(Obj: TObject): Boolean;
var
  OtherSet: TBitSet absolute Obj;
  I, N: Integer;
begin
  Result := False;
  if (Obj = nil) or (not (Obj is TBitSet)) then
    Exit;

  N := Min(Length(FBits), Length(OtherSet.FBits));

  // for any bits in common, compare
  for I := 0 to N - 1 do
  begin
    if (FBits[I] <> OtherSet.FBits[I]) then
      Exit;
  end;

  // make sure any extra bits are off
  if (Length(FBits) > N) then
  begin
    for I := N + 1 to Length(FBits) - 1 do
    begin
      if (FBits[I] <> 0) then
        Exit;
    end;
  end
  else
    if (Length(OtherSet.FBits) > N) then
    begin
      for I := N + 1 to Length(OtherSet.FBits) - 1 do
      begin
        if (OtherSet.FBits[I] <> 0) then
          Exit;
      end;
    end;

  Result := True;
end;

function TBitSet.GetIsNil: Boolean;
var
  I: Integer;
begin
  for I := Length(FBits) - 1 downto 0 do
    if (FBits[I] <> 0) then
    begin
      Result := False;
      Exit;
    end;
  Result := True;
end;

procedure TBitSet.GrowToInclude(const Bit: Integer);
var
  NewSize: Integer;
begin
  NewSize := Max(Length(FBits) shl 1,NumWordsToHold(Bit));
  SetLength(FBits,NewSize);
end;

function TBitSet.LengthInLongWords: Integer;
begin
  Result := Length(FBits);
end;

function TBitSet.Member(const El: Integer): Boolean;
var
  N: Integer;
begin
  if (El < 0) then
    Result := False
  else
  begin
    N := WordNumber(El);
    if (N >= Length(FBits)) then
      Result := False
    else
      Result := ((FBits[N] and BitMask(El)) <> 0);
  end;
end;

function TBitSet.NumBits: Integer;
begin
  Result := Length(FBits) shl LOG_BITS;
end;

class function TBitSet.NumWordsToHold(const El: Integer): Integer;
begin
  Result := (El shr LOG_BITS) + 1;
end;

procedure TBitSet.OrInPlace(const A: IBitSet);
var
  I, M: Integer;
  ABits: TUInt64Array;
begin
  if Assigned(A) then
  begin
    // If this is smaller than a, grow this first
    if (A.LengthInLongWords > Length(FBits)) then
      SetLength(FBits,A.LengthInLongWords);
    M := Min(Length(FBits), A.LengthInLongWords);
    ABits := A.ToPackedArray;
    for I := M - 1 downto 0 do
      FBits[I] := FBits[I] or ABits[I];
  end;
end;

procedure TBitSet.Remove(const El: Integer);
var
  N: Integer;
begin
  N := WordNumber(El);
  if (N < Length(FBits)) then
    FBits[N] := (FBits[N] and not BitMask(El));
end;

function TBitSet.Size: Integer;
var
  I, Bit: Integer;
  W: UInt64;
begin
  Result := 0;
  for I := Length(FBits) - 1 downto 0 do
  begin
    W := FBits[I];
    if (W <> 0) then
    begin
      for Bit := BITS - 1 downto 0 do
      begin
        if ((W and (UInt64(1) shl Bit)) <> 0) then
          Inc(Result);
      end;
    end;
  end;
end;

function TBitSet.ToArray: TIntegerArray;
var
  I, En: Integer;
begin
  SetLength(Result,Size);
  En := 0;
  for I := 0 to (Length(FBits) shl LOG_BITS) - 1 do
  begin
    if Member(I) then
    begin
      Result[En] := I;
      Inc(En);
    end;
  end;
end;

function TBitSet.ToPackedArray: TUInt64Array;
begin
  Result := FBits;
end;

function TBitSet.ToString: String;
begin
  Result := ToString(nil);
end;

function TBitSet.ToString(const TokenNames: TStringArray): String;
var
  Buf: TStringBuilder;
  I: Integer;
  HavePrintedAnElement: Boolean;
begin
  HavePrintedAnElement := False;
  Buf := TStringBuilder.Create;
  try
    Buf.Append('{');
    for I := 0 to (Length(FBits) shl LOG_BITS) - 1 do
    begin
      if Member(I) then
      begin
        if (I > 0) and HavePrintedAnElement then
          Buf.Append(',');
        if Assigned(TokenNames) then
          Buf.Append(TokenNames[I])
        else
          Buf.Append(I);
        HavePrintedAnElement := True;
      end;
    end;
    Buf.Append('}');
    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

class function TBitSet.WordNumber(const Bit: Integer): Integer;
begin
  Result := Bit shr LOG_BITS; // Bit / BITS
end;

{ TRecognizerSharedState }

constructor TRecognizerSharedState.Create;
var
  I: Integer;
begin
  inherited;
  SetLength(FFollowing,TBaseRecognizer.INITIAL_FOLLOW_STACK_SIZE);
  for I := 0 to TBaseRecognizer.INITIAL_FOLLOW_STACK_SIZE - 1 do
    FFollowing[I] := TBitSet.Create;
  FFollowingStackPointer := -1;
  FLastErrorIndex := -1;
  FTokenStartCharIndex := -1;
end;

function TRecognizerSharedState.GetBacktracking: Integer;
begin
  Result := FBacktracking;
end;

function TRecognizerSharedState.GetChannel: Integer;
begin
  Result := FChannel;
end;

function TRecognizerSharedState.GetErrorRecovery: Boolean;
begin
  Result := FErrorRecovery;
end;

function TRecognizerSharedState.GetFailed: Boolean;
begin
  Result := FFailed;
end;

function TRecognizerSharedState.GetFollowing: TBitSetArray;
begin
  Result := FFollowing;
end;

function TRecognizerSharedState.GetFollowingStackPointer: Integer;
begin
  Result := FFollowingStackPointer;
end;

function TRecognizerSharedState.GetLastErrorIndex: Integer;
begin
  Result := FLastErrorIndex;
end;

function TRecognizerSharedState.GetRuleMemo: TDictionaryArray<Integer, Integer>;
begin
  Result := FRuleMemo;
end;

function TRecognizerSharedState.GetRuleMemoCount: Integer;
begin
  Result := Length(FRuleMemo);
end;

function TRecognizerSharedState.GetSyntaxErrors: Integer;
begin
  Result := FSyntaxErrors;
end;

function TRecognizerSharedState.GetText: String;
begin
  Result := FText;
end;

function TRecognizerSharedState.GetToken: IToken;
begin
  Result := FToken;
end;

function TRecognizerSharedState.GetTokenStartCharIndex: Integer;
begin
  Result := FTokenStartCharIndex;
end;

function TRecognizerSharedState.GetTokenStartCharPositionInLine: Integer;
begin
  Result := FTokenStartCharPositionInLine;
end;

function TRecognizerSharedState.GetTokenStartLine: Integer;
begin
  Result := FTokenStartLine;
end;

function TRecognizerSharedState.GetTokenType: Integer;
begin
  Result := FTokenType;
end;

procedure TRecognizerSharedState.SetBacktracking(const Value: Integer);
begin
  FBacktracking := Value;
end;

procedure TRecognizerSharedState.SetChannel(const Value: Integer);
begin
  FChannel := Value;
end;

procedure TRecognizerSharedState.SetErrorRecovery(const Value: Boolean);
begin
  FErrorRecovery := Value;
end;

procedure TRecognizerSharedState.SetFailed(const Value: Boolean);
begin
  FFailed := Value;
end;

procedure TRecognizerSharedState.SetFollowing(const Value: TBitSetArray);
begin
  FFollowing := Value;
end;

procedure TRecognizerSharedState.SetFollowingStackPointer(const Value: Integer);
begin
  FFollowingStackPointer := Value;
end;

procedure TRecognizerSharedState.SetLastErrorIndex(const Value: Integer);
begin
  FLastErrorIndex := Value;
end;

procedure TRecognizerSharedState.SetRuleMemoCount(const Value: Integer);
begin
  SetLength(FRuleMemo, Value);
end;

procedure TRecognizerSharedState.SetSyntaxErrors(const Value: Integer);
begin
  FSyntaxErrors := Value;
end;

procedure TRecognizerSharedState.SetText(const Value: String);
begin
  FText := Value;
end;

procedure TRecognizerSharedState.SetToken(const Value: IToken);
begin
  FToken := Value;
end;

procedure TRecognizerSharedState.SetTokenStartCharIndex(const Value: Integer);
begin
  FTokenStartCharIndex := Value;
end;

procedure TRecognizerSharedState.SetTokenStartCharPositionInLine(
  const Value: Integer);
begin
  FTokenStartCharPositionInLine := Value;
end;

procedure TRecognizerSharedState.SetTokenStartLine(const Value: Integer);
begin
  FTokenStartLine := Value;
end;

procedure TRecognizerSharedState.SetTokenType(const Value: Integer);
begin
  FTokenType := Value;
end;

{ TCommonToken }

constructor TCommonToken.Create;
begin
  inherited;
  FChannel := TToken.DEFAULT_CHANNEL;
  FCharPositionInLine := -1;
  FIndex := -1;
end;

constructor TCommonToken.Create(const ATokenType: Integer);
begin
  Create;
  FTokenType := ATokenType;
end;

constructor TCommonToken.Create(const AInput: ICharStream; const ATokenType,
  AChannel, AStart, AStop: Integer);
begin
  Create;
  FInput := AInput;
  FTokenType := ATokenType;
  FChannel := AChannel;
  FStart := AStart;
  FStop := AStop;
end;

constructor TCommonToken.Create(const ATokenType: Integer; const AText: String);
begin
  Create;
  FTokenType := ATokenType;
  FChannel := TToken.DEFAULT_CHANNEL;
  FText := AText;
end;

function TCommonToken.GetChannel: Integer;
begin
  Result := FChannel;
end;

function TCommonToken.GetCharPositionInLine: Integer;
begin
  Result := FCharPositionInLine;
end;

function TCommonToken.GetInputStream: ICharStream;
begin
  Result := FInput;
end;

function TCommonToken.GetLine: Integer;
begin
  Result := FLine;
end;

function TCommonToken.GetStartIndex: Integer;
begin
  Result := FStart;
end;

function TCommonToken.GetStopIndex: Integer;
begin
  Result := FStop;
end;

function TCommonToken.GetText: String;
begin
  if (FText <> '') then
    Result := FText
  else
    if (FInput = nil) then
      Result := ''
    else
      Result := FInput.Substring(FStart, FStop);
end;

function TCommonToken.GetTokenIndex: Integer;
begin
  Result := FIndex;
end;

function TCommonToken.GetTokenType: Integer;
begin
  Result := FTokenType;
end;

procedure TCommonToken.SetChannel(const Value: Integer);
begin
  FChannel := Value;
end;

procedure TCommonToken.SetCharPositionInLine(const Value: Integer);
begin
  FCharPositionInLine := Value;
end;

procedure TCommonToken.SetInputStream(const Value: ICharStream);
begin
  FInput := Value;
end;

procedure TCommonToken.SetLine(const Value: Integer);
begin
  FLine := Value;
end;

procedure TCommonToken.SetStartIndex(const Value: Integer);
begin
  FStart := Value;
end;

procedure TCommonToken.SetStopIndex(const Value: Integer);
begin
  FStop := Value;
end;

procedure TCommonToken.SetText(const Value: String);
begin
  (* Override the text for this token.  The property getter
   * will return this text rather than pulling from the buffer.
   * Note that this does not mean that start/stop indexes are
   * not valid.  It means that the input was converted to a new
   * string in the token object.
   *)
  FText := Value;
end;

procedure TCommonToken.SetTokenIndex(const Value: Integer);
begin
  FIndex := Value;
end;

procedure TCommonToken.SetTokenType(const Value: Integer);
begin
  FTokenType := Value;
end;

function TCommonToken.ToString: String;
var
  ChannelStr, Txt: String;
begin
  if (FChannel > 0) then
    ChannelStr := ',channel=' + IntToStr(FChannel)
  else
    ChannelStr := '';

  Txt := GetText;
  if (Txt <> '') then
  begin
    Txt := ReplaceStr(Txt,#10,'\n');
    Txt := ReplaceStr(Txt,#13,'\r');
    Txt := ReplaceStr(Txt,#9,'\t');
  end else
    Txt := '<no text>';

  Result := Format('[@%d,%d:%d=''%s'',<%d>%s,%d:%d]',
    [FIndex,FStart,FStop,Txt,FTokenType,ChannelStr,FLine,FCharPositionInLine]);
end;

constructor TCommonToken.Create(const AOldToken: IToken);
var
  OldCommonToken: ICommonToken;
begin
  Create;
  FText := AOldToken.Text;
  FTokenType := AOldToken.TokenType;
  FLine := AOldToken.Line;
  FIndex := AOldToken.TokenIndex;
  FCharPositionInLine := AOldToken.CharPositionInLine;
  FChannel := AOldToken.Channel;
  if Supports(AOldToken, ICommonToken, OldCommonToken) then
  begin
    FStart := OldCommonToken.StartIndex;
    FStop := OldCommonToken.StopIndex;
  end;
end;

{ TClassicToken }

constructor TClassicToken.Create(const AOldToken: IToken);
begin
  inherited Create;
  FText := AOldToken.Text;
  FTokenType := AOldToken.TokenType;
  FLine := AOldToken.Line;
  FCharPositionInLine := AOldToken.CharPositionInLine;
  FChannel := AOldToken.Channel;
end;

constructor TClassicToken.Create(const ATokenType: Integer);
begin
  inherited Create;
  FTokenType := ATokenType;
end;

constructor TClassicToken.Create(const ATokenType: Integer; const AText: String;
  const AChannel: Integer);
begin
  inherited Create;
  FTokenType := ATokenType;
  FText := AText;
  FChannel := AChannel;
end;

constructor TClassicToken.Create(const ATokenType: Integer;
  const AText: String);
begin
  inherited Create;
  FTokenType := ATokenType;
  FText := AText;
end;

function TClassicToken.GetChannel: Integer;
begin
  Result := FChannel;
end;

function TClassicToken.GetCharPositionInLine: Integer;
begin
  Result := FCharPositionInLine;
end;

function TClassicToken.GetInputStream: ICharStream;
begin
  // No default implementation
  Result := nil;
end;

function TClassicToken.GetLine: Integer;
begin
  Result := FLine;
end;

function TClassicToken.GetText: String;
begin
  Result := FText;
end;

function TClassicToken.GetTokenIndex: Integer;
begin
  Result := FIndex;
end;

function TClassicToken.GetTokenType: Integer;
begin
  Result := FTokenType;
end;

procedure TClassicToken.SetChannel(const Value: Integer);
begin
  FChannel := Value;
end;

procedure TClassicToken.SetCharPositionInLine(const Value: Integer);
begin
  FCharPositionInLine := Value;
end;

procedure TClassicToken.SetInputStream(const Value: ICharStream);
begin
  // No default implementation
end;

procedure TClassicToken.SetLine(const Value: Integer);
begin
  FLine := Value;
end;

procedure TClassicToken.SetText(const Value: String);
begin
  FText := Value;
end;

procedure TClassicToken.SetTokenIndex(const Value: Integer);
begin
  FIndex := Value;
end;

procedure TClassicToken.SetTokenType(const Value: Integer);
begin
  FTokenType := Value;
end;

function TClassicToken.ToString: String;
var
  ChannelStr, Txt: String;
begin
  if (FChannel > 0) then
    ChannelStr := ',channel=' + IntToStr(FChannel)
  else
    ChannelStr := '';
  Txt := FText;
  if (Txt <> '') then
  begin
    Txt := ReplaceStr(Txt,#10,'\n');
    Txt := ReplaceStr(Txt,#13,'\r');
    Txt := ReplaceStr(Txt,#9,'\t');
  end else
    Txt := '<no text>';

  Result := Format('[@%d,''%s'',<%d>%s,%d:%d]',
    [FIndex,Txt,FTokenType,ChannelStr,FLine,FCharPositionInLine]);
end;

{ TToken }

class procedure TToken.Initialize;
begin
  EOF_TOKEN := TCommonToken.Create(EOF);
  INVALID_TOKEN := TCommonToken.Create(INVALID_TOKEN_TYPE);
  SKIP_TOKEN := TCommonToken.Create(INVALID_TOKEN_TYPE);
end;

{ TBaseRecognizer }

constructor TBaseRecognizer.Create;
begin
  inherited;
  FState := TRecognizerSharedState.Create;
end;

function TBaseRecognizer.AlreadyParsedRule(const Input: IIntStream;
  const RuleIndex: Integer): Boolean;
var
  StopIndex: Integer;
begin
  StopIndex := GetRuleMemoization(RuleIndex, Input.Index);
  Result := (StopIndex <> MEMO_RULE_UNKNOWN);
  if Result then
  begin
    if (StopIndex = MEMO_RULE_FAILED) then
      FState.Failed := True
    else
      Input.Seek(StopIndex + 1);  // jump to one past stop token
  end;
end;

procedure TBaseRecognizer.BeginBacktrack(const Level: Integer);
begin
  // No defeault implementation
end;

procedure TBaseRecognizer.BeginResync;
begin
  // No defeault implementation
end;

procedure TBaseRecognizer.ConsumeUntil(const Input: IIntStream;
  const TokenType: Integer);
var
  TType: Integer;
begin
  TType := Input.LA(1);
  while (TType <> TToken.EOF) and (TType <> TokenType) do
  begin
    Input.Consume;
    TType := Input.LA(1);
  end;
end;

function TBaseRecognizer.CombineFollows(const Exact: Boolean): IBitSet;
var
  I, Top: Integer;
  LocalFollowSet: IBitSet;
begin
  Top := FState.FollowingStackPointer;
  Result := TBitSet.Create;
  for I := Top downto 0 do
  begin
    LocalFollowSet := FState.Following[I];
    Result.OrInPlace(LocalFollowSet);
    if (Exact) then
    begin
      // can we see end of rule?
      if LocalFollowSet.Member(TToken.EOR_TOKEN_TYPE) then
      begin
        // Only leave EOR in set if at top (start rule); this lets
        // us know if have to include follow(start rule); i.e., EOF
        if (I > 0) then
          Result.Remove(TToken.EOR_TOKEN_TYPE);
      end
      else
        // can't see end of rule, quit
        Break;
    end;
  end;
end;

function TBaseRecognizer.ComputeContextSensitiveRuleFOLLOW: IBitSet;
begin
  Result := CombineFollows(True);
end;

function TBaseRecognizer.ComputeErrorRecoverySet: IBitSet;
begin
  Result := CombineFollows(False);
end;

procedure TBaseRecognizer.ConsumeUntil(const Input: IIntStream;
  const BitSet: IBitSet);
var
  TType: Integer;
begin
  TType := Input.LA(1);
  while (TType <> TToken.EOF) and (not BitSet.Member(TType)) do
  begin
    Input.Consume;
    TType := Input.LA(1);
  end;
end;

constructor TBaseRecognizer.Create(const AState: IRecognizerSharedState);
begin
  if (AState = nil) then
    Create
  else
  begin
    inherited Create;
    FState := AState;
  end;
end;

procedure TBaseRecognizer.DisplayRecognitionError(
  const TokenNames: TStringArray; const E: ERecognitionException);
var
  Hdr, Msg: String;
begin
  Hdr := GetErrorHeader(E);
  Msg := GetErrorMessage(E, TokenNames);
  EmitErrorMessage(Hdr + ' ' + Msg);
end;

procedure TBaseRecognizer.EmitErrorMessage(const Msg: String);
begin
  WriteLn(Msg);
end;

procedure TBaseRecognizer.EndBacktrack(const Level: Integer;
  const Successful: Boolean);
begin
  // No defeault implementation
end;

procedure TBaseRecognizer.EndResync;
begin
  // No defeault implementation
end;

function TBaseRecognizer.GetBacktrackingLevel: Integer;
begin
  Result := FState.Backtracking;
end;

function TBaseRecognizer.GetCurrentInputSymbol(
  const Input: IIntStream): IANTLRInterface;
begin
  // No defeault implementation
  Result := nil;
end;

function TBaseRecognizer.GetErrorHeader(const E: ERecognitionException): String;
begin
  Result := 'line ' + IntToStr(E.Line) + ':' + IntToStr(E.CharPositionInLine);
end;

function TBaseRecognizer.GetErrorMessage(const E: ERecognitionException;
  const TokenNames: TStringArray): String;
var
  UTE: EUnwantedTokenException absolute E;
  MTE: EMissingTokenException absolute E;
  MMTE: EMismatchedTokenException absolute E;
  MTNE: EMismatchedTreeNodeException absolute E;
  NVAE: ENoViableAltException absolute E;
  EEE: EEarlyExitException absolute E;
  MSE: EMismatchedSetException absolute E;
  MNSE: EMismatchedNotSetException absolute E;
  FPE: EFailedPredicateException absolute E;
  TokenName: String;
begin
  Result := E.Message;
  if (E is EUnwantedTokenException) then
  begin
    if (UTE.Expecting = TToken.EOF) then
      TokenName := 'EOF'
    else
      TokenName := TokenNames[UTE.Expecting];
    Result := 'extraneous input ' + GetTokenErrorDisplay(UTE.UnexpectedToken)
      + ' expecting ' + TokenName;
  end
  else
    if (E is EMissingTokenException) then
    begin
      if (MTE.Expecting = TToken.EOF) then
        TokenName := 'EOF'
      else
        TokenName := TokenNames[MTE.Expecting];
      Result := 'missing ' + TokenName + ' at ' + GetTokenErrorDisplay(E.Token);
    end
    else
      if (E is EMismatchedTokenException) then
      begin
        if (MMTE.Expecting = TToken.EOF) then
          TokenName := 'EOF'
        else
          TokenName := TokenNames[MMTE.Expecting];
        Result := 'mismatched input ' + GetTokenErrorDisplay(E.Token)
          + ' expecting ' + TokenName;
      end
      else
        if (E is EMismatchedTreeNodeException) then
        begin
          if (MTNE.Expecting = TToken.EOF) then
            Result := 'EOF'
          else
            Result := TokenNames[MTNE.Expecting];
          // The ternary operator is only necessary because of a bug in the .NET framework
          Result := 'mismatched tree node: ';
          if (MTNE.Node <> nil) and (MTNE.Node.ToString <> '') then
            Result := Result + MTNE.Node.ToString;
          Result := Result + ' expecting ' + TokenName;
        end
        else
          if (E is ENoViableAltException) then
          begin
            // for development, can add "decision=<<"+nvae.grammarDecisionDescription+">>"
            // and "(decision="+nvae.decisionNumber+") and
            // "state "+nvae.stateNumber
            Result := 'no viable alternative at input ' + GetTokenErrorDisplay(E.Token);
          end
          else
            if (E is EEarlyExitException) then
            begin
              // for development, can add "(decision="+eee.decisionNumber+")"
              Result := 'required (...)+ loop did not  match anyting at input '
                + GetTokenErrorDisplay(E.Token);
            end else
              if (E is EMismatchedSetException) then
              begin
                Result := 'mismatched input ' + GetTokenErrorDisplay(E.Token)
                  + ' expecting set ' + MSE.Expecting.ToString;
              end
              else
                if (E is EMismatchedNotSetException) then
                begin
                  Result := 'mismatched input ' + GetTokenErrorDisplay(E.Token)
                    + ' expecting set ' + MSE.Expecting.ToString;
                end
                else
                  if (E is EFailedPredicateException) then
                  begin
                    Result := 'rule ' + FPE.RuleName
                      + ' failed predicate: {' + FPE.PredicateText + '}?';
                  end;
end;

function TBaseRecognizer.GetGrammarFileName: String;
begin
  // No defeault implementation
  Result := '';
end;

function TBaseRecognizer.GetMissingSymbol(const Input: IIntStream;
  const E: ERecognitionException; const ExpectedTokenType: Integer;
  const Follow: IBitSet): IANTLRInterface;
begin
  // No defeault implementation
  Result := nil;
end;

function TBaseRecognizer.GetNumberOfSyntaxErrors: Integer;
begin
  Result := FState.SyntaxErrors;
end;

function TBaseRecognizer.GetRuleMemoization(const RuleIndex,
  RuleStartIndex: Integer): Integer;
var
  Dict: IDictionary<Integer, Integer>;
begin
  Dict := FState.RuleMemo[RuleIndex];
  if (Dict = nil) then
  begin
    Dict := TDictionary<Integer, Integer>.Create;
    FState.RuleMemo[RuleIndex] := Dict;
  end;
  if (not Dict.TryGetValue(RuleStartIndex, Result)) then
    Result := MEMO_RULE_UNKNOWN;
end;

function TBaseRecognizer.GetRuleMemoizationChaceSize: Integer;
var
  RuleMap: IDictionary<Integer, Integer>;
begin
  Result := 0;
  if Assigned(FState.RuleMemo) then
  begin
    for RuleMap in FState.RuleMemo do
      if Assigned(RuleMap) then
        Inc(Result,RuleMap.Count);  // how many input indexes are recorded?
  end;
end;

function TBaseRecognizer.GetState: IRecognizerSharedState;
begin
  Result := FState;
end;

function TBaseRecognizer.GetTokenErrorDisplay(const T: IToken): String;
begin
  Result := T.Text;
  if (Result = '') then
  begin
    if (T.TokenType = TToken.EOF) then
      Result := '<EOF>'
    else
      Result := '<' + IntToStr(T.TokenType) + '>';
  end;
  Result := ReplaceStr(Result,#10,'\n');
  Result := ReplaceStr(Result,#13,'\r');
  Result := ReplaceStr(Result,#9,'\t');
  Result := '''' + Result + '''';
end;

function TBaseRecognizer.GetTokenNames: TStringArray;
begin
  // no default implementation
  Result := nil;
end;

function TBaseRecognizer.Match(const Input: IIntStream;
  const TokenType: Integer; const Follow: IBitSet): IANTLRInterface;
begin
  Result := GetCurrentInputSymbol(Input);
  if (Input.LA(1) = TokenType) then
  begin
    Input.Consume;
    FState.ErrorRecovery := False;
    FState.Failed := False;
  end else
  begin
    if (FState.Backtracking > 0) then
      FState.Failed := True
    else
    begin
      Mismatch(Input, TokenType, Follow);
      Result := RecoverFromMismatchedToken(Input, TokenType, Follow);
    end;
  end;
end;

procedure TBaseRecognizer.MatchAny(const Input: IIntStream);
begin
  FState.ErrorRecovery := False;
  FState.Failed := False;
  Input.Consume;
end;

procedure TBaseRecognizer.Memoize(const Input: IIntStream; const RuleIndex,
  RuleStartIndex: Integer);
var
  StopTokenIndex: Integer;
  Dict: IDictionary<Integer, Integer>;
begin
  Dict := FState.RuleMemo[RuleIndex];
  if Assigned(Dict) then
  begin
    if FState.Failed then
      StopTokenIndex := MEMO_RULE_FAILED
    else
      StopTokenIndex := Input.Index - 1;
    Dict.AddOrSetValue(RuleStartIndex, StopTokenIndex);
  end;
end;

procedure TBaseRecognizer.Mismatch(const Input: IIntStream;
  const TokenType: Integer; const Follow: IBitSet);
begin
  if MismatchIsUnwantedToken(Input, TokenType) then
    raise EUnwantedTokenException.Create(TokenType, Input)
  else
    if MismatchIsMissingToken(Input, Follow) then
      raise EMissingTokenException.Create(TokenType, Input, nil)
    else
      raise EMismatchedTokenException.Create(TokenType, Input);
end;

function TBaseRecognizer.MismatchIsMissingToken(const Input: IIntStream;
  const Follow: IBitSet): Boolean;
var
  ViableTokensFollowingThisRule, Follow2: IBitSet;
begin
  if (Follow = nil) then
    // we have no information about the follow; we can only consume
    // a single token and hope for the best
    Result := False
  else
  begin
    Follow2 := Follow;
    // compute what can follow this grammar element reference
    if (Follow.Member(TToken.EOR_TOKEN_TYPE)) then
    begin
      ViableTokensFollowingThisRule := ComputeContextSensitiveRuleFOLLOW();
      Follow2 := Follow.BitSetOr(ViableTokensFollowingThisRule);
      if (FState.FollowingStackPointer >= 0) then
        // remove EOR if we're not the start symbol
        Follow2.Remove(TToken.EOR_TOKEN_TYPE);
    end;

    // if current token is consistent with what could come after set
    // then we know we're missing a token; error recovery is free to
    // "insert" the missing token

    // BitSet cannot handle negative numbers like -1 (EOF) so I leave EOR
    // in follow set to indicate that the fall of the start symbol is
    // in the set (EOF can follow).
    if (Follow2.Member(Input.LA(1)) or Follow2.Member(TToken.EOR_TOKEN_TYPE)) then
      Result := True
    else
      Result := False;
  end;
end;

function TBaseRecognizer.MismatchIsUnwantedToken(const Input: IIntStream;
  const TokenType: Integer): Boolean;
begin
  Result := (Input.LA(2) = TokenType);
end;

procedure TBaseRecognizer.PushFollow(const FSet: IBitSet);
var
  F: TBitSetArray;
  I: Integer;
begin
  if ((FState.FollowingStackPointer + 1) >= Length(FState.Following)) then
  begin
    SetLength(F, Length(FState.Following) * 2);
    FillChar(F[0], Length(F) * SizeOf(IBitSet), 0);
    for I := 0 to Length(FState.Following) - 1 do
      F[I] := FState.Following[I];
    FState.Following := F;
  end;
  FState.FollowingStackPointer := FState.FollowingStackPointer + 1;
  FState.Following[FState.FollowingStackPointer] := FSet;
end;

procedure TBaseRecognizer.Recover(const Input: IIntStream;
  const RE: ERecognitionException);
var
  FollowSet: IBitSet;
begin
  if (FState.LastErrorIndex = Input.Index) then
    // uh oh, another error at same token index; must be a case
    // where LT(1) is in the recovery token set so nothing is
    // consumed; consume a single token so at least to prevent
    // an infinite loop; this is a failsafe.
    Input.Consume;
  FState.LastErrorIndex := Input.Index;
  FollowSet := ComputeErrorRecoverySet;
  BeginResync;
  ConsumeUntil(Input,FollowSet);
  EndResync;
end;

function TBaseRecognizer.RecoverFromMismatchedSet(const Input: IIntStream;
  const E: ERecognitionException; const Follow: IBitSet): IANTLRInterface;
begin
  if MismatchIsMissingToken(Input, Follow) then
  begin
    ReportError(E);
    // we don't know how to conjure up a token for sets yet
    Result := GetMissingSymbol(Input, E, TToken.INVALID_TOKEN_TYPE, Follow);
  end
  else
  begin
    // TODO do single token deletion like above for Token mismatch
    Result := nil;
    raise E;
  end;
end;

function TBaseRecognizer.RecoverFromMismatchedToken(const Input: IIntStream;
  const TokenType: Integer; const Follow: IBitSet): IANTLRInterface;
var
  E: ERecognitionException;
begin
  // if next token is what we are looking for then "delete" this token
  if MismatchIsUnwantedToken(Input, TokenType) then
  begin
    E := EUnwantedTokenException.Create(TokenType, Input);
    BeginResync;
    Input.Consume; // simply delete extra token
    EndResync;
    ReportError(E);  // report after consuming so AW sees the token in the exception
    // we want to return the token we're actually matching
    Result := GetCurrentInputSymbol(Input);
    Input.Consume;  // move past ttype token as if all were ok
  end
  else
  begin
    // can't recover with single token deletion, try insertion
    if MismatchIsMissingToken(Input, Follow) then
    begin
      E := nil;
      Result := GetMissingSymbol(Input, E, TokenType, Follow);
      E := EMissingTokenException.Create(TokenType, Input, Result);
      ReportError(E);  // report after inserting so AW sees the token in the exception
    end
    else
    begin
      // even that didn't work; must throw the exception
      raise EMismatchedTokenException.Create(TokenType, Input);
    end;
  end;
end;

procedure TBaseRecognizer.ReportError(const E: ERecognitionException);
begin
  // if we've already reported an error and have not matched a token
  // yet successfully, don't report any errors.
  if (not FState.ErrorRecovery) then
  begin
    FState.SyntaxErrors := FState.SyntaxErrors + 1; // don't count spurious
    FState.ErrorRecovery := True;
    DisplayRecognitionError(GetTokenNames, E);
  end;
end;

procedure TBaseRecognizer.Reset;
var
  I: Integer;
begin
  // wack everything related to error recovery
  if (FState = nil) then
    Exit;  // no shared state work to do

  FState.FollowingStackPointer := -1;
  FState.ErrorRecovery := False;
  FState.LastErrorIndex := -1;
  FState.Failed := False;
  FState.SyntaxErrors := 0;

  // wack everything related to backtracking and memoization
  FState.Backtracking := 0;
  if Assigned(FState.RuleMemo) then
    for I := 0 to Length(FState.RuleMemo) - 1 do
    begin
      // wipe cache
      FState.RuleMemo[I] := nil;
    end;
end;

function TBaseRecognizer.ToStrings(const Tokens: IList<IToken>): IList<String>;
var
  Token: IToken;
begin
  if (Tokens = nil) then
    Result := nil
  else
  begin
    Result := TList<String>.Create;
    for Token in Tokens do
      Result.Add(Token.Text);
  end;
end;

procedure TBaseRecognizer.TraceIn(const RuleName: String;
  const RuleIndex: Integer; const InputSymbol: String);
begin
  Write('enter ' + RuleName + ' ' + InputSymbol);
  if (FState.Failed) then
    WriteLn(' failed=True');
  if (FState.Backtracking > 0) then
    Write(' backtracking=' + IntToStr(FState.Backtracking));
  WriteLn;
end;

procedure TBaseRecognizer.TraceOut(const RuleName: String;
  const RuleIndex: Integer; const InputSymbol: String);
begin
  Write('exit ' + RuleName + ' ' + InputSymbol);
  if (FState.Failed) then
    WriteLn(' failed=True');
  if (FState.Backtracking > 0) then
    Write(' backtracking=' + IntToStr(FState.Backtracking));
  WriteLn;
end;

{ TCommonTokenStream }

procedure TCommonTokenStream.Consume;
begin
  if (FP < FTokens.Count) then
  begin
    Inc(FP);
    FP := SkipOffTokenChannels(FP); // leave p on valid token
  end;
end;

constructor TCommonTokenStream.Create;
begin
  inherited;
  FP := -1;
  FChannel := TToken.DEFAULT_CHANNEL;
  FTokens := TList<IToken>.Create;
  FTokens.Capacity := 500;
end;

constructor TCommonTokenStream.Create(const ATokenSource: ITokenSource);
begin
  Create;
  FTokenSource := ATokenSource;
end;

procedure TCommonTokenStream.DiscardOffChannelTokens(const Discard: Boolean);
begin
  FDiscardOffChannelTokens := Discard;
end;

procedure TCommonTokenStream.DiscardTokenType(const TType: Integer);
begin
  if (FDiscardSet = nil) then
    FDiscardSet := THashList<Integer, Integer>.Create;
  FDiscardSet.Add(TType, TType);
end;

procedure TCommonTokenStream.FillBuffer;
var
  Index: Integer;
  T: IToken;
  Discard: Boolean;
begin
  Index := 0;
  T := FTokenSource.NextToken;
  while Assigned(T) and (T.TokenType <> Integer(cscEOF)) do
  begin
    Discard := False;
    // is there a channel override for token type?
    if Assigned(FChannelOverrideMap) then
      if FChannelOverrideMap.ContainsKey(T.TokenType) then
        T.Channel := FChannelOverrideMap[T.TokenType];

    if Assigned(FDiscardSet) and FDiscardSet.ContainsKey(T.TokenType) then
      Discard := True
    else
      if FDiscardOffChannelTokens and (T.Channel <> FChannel) then
        Discard := True;

    if (not Discard) then
    begin
      T.TokenIndex := Index;
      FTokens.Add(T);
      Inc(Index);
    end;

    T := FTokenSource.NextToken;
  end;
  // leave p pointing at first token on channel
  FP := 0;
  FP := SkipOffTokenChannels(FP);
end;

function TCommonTokenStream.Get(const I: Integer): IToken;
begin
  Result := FTokens[I];
end;

function TCommonTokenStream.GetSourceName: String;
begin
  Result := FTokenSource.SourceName;
end;

function TCommonTokenStream.GetTokens(const Start, Stop: Integer;
  const Types: IList<Integer>): IList<IToken>;
begin
  Result := GetTokens(Start, Stop, TBitSet.Create(Types));
end;

function TCommonTokenStream.GetTokens(const Start, Stop,
  TokenType: Integer): IList<IToken>;
begin
  Result := GetTokens(Start, Stop, TBitSet.BitSetOf(TokenType));
end;

function TCommonTokenStream.GetTokens(const Start, Stop: Integer;
  const Types: IBitSet): IList<IToken>;
var
  I, StartIndex, StopIndex: Integer;
  T: IToken;
begin
  if (FP = -1) then
    FillBuffer;
  StopIndex := Min(Stop,FTokens.Count - 1);
  StartIndex := Max(Start,0);
  if (StartIndex > StopIndex) then
    Result := nil
  else
  begin
    Result := TList<IToken>.Create;
    for I := StartIndex to StopIndex do
    begin
      T := FTokens[I];
      if (Types = nil) or Types.Member(T.TokenType) then
        Result.Add(T);
    end;
    if (Result.Count = 0) then
      Result := nil;
  end;
end;

function TCommonTokenStream.GetTokens: IList<IToken>;
begin
  if (FP = -1) then
    FillBuffer;
  Result := FTokens;
end;

function TCommonTokenStream.GetTokens(const Start,
  Stop: Integer): IList<IToken>;
begin
  Result := GetTokens(Start, Stop, IBitSet(nil));
end;

function TCommonTokenStream.GetTokenSource: ITokenSource;
begin
  Result := FTokenSource;
end;

function TCommonTokenStream.Index: Integer;
begin
  Result := FP;
end;

function TCommonTokenStream.LA(I: Integer): Integer;
begin
  Result := LT(I).TokenType;
end;

function TCommonTokenStream.LAChar(I: Integer): Char;
begin
  Result := Char(LA(I));
end;

function TCommonTokenStream.LB(const K: Integer): IToken;
var
  I, N: Integer;
begin
  if (FP = -1) then
    FillBuffer;
  if (K = 0) then
    Result := nil
  else
    if ((FP - K) < 0) then
      Result := nil
    else
    begin
      I := FP;
      N := 1;
      // find k good tokens looking backwards
      while (N <= K) do
      begin
        // skip off-channel tokens
        I := SkipOffTokenChannelsReverse(I - 1); // leave p on valid token
        Inc(N);
      end;
      if (I < 0) then
        Result := nil
      else
        Result := FTokens[I];
    end;
end;

function TCommonTokenStream.LT(const K: Integer): IToken;
var
  I, N: Integer;
begin
  if (FP = -1) then
    FillBuffer;
  if (K = 0) then
    Result := nil
  else
    if (K < 0) then
      Result := LB(-K)
    else
      if ((FP + K - 1) >= FTokens.Count) then
        Result := TToken.EOF_TOKEN
      else
      begin
        I := FP;
        N := 1;
        // find k good tokens
        while (N < K) do
        begin
          // skip off-channel tokens
          I := SkipOffTokenChannels(I + 1); // leave p on valid token
          Inc(N);
        end;
        if (I >= FTokens.Count) then
          Result := TToken.EOF_TOKEN
        else
          Result := FTokens[I];
      end;
end;

function TCommonTokenStream.Mark: Integer;
begin
  if (FP = -1) then
    FillBuffer;
  FLastMarker := Index;
  Result := FLastMarker;
end;

procedure TCommonTokenStream.Release(const Marker: Integer);
begin
  // no resources to release
end;

procedure TCommonTokenStream.Reset;
begin
  FP := 0;
  FLastMarker := 0;
end;

procedure TCommonTokenStream.Rewind(const Marker: Integer);
begin
  Seek(Marker);
end;

procedure TCommonTokenStream.Rewind;
begin
  Seek(FLastMarker);
end;

procedure TCommonTokenStream.Seek(const Index: Integer);
begin
  FP := Index;
end;

procedure TCommonTokenStream.SetTokenSource(const Value: ITokenSource);
begin
  FTokenSource := Value;
  FTokens.Clear;
  FP := -1;
  FChannel := TToken.DEFAULT_CHANNEL;
end;

procedure TCommonTokenStream.SetTokenTypeChannel(const TType, Channel: Integer);
begin
  if (FChannelOverrideMap = nil) then
    FChannelOverrideMap := TDictionary<Integer, Integer>.Create;
  FChannelOverrideMap[TType] := Channel;
end;

function TCommonTokenStream.Size: Integer;
begin
  Result := FTokens.Count;
end;

function TCommonTokenStream.SkipOffTokenChannels(const I: Integer): Integer;
var
  N: Integer;
begin
  Result := I;
  N := FTokens.Count;
  while (Result < N) and (FTokens[Result].Channel <> FChannel) do
    Inc(Result);
end;

function TCommonTokenStream.SkipOffTokenChannelsReverse(
  const I: Integer): Integer;
begin
  Result := I;
  while (Result >= 0) and (FTokens[Result].Channel <> FChannel) do
    Dec(Result);
end;

function TCommonTokenStream.ToString: String;
begin
  if (FP = -1) then
    FillBuffer;
  Result := ToString(0, FTokens.Count - 1);
end;

function TCommonTokenStream.ToString(const Start, Stop: Integer): String;
var
  I, Finish: Integer;
  Buf: TStringBuilder;
  T: IToken;
begin
  if (Start < 0) or (Stop < 0) then
    Result := ''
  else
  begin
    if (FP = -1) then
      FillBuffer;
    if (Stop >= FTokens.Count) then
      Finish := FTokens.Count - 1
    else
      Finish := Stop;
    Buf := TStringBuilder.Create;
    try
      for I := Start to Finish do
      begin
        T := FTokens[I];
        Buf.Append(T.Text);
      end;
      Result := Buf.ToString;
    finally
      Buf.Free;
    end;
  end;
end;

function TCommonTokenStream.ToString(const Start, Stop: IToken): String;
begin
  if Assigned(Start) and Assigned(Stop) then
    Result := ToString(Start.TokenIndex, Stop.TokenIndex)
  else
    Result := '';
end;

constructor TCommonTokenStream.Create(const ATokenSource: ITokenSource;
  const AChannel: Integer);
begin
  Create(ATokenSource);
  FChannel := AChannel;
end;

constructor TCommonTokenStream.Create(const ALexer: ILexer);
begin
  Create(ALexer as ITokenSource);
end;

constructor TCommonTokenStream.Create(const ALexer: ILexer;
  const AChannel: Integer);
begin
  Create(ALexer as ITokenSource, AChannel);
end;

{ TDFA }

function TDFA.Description: String;
begin
  Result := 'n/a';
end;

procedure TDFA.Error(const NVAE: ENoViableAltException);
begin
  // No default implementation
end;

function TDFA.GetRecognizer: IBaseRecognizer;
begin
  Result := IBaseRecognizer(FRecognizer);
end;

function TDFA.GetSpecialStateTransitionHandler: TSpecialStateTransitionHandler;
begin
  Result := FSpecialStateTransitionHandler;
end;

procedure TDFA.NoViableAlt(const S: Integer; const Input: IIntStream);
var
  NVAE: ENoViableAltException;
begin
  if (Recognizer.State.Backtracking > 0) then
    Recognizer.State.Failed := True
  else
  begin
    NVAE := ENoViableAltException.Create(Description, FDecisionNumber, S, Input);
    Error(NVAE);
    raise NVAE;
  end;
end;

function TDFA.Predict(const Input: IIntStream): Integer;
var
  Mark, S, SNext, SpecialState: Integer;
  C: Char;
begin
  Result := 0;
  Mark := Input.Mark; // remember where decision started in input
  S := 0; // we always start at s0
  try
    while True do
    begin
      SpecialState := FSpecial[S];
      if (SpecialState >= 0) then
      begin
        S := FSpecialStateTransitionHandler(Self, SpecialState, Input);
        if (S = -1) then
        begin
          NoViableAlt(S, Input);
          Exit;
        end;
        Input.Consume;
        Continue;
      end;

      if (FAccept[S] >= 1) then
      begin
        Result := FAccept[S];
        Exit;
      end;

      // look for a normal char transition
      C := Char(Input.LA(1)); // -1 == \uFFFF, all tokens fit in 65000 space
      if (C >= FMin[S]) and (C <= FMax[S]) then
      begin
        SNext := FTransition[S,Integer(C) - Integer(FMin[S])];  // move to next state
        if (SNext < 0) then
        begin
          // was in range but not a normal transition
          // must check EOT, which is like the else clause.
          // eot[s]>=0 indicates that an EOT edge goes to another
          // state.
          if (FEOT[S] >= 0) then  // EOT Transition to accept state?
          begin
            S := FEOT[S];
            Input.Consume;
            // TODO: I had this as return accept[eot[s]]
            // which assumed here that the EOT edge always
            // went to an accept...faster to do this, but
            // what about predicated edges coming from EOT
            // target?
            Continue;
          end;

          NoViableAlt(S, Input);
          Exit;
        end;
        S := SNext;
        Input.Consume;
        Continue;
      end;

      if (FEOT[S] >= 0) then
      begin
        // EOT Transition?
        S := FEOT[S];
        Input.Consume;
        Continue;
      end;

      if (C = Char(TToken.EOF)) and (FEOF[S] >= 0) then
      begin
        // EOF Transition to accept state?
        Result := FAccept[FEOF[S]];
        Exit;
      end;

      // not in range and not EOF/EOT, must be invalid symbol
      NoViableAlt(S, Input);
      Exit;
    end;
  finally
    Input.Rewind(Mark);
  end;
end;

procedure TDFA.SetRecognizer(const Value: IBaseRecognizer);
begin
  FRecognizer := Pointer(Value);
end;

procedure TDFA.SetSpecialStateTransitionHandler(
  const Value: TSpecialStateTransitionHandler);
begin
  FSpecialStateTransitionHandler := Value;
end;

function TDFA.SpecialStateTransition(const S: Integer;
  const Input: IIntStream): Integer;
begin
  // No default implementation
  Result := -1;
end;

function TDFA.SpecialTransition(const State, Symbol: Integer): Integer;
begin
  Result := 0;
end;

class function TDFA.UnpackEncodedString(
  const EncodedString: String): TSmallintArray;
var
  I, J, DI, Size: Integer;
  N, V: Char;
begin
  Size := 0;
  I := 1;
  while (I <= Length(EncodedString)) do
  begin
    Inc(Size,Integer(EncodedString[I]));
    Inc(I,2);
  end;

  SetLength(Result,Size);
  DI := 0;
  I := 1;
  while (I <= Length(EncodedString)) do
  begin
    N := EncodedString[I];
    V := EncodedString[I + 1];
    // add v n times to data
    for J := 1 to Integer(N) do
    begin
      Result[DI] := Smallint(V);
      Inc(DI);
    end;
    Inc(I,2);
  end;
end;

class function TDFA.UnpackEncodedStringArray(
  const EncodedStrings: array of String): TSmallintMatrix;
var
  I: Integer;
begin
  SetLength(Result,Length(EncodedStrings));
  for I := 0 to Length(EncodedStrings) - 1 do
    Result[I] := UnpackEncodedString(EncodedStrings[I]);
end;

class function TDFA.UnpackEncodedStringArray(
  const EncodedStrings: TStringArray): TSmallintMatrix;
var
  I: Integer;
begin
  SetLength(Result,Length(EncodedStrings));
  for I := 0 to Length(EncodedStrings) - 1 do
    Result[I] := UnpackEncodedString(EncodedStrings[I]);
end;

class function TDFA.UnpackEncodedStringToUnsignedChars(
  const EncodedString: String): TCharArray;
var
  I, J, DI, Size: Integer;
  N, V: Char;
begin
  Size := 0;
  I := 1;
  while (I <= Length(EncodedString)) do
  begin
    Inc(Size,Integer(EncodedString[I]));
    Inc(I,2);
  end;

  SetLength(Result,Size);
  DI := 0;
  I := 1;
  while (I <= Length(EncodedString)) do
  begin
    N := EncodedString[I];
    V := EncodedString[I + 1];
    // add v n times to data
    for J := 1 to Integer(N) do
    begin
      Result[DI] := V;
      Inc(DI);
    end;
    Inc(I,2);
  end;
end;

{ TLexer }

constructor TLexer.Create;
begin
  inherited;
end;

constructor TLexer.Create(const AInput: ICharStream);
begin
  inherited Create;
  FInput := AInput;
end;

constructor TLexer.Create(const AInput: ICharStream;
  const AState: IRecognizerSharedState);
begin
  inherited Create(AState);
  FInput := AInput;
end;

function TLexer.Emit: IToken;
begin
  Result := TCommonToken.Create(FInput, FState.TokenType, FState.Channel,
    FState.TokenStartCharIndex, GetCharIndex - 1);
  Result.Line := FState.TokenStartLine;
  Result.Text := FState.Text;
  Result.CharPositionInLine := FState.TokenStartCharPositionInLine;
  Emit(Result);
end;

procedure TLexer.Emit(const Token: IToken);
begin
  FState.Token := Token;
end;

function TLexer.GetCharErrorDisplay(const C: Integer): String;
begin
  case C of
    // TToken.EOF
    TOKEN_dot_EOF:
      Result := '<EOF>';
    10:
      Result := '\n';
    9:
      Result := '\t';
    13:
      Result := '\r';
    else
      Result := Char(C);
  end;
  Result := '''' + Result + '''';
end;

function TLexer.GetCharIndex: Integer;
begin
  Result := FInput.Index;
end;

function TLexer.GetCharPositionInLine: Integer;
begin
  Result := FInput.CharPositionInLine;
end;

function TLexer.GetCharStream: ICharStream;
begin
  Result := FInput;
end;

function TLexer.GetErrorMessage(const E: ERecognitionException;
  const TokenNames: TStringArray): String;
var
  MTE: EMismatchedTokenException absolute E;
  NVAE: ENoViableAltException absolute E;
  EEE: EEarlyExitException absolute E;
  MNSE: EMismatchedNotSetException absolute E;
  MSE: EMismatchedSetException absolute E;
  MRE: EMismatchedRangeException absolute E;
begin
  if (E is EMismatchedTokenException) then
    Result := 'mismatched character ' + GetCharErrorDisplay(E.Character)
      + ' expecting ' + GetCharErrorDisplay(MTE.Expecting)
  else
    if (E is ENoViableAltException) then
      // for development, can add "decision=<<"+nvae.grammarDecisionDescription+">>"
      // and "(decision="+nvae.decisionNumber+") and
      // "state "+nvae.stateNumber
      Result := 'no viable alternative at character ' + GetCharErrorDisplay(NVAE.Character)
    else
      if (E is EEarlyExitException) then
        // for development, can add "(decision="+eee.decisionNumber+")"
        Result := 'required (...)+ loop did not match anything at character '
          + GetCharErrorDisplay(EEE.Character)
      else
        if (E is EMismatchedNotSetException) then
          Result := 'mismatched character ' + GetCharErrorDisplay(MNSE.Character)
            + ' expecting set ' + MNSE.Expecting.ToString
        else
          if (E is EMismatchedSetException) then
            Result := 'mismatched character ' + GetCharErrorDisplay(MSE.Character)
              + ' expecting set ' + MSE.Expecting.ToString
          else
            if (E is EMismatchedRangeException) then
              Result := 'mismatched character ' + GetCharErrorDisplay(MRE.Character)
                + ' expecting set ' + GetCharErrorDisplay(MRE.A) + '..'
                + GetCharErrorDisplay(MRE.B)
            else
              Result := inherited GetErrorMessage(E, TokenNames);
end;

function TLexer.GetInput: IIntStream;
begin
  Result := FInput;
end;

function TLexer.GetLine: Integer;
begin
  Result := FInput.Line;
end;

function TLexer.GetSourceName: String;
begin
  Result := FInput.SourceName;
end;

function TLexer.GetText: String;
begin
  if (FState.Text <> '') then
    Result := FState.Text
  else
    Result := FInput.Substring(FState.TokenStartCharIndex, GetCharIndex - 1)
end;

procedure TLexer.Match(const S: String);
var
  I: Integer;
  MTE: EMismatchedTokenException;
begin
  for I := 1 to Length(S) do
  begin
    if (FInput.LA(1) <> Integer(S[I])) then
    begin
      if (FState.Backtracking > 0) then
      begin
        FState.Failed := True;
        Exit;
      end;
      MTE := EMismatchedTokenException.Create(Integer(S[I]), FInput);
      Recover(MTE); // don't really recover; just consume in lexer
      raise MTE;
    end;
    FInput.Consume;
    FState.Failed := False;
  end;
end;

procedure TLexer.Match(const C: Integer);
var
  MTE: EMismatchedTokenException;
begin
  if (FInput.LA(1) <> C) then
  begin
    if (FState.Backtracking > 0) then
    begin
      FState.Failed := True;
      Exit;
    end;
    MTE := EMismatchedTokenException.Create(C, FInput);
    Recover(MTE);
    raise MTE;
  end;
  FInput.Consume;
  FState.Failed := False;
end;

procedure TLexer.MatchAny;
begin
  FInput.Consume;
end;

procedure TLexer.MatchRange(const A, B: Integer);
var
  MRE: EMismatchedRangeException;
begin
  if (FInput.LA(1) < A) or (FInput.LA(1) > B) then
  begin
    if (FState.Backtracking > 0) then
    begin
      FState.Failed := True;
      Exit;
    end;
    MRE := EMismatchedRangeException.Create(A, B, FInput);
    Recover(MRE);
    raise MRE;
  end;
  FInput.Consume;
  FState.Failed := False;
end;

function TLexer.NextToken: IToken;
begin
  while True do
  begin
    FState.Token := nil;
    FState.Channel := TToken.DEFAULT_CHANNEL;
    FState.TokenStartCharIndex := FInput.Index;
    FState.TokenStartCharPositionInLine := FInput.CharPositionInLine;
    FState.TokenStartLine := Finput.Line;
    FState.Text := '';
    if (FInput.LA(1) = Integer(cscEOF)) then
    begin
      Result := TToken.EOF_TOKEN;
      Exit;
    end;

    try
      DoTokens;
      if (FState.Token = nil) then
        Emit
      else
        if (FState.Token = TToken.SKIP_TOKEN) then
          Continue;
      Exit(FState.Token);
    except
      on NVA: ENoViableAltException do
      begin
        ReportError(NVA);
        Recover(NVA);  // throw out current char and try again
      end;

      on RE: ERecognitionException do
      begin
        ReportError(RE);
        // Match() routine has already called Recover()
      end;
    end;
  end;
end;

procedure TLexer.Recover(const RE: ERecognitionException);
begin
  FInput.Consume;
end;

procedure TLexer.ReportError(const E: ERecognitionException);
begin
  DisplayRecognitionError(GetTokenNames, E);
end;

procedure TLexer.Reset;
begin
  inherited; // reset all recognizer state variables
  // wack Lexer state variables
  if Assigned(FInput) then
    FInput.Seek(0);  // rewind the input
  if (FState = nil) then
    Exit;  // no shared state work to do
  FState.Token := nil;
  FState.TokenType := TToken.INVALID_TOKEN_TYPE;
  FState.Channel := TToken.DEFAULT_CHANNEL;
  FState.TokenStartCharIndex := -1;
  FState.TokenStartCharPositionInLine := -1;
  FState.TokenStartLine := -1;
  FState.Text := '';
end;

procedure TLexer.SetCharStream(const Value: ICharStream);
begin
  FInput := nil;
  Reset;
  FInput := Value;
end;

procedure TLexer.SetText(const Value: String);
begin
  FState.Text := Value;
end;

procedure TLexer.Skip;
begin
  FState.Token := TToken.SKIP_TOKEN;
end;

procedure TLexer.TraceIn(const RuleName: String; const RuleIndex: Integer);
var
  InputSymbol: String;
begin
  InputSymbol := Char(FInput.LT(1)) + ' line=' + IntToStr(GetLine) + ':'
    + IntToStr(GetCharPositionInLine);
  inherited TraceIn(RuleName, RuleIndex, InputSymbol);
end;

procedure TLexer.TraceOut(const RuleName: String; const RuleIndex: Integer);
var
  InputSymbol: String;
begin
  InputSymbol := Char(FInput.LT(1)) + ' line=' + IntToStr(GetLine) + ':'
    + IntToStr(GetCharPositionInLine);
  inherited TraceOut(RuleName, RuleIndex, InputSymbol);
end;

{ TParser }

constructor TParser.Create(const AInput: ITokenStream);
begin
  inherited Create; // highlight that we go to base class to set state object
  SetTokenStream(AInput);
end;

constructor TParser.Create(const AInput: ITokenStream;
  const AState: IRecognizerSharedState);
begin
  inherited Create(AState); // share the state object with another parser
  SetTokenStream(AInput);
end;

function TParser.GetCurrentInputSymbol(
  const Input: IIntStream): IANTLRInterface;
begin
  Result := FInput.LT(1)
end;

function TParser.GetInput: IIntStream;
begin
  Result := FInput;
end;

function TParser.GetMissingSymbol(const Input: IIntStream;
  const E: ERecognitionException; const ExpectedTokenType: Integer;
  const Follow: IBitSet): IANTLRInterface;
var
  TokenText: String;
  T: ICommonToken;
  Current: IToken;
begin
  if (ExpectedTokenType = TToken.EOF) then
    TokenText := '<missing EOF>'
  else
    TokenText := '<missing ' + GetTokenNames[ExpectedTokenType] + '>';
  T := TCommonToken.Create(ExpectedTokenType, TokenText);
  Current := FInput.LT(1);
  if (Current.TokenType = TToken.EOF) then
    Current := FInput.LT(-1);
  T.Line := Current.Line;
  T.CharPositionInLine := Current.CharPositionInLine;
  T.Channel := DEFAULT_TOKEN_CHANNEL;
  Result := T;
end;

function TParser.GetSourceName: String;
begin
  Result := FInput.SourceName;
end;

function TParser.GetTokenStream: ITokenStream;
begin
  Result := FInput;
end;

procedure TParser.Reset;
begin
  inherited; // reset all recognizer state variables
  if Assigned(FInput) then
    FInput.Seek(0); // rewind the input
end;

procedure TParser.SetTokenStream(const Value: ITokenStream);
begin
  FInput := nil;
  Reset;
  FInput := Value;
end;

procedure TParser.TraceIn(const RuleName: String; const RuleIndex: Integer);
begin
  inherited TraceIn(RuleName, RuleIndex, FInput.LT(1).ToString);
end;

procedure TParser.TraceOut(const RuleName: String; const RuleIndex: Integer);
begin
  inherited TraceOut(RuleName, RuleIndex, FInput.LT(1).ToString);
end;

{ TRuleReturnScope }

function TRuleReturnScope.GetStart: IANTLRInterface;
begin
  Result := nil;
end;

function TRuleReturnScope.GetStop: IANTLRInterface;
begin
  Result := nil;
end;

function TRuleReturnScope.GetTemplate: IANTLRInterface;
begin
  Result := nil;
end;

function TRuleReturnScope.GetTree: IANTLRInterface;
begin
  Result := nil;
end;

procedure TRuleReturnScope.SetStart(const Value: IANTLRInterface);
begin
  raise EInvalidOperation.Create('Setter has not been defined for this property.');
end;

procedure TRuleReturnScope.SetStop(const Value: IANTLRInterface);
begin
  raise EInvalidOperation.Create('Setter has not been defined for this property.');
end;

procedure TRuleReturnScope.SetTree(const Value: IANTLRInterface);
begin
  raise EInvalidOperation.Create('Setter has not been defined for this property.');
end;

{ TParserRuleReturnScope }

function TParserRuleReturnScope.GetStart: IANTLRInterface;
begin
  Result := FStart;
end;

function TParserRuleReturnScope.GetStop: IANTLRInterface;
begin
  Result := FStop;
end;

procedure TParserRuleReturnScope.SetStart(const Value: IANTLRInterface);
begin
  FStart := Value as IToken;
end;

procedure TParserRuleReturnScope.SetStop(const Value: IANTLRInterface);
begin
  FStop := Value as IToken;
end;

{ TTokenRewriteStream }

procedure TTokenRewriteStream.Delete(const Start, Stop: IToken);
begin
  Delete(DEFAULT_PROGRAM_NAME, Start, Stop);
end;

procedure TTokenRewriteStream.Delete(const IndexT: IToken);
begin
  Delete(DEFAULT_PROGRAM_NAME, IndexT, IndexT);
end;

constructor TTokenRewriteStream.Create;
begin
  inherited;
  Init;
end;

constructor TTokenRewriteStream.Create(const ATokenSource: ITokenSource);
begin
  inherited Create(ATokenSource);
  Init;
end;

constructor TTokenRewriteStream.Create(const ALexer: ILexer);
begin
  Create(ALexer as ITokenSource);
end;

constructor TTokenRewriteStream.Create(const ALexer: ILexer;
  const AChannel: Integer);
begin
  Create(ALexer as ITokenSource, AChannel);
end;

function TTokenRewriteStream.CatOpText(const A, B: IANTLRInterface): IANTLRInterface;
var
  X, Y: String;
begin
  if Assigned(A) then
    X := A.ToString
  else
    X := '';

  if Assigned(B) then
    Y := B.ToString
  else
    Y := '';

  Result := TANTLRString.Create(X + Y);
end;

constructor TTokenRewriteStream.Create(const ATokenSource: ITokenSource;
  const AChannel: Integer);
begin
  inherited Create(ATokenSource, AChannel);
  Init;
end;

procedure TTokenRewriteStream.Delete(const ProgramName: String; const Start,
  Stop: IToken);
begin
  Replace(ProgramName, Start, Stop, nil);
end;

procedure TTokenRewriteStream.Delete(const ProgramName: String; const Start,
  Stop: Integer);
begin
  Replace(ProgramName, Start, Stop, nil);
end;

procedure TTokenRewriteStream.Delete(const Start, Stop: Integer);
begin
  Delete(DEFAULT_PROGRAM_NAME, Start, Stop);
end;

procedure TTokenRewriteStream.Delete(const Index: Integer);
begin
  Delete(DEFAULT_PROGRAM_NAME, Index, Index);
end;

procedure TTokenRewriteStream.DeleteProgram(const ProgramName: String);
begin
  Rollback(ProgramName, MIN_TOKEN_INDEX);
end;

procedure TTokenRewriteStream.DeleteProgram;
begin
  DeleteProgram(DEFAULT_PROGRAM_NAME);
end;

function TTokenRewriteStream.GetLastRewriteTokenIndex: Integer;
begin
  Result := GetLastRewriteTokenIndex(DEFAULT_PROGRAM_NAME);
end;

function TTokenRewriteStream.GetKindOfOps(
  const Rewrites: IList<IRewriteOperation>;
  const Kind: TGUID): IList<IRewriteOperation>;
begin
  Result := GetKindOfOps(Rewrites, Kind, Rewrites.Count);
end;

function TTokenRewriteStream.GetKindOfOps(
  const Rewrites: IList<IRewriteOperation>; const Kind: TGUID;
  const Before: Integer): IList<IRewriteOperation>;
var
  I: Integer;
  Op: IRewriteOperation;
  Obj: IInterface;
begin
  Result := TList<IRewriteOperation>.Create;
  I := 0;
  while (I < Before) and (I < Rewrites.Count) do
  begin
    Op := Rewrites[I];
    if Assigned(Op) and (Op.QueryInterface(Kind, Obj) = 0) then
      Result.Add(Op);
    Inc(I);
  end;
end;

function TTokenRewriteStream.GetLastRewriteTokenIndex(
  const ProgramName: String): Integer;
begin
  if (not FLastRewriteTokenIndexes.TryGetValue(ProgramName, Result)) then
    Result := -1;
end;

function TTokenRewriteStream.GetProgram(
  const Name: String): IList<IRewriteOperation>;
var
  InstructionStream: IList<IRewriteOperation>;
begin
  InstructionStream := FPrograms[Name];
  if (InstructionStream = nil) then
    InstructionStream := InitializeProgram(Name);
  Result := InstructionStream;
end;

procedure TTokenRewriteStream.InsertAfter(const ProgramName: String;
  const T: IToken; const Text: IANTLRInterface);
begin
  InsertAfter(ProgramName, T.TokenIndex, Text);
end;

procedure TTokenRewriteStream.Init;
var
  List: IList<IRewriteOperation>;
begin
  FPrograms := TDictionary<String, IList<IRewriteOperation>>.Create;
  List := TList<IRewriteOperation>.Create;
  List.Capacity := PROGRAM_INIT_SIZE;
  FPrograms.Add(DEFAULT_PROGRAM_NAME, List);
  FLastRewriteTokenIndexes := TDictionary<String, Integer>.Create;
end;

function TTokenRewriteStream.InitializeProgram(
  const Name: String): IList<IRewriteOperation>;
begin
  Result := TList<IRewriteOperation>.Create;
  Result.Capacity := PROGRAM_INIT_SIZE;
  FPrograms[Name] := Result;
end;

procedure TTokenRewriteStream.InsertAfter(const ProgramName: String;
  const Index: Integer; const Text: IANTLRInterface);
begin
  // to insert after, just insert before next index (even if past end)
  InsertBefore(ProgramName, Index + 1, Text);
end;

procedure TTokenRewriteStream.InsertAfter(const T: IToken;
  const Text: IANTLRInterface);
begin
  InsertAfter(DEFAULT_PROGRAM_NAME, T, Text);
end;

procedure TTokenRewriteStream.InsertAfter(const Index: Integer;
  const Text: IANTLRInterface);
begin
  InsertAfter(DEFAULT_PROGRAM_NAME, Index, Text);
end;

procedure TTokenRewriteStream.InsertBefore(const Index: Integer;
  const Text: IANTLRInterface);
begin
  InsertBefore(DEFAULT_PROGRAM_NAME, Index, Text);
end;

procedure TTokenRewriteStream.InsertBefore(const ProgramName: String;
  const T: IToken; const Text: IANTLRInterface);
begin
  InsertBefore(ProgramName, T.TokenIndex, Text);
end;

procedure TTokenRewriteStream.InsertBefore(const ProgramName: String;
  const Index: Integer; const Text: IANTLRInterface);
var
  Op: IRewriteOperation;
begin
  Op := TInsertBeforeOp.Create(Index, Text, Self);
  GetProgram(ProgramName).Add(Op);
end;

procedure TTokenRewriteStream.InsertBefore(const T: IToken;
  const Text: IANTLRInterface);
begin
  InsertBefore(DEFAULT_PROGRAM_NAME, T, Text);
end;

procedure TTokenRewriteStream.Replace(const Start, Stop: IToken;
  const Text: IANTLRInterface);
begin
  Replace(DEFAULT_PROGRAM_NAME, Stop, Stop, Text);
end;

procedure TTokenRewriteStream.Replace(const IndexT: IToken;
  const Text: IANTLRInterface);
begin
  Replace(DEFAULT_PROGRAM_NAME, IndexT, IndexT, Text);
end;

procedure TTokenRewriteStream.Replace(const ProgramName: String; const Start,
  Stop: Integer; const Text: IANTLRInterface);
var
  Op: IRewriteOperation;
  Rewrites: IList<IRewriteOperation>;
begin
  if (Start > Stop) or (Start < 0) or (Stop < 0) or (Stop >= GetTokens.Count) then
    raise EArgumentOutOfRangeException.Create('replace: range invalid: '
      + IntToStr(Start) + '..' + IntToStr(Stop) + '(size='
      + IntToStr(GetTokens.Count) + ')');

  Op := TReplaceOp.Create(Start, Stop, Text, Self);
  Rewrites := GetProgram(ProgramName);
  Op.InstructionIndex := Rewrites.Count;
  Rewrites.Add(Op);
end;

function TTokenRewriteStream.ReduceToSingleOperationPerIndex(
  const Rewrites: IList<IRewriteOperation>): IDictionary<Integer, IRewriteOperation>;
var
  I, J: Integer;
  Op: IRewriteOperation;
  ROp, PrevROp: IReplaceOp;
  IOp, PrevIOp: IInsertBeforeOp;
  Inserts, PrevInserts, PrevReplaces: IList<IRewriteOperation>;
  Disjoint, Same: Boolean;
begin
  // WALK REPLACES
  for I := 0 to Rewrites.Count - 1 do
  begin
    Op := Rewrites[I];
    if (Op = nil) then
      Continue;
    if (not Supports(Op, IReplaceOp, ROp)) then
      Continue;

    // Wipe prior inserts within range
    Inserts := GetKindOfOps(Rewrites, IInsertBeforeOp, I);
    for J := 0 to Inserts.Count - 1 do
    begin
      IOp := Inserts[J] as IInsertBeforeOp;
      if (IOp.Index >= ROp.Index) and (IOp.Index <= ROp.LastIndex) then
      begin
        // delete insert as it's a no-op.
        Rewrites[IOp.InstructionIndex] := nil;
      end;
    end;

    // Drop any prior replaces contained within
    PrevReplaces := GetKindOfOps(Rewrites, IReplaceOp, I);
    for J := 0 to PrevReplaces.Count - 1 do
    begin
      PrevROp := PrevReplaces[J] as IReplaceOp;
      if (PrevROp.Index >= ROp.Index) and (PrevROp.LastIndex <= ROp.LastIndex) then
      begin
        // delete replace as it's a no-op.
        Rewrites[PrevROp.InstructionIndex] := nil;
        Continue;
      end;
      // throw exception unless disjoint or identical
      Disjoint := (PrevROp.LastIndex < ROp.Index) or (PrevROp.Index > ROp.LastIndex);
      Same := (PrevROp.Index = ROp.Index) and (PrevROp.LastIndex = ROp.LastIndex);
      if (not Disjoint) and (not Same) then
        raise EArgumentOutOfRangeException.Create('replace of boundaries of '
          + ROp.ToString + ' overlap with previous ' + PrevROp.ToString);
    end;
  end;

  // WALK INSERTS
  for I := 0 to Rewrites.Count - 1 do
  begin
    Op := Rewrites[I];
    if (Op = nil) then
      Continue;
    if (not Supports(Op, IInsertBeforeOp, IOp)) then
      Continue;

    // combine current insert with prior if any at same index
    PrevInserts := GetKindOfOps(Rewrites, IInsertBeforeOp, I);
    for J := 0 to PrevInserts.Count - 1 do
    begin
      PrevIOp := PrevInserts[J] as IInsertBeforeOp;
      if (PrevIOp.Index = IOp.Index) then
      begin
        // combine objects
        // convert to strings...we're in process of toString'ing
        // whole token buffer so no lazy eval issue with any templates
        IOp.Text := CatOpText(IOp.Text, PrevIOp.Text);
        // delete redundant prior insert
        Rewrites[PrevIOp.InstructionIndex] := nil;
      end;
    end;

    // look for replaces where iop.index is in range; error
    PrevReplaces := GetKindOfOps(Rewrites, IReplaceOp, I);
    for J := 0 to PrevReplaces.Count - 1 do
    begin
      Rop := PrevReplaces[J] as IReplaceOp;
      if (IOp.Index = ROp.Index) then
      begin
        ROp.Text := CatOpText(IOp.Text, ROp.Text);
        Rewrites[I] := nil;  // delete current insert
        Continue;
      end;
      if (IOp.Index >= ROp.Index) and (IOp.Index <= ROp.LastIndex) then
        raise EArgumentOutOfRangeException.Create('insert op '
          + IOp.ToString + ' within boundaries of previous ' + ROp.ToString);
    end;
  end;

  Result := TDictionary<Integer, IRewriteOperation>.Create;
  for Op in Rewrites do
  begin
    if (Op = nil) then
      Continue; // ignore deleted ops
    if (Result.ContainsKey(Op.Index)) then
      raise Exception.Create('should only be one op per index');
    Result.Add(Op.Index, Op);
  end;
end;

procedure TTokenRewriteStream.Replace(const ProgramName: String; const Start,
  Stop: IToken; const Text: IANTLRInterface);
begin
  Replace(ProgramName, Start.TokenIndex, Stop.TokenIndex, Text);
end;

procedure TTokenRewriteStream.Replace(const Index: Integer;
  const Text: IANTLRInterface);
begin
  Replace(DEFAULT_PROGRAM_NAME, Index, Index, Text);
end;

procedure TTokenRewriteStream.Replace(const Start, Stop: Integer;
  const Text: IANTLRInterface);
begin
  Replace(DEFAULT_PROGRAM_NAME, Start, Stop, Text);
end;

procedure TTokenRewriteStream.Rollback(const InstructionIndex: Integer);
begin
  Rollback(DEFAULT_PROGRAM_NAME, InstructionIndex);
end;

procedure TTokenRewriteStream.Rollback(const ProgramName: String;
  const InstructionIndex: Integer);
var
  InstructionStream: IList<IRewriteOperation>;
begin
  InstructionStream := FPrograms[ProgramName];
  if Assigned(InstructionStream) then
    FPrograms[ProgramName] := InstructionStream.GetRange(MIN_TOKEN_INDEX,
      InstructionIndex - MIN_TOKEN_INDEX);
end;

procedure TTokenRewriteStream.SetLastRewriteTokenIndex(
  const ProgramName: String; const I: Integer);
begin
  FLastRewriteTokenIndexes[ProgramName] := I;
end;

function TTokenRewriteStream.ToDebugString: String;
begin
  Result := ToDebugString(MIN_TOKEN_INDEX, Size - 1);
end;

function TTokenRewriteStream.ToDebugString(const Start, Stop: Integer): String;
var
  Buf: TStringBuilder;
  I: Integer;
begin
  Buf := TStringBuilder.Create;
  try
    if (Start >= MIN_TOKEN_INDEX) then
      for I := Start to Min(Stop,GetTokens.Count - 1) do
        Buf.Append(Get(I).ToString);
  finally
    Buf.Free;
  end;
end;

function TTokenRewriteStream.ToOriginalString: String;
begin
  Result := ToOriginalString(MIN_TOKEN_INDEX, Size - 1);
end;

function TTokenRewriteStream.ToOriginalString(const Start,
  Stop: Integer): String;
var
  Buf: TStringBuilder;
  I: Integer;
begin
  Buf := TStringBuilder.Create;
  try
    if (Start >= MIN_TOKEN_INDEX) then
      for I := Start to Min(Stop, GetTokens.Count - 1) do
        Buf.Append(Get(I).Text);
    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

function TTokenRewriteStream.ToString: String;
begin
  Result := ToString(MIN_TOKEN_INDEX, Size - 1);
end;

function TTokenRewriteStream.ToString(const ProgramName: String): String;
begin
  Result := ToString(ProgramName, MIN_TOKEN_INDEX, Size - 1);
end;

function TTokenRewriteStream.ToString(const ProgramName: String; const Start,
  Stop: Integer): String;
var
  Rewrites: IList<IRewriteOperation>;
  I, StartIndex, StopIndex: Integer;
  IndexToOp: IDictionary<Integer, IRewriteOperation>;
  Buf: TStringBuilder;
  Tokens: IList<IToken>;
  T: IToken;
  Op: IRewriteOperation;
  Pair: TPair<Integer, IRewriteOperation>;
begin
  Rewrites := FPrograms[ProgramName];
  Tokens := GetTokens;
  // ensure start/end are in range
  StopIndex := Min(Stop,Tokens.Count - 1);
  StartIndex := Max(Start,0);

  if (Rewrites = nil) or (Rewrites.Count = 0) then
  begin
     // no instructions to execute
    Result := ToOriginalString(StartIndex, StopIndex);
    Exit;
  end;

  Buf := TStringBuilder.Create;
  try
    // First, optimize instruction stream
    IndexToOp := ReduceToSingleOperationPerIndex(Rewrites);

    // Walk buffer, executing instructions and emitting tokens
    I := StartIndex;
    while (I <= StopIndex) and (I < Tokens.Count) do
    begin
      if (not IndexToOp.TryGetValue(I, Op)) then
        Op := nil;
      IndexToOp.Remove(I); // remove so any left have index size-1
      T := Tokens[I];
      if (Op = nil) then
      begin
        // no operation at that index, just dump token
        Buf.Append(T.Text);
        Inc(I); // move to next token
      end
      else
        I := Op.Execute(Buf); // execute operation and skip
    end;

    // include stuff after end if it's last index in buffer
    // So, if they did an insertAfter(lastValidIndex, "foo"), include
    // foo if end==lastValidIndex.
    if (StopIndex = Tokens.Count - 1) then
    begin
      // Scan any remaining operations after last token
      // should be included (they will be inserts).
      for Pair in IndexToOp do
      begin
        if (Pair.Value.Index >= Tokens.Count - 1) then
          Buf.Append(Pair.Value.Text.ToString);
      end;
    end;
    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

function TTokenRewriteStream.ToString(const Start, Stop: Integer): String;
begin
  Result := ToString(DEFAULT_PROGRAM_NAME, Start, Stop);
end;

procedure TTokenRewriteStream.InsertBefore(const Index: Integer;
  const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  InsertBefore(Index, S);
end;

procedure TTokenRewriteStream.InsertBefore(const T: IToken; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  InsertBefore(T, S);
end;

procedure TTokenRewriteStream.InsertBefore(const ProgramName: String;
  const Index: Integer; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  InsertBefore(ProgramName, Index, S);
end;

procedure TTokenRewriteStream.InsertBefore(const ProgramName: String;
  const T: IToken; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  InsertBefore(ProgramName, T, S);
end;

procedure TTokenRewriteStream.InsertAfter(const Index: Integer;
  const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  InsertAfter(Index,S);
end;

procedure TTokenRewriteStream.InsertAfter(const T: IToken; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  InsertAfter(T,S);
end;

procedure TTokenRewriteStream.InsertAfter(const ProgramName: String;
  const Index: Integer; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  InsertAfter(ProgramName,Index,S);
end;

procedure TTokenRewriteStream.InsertAfter(const ProgramName: String;
  const T: IToken; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  InsertAfter(ProgramName,T,S);
end;

procedure TTokenRewriteStream.Replace(const IndexT: IToken; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  Replace(IndexT, S);
end;

procedure TTokenRewriteStream.Replace(const Start, Stop: Integer;
  const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  Replace(Start, Stop, S);
end;

procedure TTokenRewriteStream.Replace(const Index: Integer; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  Replace(Index, S);
end;

procedure TTokenRewriteStream.Replace(const ProgramName: String; const Start,
  Stop: IToken; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  Replace(ProgramName, Start, Stop, S);
end;

procedure TTokenRewriteStream.Replace(const ProgramName: String; const Start,
  Stop: Integer; const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  Replace(ProgramName, Start, Stop, S);
end;

procedure TTokenRewriteStream.Replace(const Start, Stop: IToken;
  const Text: String);
var
  S: IANTLRString;
begin
  S := TANTLRString.Create(Text);
  Replace(Start, Stop, S);
end;

{ TTokenRewriteStream.TRewriteOperation }

constructor TTokenRewriteStream.TRewriteOperation.Create(const AIndex: Integer;
  const AText: IANTLRInterface; const AParent: ITokenRewriteStream);
begin
  inherited Create;
  FIndex := AIndex;
  FText := AText;
  FParent := Pointer(AParent);
end;

function TTokenRewriteStream.TRewriteOperation.Execute(
  const Buf: TStringBuilder): Integer;
begin
  Result := FIndex;
end;

function TTokenRewriteStream.TRewriteOperation.GetIndex: Integer;
begin
  Result := FIndex;
end;

function TTokenRewriteStream.TRewriteOperation.GetInstructionIndex: Integer;
begin
  Result := FInstructionIndex;
end;

function TTokenRewriteStream.TRewriteOperation.GetParent: ITokenRewriteStream;
begin
  Result := ITokenRewriteStream(FParent);
end;

function TTokenRewriteStream.TRewriteOperation.GetText: IANTLRInterface;
begin
  Result := FText;
end;

procedure TTokenRewriteStream.TRewriteOperation.SetIndex(const Value: Integer);
begin
  FIndex := Value;
end;

procedure TTokenRewriteStream.TRewriteOperation.SetInstructionIndex(
  const Value: Integer);
begin
  FInstructionIndex := Value;
end;

procedure TTokenRewriteStream.TRewriteOperation.SetParent(
  const Value: ITokenRewriteStream);
begin
  FParent := Pointer(Value);
end;

procedure TTokenRewriteStream.TRewriteOperation.SetText(
  const Value: IANTLRInterface);
begin
  FText := Value;
end;

function TTokenRewriteStream.TRewriteOperation.ToString: String;
var
  OpName: String;
  DollarIndex: Integer;
begin
  OpName := ClassName;
  DollarIndex := Pos('$',OpName) - 1; // Delphi strings are 1-based
  if (DollarIndex >= 0) then
    OpName := Copy(OpName,DollarIndex + 1,Length(OpName) - (DollarIndex + 1));
  Result := '<' + OpName + '@' + IntToStr(FIndex) + ':"' + FText.ToString + '">';
end;

{ TTokenRewriteStream.TRewriteOpComparer<T> }

function TTokenRewriteStream.TRewriteOpComparer<T>.Compare(const Left,
  Right: T): Integer;
begin
  if (Left.GetIndex < Right.GetIndex) then
    Result := -1
  else
    if (Left.GetIndex > Right.GetIndex) then
      Result := 1
    else
      Result := 0;
end;

{ TTokenRewriteStream.TInsertBeforeOp }

function TTokenRewriteStream.TInsertBeforeOp.Execute(
  const Buf: TStringBuilder): Integer;
begin
  Buf.Append(Text.ToString);
  Buf.Append(Parent.Get(Index).Text);
  Result := Index + 1;
end;

{ TTokenRewriteStream.TReplaceOp }

constructor TTokenRewriteStream.TReplaceOp.Create(const AStart, AStop: Integer;
  const AText: IANTLRInterface; const AParent: ITokenRewriteStream);
begin
  inherited Create(AStart, AText, AParent);
  FLastIndex := AStop;
end;

function TTokenRewriteStream.TReplaceOp.Execute(
  const Buf: TStringBuilder): Integer;
begin
  if (Text <> nil) then
    Buf.Append(Text.ToString);
  Result := FLastIndex + 1;
end;

function TTokenRewriteStream.TReplaceOp.GetLastIndex: Integer;
begin
  Result := FLastIndex;
end;

procedure TTokenRewriteStream.TReplaceOp.SetLastIndex(const Value: Integer);
begin
  FLastIndex := Value;
end;

function TTokenRewriteStream.TReplaceOp.ToString: String;
begin
  Result := '<ReplaceOp@' + IntToStr(Index) + '..' + IntToStr(FLastIndex)
    + ':"' + Text.ToString + '">';
end;

{ TTokenRewriteStream.TDeleteOp }

function TTokenRewriteStream.TDeleteOp.ToString: String;
begin
  Result := '<DeleteOp@' + IntToStr(Index) + '..' + IntToStr(FLastIndex) + '>';
end;

{ Utilities }

var
  EmptyToken: IToken = nil;
  EmptyRuleReturnScope: IRuleReturnScope = nil;

function Def(const X: IToken): IToken; overload;
begin
  if Assigned(X) then
    Result := X
  else
  begin
    if (EmptyToken = nil) then
      EmptyToken := TCommonToken.Create;
    Result := EmptyToken;
  end;
end;

function Def(const X: IRuleReturnScope): IRuleReturnScope;
begin
  if Assigned(X) then
    Result := X
  else
  begin
    if (EmptyRuleReturnScope = nil) then
      EmptyRuleReturnScope := TRuleReturnScope.Create;
    Result := EmptyRuleReturnScope;
  end;
end;

initialization
  TToken.Initialize;

end.
