unit Antlr.Runtime.Tests;

interface

uses
  Classes,
  SysUtils,
  TestFramework,
  Antlr.Runtime;

type
  // Test methods for class IANTLRStringStream
  TestANTLRStringStream = class(TTestCase)
  strict private
    const
      NL = #13#10;
      GRAMMARSTR = ''
        + 'parser grammar p;' + NL
        + 'prog : WHILE ID LCURLY (assign)* RCURLY EOF;' + NL
        + 'assign : ID ASSIGN expr SEMI ;' + NL
        + 'expr : INT | FLOAT | ID ;' + NL;
  public
    procedure SetUp; override;
    procedure TearDown; override;
  published
    procedure TestSizeOnEmptyANTLRStringStream;
    procedure TestSizeOnANTLRStringStream;
    procedure TestConsumeOnANTLRStringStream;
    procedure TestResetOnANTLRStringStream;
    procedure TestSubstringOnANTLRStringStream;
  end;

implementation

{ TestANTLRStringStream }

procedure TestANTLRStringStream.SetUp;
begin
end;

procedure TestANTLRStringStream.TearDown;
begin
end;

procedure TestANTLRStringStream.TestConsumeOnANTLRStringStream;
var
  Stream: IANTLRStringStream;
begin
  Stream := TANTLRStringStream.Create('One'#13#10'Two');
  CheckEquals(0, Stream.Index);
  CheckEquals(0, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);

  Stream.Consume; // O
  CheckEquals(1, Stream.Index);
  CheckEquals(1, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);

  Stream.Consume; // n
  CheckEquals(2, Stream.Index);
  CheckEquals(2, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);

  Stream.Consume; // e
  CheckEquals(3, Stream.Index);
  CheckEquals(3, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);

  Stream.Consume; // #13
  CheckEquals(4, Stream.Index);
  CheckEquals(4, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);

  Stream.Consume; // #10
  CheckEquals(5, Stream.Index);
  CheckEquals(0, Stream.CharPositionInLine);
  CheckEquals(2, Stream.Line);

  Stream.Consume; // T
  CheckEquals(6, Stream.Index);
  CheckEquals(1, Stream.CharPositionInLine);
  CheckEquals(2, Stream.Line);

  Stream.Consume; // w
  CheckEquals(7, Stream.Index);
  CheckEquals(2, Stream.CharPositionInLine);
  CheckEquals(2, Stream.Line);

  Stream.Consume; // o
  CheckEquals(8, Stream.Index);
  CheckEquals(3, Stream.CharPositionInLine);
  CheckEquals(2, Stream.Line);

  Stream.Consume; // EOF
  CheckEquals(8, Stream.Index);
  CheckEquals(3, Stream.CharPositionInLine);
  CheckEquals(2, Stream.Line);

  Stream.Consume; // EOF
  CheckEquals(8, Stream.Index);
  CheckEquals(3, Stream.CharPositionInLine);
  CheckEquals(2, Stream.Line);
end;

procedure TestANTLRStringStream.TestResetOnANTLRStringStream;
var
  Stream: IANTLRStringStream;
begin
  Stream := TANTLRStringStream.Create('One'#13#10'Two');
  CheckEquals(0, Stream.Index);
  CheckEquals(0, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);

  Stream.Consume; // O
  Stream.Consume; // n

  CheckEquals(Ord('e'), Stream.LA(1));
  CheckEquals(2, Stream.Index);

  Stream.Reset;
  CheckEquals(Ord('O'), Stream.LA(1));
  CheckEquals(0, Stream.Index);
  CheckEquals(0, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);
  Stream.Consume; // O

  CheckEquals(Ord('n'), Stream.LA(1));
  CheckEquals(1, Stream.Index);
  CheckEquals(1, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);
  Stream.Consume; // n

  CheckEquals(Ord('e'), Stream.LA(1));
  CheckEquals(2, Stream.Index);
  CheckEquals(2, Stream.CharPositionInLine);
  CheckEquals(1, Stream.Line);
  Stream.Consume; // n
end;

procedure TestANTLRStringStream.TestSizeOnANTLRStringStream;
var
  S1, S2, S3: IANTLRStringStream;
begin
  S1 := TANTLRStringStream.Create('lexer'#13#10);
  CheckEquals(7, S1.Size);

  S2 := TANTLRStringStream.Create(GRAMMARSTR);
  CheckEquals(Length(GRAMMARSTR), S2.Size);

  S3 := TANTLRStringStream.Create('grammar P;');
  CheckEquals(10, S3.Size);
end;

procedure TestANTLRStringStream.TestSizeOnEmptyANTLRStringStream;
var
  S1: IANTLRStringStream;
begin
  S1 := TANTLRStringStream.Create('');
  CheckEquals(0, S1.Size);
  CheckEquals(0, S1.Index);
end;

procedure TestANTLRStringStream.TestSubstringOnANTLRStringStream;
var
  Stream: IANTLRStringStream;
begin
  Stream := TANTLRStringStream.Create('One'#13#10'Two'#13#10'Three');

  CheckEquals('Two', Stream.Substring(5, 7));
  CheckEquals('One', Stream.Substring(0, 2));
  CheckEquals('Three', Stream.Substring(10, 14));

  Stream.Consume;

  CheckEquals('Two', Stream.Substring(5, 7));
  CheckEquals('One', Stream.Substring(0, 2));
  CheckEquals('Three', Stream.Substring(10, 14));
end;

initialization
  // Register any test cases with the test runner
  RegisterTest(TestANTLRStringStream.Suite);
end.
