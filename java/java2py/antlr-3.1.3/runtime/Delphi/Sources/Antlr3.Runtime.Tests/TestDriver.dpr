program TestDriver;
{

  Delphi DUnit Test Project
  -------------------------
  This project contains the DUnit test framework and the GUI/Console test runners.
  Add "CONSOLE_TESTRUNNER" to the conditional defines entry in the project options
  to use the console test runner.  Otherwise the GUI test runner will be used by
  default.

}

{$IFDEF CONSOLE_TESTRUNNER}
{$APPTYPE CONSOLE}
{$ENDIF}

uses
  Forms,
  TestFramework,
  GUITestRunner,
  TextTestRunner,
  Antlr.Runtime.Tools.Tests in 'Antlr.Runtime.Tools.Tests.pas',
  Antlr.Runtime.Collections.Tests in 'Antlr.Runtime.Collections.Tests.pas',
  Antlr.Runtime.Tree.Tests in 'Antlr.Runtime.Tree.Tests.pas',
  Antlr.Runtime.Tests in 'Antlr.Runtime.Tests.pas';

{$R *.RES}

begin
  ReportMemoryLeaksOnShutdown := True;
  Application.Initialize;
  if IsConsole then
    TextTestRunner.RunRegisteredTests
  else
    GUITestRunner.RunRegisteredTests;
end.

