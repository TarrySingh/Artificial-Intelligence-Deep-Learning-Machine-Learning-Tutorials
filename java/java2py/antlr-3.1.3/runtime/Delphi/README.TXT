==================================================
ANTLR v3 Delphi Code generator and Runtime library
==================================================

October 27, 2008
Erik van Bilsen (erik AT bilsen DOT com)

Please see LICENSE.TXT for the full text of the license and NOTICE.TXT
for attribution notices.

Architecture
============
The Delphi target consists of a set of code generation templates and a runtime 
library (written in Delphi 2009) for the Win32 platform.
The Delphi code generation targets and the runtime library are modeled on the 
C# version.

You need to use Delphi 2009 or a later version to be able to use this target. 
You will not be able to compile generated code with older versions of Delphi.
The reason for this is that this Delphi target uses a lot of new Delphi 
language features such as generics.

To use the Delphi target, you only need to put the runtime source code in a 
directory of your choice, and add this directory to you Delphi library path or 
to your project's search path.

The runtime consists of the following units:
-Antlr.Runtime: the main Antlr unit that contains the parser and lexer classes
-Antlr.Runtime.Tree: the tree parser class and other tree related classes
-Antlr.Runtime.Collections: several collection utilities
-Antlr.Runtime.Tools: this is a special Delphi addition to the runtime 
 containing several helper classes and utilities
You will find these files in the "Antlr3.Runtime" subdirectory.

In your projects, you usually only need to use the Antlr.Runtime unit, and the
Antlr.Runtime.Tree unit for tree parsers.
This target does not require any third party libraries, and you do not have to
deploy any DLLs or other files with your ANTLR Delphi projects.

Please note that this Delphi target does not support StringTemplate output, but
it does support all other output types, including AST output.

Status
======
As of October 2008, the Delphi target is in sync with ANTLR 3.1.

This version passes all the unit tests (which you can find in the
"Antlr3.Runtime.Tests" subdirectory) without any memory leaks.
Also, all the grammar samples in the "examples-v3\Delphi" directory function
correctly and without any memory leaks.

Performance
===========
This target should perform reasonably well compared to other ANTLR targets.
For some grammars, especially tree grammars, the code that is generated is not
as efficient as for other targets. This has to do with the way the code is
generated to work around some issues with the Delphi language. But even with
these workarounds, the target performs within reasonable boundaries.

Usage
=====
Here is a short list of Delphi specific issues you need to take into account
when using this target. Please check out the Delphi sample grammars in the
"examples-v3" archive for examples of all the issues described below. And these
examples are a great way to get started with ANTLR.

Specify that Delphi code should be generated for a grammar
----------------------------------------------------------
To specify that the ANTLR tool should generate Delphi (2009) code (rather than
the default of generating Java code) for a grammar, set the grammar-level option
named "language" to the value "Delphi" as shown below:

  grammar MyGrammar;

  options {
  	language=Delphi;
  }
  ...

For the example grammar named MyGrammar above, the grammar file would typically
be named MyGrammar.g. The grammar filename (excluding the extension) must match
the grammar name as declared with the grammar directive in the file.

Use Delphi code in actions
--------------------------
Obviously, any custom actions inside your grammars should be written in the
Delphi language. This also applies to less obvious actions like
{$channel=HIDDEN;}, which should be written as {$channel:=HIDDEN;} (with the
colon before the equals sign).

Rule names must not be case sensitive
-------------------------------------
Since the Delphi language is not case sensitive, you must take care that the
names of rules in your grammars differ by more than only case. For example, if
you have a parser rule called "expression", then you shouldn't have a lexer rule
called "EXPRESSION" or "Expression" or any other combination of upper- and lower
case characters that math the same word. ANTLR will still be able to generate
Delphi code for this, but you will not be able to compile it because of
duplicate identifiers.

The @members grammar action
---------------------------
The Delphi target does not recognize the default @members grammar action. It
uses the following three grammar actions instead (see the C and Java sample
grammars for examples):

@memberDeclarations: use this action that declare members in the generated
parser/lexer class. For example:

  @memberDeclarations {
	enumIsKeyword: Boolean;
	function isTypeName(const name: String): Boolean;
  }

  These declarations will appear inside the parser/lexer class declaration.

@memberInitializations: use this action to initialize variables declared in the
@memberDeclarations action. For example:

  @memberInitializations {
	enumIsKeyword := True;
  }

  These statements will appear inside the constructor of the parser/lexer class.

@memberImplementations: use this action for any code that must appear in the
parser class implementation. For example:

  @memberImplementations {
  function TCParser.isTypeName(const name: String): Boolean;
  begin
    Result := [...]
  end;
  }

  The code inside this action appears as-is inside the implementation section of
  the parser/lexer unit. This means that you need to specify the full name of
  the method, including the parser/lexer class name (eg. TCParser.isTypeName).
  The class name is based on the name of the grammar, and whether it is a parser
  or lexer. So, if your grammar is called "MyGrammar", then the lexer class will
  be called TMyGrammarLexer and the parser class will be called
  TMyGrammarParser.

The @vars grammar action
------------------------
ANTLR supports an @init (and @after) grammar action for any code you want to
execute at the beginning (or end) of a rule. If that code, or any other code
inside the rule, makes use of a local variable, then you need to declare that
variable first. The Delphi target adds the @vars grammar action for this
purpose. You can declare any local variables inside this action, as in this
example (taken from the Python example grammar):

  LEADING_WS
  @vars {
    spaces: Integer;
    S: String;
  }
  @init {
    spaces := 0;
  }

The variables you declare in the @vars action will appear inside the "var"
declaration block of the method for the rule (in this case for the
LEADING_WS rule).

The @usesInterface and @usedImplementation grammar actions
----------------------------------------------------------
If you need to add units to the uses clause of the generated units, then you can
use the @usesInterface and @usesImplementation grammar actions. For example, if
some code inside the grammer rules needs access to the Delphi TStringList class,
then you will need to use the Classes unit.
Use the @usesInterface action if you need the units to appear in the interface
part, or @usesImplementation if you only need a unit inside the implementation.
For example:

  @usesImplementation {
    Classes,
    Generics.Collections,
  }

Note that you need to add a comma after each unit in this list. The Delphi units
SysUtils, StrUtils and Math are added to the uses clause automatically.
Also note that you will usually put the @usesInterface/@usesImplementation
actions at the top of your grammar file, like you would the with the @header
action for other language targets.

The Delphi target is interface based
------------------------------------
All classes inside the Delphi ANTLR runtime use object interfaces. This greatly
simplifies memory management and makes using the runtime much easier. This means
that you will never declare class variables for ANTLR objects, but only use
interface variables. For example, a typical test rig in Delphi looks like this
(taken from the SimpleC example):

  procedure Run(const InputFilename: String);
  var
    Input: ICharStream;
    Lex: ISimpleCLexer;
    Tokens: ICommonTokenStream;
    Parser: ISimpleCParser;
    R: Iprog_return;
  begin
    Input := TANTLRFileStream.Create(InputFilename);
    Lex := TSimpleCLexer.Create(Input);
    Tokens := TCommonTokenStream.Create(Lex);
    Parser := TSimpleCParser.Create(Tokens);
    R := Parser.prog;
    WriteLn('tree=' + (R.Tree as ITree).ToStringTree);
  end;

Note that all variables are declared as interface variables (starting with a
capital I) instead of class variables (with a capital T). And there is no need
to destroy these objects yourself (there are no calls to Free and no
try..finally blocks to protect these resources).

If you are new to interface-based programming, then don't worry: just remember
to declare all ANTLR objects using interface variables, and don't call Free
on them.

Note that the C# and Java versions of the tree creation classes use the general
Object type for tree nodes. In the Delphi version, tree nodes are of type
IANTLRInterface, and can be implemented in various class (like TCommonTree).

Antlr.Runtime.Tools
-------------------
This unit contains some classes and interfaces you may find useful inside ANTLR
projects. Also, this unit contains declarations for the IANTLRInterface
interface and TANTLRObject class. All ANTLR classes derive from TANTLRObject and
implement the IANTLRInterface interface.

Other interfaces/classes you may find useful are:

* IANTLRString (implemented in TANTLRString): a wrapper around a Delphi string
  that allows you to treat a string as a regular ANTLR object.

* IList<T> (implemented in TList<T>): a generic list containing elements of
  type <T>. For example, you can create a list of Integers like this:

  var
    List: IList<Integer>;
  begin
    List := TList<Integer>.Create;
    List.Add(123);
  end;

  Note that this is basically the same TList<T> declared in Delphi's unit
  Generics.Collections, but it implements the IList<T> interface.

* IDictionary<TKey, TValue> (implemented in TDictionary<TKey, TValue>): a
  generic dictionary that maps elements of type <TKey> to <TValue>. For example,
  to map Strings to TANTLRObjects, use:

  var
    Map: IDictionary<String, IANTLRInterface>
  begin
    Map := TDictionary<String, IANTLRInterface>.Create;
    Map.Add('foo', TANTLRObject.Create);
  end;

  Again, this class is similar to Delphi's TDictionary, but it implements the
  IDictionary<TKey, TValue> interface.



Erik van Bilsen
