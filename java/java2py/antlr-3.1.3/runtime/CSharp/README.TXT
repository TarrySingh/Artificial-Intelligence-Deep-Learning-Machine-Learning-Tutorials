ANTLR v3.1 .NET Runtime Library (for us with the ANTLR C# Code Generator)

29 September, 2007

Kunle Odutola		: kunle UNDERSCORE odutola AT hotmail DOT com
Micheal Jordan

1. INTRODUCTION 

The ANTLR v3.1 .NET Runtime Library extend the ANTLR language processing 
tools generator to the C#/CLI platforms such as Microsoft .NET, 
Novell/Ximian Mono and dotGNU. It is written in the C# programming language 
and was designed specifically for use with the ANTLR C# Code Generation
target but, it would work equally well with a VB.NET, C++/CLI or indeed 
IronPython code generator were such a thing to be developed for ANTLR v3.1.

We hope you find the ANTLR v3.1 .NET Runtime Library delightful and useful 
even but, as per the license under which you may use it, this software is not 
guaranteed to work.

Please see LICENSE.TXT for the full text of the license and NOTICE.TXT 
for attribution notices.


2. WHAT'S IN THE PACK?

This distribution contains three projects, the project files needed to 
build them with Microsoft Visual Studio 2003 & 2005 and, Nant build files.

The projects are:

	1. Antlr3.Runtime           - the ANTLR v3.1 .NET Runtime Library

	2. Antlr3.Utility           - the ANTLR v3.1 .NET Runtime Utility Library

	3. Antlr3.Runtime.Tests     - the ANTLR v3.1 .NET Runtime Library Tests

In addition the Libraries sub-directory contains externals dependencies.

2.1 Dependencies

   1. Antlr3.Runtime           - none

   2. Antlr3.Utility           - Antlr3.Runtime.dll
                                 StringTemplate.dll
                                 antlr.runtime.dll

   3. Antlr3.Runtime.Tests     - Antlr3.Runtime.dll
                                 StringTemplate.dll
                                 antlr.runtime.dll

In addition, Antlr3.Runtime.Tests has a dependency on the MbUnit v2.4 dlls.


3. USING The ANTLR v3.1 .NET Runtime Library

Tou use the ANTLR v3.1 .NET Runtime Library in your projects, add a 
reference to the following file in your projects:
  - Antlr3.Runtime.dll
  
If you are using StringTemplate out in your grammar, add the following 
files too:  
  - StringTemplate.dll 
  - antlr.runtime.dll

You can find examples of using ANTLR v3.1 and the ANTLR v3.1 .NET Runtime 
Library at:
  http://www.antlr.org/download/examples-v3.tar.gz


4. BUILDING The ANTLR v3.1 .NET Runtime Library

If you wish to re-build The ANTLR v3.1 .NET Runtime Library for any reason, this 
is what you need to know.

    nant clean
	nant release -t:net-1.1
or
    nant clean
	nant release -t:net-2.0


5. ANTLR v3.1 .NET Runtime Library STATUS

This release of the ANTLR v3.1 .NET Runtime Library is a stable beta release.

There are currently 0 failures in the unit test suite.


Don't forget to visit the www.antlr.org for further info. The mailing lists 
are (currently) low volume and have a very high Signal-to-Noise ratio. We'd 
like to hear about how you're using ANTLR v3.1 and the .NET Runtime Library.


7. CONTRIBUTORS

Kunle Odutola
Micheal Jordan


Enjoy!

Kunle Odutola
