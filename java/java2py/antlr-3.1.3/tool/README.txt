ANTLR v3.1.3
March 17, 2009

Terence Parr, parrt at cs usfca edu
ANTLR project lead and supreme dictator for life
University of San Francisco

INTRODUCTION 

Welcome to ANTLR v3!  ANTLR (ANother Tool for Language Recognition) is
a language tool that provides a framework for constructing
recognizers, interpreters, compilers, and translators from grammatical
descriptions containing actions in a variety of target
languages. ANTLR provides excellent support for tree construction,
tree walking, translation, error recovery, and error reporting. I've
been working on parser generators for 20 years and on this particular
version of ANTLR for 5 years.

You should use v3 in conjunction with ANTLRWorks:

    http://www.antlr.org/works/index.html 

and gUnit (grammar unit testing tool included in distribution):

    http://www.antlr.org/wiki/display/ANTLR3/gUnit+-+Grammar+Unit+Testing

The book will also help you a great deal (printed May 15, 2007); you
can also buy the PDF:

    http://www.pragmaticprogrammer.com/titles/tpantlr/index.html

(New book coming out in beta Summer 2009: "Language Design Patterns")

See the getting started document:

    http://www.antlr.org/wiki/display/ANTLR3/FAQ+-+Getting+Started

You also have the examples plus the source to guide you.

See the wiki FAQ:

    http://www.antlr.org/wiki/display/ANTLR3/ANTLR+v3+FAQ

and general doc root:

    http://www.antlr.org/wiki/display/ANTLR3/ANTLR+3+Wiki+Home

Please help add/update FAQ entries.

If all else fails, you can buy support or ask the antlr-interest list:

    http://www.antlr.org/support.html

Per the license in LICENSE.txt, this software is not guaranteed to
work and might even destroy all life on this planet:

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------

EXAMPLES

ANTLR v3 sample grammars:

    http://www.antlr.org/download/examples-v3.tar.gz

Also check out Mantra Programming Language for a prototype (work in
progress) using v3:

    http://www.linguamantra.org/

----------------------------------------------------------------------

What is ANTLR?

ANTLR stands for (AN)other (T)ool for (L)anguage (R)ecognition and was
originally known as PCCTS.  ANTLR is a language tool that provides a
framework for constructing recognizers, compilers, and translators
from grammatical descriptions containing actions.  Target language list:

http://www.antlr.org/wiki/display/ANTLR3/Code+Generation+Targets

----------------------------------------------------------------------

How is ANTLR v3 different than ANTLR v2?

See "What is the difference between ANTLR v2 and v3?"

    http://www.antlr.org/wiki/pages/viewpage.action?pageId=719

See migration guide:

    http://www.antlr.org/wiki/display/ANTLR3/Migrating+from+ANTLR+2+to+ANTLR+3

----------------------------------------------------------------------

How do I install this damn thing?

Just untar and you'll get:

antlr-3.1.3/README.txt (this file)
antlr-3.1.3/LICENSE.txt
antlr-3.1.3/src/main/java/org/antlr/...
antlr-3.1.3/lib/stringtemplate-3.2.jar
antlr-3.1.3/lib/antlr-2.7.7.jar (ANTLR v3 currently written in v2)
antlr-3.1.3/lib/antlr-3.1.3.jar (all jars combined, runtime and tools)
antlr-3.1.3/lib/antlr-runtime-3.1.3.jar (only what is needed to use ANTLR parsers)

Then you need to add all the jars in lib to your CLASSPATH.

Please see the FAQ

    http://www.antlr.org/wiki/display/ANTLR3/ANTLR+v3+FAQ
