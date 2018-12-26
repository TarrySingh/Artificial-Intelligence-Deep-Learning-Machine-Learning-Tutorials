/*
[The "BSD licence"]
Copyright (c) 2005-2008 Terence Parr
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
derived from this software without specific prior written permission.

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
 */
package org.antlr.tool;

import java.io.Reader;
import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.File;
import java.util.List;
import java.util.ArrayList;

/** Load a grammar file and scan it just until we learn a few items
 *  of interest.  Currently: name, type, imports, tokenVocab, language option.
 *
 *  GrammarScanner (at bottom of this class) converts grammar to stuff like:
 *
 *   grammar Java ; options { backtrack true memoize true }
 *   import JavaDecl JavaAnnotations JavaExpr ;
 *   ... : ...
 *
 *  First ':' or '@' indicates we can stop looking for imports/options.
 *
 *  Then we just grab interesting grammar properties.
 */
public class GrammarSpelunker {
    protected String grammarFileName;
    protected String token;
    protected Scanner scanner;

    // grammar info / properties
    protected String grammarModifier;
    protected String grammarName;
    protected String tokenVocab;
    protected String language = "Java"; // default
    protected String inputDirectory;
    protected List<String> importedGrammars;

    public GrammarSpelunker(String inputDirectory, String grammarFileName) {
        this.inputDirectory = inputDirectory;
        this.grammarFileName = grammarFileName;
    }

    void consume() throws IOException { token = scanner.nextToken(); }

    protected void match(String expecting) throws IOException {
        //System.out.println("match "+expecting+"; is "+token);
        if ( token.equals(expecting) ) consume();
        else throw new Error("Error parsing "+grammarFileName+": '"+token+
                             "' not expected '"+expecting+"'");
    }

    public void parse() throws IOException {
        Reader r = new FileReader((inputDirectory != null ? inputDirectory + File.separator : "") + grammarFileName);
        BufferedReader br = new BufferedReader(r);
        try {
            scanner = new Scanner(br);
            consume();
            grammarHeader();
            // scan until imports or options
            while ( token!=null && !token.equals("@") && !token.equals(":") &&
                    !token.equals("import") && !token.equals("options") )
            {
                consume();
            }
            if ( token.equals("options") ) options();
            // scan until options or first rule
            while ( token!=null && !token.equals("@") && !token.equals(":") &&
                    !token.equals("import") )
            {
                consume();
            }
            if ( token.equals("import") ) imports();
            // ignore rest of input; close up shop
        }
        finally {
            if ( br!=null ) br.close();
        }
    }

    protected void grammarHeader() throws IOException {
        if ( token==null ) return;
        if ( token.equals("tree") || token.equals("parser") || token.equals("lexer") ) {
            grammarModifier=token;
            consume();
        }
        match("grammar");
        grammarName = token;
        consume(); // move beyond name
    }

    // looks like "options { backtrack true ; tokenVocab MyTokens ; }"
    protected void options() throws IOException {
        match("options");
        match("{");
        while ( token!=null && !token.equals("}") ) {
            String name = token;
            consume();
            String value = token;
            consume();
            match(";");
            if ( name.equals("tokenVocab") ) tokenVocab = value;
            if ( name.equals("language") ) language = value;
        }
        match("}");
    }

    // looks like "import JavaDecl JavaAnnotations JavaExpr ;"
    protected void imports() throws IOException {
        match("import");
        importedGrammars = new ArrayList<String>();
        while ( token!=null && !token.equals(";") ) {
            importedGrammars.add(token);
            consume();
        }
        match(";");
        if ( importedGrammars.size()==0 ) importedGrammars = null;
    }

    public String getGrammarModifier() { return grammarModifier; }

    public String getGrammarName() { return grammarName; }

    public String getTokenVocab() { return tokenVocab; }

    public String getLanguage() { return language; }

    public List<String> getImportedGrammars() { return importedGrammars; }

    /** Strip comments and then return stream of words and
     *  tokens {';', ':', '{', '}'}
     */ 
    public static class Scanner {
        public static final int EOF = -1;
        Reader input;
        int c;

        public Scanner(Reader input) throws IOException {
            this.input = input;
            consume();
        }

        boolean isDIGIT() { return c>='0'&&c<='9'; }
        boolean isID_START() { return c>='a'&&c<='z' || c>='A'&&c<='Z'; }
        boolean isID_LETTER() { return isID_START() || c>='0'&&c<='9' || c=='_'; }
        
        void consume() throws IOException { c = input.read(); }

        public String nextToken() throws IOException {
            while ( c!=EOF ) {
                //System.out.println("check "+(char)c);
                switch ( c ) {
                    case ';' : consume(); return ";";
                    case '{' : consume(); return "{";
                    case '}' : consume(); return "}";
                    case ':' : consume(); return ":";
                    case '@' : consume(); return "@";
                    case '/' : COMMENT(); break;
                    case '\'': return STRING();
                    default:
                        if ( isID_START() ) return ID();
                        else if ( isDIGIT() ) return INT();
                        consume(); // ignore anything else
                }
            }
            return null;
        }

        /** NAME : LETTER+ ; // NAME is sequence of >=1 letter */
        String ID() throws IOException {
            StringBuffer buf = new StringBuffer();
            while ( c!=EOF && isID_LETTER() ) { buf.append((char)c); consume(); }
            return buf.toString();
        }

        String INT() throws IOException {
            StringBuffer buf = new StringBuffer();
            while ( c!=EOF && isDIGIT() ) { buf.append((char)c); consume(); }
            return buf.toString();
        }

        String STRING() throws IOException {
            StringBuffer buf = new StringBuffer();
            consume();
            while ( c!=EOF && c!='\'' ) {
                if ( c=='\\' ) {
                    buf.append((char)c);
                    consume();
                }
                buf.append((char)c);
                consume();
            }
            consume(); // scan past '
            return buf.toString();
        }

        void COMMENT() throws IOException {
            if ( c=='/' ) {
                consume();
                if ( c=='*' ) {
                    consume();
        scarf:
                    while ( true ) {
                        if ( c=='*' ) {
                            consume();
                            if ( c=='/' ) { consume(); break scarf; }
                        }
                        else {
                            while ( c!=EOF && c!='*' ) consume();
                        }
                    }
                }
                else if ( c=='/' ) {
                    while ( c!=EOF && c!='\n' ) consume();
                }
            }
        }
    }

    /** Tester; Give grammar filename as arg */
    public static void main(String[] args) throws IOException {
        GrammarSpelunker g = new GrammarSpelunker(".", args[0]);
        g.parse();
        System.out.println(g.grammarModifier+" grammar "+g.grammarName);
        System.out.println("language="+g.language);
        System.out.println("tokenVocab="+g.tokenVocab);
        System.out.println("imports="+g.importedGrammars);
    }
}
