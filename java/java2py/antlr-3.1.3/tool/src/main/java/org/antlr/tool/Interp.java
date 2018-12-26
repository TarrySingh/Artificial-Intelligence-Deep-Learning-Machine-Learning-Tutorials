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

import org.antlr.runtime.ANTLRFileStream;
import org.antlr.runtime.CharStream;
import org.antlr.runtime.CommonTokenStream;
import org.antlr.runtime.tree.ParseTree;
import org.antlr.Tool;

import java.util.StringTokenizer;
import java.util.List;
import java.io.FileReader;
import java.io.BufferedReader;

/** Interpret any ANTLR grammar:
 *
 *  java Interp file.g tokens-to-ignore start-rule input-file
 *
 *  java Interp C.g 'WS COMMENT' program t.c
 *
 *  where the WS and COMMENT are the names of tokens you want to have
 *  the parser ignore.
 */
public class Interp {
	// pass me a java file to parse
	public static void main(String[] args) throws Exception {
		if ( args.length!=4 ) {
			System.err.println("java Interp file.g tokens-to-ignore start-rule input-file");
			return;
		}
		String grammarFileName = args[0];
		String ignoreTokens = args[1];
		String startRule = args[2];
		String inputFileName = args[3];

		// TODO: using wrong constructor now
		Tool tool = new Tool();
		CompositeGrammar composite = new CompositeGrammar();
		Grammar parser = new Grammar(tool, grammarFileName, composite);
		composite.setDelegationRoot(parser);
		FileReader fr = new FileReader(grammarFileName);
		BufferedReader br = new BufferedReader(fr);
		parser.parseAndBuildAST(br);
		br.close();

		parser.composite.assignTokenTypes();
		parser.composite.defineGrammarSymbols();
		parser.composite.createNFAs();

		List leftRecursiveRules = parser.checkAllRulesForLeftRecursion();
		if ( leftRecursiveRules.size()>0 ) {
			return;
		}

		if ( parser.getRule(startRule)==null ) {
			System.out.println("undefined start rule "+startRule);
			return;
		}

		String lexerGrammarText = parser.getLexerGrammar();
		Grammar lexer = new Grammar();
		lexer.importTokenVocabulary(parser);
		lexer.fileName = grammarFileName;
		lexer.setTool(tool);
		if ( lexerGrammarText!=null ) {
			lexer.setGrammarContent(lexerGrammarText);
		}
		else {
			System.err.println("no lexer grammar found in "+grammarFileName);
		}
		lexer.composite.createNFAs();
		
		CharStream input =
			new ANTLRFileStream(inputFileName);
		Interpreter lexEngine = new Interpreter(lexer, input);
		CommonTokenStream tokens = new CommonTokenStream(lexEngine);
		StringTokenizer tk = new StringTokenizer(ignoreTokens, " ");
		while ( tk.hasMoreTokens() ) {
			String tokenName = tk.nextToken();
			tokens.setTokenTypeChannel(lexer.getTokenType(tokenName), 99);
		}

		if ( parser.getRule(startRule)==null ) {
			System.err.println("Rule "+startRule+" does not exist in "+grammarFileName);
			return;
		}
		Interpreter parseEngine = new Interpreter(parser, tokens);
		ParseTree t = parseEngine.parse(startRule);
		System.out.println(t.toStringTree());
	}
}
