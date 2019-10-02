/*
 [The "BSD licence"]
 Copyright (c) 2005-2006 Terence Parr
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
package org.antlr.runtime {
    import org.antlr.runtime.tree.TreeAdaptor;
    
	/** A parser for TokenStreams.  "parser grammars" result in a subclass
	 *  of this.
	 */
	public class Parser extends BaseRecognizer {
	    protected var input:TokenStream;
	
		public function Parser(input:TokenStream, state:RecognizerSharedState = null) {
			super(state);
			tokenStream = input;
	    }
	
		public override function reset():void {
			super.reset(); // reset all recognizer state variables
			if ( input!=null ) {
				input.seek(0); // rewind the input
			}
		}

		protected override function getCurrentInputSymbol(input:IntStream):Object {
			return TokenStream(input).LT(1);
		}
	
		protected override function getMissingSymbol(input:IntStream,
										    e:RecognitionException,
										    expectedTokenType:int,
										    follow:BitSet):Object {
		    var tokenText:String = null;
            if ( expectedTokenType==TokenConstants.EOF ) tokenText = "<missing EOF>";
            else tokenText = "<missing "+tokenNames[expectedTokenType]+">";
			var t:CommonToken = new CommonToken(expectedTokenType, tokenText);
			var current:Token = TokenStream(input).LT(1);
			if ( current.type == TokenConstants.EOF ) {
				current = TokenStream(input).LT(-1);
			}
			t.line = current.line;
			t.charPositionInLine = current.charPositionInLine;
			t.channel = DEFAULT_TOKEN_CHANNEL;
			return t;
		}
	
		/** Set the token stream and reset the parser */
		public function set tokenStream(input:TokenStream):void {
			this.input = null;
			reset();
			this.input = input;
		}
		
	    public function get tokenStream():TokenStream {
			return input;
		}
	
		public override function get sourceName():String {
			return input.sourceName;
		}
		
		public function set treeAdaptor(adaptor:TreeAdaptor):void {
		    // do nothing, implemented in generated code
		}
		
		public function get treeAdaptor():TreeAdaptor {
		    // implementation provided in generated code
		    return null;
		}
		
		public function traceIn(ruleName:String, ruleIndex:int):void  {
			super.traceInSymbol(ruleName, ruleIndex, input.LT(1));
		}
	
		public function traceOut(ruleName:String, ruleIndex:int):void {
			super.traceOutSymbol(ruleName, ruleIndex, input.LT(1));
		}
	}

}