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
	
	/** A lexer is recognizer that draws input symbols from a character stream.
	 *  lexer grammars result in a subclass of this object. A Lexer object
	 *  uses simplified match() and error recovery mechanisms in the interest
	 *  of speed.
	 */
	public class Lexer extends BaseRecognizer implements TokenSource {
		/** Where is the lexer drawing characters from? */
	    protected var input:CharStream;
	
		public function Lexer(input:CharStream = null, state:RecognizerSharedState = null) {
		    super(state);
			this.input = input;
		}
		
		public override function reset():void {
			super.reset(); // reset all recognizer state variables
    		// wack Lexer state variables
    		if ( input!=null ) {
    			input.seek(0); // rewind the input
    		}
    		if ( state==null ) {
    			return; // no shared state work to do
    		}
    		state.token = null;
    		state.type = TokenConstants.INVALID_TOKEN_TYPE;
    		state.channel = TokenConstants.DEFAULT_CHANNEL;
    		state.tokenStartCharIndex = -1;
    		state.tokenStartCharPositionInLine = -1;
    		state.tokenStartLine = -1;
    		state.text = null;
		}
	
		/** Return a token from this source; i.e., match a token on the char
		 *  stream.
		 */
	    public function nextToken():Token {
			while (true) {
				state.token = null;
				state.channel = TokenConstants.DEFAULT_CHANNEL;
				state.tokenStartCharIndex = input.index;
				state.tokenStartCharPositionInLine = input.charPositionInLine;
				state.tokenStartLine = input.line;
				state.text = null;
				if ( input.LA(1)==CharStreamConstants.EOF ) {
	                return TokenConstants.EOF_TOKEN;
	            }
	            try {
	                mTokens();
					if ( state.token==null ) {
						emit();
					}
					else if ( state.token==TokenConstants.SKIP_TOKEN ) {
						continue;
					}
					return state.token;
				}
	            catch (nva:NoViableAltException) {
    				reportError(nva);
    				recover(nva); // throw out current char and try again
    			}
    			catch (re:RecognitionException) {
    				reportError(re);
    				// match() routine has already called recover()
    			}
	        }
	        // Can't happen, but will quiet complier error
	        return null;
	    }
	
		/** Instruct the lexer to skip creating a token for current lexer rule
		 *  and look for another token.  nextToken() knows to keep looking when
		 *  a lexer rule finishes with token set to SKIP_TOKEN.  Recall that
		 *  if token==null at end of any token rule, it creates one for you
		 *  and emits it.
		 */
		public function skip():void {
			state.token = TokenConstants.SKIP_TOKEN;
		}
	
		/** This is the lexer entry point that sets instance var 'token' */
		public function mTokens():void {
			// abstract function
			throw new Error("Not implemented");
		} 	
	
		/** Set the char stream and reset the lexer */
		public function set charStream(input:CharStream):void {
			this.input = null;
			reset();
			this.input = input;
		}
		
		public function get charStream():CharStream {
			return input;
		}
	
		public override function get sourceName():String {
			return input.sourceName;
		}
		
		/** Currently does not support multiple emits per nextToken invocation
		 *  for efficiency reasons.  Subclass and override this method and
		 *  nextToken (to push tokens into a list and pull from that list rather
		 *  than a single variable as this implementation does).
		 */
		public function emitToken(token:Token):void {
			state.token = token;
		}
	
		/** The standard method called to automatically emit a token at the
		 *  outermost lexical rule.  The token object should point into the
		 *  char buffer start..stop.  If there is a text override in 'text',
		 *  use that to set the token's text.  Override this method to emit
		 *  custom Token objects.
		 */
		public function emit():Token {
			var t:Token = CommonToken.createFromStream(input, state.type, state.channel, state.tokenStartCharIndex, charIndex - 1);
			t.line = state.tokenStartLine;
			t.text = state.text;
			t.charPositionInLine = state.tokenStartCharPositionInLine;
			emitToken(t);
			return t;
		}
	
		public function matchString(s:String):void {
	        var i:int = 0;
	        while ( i<s.length ) {
	            if ( input.LA(1) != s.charCodeAt(i) ) {
					if ( state.backtracking>0 ) {
						state.failed = true;
						return;
					}
					var mte:MismatchedTokenException =
						new MismatchedTokenException(s.charCodeAt(i), input);
					recover(mte);
					throw mte;
	            }
	            i++;
	            input.consume();
				state.failed = false;
	        }
	    }
	
	    public function matchAny():void {
	        input.consume();
	    }
	
	    public function match(c:int):void {
	        if ( input.LA(1)!=c ) {
				if ( state.backtracking>0 ) {
					state.failed = true;
					return;
				}
				var mte:MismatchedTokenException =
					new MismatchedTokenException(c, input);
				recover(mte);  // don't really recover; just consume in lexer
				throw mte;
	        }
	        input.consume();
			state.failed = false;
	    }
	
	    public function matchRange(a:int, b:int):void
		{
	        if ( input.LA(1)<a || input.LA(1)>b ) {
				if ( state.backtracking>0 ) {
					state.failed = true;
					return;
				}
	            var mre:MismatchedRangeException =
					new MismatchedRangeException(a,b,input);
				recover(mre);
				throw mre;
	        }
	        input.consume();
			state.failed = false;
	    }
	
	    public function get line():int {
	        return input.line;
	    }
	
	    public function get charPositionInLine():int {
	        return input.charPositionInLine;
	    }
	
		/** What is the index of the current character of lookahead? */
		public function get charIndex():int {
			return input.index;
		}
	
		/** Return the text matched so far for the current token or any
		 *  text override.
		 */
		public function get text():String {
			if ( state.text!=null ) {
				return state.text;
			}
			return input.substring(state.tokenStartCharIndex, charIndex-1);
		}
	
		/** Set the complete text of this token; it wipes any previous
		 *  changes to the text.
		 */
		public function set text(text:String):void {
			state.text = text;
		}
	
		public override function reportError(e:RecognitionException):void {
			displayRecognitionError(this.tokenNames, e);
		}
	
		public override function getErrorMessage(e:RecognitionException, tokenNames:Array):String {
			var msg:String = null;
			if ( e is MismatchedTokenException ) {
				var mte:MismatchedTokenException = MismatchedTokenException(e);
				msg = "mismatched character "+getCharErrorDisplay(e.c)+" expecting "+getCharErrorDisplay(mte.expecting);
			}
			else if ( e is NoViableAltException ) {
				var nvae:NoViableAltException = NoViableAltException(e);
				// for development, can add "decision=<<"+nvae.grammarDecisionDescription+">>"
				// and "(decision="+nvae.decisionNumber+") and
				// "state "+nvae.stateNumber
				msg = "no viable alternative at character "+getCharErrorDisplay(e.c);
			}
			else if ( e is EarlyExitException ) {
				var eee:EarlyExitException = EarlyExitException(e);
				// for development, can add "(decision="+eee.decisionNumber+")"
				msg = "required (...)+ loop did not match anything at character "+getCharErrorDisplay(e.c);
			}
			else if ( e is MismatchedNotSetException ) {
				var mnse:MismatchedNotSetException = MismatchedNotSetException(e);
				msg = "mismatched character "+getCharErrorDisplay(e.c)+" expecting set "+mnse.expecting;
			}
			else if ( e is MismatchedSetException ) {
				var mse:MismatchedSetException = MismatchedSetException(e);
				msg = "mismatched character "+getCharErrorDisplay(e.c)+" expecting set "+mse.expecting;
			}
			else if ( e is MismatchedRangeException ) {
				var mre:MismatchedRangeException = MismatchedRangeException(e);
				msg = "mismatched character "+getCharErrorDisplay(e.c)+" expecting set "+
					getCharErrorDisplay(mre.a)+".."+getCharErrorDisplay(mre.b);
			}
			else {
				msg = super.getErrorMessage(e, tokenNames);
			}
			return msg;
		}
	
		public function getCharErrorDisplay(c:int):String {
			var s:String = String.fromCharCode(c);
			switch ( c ) {
				case TokenConstants.EOF :
					s = "<EOF>";
					break;
				case '\n' :
					s = "\\n";
					break;
				case '\t' :
					s = "\\t";
					break;
				case '\r' :
					s = "\\r";
					break;
			}
			return "'"+s+"'";
		}
	
		/** Lexers can normally match any char in it's vocabulary after matching
		 *  a token, so do the easy thing and just kill a character and hope
		 *  it all works out.  You can instead use the rule invocation stack
		 *  to do sophisticated error recovery if you are in a fragment rule.
		 * 
		 *  @return This method should return the exception it was provided as an
		 *  argument.  This differs from the Java runtime so that an exception variable
		 *  does not need to be declared in the generated code, thus reducing a large
		 *  number of compiler warnings in generated code.
		 */
		public function recover(re:RecognitionException):RecognitionException {
			input.consume();
			return re;
		}
	
		public function traceIn(ruleName:String, ruleIndex:int):void {
			var inputSymbol:String = String.fromCharCode(input.LT(1))+" line="+ line +":"+ charPositionInLine;
			super.traceInSymbol(ruleName, ruleIndex, inputSymbol);
		}
	
		public function traceOut(ruleName:String, ruleIndex:int):void {
			var inputSymbol:String = String.fromCharCode(input.LT(1))+" line="+ line +":"+ charPositionInLine;
			super.traceOutSymbol(ruleName, ruleIndex, inputSymbol);
		}
	}
}