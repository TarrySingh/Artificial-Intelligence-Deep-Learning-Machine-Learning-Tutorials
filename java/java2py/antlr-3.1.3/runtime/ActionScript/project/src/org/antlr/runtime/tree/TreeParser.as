/*
 [The "BSD licence"]
 Copyright (c) 2005-2007 Terence Parr
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
package org.antlr.runtime.tree {
	import org.antlr.runtime.*;
	
	/** A parser for a stream of tree nodes.  "tree grammars" result in a subclass
	 *  of this.  All the error reporting and recovery is shared with Parser via
	 *  the BaseRecognizer superclass.
	*/
	public class TreeParser extends BaseRecognizer {
		public static const DOWN:int = TokenConstants.DOWN;
		public static const UP:int = TokenConstants.UP;
	
		protected var input:TreeNodeStream;
	
		public function TreeParser(input:TreeNodeStream, state:RecognizerSharedState = null) {
			super(state);
			treeNodeStream = input;
		}
	
		public override function reset():void {
			super.reset(); // reset all recognizer state variables
			if ( input!=null ) {
				input.seek(0); // rewind the input
			}
		}
	
		/** Set the input stream */
		public function set treeNodeStream(input:TreeNodeStream):void {
			this.input = input;
		}
	
		public function get treeNodeStream():TreeNodeStream {
			return input;
		}
		
		public override function get sourceName():String {
			return input.sourceName;
		}
	
		protected override function getCurrentInputSymbol(input:IntStream):Object {
			return TreeNodeStream(input).LT(1);
		}
	
		protected override function getMissingSymbol(input:IntStream,
										   e:RecognitionException,
										   expectedTokenType:int,
										   follow:BitSet):Object {
			var tokenText:String =
				"<missing "+tokenNames[expectedTokenType]+">";
			return CommonTree.createFromToken(new CommonToken(expectedTokenType, tokenText));
		}	
		
		/** Match '.' in tree parser has special meaning.  Skip node or
		 *  entire tree if node has children.  If children, scan until
		 *  corresponding UP node.
		 */
		public function matchAny(ignore:IntStream):void { // ignore stream, copy of this.input
			state.errorRecovery = false;
			state.failed = false;
			var look:Object = input.LT(1);
			if ( input.treeAdaptor.getChildCount(look)==0 ) {
				input.consume(); // not subtree, consume 1 node and return
				return;
			}
			// current node is a subtree, skip to corresponding UP.
			// must count nesting level to get right UP
			var level:int=0;
			var tokenType:int = input.treeAdaptor.getType(look);
			while ( tokenType!=TokenConstants.EOF && !(tokenType==UP && level==0) ) {
				input.consume();
				look = input.LT(1);
				tokenType = input.treeAdaptor.getType(look);
				if ( tokenType == DOWN ) {
					level++;
				}
				else if ( tokenType == UP ) {
					level--;
				}
			}
			input.consume(); // consume UP
		}
	
		/** We have DOWN/UP nodes in the stream that have no line info; override.
		 *  plus we want to alter the exception type. Don't try to recover
	 	 *  from tree parser errors inline...
		 */
		protected override function mismatch(input:IntStream, ttype:int, follow:BitSet):void {
			throw new MismatchedTreeNodeException(ttype, TreeNodeStream(input));
		}
	
		/** Prefix error message with the grammar name because message is
		 *  always intended for the programmer because the parser built
		 *  the input tree not the user.
		 */
		public override function getErrorHeader(e:RecognitionException):String {
			return grammarFileName+": node from "+
				   (e.approximateLineInfo?"after ":"")+"line "+e.line+":"+e.charPositionInLine;
		}
	
		/** Tree parsers parse nodes they usually have a token object as
		 *  payload. Set the exception token and do the default behavior.
		 */
		public override function getErrorMessage(e:RecognitionException, tokenNames:Array):String {
			if ( this is TreeParser ) {
				var adaptor:TreeAdaptor = TreeNodeStream(e.input).treeAdaptor;
				e.token = adaptor.getToken(e.node);
				if ( e.token==null ) { // could be an UP/DOWN node
					e.token = new CommonToken(adaptor.getType(e.node),
											  adaptor.getText(e.node));
				}
			}
			return super.getErrorMessage(e, tokenNames);
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
	
		public function traceOut(ruleName:String, ruleIndex:int):void  {
			super.traceOutSymbol(ruleName, ruleIndex, input.LT(1));
		}
	}

}