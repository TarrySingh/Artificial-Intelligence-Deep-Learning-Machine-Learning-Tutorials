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
package org.antlr.runtime {
	public interface Token {
		
		/** Get the text of the token */
		function get text():String;
		function set text(text:String):void;
	
		function get type():int;
		function set type(ttype:int):void;
		
		/**  The line number on which this token was matched; line=1..n */
		function get line():int;
	    function set line(line:int):void;
	
		/** The index of the first character relative to the beginning of the line 0..n-1 */
		function get charPositionInLine():int;
		function set charPositionInLine(pos:int):void;
	
		function get channel():int;
		function set channel(channel:int):void;
	
		/** An index from 0..n-1 of the token object in the input stream.
		 *  This must be valid in order to use the ANTLRWorks debugger.
		 */
		function get tokenIndex():int;
		function set tokenIndex(index:int):void;
		
		/** From what character stream was this token created?  You don't have to
		 *  implement but it's nice to know where a Token comes from if you have
		 *  include files etc... on the input.
		 */
		function get inputStream():CharStream;
		function set inputStream(input:CharStream):void;
	
	}

}