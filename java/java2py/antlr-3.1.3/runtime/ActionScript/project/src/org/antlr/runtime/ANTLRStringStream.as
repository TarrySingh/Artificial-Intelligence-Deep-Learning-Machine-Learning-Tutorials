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

	public class ANTLRStringStream implements CharStream {
		/** The data being scanned */
		protected var data:String;
	
		/** How many characters are actually in the buffer */
		protected var n:int;
	
		/** 0..n-1 index into string of next char */
		protected var p:int = 0;
	
		/** line number 1..n within the input */
		protected var _line:int = 1;
	
		/** The index of the character relative to the beginning of the line 0..n-1 */
		protected var _charPositionInLine:int = 0;
	
		/** tracks how deep mark() calls are nested */
		protected var markDepth:int = 0;
	
		/** A list of CharStreamState objects that tracks the stream state
		 *  values line, charPositionInLine, and p that can change as you
		 *  move through the input stream.  Indexed from 1..markDepth.
	     *  A null is kept @ index 0.  Create upon first call to mark().
		 */
		protected var markers:Array;
	
		/** Track the last mark() call result value for use in rewind(). */
		protected var lastMarker:int;
	
		protected var _sourceName:String;
		
	    protected var _lineDelimiter:String;
	    
		/** Copy data in string to a local char array */
		public function ANTLRStringStream(input:String = null, lineDelimiter:String = "\n") {
			this._lineDelimiter = lineDelimiter;
			if (input != null) {
				this.data = input;
				this.n = input.length;
			}
		}
	
		/** Reset the stream so that it's in the same state it was
		 *  when the object was created *except* the data array is not
		 *  touched.
		 */
		public function reset():void {
			p = 0;
			_line = 1;
			_charPositionInLine = 0;
			markDepth = 0;
		}
	
	    public function consume():void {
	        if ( p < n ) {
				_charPositionInLine++;
				if ( data.charAt(p)==_lineDelimiter ) {
					_line++;
					_charPositionInLine=0;
				}
	            p++;
	        }
	    }
	
	    public function LA(i:int):int {
			if ( i==0 ) {
				return 0; // undefined
			}
			if ( i<0 ) {
				i++; // e.g., translate LA(-1) to use offset i=0; then data[p+0-1]
				if ( (p+i-1) < 0 ) {
					return CharStreamConstants.EOF; // invalid; no char before first char
				}
			}
	
			if ( (p+i-1) >= n ) {
	            return CharStreamConstants.EOF;
	        }
			return data.charCodeAt(p+i-1);
	    }
	
		public function LT(i:int):int {
			return LA(i);
		}
	
		/** Return the current input symbol index 0..n where n indicates the
	     *  last symbol has been read.  The index is the index of char to
		 *  be returned from LA(1).
	     */
	    public function get index():int {
	        return p;
	    }
	
		public function get size():int {
			return n;
		}
	
		public function mark():int {
	        if ( markers==null ) {
	            markers = new Array();
	            markers.push(null); // depth 0 means no backtracking, leave blank
	        }
	        markDepth++;
			var state:CharStreamState = null;
			if ( markDepth>=markers.length ) {
				state = new CharStreamState();
				markers.push(state);
			}
			else {
				state = CharStreamState(markers[markDepth]);
			}
			state.p = p;
			state.line = _line;
			state.charPositionInLine = _charPositionInLine;
			lastMarker = markDepth;
			return markDepth;
	    }
	
	    public function rewindTo(m:int):void {
			var state:CharStreamState = CharStreamState(markers[m]);
			// restore stream state
			seek(state.p);
			_line = state.line;
			_charPositionInLine = state.charPositionInLine;
			release(m);
		}
	
		public function rewind():void {
			rewindTo(lastMarker);
		}
	
		public function release(marker:int):void {
			// unwind any other markers made after m and release m
			markDepth = marker;
			// release this marker
			markDepth--;
		}
	
		/** consume() ahead until p==index; can't just set p=index as we must
		 *  update line and charPositionInLine.
		 */
		public function seek(index:int):void {
			if ( index<=p ) {
				p = index; // just jump; don't update stream state (line, ...)
				return;
			}
			// seek forward, consume until p hits index
			while ( p<index ) {
				consume();
			}
		}
	
		public function substring(start:int, stop:int):String {
			return data.substring(start, stop + 1);
		}
	
		public function get line():int {
			return _line;
		}
	
		public function get charPositionInLine():int {
			return _charPositionInLine;
		}
	
		public function set line(line:int):void {
			this._line = line;
		}
	
		public function set charPositionInLine(pos:int):void {
			this._charPositionInLine = pos;
		}
		
		public function get sourceName():String {
			return _sourceName;
		}
		
		public function set sourceName(sourceName:String):void {
			_sourceName = sourceName;
		}
		
	}

}