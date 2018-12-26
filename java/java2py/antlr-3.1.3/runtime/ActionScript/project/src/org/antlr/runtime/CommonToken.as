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
	
	

public class CommonToken implements Token {
	protected var _type:int;
	protected var _line:int;
	protected var _charPositionInLine:int = -1; // set to invalid position
	protected var _channel:int = TokenConstants.DEFAULT_CHANNEL;
	protected var _input:CharStream;
	
	/** We need to be able to change the text once in a while.  If
	 *  this is non-null, then getText should return this.  Note that
	 *  start/stop are not affected by changing this.
	 */
	protected var _text:String;

	/** What token number is this from 0..n-1 tokens; < 0 implies invalid index */
	protected var _index:int = -1;

	/** The char position into the input buffer where this token starts */
	protected var _start:int;

	/** The char position into the input buffer where this token stops */
	protected var _stop:int;

	public function CommonToken(type:int, text:String = null) {
		this._type = type;
		this._text = text;
	}

	public static function createFromStream(input:CharStream, type:int, channel:int, start:int, stop:int):CommonToken {
		var token:CommonToken = new CommonToken(type);
		token._input = input;
		token._channel = channel;
		token._start = start;
		token._stop = stop;
		return token;
	}
	
	public static function cloneToken(oldToken:Token):CommonToken {
		var token:CommonToken = new CommonToken(oldToken.type, oldToken.text);
		token._line = oldToken.line;
		token._index = oldToken.tokenIndex;
		token._charPositionInLine = oldToken.charPositionInLine;
		token._channel = oldToken.channel;
		if ( oldToken is CommonToken ) {
			token._start = CommonToken(oldToken).startIndex;
			token._stop = CommonToken(oldToken).stopIndex;
		}
		return token;
	}

	public function get type():int {
		return _type;
	}

	public function set line(line:int):void {
		_line = line;
	}

	public function get text():String {
		if ( _text!=null ) {
			return _text;
		}
		if ( _input==null ) {
			return null;
		}
		_text = _input.substring(_start, _stop);
		return _text;
	}

	/** Override the text for this token.  getText() will return this text
	 *  rather than pulling from the buffer.  Note that this does not mean
	 *  that start/stop indexes are not valid.  It means that that input
	 *  was converted to a new string in the token object.
	 */
	public function set text(text:String):void {
		_text = text;
	}

	public function get line():int {
		return _line;
	}

	public function get charPositionInLine():int {
		return _charPositionInLine;
	}

	public function set charPositionInLine(charPositionInLine:int):void {
		_charPositionInLine = charPositionInLine;
	}

	public function get channel():int {
		return _channel;
	}

	public function set channel(channel:int):void {
		_channel = channel;
	}

	public function set type(type:int):void {
		_type = type;
	}

	public function get startIndex():int {
		return _start;
	}

	public function set startIndex(start:int):void {
		_start = start;
	}

	public function get stopIndex():int {
		return _stop;
	}

	public function set stopIndex(stop:int):void {
		_stop = stop;
	}

	public function get tokenIndex():int {
		return _index;
	}

	public function set tokenIndex(index:int):void {
		_index = index;
	}

	public function get inputStream():CharStream {
		return _input;
	}
	
	public function set inputStream(input:CharStream):void {
		_input = input;
	}
	
	public function toString():String {
		var channelStr:String = "";
		if ( channel>0 ) {
			channelStr=",channel="+channel;
		}
		var txt:String = text;
		if ( txt!=null ) {
			txt = txt.replace("\n", "\\\\n");
			txt = txt.replace("\r", "\\\\r");
			txt = txt.replace("\t", "\\\\t");
		}
		else {
			txt = "<no text>";
		}
		return "[@"+tokenIndex+","+startIndex+":"+stopIndex+"='"+txt+"',<"+type+">"+channelStr+","+line+":"+charPositionInLine+"]";
	}
}

}