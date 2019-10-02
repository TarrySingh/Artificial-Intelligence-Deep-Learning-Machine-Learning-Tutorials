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
package org.antlr.runtime;

/** A Token object like we'd use in ANTLR 2.x; has an actual string created
 *  and associated with this object.  These objects are needed for imaginary
 *  tree nodes that have payload objects.  We need to create a Token object
 *  that has a string; the tree node will point at this token.  CommonToken
 *  has indexes into a char stream and hence cannot be used to introduce
 *  new strings.
 */
public class ClassicToken implements Token {
	protected String text;
	protected int type;
	protected int line;
	protected int charPositionInLine;
	protected int channel=DEFAULT_CHANNEL;

	/** What token number is this from 0..n-1 tokens */
	protected int index;

	public ClassicToken(int type) {
		this.type = type;
	}

	public ClassicToken(Token oldToken) {
		text = oldToken.getText();
		type = oldToken.getType();
		line = oldToken.getLine();
		charPositionInLine = oldToken.getCharPositionInLine();
		channel = oldToken.getChannel();
	}

	public ClassicToken(int type, String text) {
		this.type = type;
		this.text = text;
	}

	public ClassicToken(int type, String text, int channel) {
		this.type = type;
		this.text = text;
		this.channel = channel;
	}

	public int getType() {
		return type;
	}

	public void setLine(int line) {
		this.line = line;
	}

	public String getText() {
		return text;
	}

	public void setText(String text) {
		this.text = text;
	}

	public int getLine() {
		return line;
	}

	public int getCharPositionInLine() {
		return charPositionInLine;
	}

	public void setCharPositionInLine(int charPositionInLine) {
		this.charPositionInLine = charPositionInLine;
	}

	public int getChannel() {
		return channel;
	}

	public void setChannel(int channel) {
		this.channel = channel;
	}

	public void setType(int type) {
		this.type = type;
	}

	public int getTokenIndex() {
		return index;
	}

	public void setTokenIndex(int index) {
		this.index = index;
	}

	public CharStream getInputStream() {
		return null;
	}

	public void setInputStream(CharStream input) {
	}
	
	public String toString() {
		String channelStr = "";
		if ( channel>0 ) {
			channelStr=",channel="+channel;
		}
		String txt = getText();
		if ( txt!=null ) {
			txt = txt.replaceAll("\n","\\\\n");
			txt = txt.replaceAll("\r","\\\\r");
			txt = txt.replaceAll("\t","\\\\t");
		}
		else {
			txt = "<no text>";
		}
		return "[@"+getTokenIndex()+",'"+txt+"',<"+type+">"+channelStr+","+line+":"+getCharPositionInLine()+"]";
	}
}
