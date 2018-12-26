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

	/** The most common stream of tokens is one where every token is buffered up
	 *  and tokens are prefiltered for a certain channel (the parser will only
	 *  see these tokens and cannot change the filter channel number during the
	 *  parse).
	 *
	 *  TODO: how to access the full token stream?  How to track all tokens matched per rule?
	 */
	public class CommonTokenStream implements TokenStream {
	    protected var _tokenSource:TokenSource;

		/** Record every single token pulled from the source so we can reproduce
		 *  chunks of it later.
		 */
		protected var tokens:Array = new Array();
	
		/** Map<tokentype, channel> to override some Tokens' channel numbers */
		protected var channelOverrideMap:Array;
	
		/** Set<tokentype>; discard any tokens with this type */
		protected var discardSet:Array;
	
		/** Skip tokens on any channel but this one; this is how we skip whitespace... */
		protected var channel:int = TokenConstants.DEFAULT_CHANNEL;
	
		/** By default, track all incoming tokens */
		protected var _discardOffChannelTokens:Boolean = false;
	
		/** Track the last mark() call result value for use in rewind(). */
		protected var lastMarker:int;
	
		/** The index into the tokens list of the current token (next token
	     *  to consume).  p==-1 indicates that the tokens list is empty
	     */
	    protected var p:int = -1;

		public function CommonTokenStream(tokenSource:TokenSource = null, channel:int = TokenConstants.DEFAULT_CHANNEL) {
			_tokenSource = tokenSource;
			this.channel = channel;
		}
	
		/** Reset this token stream by setting its token source. */
		public function set tokenSource(tokenSource:TokenSource):void {
			_tokenSource = tokenSource;
			tokens.clear();
			p = -1;
			channel = TokenConstants.DEFAULT_CHANNEL;
		}
	
		/** Load all tokens from the token source and put in tokens.
		 *  This is done upon first LT request because you might want to
		 *  set some token type / channel overrides before filling buffer.
		 */
		protected function fillBuffer():void {
			var index:int = 0;
			var t:Token = tokenSource.nextToken();
			while ( t!=null && t.type != CharStreamConstants.EOF ) {
				var discard:Boolean = false;
				// is there a channel override for token type?
				if ( channelOverrideMap != null ) {
					if (channelOverrideMap[t.type] != undefined) {
						t.channel = channelOverrideMap[t.type];
					}
				}
				if ( discardSet !=null &&
					 discardSet[t.type] == true )
				{
					discard = true;
				}
				else if ( _discardOffChannelTokens && t.channel != this.channel ) {
					discard = true;
				}
				if ( !discard )	{
					t.tokenIndex = index;
					tokens.push(t);
					index++;
				}
				t = tokenSource.nextToken();
			}
			// leave p pointing at first token on channel
			p = 0;
			p = skipOffTokenChannels(p);
	    }
	
		/** Move the input pointer to the next incoming token.  The stream
		 *  must become active with LT(1) available.  consume() simply
		 *  moves the input pointer so that LT(1) points at the next
		 *  input symbol. Consume at least one token.
		 *
		 *  Walk past any token not on the channel the parser is listening to.
		 */
		public function consume():void {
			if ( p<tokens.length ) {
	            p++;
				p = skipOffTokenChannels(p); // leave p on valid token
	        }
	    }
	
		/** Given a starting index, return the index of the first on-channel
		 *  token.
		 */
		protected function skipOffTokenChannels(i:int):int {
			var n:int = tokens.length;
			while ( i<n && (Token(tokens[i])).channel != channel) {
				i++;
			}
			return i;
		}
	
		protected function skipOffTokenChannelsReverse(i:int):int {
			while ( i>= 0 && (Token(tokens[i])).channel != channel) {
				i--;
			}
			return i;
		}
	
		/** A simple filter mechanism whereby you can tell this token stream
		 *  to force all tokens of type ttype to be on channel.  For example,
		 *  when interpreting, we cannot exec actions so we need to tell
		 *  the stream to force all WS and NEWLINE to be a different, ignored
		 *  channel.
		 */
		public function setTokenTypeChannel(ttype:int, channel:int):void {
			if ( channelOverrideMap==null ) {
				channelOverrideMap = new Array();
			}
	        channelOverrideMap[ttype] = channel;
		}
	
		public function discardTokenType(ttype:int):void {
			if ( discardSet==null ) {
				discardSet = new Array();
			}
	        discardSet[ttype] = true;
		}
	
		public function discardOffChannelTokens(discardOffChannelTokens:Boolean):void {
			_discardOffChannelTokens = discardOffChannelTokens;
		}
	
		public function getTokens():Array {
			if ( p == -1 ) {
				fillBuffer();
			}
			return tokens;
		}
	
		public function getTokensRange(start:int, stop:int):Array {
			return getTokensBitSet(start, stop, null);
		}
	
		/** Given a start and stop index, return a List of all tokens in
		 *  the token type BitSet.  Return null if no tokens were found.  This
		 *  method looks at both on and off channel tokens.
		 * 
		 * Renamed from getTokens
		 */
		public function getTokensBitSet(start:int, stop:int, types:BitSet):Array {
			if ( p == -1 ) {
				fillBuffer();
			}
			if ( stop>=tokens.length ) {
				stop=tokens.length-1;
			}
			if ( start<0 ) {
				start=0;
			}
			if ( start>stop ) {
				return null;
			}
	
			// list = tokens[start:stop]:{Token t, t.getType() in types}
			var filteredTokens:Array = new Array();
			for (var i:int=start; i<=stop; i++) {
				var t:Token = Token(tokens[i]);
				if ( types==null || types.member(t.type) ) {
					filteredTokens.push(t);
				}
			}
			if ( filteredTokens.length==0 ) {
				filteredTokens = null;
			}
			return filteredTokens;
		}
	
		public function getTokensArray(start:int, stop:int, types:Array):Array {
			return getTokensBitSet(start,stop,new BitSet(types));
		}
	
		public function getTokensInt(start:int, stop:int, ttype:int):Array {
			return getTokensBitSet(start,stop,BitSet.of(ttype));
		}
	
		/** Get the ith token from the current position 1..n where k=1 is the
		 *  first symbol of lookahead.
		 */
		public function LT(k:int):Token {
			if ( p == -1 ) {
				fillBuffer();
			}
			if ( k==0 ) {
				return null;
			}
			if ( k<0 ) {
				return LB(-k);
			}
			//System.out.print("LT(p="+p+","+k+")=");
			if ( (p+k-1) >= tokens.length ) {
				return TokenConstants.EOF_TOKEN;
			}
			//System.out.println(tokens.get(p+k-1));
			var i:int = p;
			var n:int = 1;
			// find k good tokens
			while ( n<k ) {
				// skip off-channel tokens
				i = skipOffTokenChannels(i+1); // leave p on valid token
				n++;
			}
			if ( i>=tokens.length ) {
				return TokenConstants.EOF_TOKEN;
			}
	        return Token(tokens[i]);
	    }
	
		/** Look backwards k tokens on-channel tokens */
		protected function LB(k:int):Token {
			//System.out.print("LB(p="+p+","+k+") ");
			if ( p == -1 ) {
				fillBuffer();
			}
			if ( k==0 ) {
				return null;
			}
			if ( (p-k)<0 ) {
				return null;
			}
	
			var i:int = p;
			var n:int = 1;
			// find k good tokens looking backwards
			while ( n<=k ) {
				// skip off-channel tokens
				i = skipOffTokenChannelsReverse(i-1); // leave p on valid token
				n++;
			}
			if ( i<0 ) {
				return null;
			}
			return Token(tokens[i]);
		}
	
		/** Return absolute token i; ignore which channel the tokens are on;
		 *  that is, count all tokens not just on-channel tokens.
		 */
		public function getToken(i:int):Token {
			return Token(tokens[i]);
		}
	
	    public function LA(i:int):int {
	        return LT(i).type;
	    }
	
	    public function mark():int {
			if ( p == -1 ) {
				fillBuffer();
			}
			lastMarker = index;
			return lastMarker;
		}
	
		public function release(marker:int):void {
			// no resources to release
		}
	
		public function get size():int {
			return tokens.length;
		}
	
	    public function get index():int {
	        return p;
	    }
	
		public function reset():void {
			p = 0;
			lastMarker = 0;
		}
	
		public function rewindTo(marker:int):void {
			seek(marker);
		}
	
		public function rewind():void {
			seek(lastMarker);
		}
	
		public function seek(index:int):void {
			p = index;
		}
	
		public function get tokenSource():TokenSource {
			return _tokenSource;
		}
	
		public function get sourceName():String {
			return tokenSource.sourceName;
		}
		
		public function toString():String {
			if ( p == -1 ) {
				fillBuffer();
			}
			return toStringWithRange(0, tokens.length-1);
		}
	
		public function toStringWithRange(start:int, stop:int):String {
			if ( start<0 || stop<0 ) {
				return null;
			}
			if ( p == -1 ) {
				fillBuffer();
			}
			if ( stop>=tokens.length ) {
				stop = tokens.length-1;
			}
	 		var buf:String = "";
			for (var i:int = start; i <= stop; i++) {
				var t:Token = Token(tokens[i]);
				buf += t.text;
			}
			return buf.toString();
		}
	
		public function toStringWithTokenRange(start:Token, stop:Token):String {
			if ( start!=null && stop!=null ) {
				return toStringWithRange(start.tokenIndex, stop.tokenIndex);
			}
			return null;
		}
	}

	
}