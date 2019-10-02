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

import java.util.*;

/** The most common stream of tokens is one where every token is buffered up
 *  and tokens are prefiltered for a certain channel (the parser will only
 *  see these tokens and cannot change the filter channel number during the
 *  parse).
 *
 *  TODO: how to access the full token stream?  How to track all tokens matched per rule?
 */
public class CommonTokenStream implements TokenStream {
    protected TokenSource tokenSource;

	/** Record every single token pulled from the source so we can reproduce
	 *  chunks of it later.
	 */
	protected List tokens;

	/** Map<tokentype, channel> to override some Tokens' channel numbers */
	protected Map channelOverrideMap;

	/** Set<tokentype>; discard any tokens with this type */
	protected Set discardSet;

	/** Skip tokens on any channel but this one; this is how we skip whitespace... */
	protected int channel = Token.DEFAULT_CHANNEL;

	/** By default, track all incoming tokens */
	protected boolean discardOffChannelTokens = false;

	/** Track the last mark() call result value for use in rewind(). */
	protected int lastMarker;

	/** The index into the tokens list of the current token (next token
     *  to consume).  p==-1 indicates that the tokens list is empty
     */
    protected int p = -1;

	public CommonTokenStream() {
		tokens = new ArrayList(500);
	}

	public CommonTokenStream(TokenSource tokenSource) {
	    this();
		this.tokenSource = tokenSource;
	}

	public CommonTokenStream(TokenSource tokenSource, int channel) {
		this(tokenSource);
		this.channel = channel;
	}

	/** Reset this token stream by setting its token source. */
	public void setTokenSource(TokenSource tokenSource) {
		this.tokenSource = tokenSource;
		tokens.clear();
		p = -1;
		channel = Token.DEFAULT_CHANNEL;
	}

	/** Load all tokens from the token source and put in tokens.
	 *  This is done upon first LT request because you might want to
	 *  set some token type / channel overrides before filling buffer.
	 */
	protected void fillBuffer() {
		int index = 0;
		Token t = tokenSource.nextToken();
		while ( t!=null && t.getType()!=CharStream.EOF ) {
			boolean discard = false;
			// is there a channel override for token type?
			if ( channelOverrideMap!=null ) {
				Integer channelI = (Integer)
					channelOverrideMap.get(new Integer(t.getType()));
				if ( channelI!=null ) {
					t.setChannel(channelI.intValue());
				}
			}
			if ( discardSet!=null &&
				 discardSet.contains(new Integer(t.getType())) )
			{
				discard = true;
			}
			else if ( discardOffChannelTokens && t.getChannel()!=this.channel ) {
				discard = true;
			}
			if ( !discard )	{
				t.setTokenIndex(index);
				tokens.add(t);
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
	public void consume() {
		if ( p<tokens.size() ) {
            p++;
			p = skipOffTokenChannels(p); // leave p on valid token
        }
    }

	/** Given a starting index, return the index of the first on-channel
	 *  token.
	 */
	protected int skipOffTokenChannels(int i) {
		int n = tokens.size();
		while ( i<n && ((Token)tokens.get(i)).getChannel()!=channel ) {
			i++;
		}
		return i;
	}

	protected int skipOffTokenChannelsReverse(int i) {
		while ( i>=0 && ((Token)tokens.get(i)).getChannel()!=channel ) {
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
	public void setTokenTypeChannel(int ttype, int channel) {
		if ( channelOverrideMap==null ) {
			channelOverrideMap = new HashMap();
		}
        channelOverrideMap.put(new Integer(ttype), new Integer(channel));
	}

	public void discardTokenType(int ttype) {
		if ( discardSet==null ) {
			discardSet = new HashSet();
		}
        discardSet.add(new Integer(ttype));
	}

	public void discardOffChannelTokens(boolean discardOffChannelTokens) {
		this.discardOffChannelTokens = discardOffChannelTokens;
	}

	public List getTokens() {
		if ( p == -1 ) {
			fillBuffer();
		}
		return tokens;
	}

	public List getTokens(int start, int stop) {
		return getTokens(start, stop, (BitSet)null);
	}

	/** Given a start and stop index, return a List of all tokens in
	 *  the token type BitSet.  Return null if no tokens were found.  This
	 *  method looks at both on and off channel tokens.
	 */
	public List getTokens(int start, int stop, BitSet types) {
		if ( p == -1 ) {
			fillBuffer();
		}
		if ( stop>=tokens.size() ) {
			stop=tokens.size()-1;
		}
		if ( start<0 ) {
			start=0;
		}
		if ( start>stop ) {
			return null;
		}

		// list = tokens[start:stop]:{Token t, t.getType() in types}
		List filteredTokens = new ArrayList();
		for (int i=start; i<=stop; i++) {
			Token t = (Token)tokens.get(i);
			if ( types==null || types.member(t.getType()) ) {
				filteredTokens.add(t);
			}
		}
		if ( filteredTokens.size()==0 ) {
			filteredTokens = null;
		}
		return filteredTokens;
	}

	public List getTokens(int start, int stop, List types) {
		return getTokens(start,stop,new BitSet(types));
	}

	public List getTokens(int start, int stop, int ttype) {
		return getTokens(start,stop,BitSet.of(ttype));
	}

	/** Get the ith token from the current position 1..n where k=1 is the
	 *  first symbol of lookahead.
	 */
	public Token LT(int k) {
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
		if ( (p+k-1) >= tokens.size() ) {
			return Token.EOF_TOKEN;
		}
		//System.out.println(tokens.get(p+k-1));
		int i = p;
		int n = 1;
		// find k good tokens
		while ( n<k ) {
			// skip off-channel tokens
			i = skipOffTokenChannels(i+1); // leave p on valid token
			n++;
		}
		if ( i>=tokens.size() ) {
			return Token.EOF_TOKEN;
		}
        return (Token)tokens.get(i);
    }

	/** Look backwards k tokens on-channel tokens */
	protected Token LB(int k) {
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

		int i = p;
		int n = 1;
		// find k good tokens looking backwards
		while ( n<=k ) {
			// skip off-channel tokens
			i = skipOffTokenChannelsReverse(i-1); // leave p on valid token
			n++;
		}
		if ( i<0 ) {
			return null;
		}
		return (Token)tokens.get(i);
	}

	/** Return absolute token i; ignore which channel the tokens are on;
	 *  that is, count all tokens not just on-channel tokens.
	 */
	public Token get(int i) {
		return (Token)tokens.get(i);
	}

    public int LA(int i) {
        return LT(i).getType();
    }

    public int mark() {
		if ( p == -1 ) {
			fillBuffer();
		}
		lastMarker = index();
		return lastMarker;
	}

	public void release(int marker) {
		// no resources to release
	}

	public int size() {
		return tokens.size();
	}

    public int index() {
        return p;
    }

	public void rewind(int marker) {
		seek(marker);
	}

	public void rewind() {
		seek(lastMarker);
	}

	public void reset() {
		p = 0;
		lastMarker = 0;
	}
	
	public void seek(int index) {
		p = index;
	}

	public TokenSource getTokenSource() {
		return tokenSource;
	}

	public String getSourceName() {
		return getTokenSource().getSourceName();
	}

	public String toString() {
		if ( p == -1 ) {
			fillBuffer();
		}
		return toString(0, tokens.size()-1);
	}

	public String toString(int start, int stop) {
		if ( start<0 || stop<0 ) {
			return null;
		}
		if ( p == -1 ) {
			fillBuffer();
		}
		if ( stop>=tokens.size() ) {
			stop = tokens.size()-1;
		}
 		StringBuffer buf = new StringBuffer();
		for (int i = start; i <= stop; i++) {
			Token t = (Token)tokens.get(i);
			buf.append(t.getText());
		}
		return buf.toString();
	}

	public String toString(Token start, Token stop) {
		if ( start!=null && stop!=null ) {
			return toString(start.getTokenIndex(), stop.getTokenIndex());
		}
		return null;
	}
}
