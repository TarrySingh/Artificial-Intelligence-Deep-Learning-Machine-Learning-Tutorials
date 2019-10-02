package org.antlr.runtime {
	public class TokenConstants	{
		public static const EOR_TOKEN_TYPE:int = 1;
	
		/** imaginary tree navigation type; traverse "get child" link */
		public static const DOWN:int = 2;
		/** imaginary tree navigation type; finish with a child list */
		public static const UP:int = 3;
	
		public static const MIN_TOKEN_TYPE:int = UP+1;
	
	    public static const EOF:int = CharStreamConstants.EOF;
		public static const EOF_TOKEN:Token = new CommonToken(EOF);
		
		public static const INVALID_TOKEN_TYPE:int = 0;
		public static const INVALID_TOKEN:Token = new CommonToken(INVALID_TOKEN_TYPE);
	
		/** In an action, a lexer rule can set token to this SKIP_TOKEN and ANTLR
		 *  will avoid creating a token for this symbol and try to fetch another.
		 */
		public static const SKIP_TOKEN:Token = new CommonToken(INVALID_TOKEN_TYPE);
	
		/** All tokens go to the parser (unless skip() is called in that rule)
		 *  on a particular "channel".  The parser tunes to a particular channel
		 *  so that whitespace etc... can go to the parser on a "hidden" channel.
		 */
		public static const DEFAULT_CHANNEL:int = 0;
		
		/** Anything on different channel than DEFAULT_CHANNEL is not parsed
		 *  by parser.
		 */
		public static const HIDDEN_CHANNEL:int = 99;

	}
}