package org.antlr.runtime.tree
{
	import org.antlr.runtime.TokenConstants;
	
	public class TreeConstants {
		public static const INVALID_NODE:CommonTree = CommonTree.createFromToken(TokenConstants.INVALID_TOKEN);
	}
}