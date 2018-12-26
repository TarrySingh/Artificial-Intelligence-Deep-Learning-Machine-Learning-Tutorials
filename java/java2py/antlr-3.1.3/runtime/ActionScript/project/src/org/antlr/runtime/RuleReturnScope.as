package org.antlr.runtime
{
	/** Rules can return start/stop info as well as possible trees and templates */
	public class RuleReturnScope {
		/** Return the start token or tree */
		public function get start():Object { return null; }
		/** Return the stop token or tree */
		public function get stop():Object { return null; }
		/** Has a value potentially if output=AST; */
		public function get tree():Object { return null; }
	}

}