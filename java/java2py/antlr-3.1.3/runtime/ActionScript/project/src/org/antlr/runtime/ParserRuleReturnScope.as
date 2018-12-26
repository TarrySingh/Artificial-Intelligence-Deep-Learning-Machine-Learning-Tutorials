package org.antlr.runtime
{
	/** Rules that return more than a single value must return an object
	 *  containing all the values.  Besides the properties defined in
	 *  RuleLabelScope.predefinedRulePropertiesScope there may be user-defined
	 *  return values.  This class simply defines the minimum properties that
	 *  are always defined and methods to access the others that might be
	 *  available depending on output option such as template and tree.
	 *
	 *  Note text is not an actual property of the return value, it is computed
	 *  from start and stop using the input stream's toString() method.  I
	 *  could add a ctor to this so that we can pass in and store the input
	 *  stream, but I'm not sure we want to do that.  It would seem to be undefined
	 *  to get the .text property anyway if the rule matches tokens from multiple
	 *  input streams.
	 *
	 *  I do not use getters for fields of objects that are used simply to
	 *  group values such as this aggregate.
	 */
	public class ParserRuleReturnScope extends RuleReturnScope {
		private var _startToken:Token;
		private var _stopToken:Token;
		private var _tree:Object;  // if output=AST this contains the tree
		private var _values:Object = new Object(); // contains the return values
			
		public override function get start():Object {
			return _startToken;
		}

		public function set start(token:Object):void {
			_startToken = Token(token);
		}		
		
		public override function get stop():Object { 
			return _stopToken;
		}
		
		public function set stop(token:Object):void {
			_stopToken = Token(token);
		}
		
		/** Has a value potentially if output=AST; */
		public override function get tree():Object {
			 return _tree;
		}
		
		public function set tree(tree:Object):void {
			_tree = tree;
		}
		
		public function get values():Object {
			return _values;
		}
	}
}