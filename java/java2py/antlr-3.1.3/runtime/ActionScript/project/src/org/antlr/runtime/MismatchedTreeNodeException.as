package org.antlr.runtime {
	import org.antlr.runtime.tree.TreeNodeStream;
	
	public class MismatchedTreeNodeException extends RecognitionException {
		public var expecting:int;
	
		public function MismatchedTreeNodeException(expecting:int, input:TreeNodeStream) {
			super(input);
			this.expecting = expecting;
		}
	
		public function toString():String {
			return "MismatchedTreeNodeException("+unexpectedType+"!="+expecting+")";
		}
	}
}