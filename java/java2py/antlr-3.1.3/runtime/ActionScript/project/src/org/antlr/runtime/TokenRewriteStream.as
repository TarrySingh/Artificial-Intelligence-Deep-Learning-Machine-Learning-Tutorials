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
	import flash.utils.getQualifiedClassName;
	
	
	/** Useful for dumping out the input stream after doing some
	 *  augmentation or other manipulations.
	 *
	 *  You can insert stuff, replace, and delete chunks.  Note that the
	 *  operations are done lazily--only if you convert the buffer to a
	 *  String.  This is very efficient because you are not moving data around
	 *  all the time.  As the buffer of tokens is converted to strings, the
	 *  toString() method(s) check to see if there is an operation at the
	 *  current index.  If so, the operation is done and then normal String
	 *  rendering continues on the buffer.  This is like having multiple Turing
	 *  machine instruction streams (programs) operating on a single input tape. :)
	 *
	 *  Since the operations are done lazily at toString-time, operations do not
	 *  screw up the token index values.  That is, an insert operation at token
	 *  index i does not change the index values for tokens i+1..n-1.
	 *
	 *  Because operations never actually alter the buffer, you may always get
	 *  the original token stream back without undoing anything.  Since
	 *  the instructions are queued up, you can easily simulate transactions and
	 *  roll back any changes if there is an error just by removing instructions.
	 *  For example,
	 *
	 *   var input:CharStream = new ANTLRFileStream("input");
	 *   var lex:TLexer = new TLexer(input);
	 *   var tokens:TokenRewriteStream = new TokenRewriteStream(lex);
	 *   var parser:T = new T(tokens);
	 *   parser.startRule();
	 *
	 * 	 Then in the rules, you can execute
	 *      var t:Token t, u:Token;
	 *      ...
	 *      input.insertAfter(t, "text to put after t");}
	 * 		input.insertAfter(u, "text after u");}
	 * 		trace(tokens.toString());
	 *
	 *  Actually, you have to cast the 'input' to a TokenRewriteStream. :(
	 *
	 *  You can also have multiple "instruction streams" and get multiple
	 *  rewrites from a single pass over the input.  Just name the instruction
	 *  streams and use that name again when printing the buffer.  This could be
	 *  useful for generating a C file and also its header file--all from the
	 *  same buffer:
	 *
	 *      tokens.insertAfter("pass1", t, "text to put after t");}
	 * 		tokens.insertAfter("pass2", u, "text after u");}
	 * 		trace(tokens.toString("pass1"));
	 * 		trace(tokens.toString("pass2"));
	 *
	 *  If you don't use named rewrite streams, a "default" stream is used as
	 *  the first example shows.
	 */
	public class TokenRewriteStream extends CommonTokenStream {
		public static const DEFAULT_PROGRAM_NAME:String = "default";
		public static const MIN_TOKEN_INDEX:int = 0;
	
		/** You may have multiple, named streams of rewrite operations.
		 *  I'm calling these things "programs."
		 *  Maps String (name) -> rewrite (List)
		 */
		protected var programs:Object = new Object();
	
		/** Map String (program name) -> Integer index */
		protected var lastRewriteTokenIndexes:Object = new Object();
	
		public function TokenRewriteStream(tokenSource:TokenSource = null, channel:int = TokenConstants.DEFAULT_CHANNEL) {
			super(tokenSource, channel);
			programs[DEFAULT_PROGRAM_NAME] = new Array();
		}
	
	    /** Rollback the instruction stream for a program so that
		 *  the indicated instruction (via instructionIndex) is no
		 *  longer in the stream.  UNTESTED!
		 */
		public function rollback(instructionIndex:int, programName:String = DEFAULT_PROGRAM_NAME):void {
			var isn:Array = programs[programName] as Array;
			if ( isn != null ) {
				programs[programName] = isn.slice(MIN_TOKEN_INDEX,instructionIndex);
			}
		}
	
		/** Reset the program so that no instructions exist */
		public function deleteProgram(programName:String = DEFAULT_PROGRAM_NAME):void {
			rollback(MIN_TOKEN_INDEX, programName);
		}	
	
		public function insertAfterToken(t:Token, text:Object, programName:String = DEFAULT_PROGRAM_NAME):void {
			insertAfter(t.tokenIndex, text, programName);
		}
	
		public function insertAfter(index:int, text:Object, programName:String = DEFAULT_PROGRAM_NAME):void {
			// to insert after, just insert before next index (even if past end)
			insertBefore(index+1, text, programName);
		}
	
		public function insertBeforeToken(t:Token, text:Object, programName:String = DEFAULT_PROGRAM_NAME):void {
			insertBefore(t.tokenIndex, text, programName);
		}
	
		public function insertBefore(index:int, text:Object, programName:String = DEFAULT_PROGRAM_NAME):void {
			var op:RewriteOperation = new InsertBeforeOp(index,text);
			var rewrites:Array = getProgram(programName);
			op.instructionIndex = rewrites.length;
			rewrites.push(op);
		}
			
		public function replace(index:int, text:Object, programName:String = DEFAULT_PROGRAM_NAME):void {
			replaceRange(index, index, text, programName);
		}
	
		public function replaceRange(fromIndex:int, toIndex:int, text:Object, programName:String = DEFAULT_PROGRAM_NAME):void {
			if ( fromIndex > toIndex || fromIndex<0 || toIndex<0 || toIndex >= tokens.length ) {
				throw new Error("replace: range invalid: "+fromIndex+".."+toIndex+"(size="+tokens.length+")");
			}
			var op:RewriteOperation = new ReplaceOp(fromIndex, toIndex, text);
			var rewrites:Array = getProgram(programName);
			op.instructionIndex = rewrites.length;
			rewrites.push(op);
		}
	
		public function replaceToken(indexT:Token, text:Object, programName:String = DEFAULT_PROGRAM_NAME):void {
			replaceTokenRange(indexT, indexT, text, programName);
		}
	
		public function replaceTokenRange(fromToken:Token, toToken:Token, text:Object, programName:String = DEFAULT_PROGRAM_NAME):void {
			replaceRange(fromToken.tokenIndex, toToken.tokenIndex, text, programName);
		}
	
		public function remove(index:int, programName:String = DEFAULT_PROGRAM_NAME):void {
			removeRange(index, index, programName);
		}
	
		public function removeRange(fromIndex:int, toIndex:int, programName:String = DEFAULT_PROGRAM_NAME):void {
			replaceRange(fromIndex, toIndex, null, programName);
		}
	
		public function removeToken(token:Token, programName:String = DEFAULT_PROGRAM_NAME):void {
			removeTokenRange(token, token, programName);
		}
	
		public function removeTokenRange(fromToken:Token, toToken:Token, programName:String = DEFAULT_PROGRAM_NAME):void {
			replaceTokenRange(fromToken, toToken, null, programName);
		}
	
		public function getLastRewriteTokenIndex(programName:String = DEFAULT_PROGRAM_NAME):int {
			var i:* = lastRewriteTokenIndexes[programName];
			if ( i == undefined ) {
				return -1;
			}
			return i as int;
		}
	
		protected function setLastRewriteTokenIndex(programName:String, i:int):void {
			lastRewriteTokenIndexes[programName] = i;
		}
	
		protected function getProgram(name:String):Array {
			var isn:Array = programs[name] as Array;
			if ( isn==null ) {
				isn = initializeProgram(name);
			}
			return isn;
		}
	
		private function initializeProgram(name:String):Array {
			var isn:Array = new Array();
			programs[name] =  isn;
			return isn;
		}
	
		public function toOriginalString():String {
			return toOriginalStringWithRange(MIN_TOKEN_INDEX, size-1);
		}
	
		public function toOriginalStringWithRange(start:int, end:int):String {
			var buf:String = new String();
			for (var i:int=start; i>=MIN_TOKEN_INDEX && i<=end && i<tokens.length; i++) {
				buf += getToken(i).text;
			}
			return buf.toString();
		}
	
		public override function toString():String {
			return toStringWithRange(MIN_TOKEN_INDEX, size-1);
		}
	
		public override function toStringWithRange(start:int, end:int):String {
			return toStringWithRangeAndProgram(start, end, DEFAULT_PROGRAM_NAME);
		}
		
		public function toStringWithRangeAndProgram(start:int, end:int, programName:String):String {
			var rewrites:Array = programs[programName] as Array;
			
			// ensure start/end are in range
	        if ( end > tokens.length-1 ) end = tokens.length-1;
	        if ( start < 0 ) start = 0;
        
			if ( rewrites==null || rewrites.length==0 ) {
				return toOriginalStringWithRange(start,end); // no instructions to execute
			}
			var state:RewriteState = new RewriteState();
			state.tokens = tokens;
			
			// First, optimize instruction stream
	        var indexToOp:Array = reduceToSingleOperationPerIndex(rewrites);

	        // Walk buffer, executing instructions and emitting tokens
	        var i:int = start;
	        while ( i <= end && i < tokens.length ) {
	            var op:RewriteOperation = RewriteOperation(indexToOp[i]);
	            indexToOp[i] = undefined; // remove so any left have index size-1
	            var t:Token = Token(tokens[i]);
	            if ( op==null ) {
	                // no operation at that index, just dump token
	                state.buf += t.text;
	                i++; // move to next token
	            }
	            else {
	                i = op.execute(state); // execute operation and skip
	            }
	        }
	        
	        // include stuff after end if it's last index in buffer
	        // So, if they did an insertAfter(lastValidIndex, "foo"), include
	        // foo if end==lastValidIndex.
	        if ( end==tokens.length-1 ) {
	            // Scan any remaining operations after last token
	            // should be included (they will be inserts).
	            for each (op in indexToOp) {
	            	if (op == null) continue;
	                if ( op.index >= tokens.length-1 ) state.buf += op.text;
	            }
	        }
	        
	        return state.buf;
		}
		
	    /** We need to combine operations and report invalid operations (like
	     *  overlapping replaces that are not completed nested).  Inserts to
	     *  same index need to be combined etc...   Here are the cases:
	     *
	     *  I.i.u I.j.v                             leave alone, nonoverlapping
	     *  I.i.u I.i.v                             combine: Iivu
	     *
	     *  R.i-j.u R.x-y.v | i-j in x-y            delete first R
	     *  R.i-j.u R.i-j.v                         delete first R
	     *  R.i-j.u R.x-y.v | x-y in i-j            ERROR
	     *  R.i-j.u R.x-y.v | boundaries overlap    ERROR
	     *
	     *  I.i.u R.x-y.v | i in x-y                delete I
	     *  I.i.u R.x-y.v | i not in x-y            leave alone, nonoverlapping
	     *  R.x-y.v I.i.u | i in x-y                ERROR
	     *  R.x-y.v I.x.u                           R.x-y.uv (combine, delete I)
	     *  R.x-y.v I.i.u | i not in x-y            leave alone, nonoverlapping
	     *
	     *  I.i.u = insert u before op @ index i
	     *  R.x-y.u = replace x-y indexed tokens with u
	     *
	     *  First we need to examine replaces.  For any replace op:
	     *
	     *      1. wipe out any insertions before op within that range.
	     *      2. Drop any replace op before that is contained completely within
	     *         that range.
	     *      3. Throw exception upon boundary overlap with any previous replace.
	     *
	     *  Then we can deal with inserts:
	     *
	     *      1. for any inserts to same index, combine even if not adjacent.
	     *      2. for any prior replace with same left boundary, combine this
	     *         insert with replace and delete this replace.
	     *      3. throw exception if index in same range as previous replace
	     *
	     *  Don't actually delete; make op null in list. Easier to walk list.
	     *  Later we can throw as we add to index -> op map.
	     *
	     *  Note that I.2 R.2-2 will wipe out I.2 even though, technically, the
	     *  inserted stuff would be before the replace range.  But, if you
	     *  add tokens in front of a method body '{' and then delete the method
	     *  body, I think the stuff before the '{' you added should disappear too.
	     *
	     *  Return a map from token index to operation.
	     */
	    protected function reduceToSingleOperationPerIndex(rewrites:Array):Array {
	        //System.out.println("rewrites="+rewrites);
	
	        // WALK REPLACES
	        for (var i:int = 0; i < rewrites.length; i++) {
	            var op:RewriteOperation = RewriteOperation(rewrites[i]);
	            if ( op==null ) continue;
	            if ( !(op is ReplaceOp) ) continue;
	            var rop:ReplaceOp = ReplaceOp(rewrites[i]);
	            // Wipe prior inserts within range
	            var inserts:Array = getKindOfOps(rewrites, InsertBeforeOp, i);
	            for (var j:int = 0; j < inserts.length; j++) {
	                var iop:InsertBeforeOp = InsertBeforeOp(inserts[j]);
	                if ( iop.index >= rop.index && iop.index <= rop.lastIndex ) {
	                    rewrites[iop.instructionIndex] = null;  // delete insert as it's a no-op.
	                }
	            }
	            // Drop any prior replaces contained within
	            var prevReplaces:Array = getKindOfOps(rewrites, ReplaceOp, i);
	            for (j = 0; j < prevReplaces.length; j++) {
	                var prevRop:ReplaceOp = ReplaceOp(prevReplaces[j]);
	                if ( prevRop.index>=rop.index && prevRop.lastIndex <= rop.lastIndex ) {
	                    rewrites[prevRop.instructionIndex] = null;  // delete replace as it's a no-op.
	                    continue;
	                }
	                // throw exception unless disjoint or identical
	                var disjoint:Boolean =
	                    prevRop.lastIndex<rop.index || prevRop.index > rop.lastIndex;
	                var same:Boolean =
	                    prevRop.index==rop.index && prevRop.lastIndex==rop.lastIndex;
	                if ( !disjoint && !same ) {
	                    throw new Error("replace op boundaries of "+rop+
	                                                       " overlap with previous "+prevRop);
	                }
	            }
	        }
	
	        // WALK INSERTS
	        for (i = 0; i < rewrites.length; i++) {
	            op = RewriteOperation(rewrites[i]);
	            if ( op==null ) continue;
	            if ( !(op is InsertBeforeOp) ) continue;
	            iop = InsertBeforeOp(rewrites[i]);
	            // combine current insert with prior if any at same index
	            var prevInserts:Array = getKindOfOps(rewrites, InsertBeforeOp, i);
	            for (j = 0; j < prevInserts.length; j++) {
	                var prevIop:InsertBeforeOp = InsertBeforeOp(prevInserts[j]);
	                if ( prevIop.index == iop.index ) { // combine objects
	                    // convert to strings...we're in process of toString'ing
	                    // whole token buffer so no lazy eval issue with any templates
	                    iop.text = catOpText(iop.text,prevIop.text);
	                    rewrites[prevIop.instructionIndex] = null;  // delete redundant prior insert
	                }
	            }
	            // look for replaces where iop.index is in range; error
	            prevReplaces = getKindOfOps(rewrites, ReplaceOp, i);
	            for (j = 0; j < prevReplaces.length; j++) {
	                rop = ReplaceOp(prevReplaces[j]);
	                if ( iop.index == rop.index ) {
	                    rop.text = catOpText(iop.text,rop.text);
	                    rewrites[i] = null;  // delete current insert
	                    continue;
	                }
	                if ( iop.index >= rop.index && iop.index <= rop.lastIndex ) {
	                    throw new Error("insert op "+iop+
	                                                       " within boundaries of previous "+rop);
	                }
	            }
	        }
	        // System.out.println("rewrites after="+rewrites);
	        var m:Array = new Array();
	        for (i = 0; i < rewrites.length; i++) {
	            op = RewriteOperation(rewrites[i]);
	            if ( op==null ) continue; // ignore deleted ops
	            if ( m[op.index] != undefined ) {
	                throw new Error("should only be one op per index");
	            }
	            m[op.index] = op;
	        }
	        //System.out.println("index to op: "+m);
	        return m;
	    }
	
	    protected function catOpText(a:Object, b:Object):String {
	        var x:String = "";
	        var y:String = "";
	        if ( a!=null ) x = a.toString();
	        if ( b!=null ) y = b.toString();
	        return x+y;
	    }
	    
	    /** Get all operations before an index of a particular kind */
	    protected function getKindOfOps(rewrites:Array, kind:Class, before:int = -1):Array {
	    	if (before == -1) {
	    		before = rewrites.length;
	    	}
	    	var ops:Array = new Array();
	        for (var i:int=0; i<before && i<rewrites.length; i++) {
	            var op:RewriteOperation = RewriteOperation(rewrites[i]);
	            if ( op==null ) continue; // ignore deleted
	            if ( getQualifiedClassName(op) == getQualifiedClassName(kind) ) ops.push(op);
	        }       
	        return ops;
	    }


		public function toDebugString():String {
			return toDebugStringWithRange(MIN_TOKEN_INDEX, size-1);
		}
	
		public function toDebugStringWithRange(start:int, end:int):String {
			var buf:String = new String();
			for (var i:int=start; i>=MIN_TOKEN_INDEX && i<=end && i<tokens.length; i++) {
				buf += getToken(i);
			}
			return buf;
		}
		

	}
}
	import org.antlr.runtime.Token;
	

// Define the rewrite operation hierarchy

class RewriteState {
	public var buf:String = new String();
	public var tokens:Array;
}

class RewriteOperation {
	/** What index into rewrites List are we? */
    internal var instructionIndex:int;
    /** Token buffer index. */
	public var index:int;
	internal var text:Object;
	public function RewriteOperation(index:int, text:Object) {
		this.index = index;
		this.text = text;
	}
	/** Execute the rewrite operation by possibly adding to the buffer.
	 *  Return the index of the next token to operate on.
	 */
	public function execute(state:RewriteState):int {
		return index;
	}
}

class InsertBeforeOp extends RewriteOperation {
	public function InsertBeforeOp(index:int, text:Object) {
		super(index,text);
	}
	
	public override function execute(state:RewriteState):int {
		state.buf += text;
		state.buf += Token(state.tokens[index]).text;
		return index + 1;
	}
	
	public function toString():String {
		return "<InsertBeforeOp@" + index + ":\"" + text + "\">";
	}
}

/** I'm going to try replacing range from x..y with (y-x)+1 ReplaceOp
 *  instructions.
 */
class ReplaceOp extends RewriteOperation {
	public var lastIndex:int;
	
	public function ReplaceOp(fromIndex:int, toIndex:int, text:Object) {
		super(fromIndex, text);
		lastIndex = toIndex;
	}
	
	public override function execute(state:RewriteState):int {
		if ( text!=null ) {
			state.buf += text;
		}
		return lastIndex+1;
	}
	
	public function toString():String {
		return "<ReplaceOp@" + index + ".." + lastIndex + ":\"" + text + "\">";
	}
}

class DeleteOp extends ReplaceOp {
	public function DeleteOp(fromIndex:int, toIndex:int) {
		super(fromIndex, toIndex, null);
	}
	
	public override function toString():String {
		return "<DeleteOp@" + index + ".." + lastIndex + ">";
	}
}
