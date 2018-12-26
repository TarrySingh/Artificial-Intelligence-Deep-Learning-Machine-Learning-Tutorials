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
package org.antlr.analysis;

import org.antlr.tool.GrammarAST;
import org.antlr.tool.Grammar;

public class PredicateLabel extends Label {
	/** A tree of semantic predicates from the grammar AST if label==SEMPRED.
	 *  In the NFA, labels will always be exactly one predicate, but the DFA
	 *  may have to combine a bunch of them as it collects predicates from
	 *  multiple NFA configurations into a single DFA state.
	 */
	protected SemanticContext semanticContext;
	
	/** Make a semantic predicate label */
	public PredicateLabel(GrammarAST predicateASTNode) {
		super(SEMPRED);
		this.semanticContext = new SemanticContext.Predicate(predicateASTNode);
	}

	/** Make a semantic predicates label */
	public PredicateLabel(SemanticContext semCtx) {
		super(SEMPRED);
		this.semanticContext = semCtx;
	}

	public int hashCode() {
		return semanticContext.hashCode();
	}

	public boolean equals(Object o) {
		if ( o==null ) {
			return false;
		}
		if ( this == o ) {
			return true; // equals if same object
		}
		if ( !(o instanceof PredicateLabel) ) {
			return false;
		}
		return semanticContext.equals(((PredicateLabel)o).semanticContext);
	}

	public boolean isSemanticPredicate() {
		return true;
	}

	public SemanticContext getSemanticContext() {
		return semanticContext;
	}

	public String toString() {
		return "{"+semanticContext+"}?";
	}

	public String toString(Grammar g) {
		return toString();
	}
}
