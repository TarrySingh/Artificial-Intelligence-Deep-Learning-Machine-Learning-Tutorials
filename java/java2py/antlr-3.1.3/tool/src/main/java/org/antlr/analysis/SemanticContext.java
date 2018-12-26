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
package org.antlr.analysis;

import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.StringTemplateGroup;
import org.antlr.codegen.CodeGenerator;
import org.antlr.grammar.v2.ANTLRParser;
import org.antlr.tool.GrammarAST;
import org.antlr.tool.Grammar;
import java.util.Set;
import java.util.HashSet;
import java.util.Iterator;

/** A binary tree structure used to record the semantic context in which
 *  an NFA configuration is valid.  It's either a single predicate or
 *  a tree representing an operation tree such as: p1&&p2 or p1||p2.
 *
 *  For NFA o-p1->o-p2->o, create tree AND(p1,p2).
 *  For NFA (1)-p1->(2)
 *           |       ^
 *           |       |
 *          (3)-p2----
 *  we will have to combine p1 and p2 into DFA state as we will be
 *  adding NFA configurations for state 2 with two predicates p1,p2.
 *  So, set context for combined NFA config for state 2: OR(p1,p2).
 *
 *  I have scoped the AND, NOT, OR, and Predicate subclasses of
 *  SemanticContext within the scope of this outer class.
 *
 *  July 7, 2006: TJP altered OR to be set of operands. the Binary tree
 *  made it really hard to reduce complicated || sequences to their minimum.
 *  Got huge repeated || conditions.
 */
public abstract class SemanticContext {
	/** Create a default value for the semantic context shared among all
	 *  NFAConfigurations that do not have an actual semantic context.
	 *  This prevents lots of if!=null type checks all over; it represents
	 *  just an empty set of predicates.
	 */
	public static final SemanticContext EMPTY_SEMANTIC_CONTEXT = new Predicate();

	/** Given a semantic context expression tree, return a tree with all
	 *  nongated predicates set to true and then reduced.  So p&&(q||r) would
	 *  return p&&r if q is nongated but p and r are gated.
	 */
	public abstract SemanticContext getGatedPredicateContext();

	/** Generate an expression that will evaluate the semantic context,
	 *  given a set of output templates.
	 */
	public abstract StringTemplate genExpr(CodeGenerator generator,
										   StringTemplateGroup templates,
										   DFA dfa);

	public abstract boolean isSyntacticPredicate();

	/** Notify the indicated grammar of any syn preds used within this context */
	public void trackUseOfSyntacticPredicates(Grammar g) {
	}

	public static class Predicate extends SemanticContext {
		/** The AST node in tree created from the grammar holding the predicate */
		public GrammarAST predicateAST;

		/** Is this a {...}?=> gating predicate or a normal disambiguating {..}?
		 *  If any predicate in expression is gated, then expression is considered
		 *  gated.
		 *
		 *  The simple Predicate object's predicate AST's type is used to set
		 *  gated to true if type==GATED_SEMPRED.
		 */
		protected boolean gated = false;

		/** syntactic predicates are converted to semantic predicates
		 *  but synpreds are generated slightly differently.
		 */
		protected boolean synpred = false;

		public static final int INVALID_PRED_VALUE = -1;
		public static final int FALSE_PRED = 0;
		public static final int TRUE_PRED = 1;

		/** sometimes predicates are known to be true or false; we need
		 *  a way to represent this without resorting to a target language
		 *  value like true or TRUE.
		 */
		protected int constantValue = INVALID_PRED_VALUE;

		public Predicate() {
			predicateAST = new GrammarAST();
			this.gated=false;
		}

		public Predicate(GrammarAST predicate) {
			this.predicateAST = predicate;
			this.gated =
				predicate.getType()==ANTLRParser.GATED_SEMPRED ||
				predicate.getType()==ANTLRParser.SYN_SEMPRED ;
			this.synpred =
				predicate.getType()==ANTLRParser.SYN_SEMPRED ||
				predicate.getType()==ANTLRParser.BACKTRACK_SEMPRED;
		}

		public Predicate(Predicate p) {
			this.predicateAST = p.predicateAST;
			this.gated = p.gated;
			this.synpred = p.synpred;
			this.constantValue = p.constantValue;
		}

		/** Two predicates are the same if they are literally the same
		 *  text rather than same node in the grammar's AST.
		 *  Or, if they have the same constant value, return equal.
		 *  As of July 2006 I'm not sure these are needed.
		 */
		public boolean equals(Object o) {
			if ( !(o instanceof Predicate) ) {
				return false;
			}
			return predicateAST.getText().equals(((Predicate)o).predicateAST.getText());
		}

		public int hashCode() {
			if ( predicateAST ==null ) {
				return 0;
			}
			return predicateAST.getText().hashCode();
		}

		public StringTemplate genExpr(CodeGenerator generator,
									  StringTemplateGroup templates,
									  DFA dfa)
		{
			StringTemplate eST = null;
			if ( templates!=null ) {
				if ( synpred ) {
					eST = templates.getInstanceOf("evalSynPredicate");
				}
				else {
					eST = templates.getInstanceOf("evalPredicate");
					generator.grammar.decisionsWhoseDFAsUsesSemPreds.add(dfa);
				}
				String predEnclosingRuleName = predicateAST.enclosingRuleName;
				/*
				String decisionEnclosingRuleName =
					dfa.getNFADecisionStartState().getEnclosingRule();
				// if these rulenames are diff, then pred was hoisted out of rule
				// Currently I don't warn you about this as it could be annoying.
				// I do the translation anyway.
				*/
				//eST.setAttribute("pred", this.toString());
				if ( generator!=null ) {
					eST.setAttribute("pred",
									 generator.translateAction(predEnclosingRuleName,predicateAST));
				}
			}
			else {
				eST = new StringTemplate("$pred$");
				eST.setAttribute("pred", this.toString());
				return eST;
			}
			if ( generator!=null ) {
				String description =
					generator.target.getTargetStringLiteralFromString(this.toString());
				eST.setAttribute("description", description);
			}
			return eST;
		}

		public SemanticContext getGatedPredicateContext() {
			if ( gated ) {
				return this;
			}
			return null;
		}

		public boolean isSyntacticPredicate() {
			return predicateAST !=null &&
				( predicateAST.getType()==ANTLRParser.SYN_SEMPRED ||
				  predicateAST.getType()==ANTLRParser.BACKTRACK_SEMPRED );
		}

		public void trackUseOfSyntacticPredicates(Grammar g) {
			if ( synpred ) {
				g.synPredNamesUsedInDFA.add(predicateAST.getText());
			}
		}

		public String toString() {
			if ( predicateAST ==null ) {
				return "<nopred>";
			}
			return predicateAST.getText();
		}
	}

	public static class TruePredicate extends Predicate {
		public TruePredicate() {
			super();
			this.constantValue = TRUE_PRED;
		}

		public StringTemplate genExpr(CodeGenerator generator,
									  StringTemplateGroup templates,
									  DFA dfa)
		{
			if ( templates!=null ) {
				return templates.getInstanceOf("true");
			}
			return new StringTemplate("true");
		}

		public String toString() {
			return "true"; // not used for code gen, just DOT and print outs
		}
	}

	/*
	public static class FalsePredicate extends Predicate {
		public FalsePredicate() {
			super();
			this.constantValue = FALSE_PRED;
		}
		public StringTemplate genExpr(CodeGenerator generator,
									  StringTemplateGroup templates,
									  DFA dfa)
		{
			if ( templates!=null ) {
				return templates.getInstanceOf("false");
			}
			return new StringTemplate("false");
		}
		public String toString() {
			return "false"; // not used for code gen, just DOT and print outs
		}
	}
	*/

	public static class AND extends SemanticContext {
		protected SemanticContext left,right;
		public AND(SemanticContext a, SemanticContext b) {
			this.left = a;
			this.right = b;
		}
		public StringTemplate genExpr(CodeGenerator generator,
									  StringTemplateGroup templates,
									  DFA dfa)
		{
			StringTemplate eST = null;
			if ( templates!=null ) {
				eST = templates.getInstanceOf("andPredicates");
			}
			else {
				eST = new StringTemplate("($left$&&$right$)");
			}
			eST.setAttribute("left", left.genExpr(generator,templates,dfa));
			eST.setAttribute("right", right.genExpr(generator,templates,dfa));
			return eST;
		}
		public SemanticContext getGatedPredicateContext() {
			SemanticContext gatedLeft = left.getGatedPredicateContext();
			SemanticContext gatedRight = right.getGatedPredicateContext();
			if ( gatedLeft==null ) {
				return gatedRight;
			}
			if ( gatedRight==null ) {
				return gatedLeft;
			}
			return new AND(gatedLeft, gatedRight);
		}
		public boolean isSyntacticPredicate() {
			return left.isSyntacticPredicate()||right.isSyntacticPredicate();
		}
		public void trackUseOfSyntacticPredicates(Grammar g) {
			left.trackUseOfSyntacticPredicates(g);
			right.trackUseOfSyntacticPredicates(g);
		}
		public String toString() {
			return "("+left+"&&"+right+")";
		}
	}

	public static class OR extends SemanticContext {
		protected Set operands;
		public OR(SemanticContext a, SemanticContext b) {
			operands = new HashSet();
			if ( a instanceof OR ) {
				operands.addAll(((OR)a).operands);
			}
			else if ( a!=null ) {
				operands.add(a);
			}
			if ( b instanceof OR ) {
				operands.addAll(((OR)b).operands);
			}
			else if ( b!=null ) {
				operands.add(b);
			}
		}
		public StringTemplate genExpr(CodeGenerator generator,
									  StringTemplateGroup templates,
									  DFA dfa)
		{
			StringTemplate eST = null;
			if ( templates!=null ) {
				eST = templates.getInstanceOf("orPredicates");
			}
			else {
				eST = new StringTemplate("($first(operands)$$rest(operands):{o | ||$o$}$)");
			}
			for (Iterator it = operands.iterator(); it.hasNext();) {
				SemanticContext semctx = (SemanticContext) it.next();
				eST.setAttribute("operands", semctx.genExpr(generator,templates,dfa));
			}
			return eST;
		}
		public SemanticContext getGatedPredicateContext() {
			SemanticContext result = null;
			for (Iterator it = operands.iterator(); it.hasNext();) {
				SemanticContext semctx = (SemanticContext) it.next();
				SemanticContext gatedPred = semctx.getGatedPredicateContext();
				if ( gatedPred!=null ) {
					result = or(result, gatedPred);
					// result = new OR(result, gatedPred);
				}
			}
			return result;
		}
		public boolean isSyntacticPredicate() {
			for (Iterator it = operands.iterator(); it.hasNext();) {
				SemanticContext semctx = (SemanticContext) it.next();
				if ( semctx.isSyntacticPredicate() ) {
					return true;
				}
			}
			return false;
		}
		public void trackUseOfSyntacticPredicates(Grammar g) {
			for (Iterator it = operands.iterator(); it.hasNext();) {
				SemanticContext semctx = (SemanticContext) it.next();
				semctx.trackUseOfSyntacticPredicates(g);
			}
		}
		public String toString() {
			StringBuffer buf = new StringBuffer();
			buf.append("(");
			int i = 0;
			for (Iterator it = operands.iterator(); it.hasNext();) {
				SemanticContext semctx = (SemanticContext) it.next();
				if ( i>0 ) {
					buf.append("||");
				}
				buf.append(semctx.toString());
				i++;
			}
			buf.append(")");
			return buf.toString();
		}
	}

	public static class NOT extends SemanticContext {
		protected SemanticContext ctx;
		public NOT(SemanticContext ctx) {
			this.ctx = ctx;
		}
		public StringTemplate genExpr(CodeGenerator generator,
									  StringTemplateGroup templates,
									  DFA dfa)
		{
			StringTemplate eST = null;
			if ( templates!=null ) {
				eST = templates.getInstanceOf("notPredicate");
			}
			else {
				eST = new StringTemplate("?!($pred$)");
			}
			eST.setAttribute("pred", ctx.genExpr(generator,templates,dfa));
			return eST;
		}
		public SemanticContext getGatedPredicateContext() {
			SemanticContext p = ctx.getGatedPredicateContext();
			if ( p==null ) {
				return null;
			}
			return new NOT(p);
		}
		public boolean isSyntacticPredicate() {
			return ctx.isSyntacticPredicate();
		}
		public void trackUseOfSyntacticPredicates(Grammar g) {
			ctx.trackUseOfSyntacticPredicates(g);
		}

		public boolean equals(Object object) {
			if ( !(object instanceof NOT) ) {
				return false;
			}
			return this.ctx.equals(((NOT)object).ctx);
		}

		public String toString() {
			return "!("+ctx+")";
		}
	}

	public static SemanticContext and(SemanticContext a, SemanticContext b) {
		//System.out.println("AND: "+a+"&&"+b);
		if ( a==EMPTY_SEMANTIC_CONTEXT || a==null ) {
			return b;
		}
		if ( b==EMPTY_SEMANTIC_CONTEXT || b==null ) {
			return a;
		}
		if ( a.equals(b) ) {
			return a; // if same, just return left one
		}
		//System.out.println("## have to AND");
		return new AND(a,b);
	}

	public static SemanticContext or(SemanticContext a, SemanticContext b) {
		//System.out.println("OR: "+a+"||"+b);
		if ( a==EMPTY_SEMANTIC_CONTEXT || a==null ) {
			return b;
		}
		if ( b==EMPTY_SEMANTIC_CONTEXT || b==null ) {
			return a;
		}
		if ( a instanceof TruePredicate ) {
			return a;
		}
		if ( b instanceof TruePredicate ) {
			return b;
		}
		if ( a instanceof NOT && b instanceof Predicate ) {
			NOT n = (NOT)a;
			// check for !p||p
			if ( n.ctx.equals(b) ) {
				return new TruePredicate();
			}
		}
		else if ( b instanceof NOT && a instanceof Predicate ) {
			NOT n = (NOT)b;
			// check for p||!p
			if ( n.ctx.equals(a) ) {
				return new TruePredicate();
			}
		}
		else if ( a.equals(b) ) {
			return a;
		}
		//System.out.println("## have to OR");
		return new OR(a,b);
	}

	public static SemanticContext not(SemanticContext a) {
		return new NOT(a);
	}

}
