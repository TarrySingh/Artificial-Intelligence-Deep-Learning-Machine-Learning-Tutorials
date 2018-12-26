header {
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
package org.antlr.grammar.v2;
import java.util.*;
import org.antlr.analysis.*;
import org.antlr.misc.*;
import org.antlr.tool.*;
}

/** Build an NFA from a tree representing an ANTLR grammar. */
class TreeToNFAConverter extends TreeParser;

options {
	importVocab = ANTLR;
	ASTLabelType = "GrammarAST";
}

{

/** Factory used to create nodes and submachines */
protected NFAFactory factory = null;

/** Which NFA object are we filling in? */
protected NFA nfa = null;

/** Which grammar are we converting an NFA for? */
protected Grammar grammar = null;

protected String currentRuleName = null;

protected int outerAltNum = 0;
protected int blockLevel = 0;

public TreeToNFAConverter(Grammar g, NFA nfa, NFAFactory factory) {
	this();
	this.grammar = g;
	this.nfa = nfa;
	this.factory = factory;
}

/*
protected void init() {
    // define all the rule begin/end NFAStates to solve forward reference issues
    Collection rules = grammar.getRules();
    for (Iterator itr = rules.iterator(); itr.hasNext();) {
		Rule r = (Rule) itr.next();
        String ruleName = r.name;
        NFAState ruleBeginState = factory.newState();
        ruleBeginState.setDescription("rule "+ruleName+" start");
		ruleBeginState.enclosingRule = r;
        r.startState = ruleBeginState;
        NFAState ruleEndState = factory.newState();
        ruleEndState.setDescription("rule "+ruleName+" end");
        ruleEndState.setAcceptState(true);
		ruleEndState.enclosingRule = r;
        r.stopState = ruleEndState;
    }
}
*/

protected void addFollowTransition(String ruleName, NFAState following) {
     //System.out.println("adding follow link to rule "+ruleName);
     // find last link in FOLLOW chain emanating from rule
     Rule r = grammar.getRule(ruleName);
     NFAState end = r.stopState;
     while ( end.transition(1)!=null ) {
         end = (NFAState)end.transition(1).target;
     }
     if ( end.transition(0)!=null ) {
         // already points to a following node
         // gotta add another node to keep edges to a max of 2
         NFAState n = factory.newState();
         Transition e = new Transition(Label.EPSILON, n);
         end.addTransition(e);
         end = n;
     }
     Transition followEdge = new Transition(Label.EPSILON, following);
     end.addTransition(followEdge);
}

protected void finish() {
    List rules = new LinkedList();
    rules.addAll(grammar.getRules());
    int numEntryPoints = factory.build_EOFStates(rules);
    if ( numEntryPoints==0 ) {
        ErrorManager.grammarWarning(ErrorManager.MSG_NO_GRAMMAR_START_RULE,
                                   grammar,
                                   null,
                                   grammar.name);
    }
}

    public void reportError(RecognitionException ex) {
		Token token = null;
		if ( ex instanceof MismatchedTokenException ) {
			token = ((MismatchedTokenException)ex).token;
		}
		else if ( ex instanceof NoViableAltException ) {
			token = ((NoViableAltException)ex).token;
		}
        ErrorManager.syntaxError(
            ErrorManager.MSG_SYNTAX_ERROR,
            grammar,
            token,
            "buildnfa: "+ex.toString(),
            ex);
    }
}

grammar
    :   ( #( LEXER_GRAMMAR grammarSpec )
	    | #( PARSER_GRAMMAR grammarSpec )
	    | #( TREE_GRAMMAR grammarSpec )
	    | #( COMBINED_GRAMMAR grammarSpec )
	    )
        {finish();}
    ;

attrScope
	:	#( "scope" ID ACTION )
	;

grammarSpec
	:	ID
		(cmt:DOC_COMMENT)?
        ( #(OPTIONS .) )?
        ( #("import" .) )?
        ( #(TOKENS .) )?
        (attrScope)*
        (AMPERSAND)* // skip actions
        rules
	;

rules
    :   ( rule )+
    ;

rule
{
    StateCluster g=null;
    StateCluster b = null;
    String r=null;
}
    :   #( RULE id:ID {r=#id.getText();}
		{
        currentRuleName = r;
        factory.setCurrentRule(grammar.getLocallyDefinedRule(r));
        }
		(modifier)?
        (ARG (ARG_ACTION)?)
        (RET (ARG_ACTION)?)
		( OPTIONS )?
		( ruleScopeSpec )?
		   (AMPERSAND)*
		   {GrammarAST blk = (GrammarAST)_t;}
		   b=block
           (exceptionGroup)?
           EOR
           {
                if ( blk.getSetValue() !=null ) {
                    // if block comes back as a set not BLOCK, make it
                    // a single ALT block
                    b = factory.build_AlternativeBlockFromSet(b);
                }
				if ( Character.isLowerCase(r.charAt(0)) ||
					 grammar.type==Grammar.LEXER )
				{
					// attach start node to block for this rule
                    Rule thisR = grammar.getLocallyDefinedRule(r);
					NFAState start = thisR.startState;
					start.associatedASTNode = #id;
					start.addTransition(new Transition(Label.EPSILON, b.left));

					// track decision if > 1 alts
					if ( grammar.getNumberOfAltsForDecisionNFA(b.left)>1 ) {
						b.left.setDescription(grammar.grammarTreeToString(#rule,false));
						b.left.setDecisionASTNode(blk);
						int d = grammar.assignDecisionNumber( b.left );
						grammar.setDecisionNFA( d, b.left );
                    	grammar.setDecisionBlockAST(d, blk);
					}

					// hook to end of rule node
					NFAState end = thisR.stopState;
					b.right.addTransition(new Transition(Label.EPSILON,end));
				}
           }
         )
    ;

modifier
	:	"protected"
	|	"public"
	|	"private"
	|	"fragment"
	;

ruleScopeSpec
 	:	#( "scope" (ACTION)? ( ID )* )
 	;

block returns [StateCluster g = null]
{
    StateCluster a = null;
    List alts = new LinkedList();
    this.blockLevel++;
    if ( this.blockLevel==1 ) {this.outerAltNum=1;}
}
    :   {grammar.isValidSet(this,#block) &&
		 !currentRuleName.equals(Grammar.ARTIFICIAL_TOKENS_RULENAME)}?
		g=set
        {this.blockLevel--;}

    |	#( BLOCK ( OPTIONS )?
           ( a=alternative rewrite
             {
             alts.add(a);
             if ( this.blockLevel==1 ) {this.outerAltNum++;}
             }
           )+ 
           EOB
        )
        {g = factory.build_AlternativeBlock(alts);}
        {this.blockLevel--;}
    ;

alternative returns [StateCluster g=null]
{
    StateCluster e = null;
}
    :   #( ALT (e=element {g = factory.build_AB(g,e);} )+ )
        {
        if (g==null) { // if alt was a list of actions or whatever
            g = factory.build_Epsilon();
        }
        else {
        	factory.optimizeAlternative(g);
        }
        }
    ;

exceptionGroup
	:	( exceptionHandler )+ (finallyClause)?
	|	finallyClause
    ;

exceptionHandler
    :    #("catch" ARG_ACTION ACTION)
    ;

finallyClause
    :    #("finally" ACTION)
    ;

rewrite
	:	(
			{
			if ( grammar.getOption("output")==null ) {
				ErrorManager.grammarError(ErrorManager.MSG_REWRITE_OR_OP_WITH_NO_OUTPUT_OPTION,
										  grammar, #rewrite.token, currentRuleName);
			}
			}
			#( REWRITE (SEMPRED)? (ALT|TEMPLATE|ACTION|ETC) )
		)*
	;

element returns [StateCluster g=null]
    :   #(ROOT g=element)
    |   #(BANG g=element)
    |	#(ASSIGN ID g=element)
    |	#(PLUS_ASSIGN ID g=element)
    |   #(RANGE a:atom[null] b:atom[null])
        {g = factory.build_Range(grammar.getTokenType(#a.getText()),
                                 grammar.getTokenType(#b.getText()));}
    |   #(CHAR_RANGE c1:CHAR_LITERAL c2:CHAR_LITERAL)
        {
        if ( grammar.type==Grammar.LEXER ) {
        	g = factory.build_CharRange(#c1.getText(), #c2.getText());
        }
        }
    |   g=atom_or_notatom
    |   g=ebnf
    |   g=tree
    |   #( SYNPRED block )
    |   ACTION {g = factory.build_Action(#ACTION);}
    |   FORCED_ACTION {g = factory.build_Action(#FORCED_ACTION);}
    |   pred:SEMPRED {g = factory.build_SemanticPredicate(#pred);}
    |   spred:SYN_SEMPRED {g = factory.build_SemanticPredicate(#spred);}
    |   bpred:BACKTRACK_SEMPRED {g = factory.build_SemanticPredicate(#bpred);}
    |   gpred:GATED_SEMPRED {g = factory.build_SemanticPredicate(#gpred);}
    |   EPSILON {g = factory.build_Epsilon();}
    ;

ebnf returns [StateCluster g=null]
{
    StateCluster b = null;
    GrammarAST blk = #ebnf;
    if ( blk.getType()!=BLOCK ) {
    	blk = (GrammarAST)blk.getFirstChild();
    }
    GrammarAST eob = blk.getLastChild();
}
    :   {grammar.isValidSet(this,#ebnf)}? g=set

    |	b=block
        {
        // track decision if > 1 alts
        if ( grammar.getNumberOfAltsForDecisionNFA(b.left)>1 ) {
            b.left.setDescription(grammar.grammarTreeToString(blk,false));
            b.left.setDecisionASTNode(blk);
            int d = grammar.assignDecisionNumber( b.left );
            grammar.setDecisionNFA( d, b.left );
            grammar.setDecisionBlockAST(d, blk);
        }
        g = b;
        }
    |   #( OPTIONAL b=block )
        {
        if ( blk.getSetValue() !=null ) {
            // if block comes back SET not BLOCK, make it
            // a single ALT block
            b = factory.build_AlternativeBlockFromSet(b);
        }
        g = factory.build_Aoptional(b);
    	g.left.setDescription(grammar.grammarTreeToString(#ebnf,false));
        // there is always at least one alt even if block has just 1 alt
        int d = grammar.assignDecisionNumber( g.left );
		grammar.setDecisionNFA(d, g.left);
        grammar.setDecisionBlockAST(d, blk);
        g.left.setDecisionASTNode(#ebnf);
    	}
    |   #( CLOSURE b=block )
        {
        if (  blk.getSetValue() !=null ) {
            b = factory.build_AlternativeBlockFromSet(b);
        }
        g = factory.build_Astar(b);
		// track the loop back / exit decision point
    	b.right.setDescription("()* loopback of "+grammar.grammarTreeToString(#ebnf,false));
        int d = grammar.assignDecisionNumber( b.right );
		grammar.setDecisionNFA(d, b.right);
        grammar.setDecisionBlockAST(d, blk);
        b.right.setDecisionASTNode(eob);
        // make block entry state also have same decision for interpreting grammar
        NFAState altBlockState = (NFAState)g.left.transition(0).target;
        altBlockState.setDecisionASTNode(#ebnf);
        altBlockState.setDecisionNumber(d);
        g.left.setDecisionNumber(d); // this is the bypass decision (2 alts)
        g.left.setDecisionASTNode(#ebnf);
    	}
    |   #( POSITIVE_CLOSURE b=block )
        {
        if ( blk.getSetValue() !=null ) {
            b = factory.build_AlternativeBlockFromSet(b);
        }
        g = factory.build_Aplus(b);
        // don't make a decision on left edge, can reuse loop end decision
		// track the loop back / exit decision point
    	b.right.setDescription("()+ loopback of "+grammar.grammarTreeToString(#ebnf,false));
        int d = grammar.assignDecisionNumber( b.right );
		grammar.setDecisionNFA(d, b.right);
        grammar.setDecisionBlockAST(d, blk);
        b.right.setDecisionASTNode(eob);
        // make block entry state also have same decision for interpreting grammar
        NFAState altBlockState = (NFAState)g.left.transition(0).target;
        altBlockState.setDecisionASTNode(#ebnf);
        altBlockState.setDecisionNumber(d);
        }
    ;

tree returns [StateCluster g=null]
{
StateCluster e=null;
GrammarAST el=null;
StateCluster down=null, up=null;
}
	:   #( TREE_BEGIN
		   {el=(GrammarAST)_t;}
		   g=element
		   {
           down = factory.build_Atom(Label.DOWN, el);
           // TODO set following states for imaginary nodes?
           //el.followingNFAState = down.right;
		   g = factory.build_AB(g,down);
		   }
		   ( {el=(GrammarAST)_t;} e=element {g = factory.build_AB(g,e);} )*
		   {
           up = factory.build_Atom(Label.UP, el);
           //el.followingNFAState = up.right;
		   g = factory.build_AB(g,up);
		   // tree roots point at right edge of DOWN for LOOK computation later
		   #tree.NFATreeDownState = down.left;
		   }
		 )
    ;

atom_or_notatom returns [StateCluster g=null]
	:	g=atom[null]
	|	#(  n:NOT
            (  c:CHAR_LITERAL (ast1:ast_suffix)?
	           {
	            int ttype=0;
     			if ( grammar.type==Grammar.LEXER ) {
        			ttype = Grammar.getCharValueFromGrammarCharLiteral(#c.getText());
     			}
     			else {
        			ttype = grammar.getTokenType(#c.getText());
        		}
                IntSet notAtom = grammar.complement(ttype);
                if ( notAtom.isNil() ) {
                    ErrorManager.grammarError(ErrorManager.MSG_EMPTY_COMPLEMENT,
					  			              grammar,
								              #c.token,
									          #c.getText());
                }
	            g=factory.build_Set(notAtom,#n);
	           }
            |  t:TOKEN_REF (ast3:ast_suffix)?
	           {
	            int ttype=0;
                IntSet notAtom = null;
     			if ( grammar.type==Grammar.LEXER ) {
        			notAtom = grammar.getSetFromRule(this,#t.getText());
        	   		if ( notAtom==null ) {
                  		ErrorManager.grammarError(ErrorManager.MSG_RULE_INVALID_SET,
				  			              grammar,
							              #t.token,
								          #t.getText());
        	   		}
        	   		else {
	            		notAtom = grammar.complement(notAtom);
	            	}
     			}
     			else {
        			ttype = grammar.getTokenType(#t.getText());
	            	notAtom = grammar.complement(ttype);
        		}
               if ( notAtom==null || notAtom.isNil() ) {
                  ErrorManager.grammarError(ErrorManager.MSG_EMPTY_COMPLEMENT,
				  			              grammar,
							              #t.token,
								          #t.getText());
               }
	           g=factory.build_Set(notAtom,#n);
	           }
            |  g=set
	           {
	           GrammarAST stNode = (GrammarAST)n.getFirstChild();
               //IntSet notSet = grammar.complement(stNode.getSetValue());
               // let code generator complement the sets
               IntSet s = stNode.getSetValue();
               stNode.setSetValue(s);
               // let code gen do the complement again; here we compute
               // for NFA construction
               s = grammar.complement(s);
               if ( s.isNil() ) {
                  ErrorManager.grammarError(ErrorManager.MSG_EMPTY_COMPLEMENT,
				  			              grammar,
							              #n.token);
               }
	           g=factory.build_Set(s,#n);
	           }
            )
        	{#n.followingNFAState = g.right;}
         )
	;

atom[String scopeName] returns [StateCluster g=null]
    :   #( r:RULE_REF (rarg:ARG_ACTION)? (as1:ast_suffix)? )
        {
        NFAState start = grammar.getRuleStartState(scopeName,r.getText());
        if ( start!=null ) {
            Rule rr = grammar.getRule(scopeName,r.getText());
            g = factory.build_RuleRef(rr, start);
            r.followingNFAState = g.right;
            r.NFAStartState = g.left;
            if ( g.left.transition(0) instanceof RuleClosureTransition
            	 && grammar.type!=Grammar.LEXER )
            {
                addFollowTransition(r.getText(), g.right);
            }
            // else rule ref got inlined to a set
        }
        }

    |   #( t:TOKEN_REF  (targ:ARG_ACTION)? (as2:ast_suffix)? )
        {
        if ( grammar.type==Grammar.LEXER ) {
            NFAState start = grammar.getRuleStartState(scopeName,t.getText());
            if ( start!=null ) {
                Rule rr = grammar.getRule(scopeName,t.getText());
                g = factory.build_RuleRef(rr, start);
            	t.NFAStartState = g.left;
                // don't add FOLLOW transitions in the lexer;
                // only exact context should be used.
            }
        }
        else {
            g = factory.build_Atom(t);
            t.followingNFAState = g.right;
        }
        }

    |   #( c:CHAR_LITERAL  (as3:ast_suffix)? )
    	{
    	if ( grammar.type==Grammar.LEXER ) {
    		g = factory.build_CharLiteralAtom(c);
    	}
    	else {
            g = factory.build_Atom(c);
            c.followingNFAState = g.right;
    	}
    	}

    |   #( s:STRING_LITERAL  (as4:ast_suffix)? )
    	{
     	if ( grammar.type==Grammar.LEXER ) {
     		g = factory.build_StringLiteralAtom(s);
     	}
     	else {
             g = factory.build_Atom(s);
             s.followingNFAState = g.right;
     	}
     	}

    |   #( w:WILDCARD (as5:ast_suffix)? )
        {
        if ( nfa.grammar.type==Grammar.TREE_PARSER ) {
            g = factory.build_WildcardTree(#w);
        }
        else {
            g = factory.build_Wildcard(#w);
        }
        }

    |   #( DOT scope:ID g=atom[#scope.getText()] ) // scope override
	;

ast_suffix
	:	ROOT
	|	BANG
	;

set returns [StateCluster g=null]
{
IntSet elements=new IntervalSet();
#set.setSetValue(elements); // track set for use by code gen
}
	:	#( b:BLOCK
           (#(ALT (BACKTRACK_SEMPRED)? setElement[elements] EOA))+
           EOB
         )
        {
        g = factory.build_Set(elements,#b);
        #b.followingNFAState = g.right;
        #b.setSetValue(elements); // track set value of this block
        }
		//{System.out.println("set elements="+elements.toString(grammar));}
	;

setRule returns [IntSet elements=new IntervalSet()]
{IntSet s=null;}
	:	#( RULE id:ID (modifier)? ARG RET ( OPTIONS )? ( ruleScopeSpec )?
		   	(AMPERSAND)*
           	#( BLOCK ( OPTIONS )?
           	   ( #(ALT (BACKTRACK_SEMPRED)? setElement[elements] EOA) )+
           	   EOB
           	 )
           	(exceptionGroup)?
           	EOR
         )
    ;
    exception
    	catch[RecognitionException re] {throw re;}

setElement[IntSet elements]
{
    int ttype;
    IntSet ns=null;
    StateCluster gset;
}
    :   c:CHAR_LITERAL
        {
     	if ( grammar.type==Grammar.LEXER ) {
        	ttype = Grammar.getCharValueFromGrammarCharLiteral(c.getText());
     	}
     	else {
        	ttype = grammar.getTokenType(c.getText());
        }
        if ( elements.member(ttype) ) {
			ErrorManager.grammarError(ErrorManager.MSG_DUPLICATE_SET_ENTRY,
									  grammar,
									  #c.token,
									  #c.getText());
        }
        elements.add(ttype);
        }
    |   t:TOKEN_REF
        {
		if ( grammar.type==Grammar.LEXER ) {
			// recursively will invoke this rule to match elements in target rule ref
			IntSet ruleSet = grammar.getSetFromRule(this,#t.getText());
			if ( ruleSet==null ) {
				ErrorManager.grammarError(ErrorManager.MSG_RULE_INVALID_SET,
								  grammar,
								  #t.token,
								  #t.getText());
			}
			else {
				elements.addAll(ruleSet);
			}
		}
		else {
			ttype = grammar.getTokenType(t.getText());
			if ( elements.member(ttype) ) {
				ErrorManager.grammarError(ErrorManager.MSG_DUPLICATE_SET_ENTRY,
										  grammar,
										  #t.token,
										  #t.getText());
			}
			elements.add(ttype);
			}
        }

    |   s:STRING_LITERAL
        {
        ttype = grammar.getTokenType(s.getText());
        if ( elements.member(ttype) ) {
			ErrorManager.grammarError(ErrorManager.MSG_DUPLICATE_SET_ENTRY,
									  grammar,
									  #s.token,
									  #s.getText());
        }
        elements.add(ttype);
        }
    |	#(CHAR_RANGE c1:CHAR_LITERAL c2:CHAR_LITERAL)
    	{
     	if ( grammar.type==Grammar.LEXER ) {
	        int a = Grammar.getCharValueFromGrammarCharLiteral(c1.getText());
    	    int b = Grammar.getCharValueFromGrammarCharLiteral(c2.getText());
    		elements.addAll(IntervalSet.of(a,b));
     	}
    	}

	|   gset=set
        {
		Transition setTrans = gset.left.transition(0);
        elements.addAll(setTrans.label.getSet());
        }

    |   #(  NOT {ns=new IntervalSet();}
            setElement[ns]
            {
                IntSet not = grammar.complement(ns);
                elements.addAll(not);
            }
        )
    ;

/** Check to see if this block can be a set.  Can't have actions
 *  etc...  Also can't be in a rule with a rewrite as we need
 *  to track what's inside set for use in rewrite.
 */
testBlockAsSet
{
    int nAlts=0;
    Rule r = grammar.getLocallyDefinedRule(currentRuleName);
}
	:   #( BLOCK
           (   #(ALT (BACKTRACK_SEMPRED)? testSetElement {nAlts++;} EOA)
                {!r.hasRewrite(outerAltNum)}?
           )+
           EOB
        )
        {nAlts>1}? // set of 1 element is not good
	;
    exception
    	catch[RecognitionException re] {throw re;}

testSetRule
	:	#( RULE id:ID (modifier)? ARG RET ( OPTIONS )? ( ruleScopeSpec )?
		   	(AMPERSAND)*
            #( BLOCK
                ( #(ALT (BACKTRACK_SEMPRED)? testSetElement EOA) )+
                EOB
            )
           	(exceptionGroup)?
           	EOR
         )
    ;
    exception
    	catch[RecognitionException re] {throw re;}

/** Match just an element; no ast suffix etc.. */
testSetElement
{
AST r = _t;
}
    :   c:CHAR_LITERAL
    |   t:TOKEN_REF
        {
		if ( grammar.type==Grammar.LEXER ) {
	        Rule rule = grammar.getRule(#t.getText());
	        if ( rule==null ) {
	        	throw new RecognitionException("invalid rule");
	        }
			// recursively will invoke this rule to match elements in target rule ref
	        testSetRule(rule.tree);
		}
        }
    |   {grammar.type!=Grammar.LEXER}? s:STRING_LITERAL 
    |	#(CHAR_RANGE c1:CHAR_LITERAL c2:CHAR_LITERAL)
	|   testBlockAsSet
    |   #( NOT testSetElement )
    ;
    exception
     	catch[RecognitionException re] {throw re;}
