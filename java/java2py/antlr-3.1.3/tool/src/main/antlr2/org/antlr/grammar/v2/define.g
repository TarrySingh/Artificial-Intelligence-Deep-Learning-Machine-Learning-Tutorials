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
	import org.antlr.misc.*;
    import org.antlr.tool.*;
}

class DefineGrammarItemsWalker extends TreeParser;

options {
	importVocab = ANTLR;
	ASTLabelType = "GrammarAST";
    codeGenBitsetTestThreshold=999;
}

{ 

protected Grammar grammar;
protected GrammarAST root;
protected String currentRuleName;
protected GrammarAST currentRewriteBlock;
protected GrammarAST currentRewriteRule;
protected int outerAltNum = 0;
protected int blockLevel = 0;

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
            "define: "+ex.toString(),
            ex);
    }

	protected void finish() {
		trimGrammar();
	}

	/** Remove any lexer rules from a COMBINED; already passed to lexer */
	protected void trimGrammar() {
		if ( grammar.type!=Grammar.COMBINED ) {
			return;
		}
		// form is (header ... ) ( grammar ID (scope ...) ... ( rule ... ) ( rule ... ) ... )
		GrammarAST p = root;
		// find the grammar spec
		while ( !p.getText().equals("grammar") ) {
			p = (GrammarAST)p.getNextSibling();
		}
		p = (GrammarAST)p.getFirstChild(); // jump down to first child of grammar
		// look for first RULE def
		GrammarAST prev = p; // points to the ID (grammar name)
		while ( p.getType()!=RULE ) {
			prev = p;
			p = (GrammarAST)p.getNextSibling();
		}
		// prev points at last node before first rule subtree at this point
		while ( p!=null ) {
			String ruleName = p.getFirstChild().getText();
			//System.out.println("rule "+ruleName+" prev="+prev.getText());
			if ( Character.isUpperCase(ruleName.charAt(0)) ) {
				// remove lexer rule
				prev.setNextSibling(p.getNextSibling());
			}
			else {
				prev = p; // non-lexer rule; move on
			}
			p = (GrammarAST)p.getNextSibling();
		}
		//System.out.println("root after removal is: "+root.toStringList());
	}

    protected void trackInlineAction(GrammarAST actionAST) {
		Rule r = grammar.getRule(currentRuleName);
        if ( r!=null ) {
            r.trackInlineAction(actionAST);
        }
    }

}

grammar[Grammar g]
{
grammar = g;
root = #grammar;
}
    :   ( #( LEXER_GRAMMAR 	  {grammar.type = Grammar.LEXER;} 	    grammarSpec )
	    | #( PARSER_GRAMMAR   {grammar.type = Grammar.PARSER;}      grammarSpec )
	    | #( TREE_GRAMMAR     {grammar.type = Grammar.TREE_PARSER;} grammarSpec )
	    | #( COMBINED_GRAMMAR {grammar.type = Grammar.COMBINED;}    grammarSpec )
	    )
	    {finish();}
    ;

attrScope
	:	#( "scope" name:ID attrs:ACTION )
		{
		AttributeScope scope = grammar.defineGlobalScope(name.getText(),#attrs.token);
		scope.isDynamicGlobalScope = true;
		scope.addAttributes(attrs.getText(), ';');
		}
	;

grammarSpec
{
Map opts=null;
Token optionsStartToken=null;
}
	:	id:ID
		(cmt:DOC_COMMENT)?
        //(#(OPTIONS .))? // already parsed these in assign.types.g
        ( {optionsStartToken=((GrammarAST)_t).getToken();}
          optionsSpec
        )?
        (delegateGrammars)?
        (tokensSpec)?
        (attrScope)*
        (actions)?
        rules
	;

actions
	:	( action )+
	;

action
{
String scope=null;
GrammarAST nameAST=null, actionAST=null;
}
	:	#(amp:AMPERSAND id1:ID
			( id2:ID a1:ACTION
			  {scope=#id1.getText(); nameAST=#id2; actionAST=#a1;}
			| a2:ACTION
			  {scope=null; nameAST=#id1; actionAST=#a2;}
			)
		 )
		 {
		 grammar.defineNamedAction(#amp,scope,nameAST,actionAST);
		 }
	;

optionsSpec
	:	OPTIONS
	;

delegateGrammars
	:	#( "import" ( #(ASSIGN ID ID) | ID )+ )
	;

tokensSpec
	:	#( TOKENS ( tokenSpec )+ )
	;

tokenSpec
	:	t:TOKEN_REF
	|	#( ASSIGN
		   t2:TOKEN_REF
		   ( s:STRING_LITERAL
		   | c:CHAR_LITERAL
		   )
		 )
	;

rules
    :   ( rule )+
    ;

rule
{
String mod=null;
String name=null;
Map opts=null;
Rule r = null;
}
    :   #( RULE id:ID {opts = #RULE.getBlockOptions();}
           (mod=modifier)?
           #( ARG (args:ARG_ACTION)? )
           #( RET (ret:ARG_ACTION)? )
           (optionsSpec)?
			{
			name = #id.getText();
			currentRuleName = name;
			if ( Character.isUpperCase(name.charAt(0)) &&
				 grammar.type==Grammar.COMBINED )
			{
				// a merged grammar spec, track lexer rules and send to another grammar
				grammar.defineLexerRuleFoundInParser(#id.getToken(), #rule);
			}
			else {
				int numAlts = countAltsForRule(#rule);
				grammar.defineRule(#id.getToken(), mod, opts, #rule, #args, numAlts);
				r = grammar.getRule(name);
				if ( #args!=null ) {
					r.parameterScope = grammar.createParameterScope(name,#args.token);
					r.parameterScope.addAttributes(#args.getText(), ',');
				}
				if ( #ret!=null ) {
					r.returnScope = grammar.createReturnScope(name,#ret.token);
					r.returnScope.addAttributes(#ret.getText(), ',');
				}
			}
			}
           (ruleScopeSpec[r])?
		   (ruleAction[r])*
           {this.blockLevel=0;}
           b:block
           (exceptionGroup)?
           EOR
           {
           // copy rule options into the block AST, which is where
           // the analysis will look for k option etc...
           #b.setBlockOptions(opts);
           }
         )
    ;

countAltsForRule returns [int n=0]
    :   #( RULE id:ID (modifier)? ARG RET (OPTIONS)? ("scope")? (AMPERSAND)*
           #(  BLOCK (OPTIONS)? (ALT (REWRITE)* {n++;})+ EOB )
           (exceptionGroup)?
           EOR
         )
	;

ruleAction[Rule r]
	:	#(amp:AMPERSAND id:ID a:ACTION ) {if (r!=null) r.defineNamedAction(#amp,#id,#a);}
	;

modifier returns [String mod]
{
mod = #modifier.getText();
}
	:	"protected"
	|	"public"
	|	"private"
	|	"fragment"
	;

ruleScopeSpec[Rule r]
 	:	#( "scope"
 	       ( attrs:ACTION
 	         {
 	         r.ruleScope = grammar.createRuleScope(r.name,#attrs.token);
			 r.ruleScope.isDynamicRuleScope = true;
			 r.ruleScope.addAttributes(#attrs.getText(), ';');
			 }
		   )?
 	       ( uses:ID
 	         {
 	         if ( grammar.getGlobalScope(#uses.getText())==null ) {
				 ErrorManager.grammarError(ErrorManager.MSG_UNKNOWN_DYNAMIC_SCOPE,
										   grammar,
										   #uses.token,
										   #uses.getText());
	         }
 	         else {
 	         	if ( r.useScopes==null ) {r.useScopes=new ArrayList();}
 	         	r.useScopes.add(#uses.getText());
 	         }
 	         }
 	       )*
 	     )
 	;

block
{
this.blockLevel++;
if ( this.blockLevel==1 ) {this.outerAltNum=1;}
}
    :   #(  BLOCK
            (optionsSpec)?
            (blockAction)*
            ( alternative rewrite
              {if ( this.blockLevel==1 ) {this.outerAltNum++;}}
            )+
            EOB
         )
         {this.blockLevel--;}
    ;

// TODO: this does nothing now! subrules cannot have init actions. :(
blockAction
	:	#(amp:AMPERSAND id:ID a:ACTION ) // {r.defineAction(#amp,#id,#a);}
	;

alternative
{
if ( grammar.type!=Grammar.LEXER && grammar.getOption("output")!=null && blockLevel==1 ) {
	GrammarAST aRewriteNode = #alternative.findFirstType(REWRITE); // alt itself has rewrite?
	GrammarAST rewriteAST = (GrammarAST)#alternative.getNextSibling();
	// we have a rewrite if alt uses it inside subrule or this alt has one
	// but don't count -> ... rewrites, which mean "do default auto construction"
	if ( aRewriteNode!=null||
		 (rewriteAST!=null &&
		  rewriteAST.getType()==REWRITE &&
		  rewriteAST.getFirstChild()!=null &&
		  rewriteAST.getFirstChild().getType()!=ETC) )
	{
		Rule r = grammar.getRule(currentRuleName);
		r.trackAltsWithRewrites(#alternative,this.outerAltNum);
	}
}
}
    :   #( ALT (element)+ EOA )
    ;

exceptionGroup
	:	( exceptionHandler )+ (finallyClause)?
	|	finallyClause
    ;

exceptionHandler
    :   #("catch" ARG_ACTION ACTION) {trackInlineAction(#ACTION);}
    ;

finallyClause
    :    #("finally" ACTION) {trackInlineAction(#ACTION);}
    ;

element
    :   #(ROOT element)
    |   #(BANG element)
    |   atom[null]
    |   #(NOT element)
    |   #(RANGE atom[null] atom[null])
    |   #(CHAR_RANGE atom[null] atom[null])
    |	#(ASSIGN id:ID el:element)
    	{
		if ( #el.getType()==ANTLRParser.ROOT ||
             #el.getType()==ANTLRParser.BANG )
		{
            #el = (GrammarAST)#el.getFirstChild();
        }
    	if ( #el.getType()==RULE_REF) {
    		grammar.defineRuleRefLabel(currentRuleName,#id.getToken(),#el);
    	}
    	else if ( #el.getType()==WILDCARD && grammar.type==Grammar.TREE_PARSER ) {
    		grammar.defineWildcardTreeLabel(currentRuleName,#id.getToken(),#el);
    	}
    	else {
    		grammar.defineTokenRefLabel(currentRuleName,#id.getToken(),#el);
    	}
    	}
    |	#(	PLUS_ASSIGN id2:ID a2:element
    	    {
            if ( #a2.getType()==ANTLRParser.ROOT ||
                 #a2.getType()==ANTLRParser.BANG )
            {
                #a2 = (GrammarAST)#a2.getFirstChild();
            }
    	    if ( #a2.getType()==RULE_REF ) {
    	    	grammar.defineRuleListLabel(currentRuleName,#id2.getToken(),#a2);
    	    }
            else if ( #a2.getType()==WILDCARD && grammar.type==Grammar.TREE_PARSER ) {
                grammar.defineWildcardTreeListLabel(currentRuleName,#id2.getToken(),#a2);
            }
    	    else {
    	    	grammar.defineTokenListLabel(currentRuleName,#id2.getToken(),#a2);
    	    }
    	    }
         )
    |   ebnf
    |   tree
    |   #( SYNPRED block )
    |   act:ACTION
        {
        #act.outerAltNum = this.outerAltNum;
		trackInlineAction(#act);
        }
    |   act2:FORCED_ACTION
        {
        #act2.outerAltNum = this.outerAltNum;
		trackInlineAction(#act2);
        }
    |   SEMPRED
        {
        #SEMPRED.outerAltNum = this.outerAltNum;
        trackInlineAction(#SEMPRED);
        }
    |   SYN_SEMPRED
    |   BACKTRACK_SEMPRED
    |   GATED_SEMPRED
        {
        #GATED_SEMPRED.outerAltNum = this.outerAltNum;
        trackInlineAction(#GATED_SEMPRED);
        }
    |   EPSILON 
    ;

ebnf:   (dotLoop)=> dotLoop // .* or .+
    |   block
    |   #( OPTIONAL block )
    |   #( CLOSURE block )
    |   #( POSITIVE_CLOSURE block )
    ;

/** Track the .* and .+ idioms and make them nongreedy by default.
 */
dotLoop
{
    GrammarAST block = (GrammarAST)#dotLoop.getFirstChild();
}
    :   (   #( CLOSURE dotBlock )           
        |   #( POSITIVE_CLOSURE dotBlock )
        )
        {
        Map opts=new HashMap();
        opts.put("greedy", "false");
        if ( grammar.type!=Grammar.LEXER ) {
            // parser grammars assume k=1 for .* loops
            // otherwise they (analysis?) look til EOF!
            opts.put("k", Utils.integer(1));
        }
        block.setOptions(grammar,opts);
        }
    ;

dotBlock
    :   #( BLOCK #( ALT WILDCARD EOA ) EOB )
    ;

tree:   #(TREE_BEGIN element (element)*)
    ;

atom[GrammarAST scope]
    :   #( rr:RULE_REF (rarg:ARG_ACTION)? )
    	{
        grammar.altReferencesRule(currentRuleName, scope, #rr, this.outerAltNum);
		if ( #rarg!=null ) {
            #rarg.outerAltNum = this.outerAltNum;
            trackInlineAction(#rarg);
        }
        }
    |   #( t:TOKEN_REF  (targ:ARG_ACTION )? )
    	{
		if ( #targ!=null ) {
            #targ.outerAltNum = this.outerAltNum;
            trackInlineAction(#targ);
        }
    	if ( grammar.type==Grammar.LEXER ) {
    		grammar.altReferencesRule(currentRuleName, scope, #t, this.outerAltNum);
    	}
    	else {
    		grammar.altReferencesTokenID(currentRuleName, #t, this.outerAltNum);
    	}
    	}
    |   c:CHAR_LITERAL
    	{
    	if ( grammar.type!=Grammar.LEXER ) {
    		Rule rule = grammar.getRule(currentRuleName);
			if ( rule!=null ) {
				rule.trackTokenReferenceInAlt(#c, outerAltNum);
    		}
    	}
    	}
    |   s:STRING_LITERAL 
    	{
    	if ( grammar.type!=Grammar.LEXER ) {
    		Rule rule = grammar.getRule(currentRuleName);
			if ( rule!=null ) {
				rule.trackTokenReferenceInAlt(#s, outerAltNum);
    		}
    	}
    	}
    |   WILDCARD
    |   #(DOT ID atom[#ID]) // scope override on rule
    ;

ast_suffix
	:	ROOT
	|	BANG
	;

rewrite
{
currentRewriteRule = #rewrite; // has to execute during guessing
if ( grammar.buildAST() ) {
    #rewrite.rewriteRefsDeep = new HashSet<GrammarAST>();
}
}
	:	(
            #( REWRITE (pred:SEMPRED)? rewrite_alternative )
            {
            if ( #pred!=null ) {
                #pred.outerAltNum = this.outerAltNum;
                trackInlineAction(#pred);
            }
            }
        )*
        //{System.out.println("-> refs = "+#rewrite.rewriteRefs);}
	;

rewrite_block
{
GrammarAST enclosingBlock = currentRewriteBlock;
if ( inputState.guessing==0 ) {  // don't do if guessing
    currentRewriteBlock=#rewrite_block; // pts to BLOCK node
    currentRewriteBlock.rewriteRefsShallow = new HashSet<GrammarAST>();
    currentRewriteBlock.rewriteRefsDeep = new HashSet<GrammarAST>();
}
}
    :   #( BLOCK rewrite_alternative EOB )
        //{System.out.println("atoms="+currentRewriteBlock.rewriteRefs);}
        {
        // copy the element refs in this block to the surrounding block
        if ( enclosingBlock!=null ) {
            enclosingBlock.rewriteRefsDeep
                .addAll(currentRewriteBlock.rewriteRefsShallow);
        }
        currentRewriteBlock = enclosingBlock; // restore old BLOCK ptr
        }
    ;

rewrite_alternative
    :   {grammar.buildAST()}?
    	#( a:ALT ( ( rewrite_element )+ | EPSILON ) EOA )
    |	{grammar.buildTemplate()}? rewrite_template
	|	ETC {this.blockLevel==1}? // only valid as outermost rewrite

    ;

rewrite_element
    :   rewrite_atom
    |   rewrite_ebnf
    |   rewrite_tree
    ;

rewrite_ebnf
    :   #( OPTIONAL rewrite_block )
    |   #( CLOSURE rewrite_block )
    |   #( POSITIVE_CLOSURE rewrite_block )
    ;

rewrite_tree
	:   #(	TREE_BEGIN rewrite_atom ( rewrite_element )* )
    ;

rewrite_atom
{
Rule r = grammar.getRule(currentRuleName);
Set tokenRefsInAlt = r.getTokenRefsInAlt(outerAltNum);
boolean imaginary =
    #rewrite_atom.getType()==TOKEN_REF &&
    !tokenRefsInAlt.contains(#rewrite_atom.getText());
if ( !imaginary && grammar.buildAST() &&
     (#rewrite_atom.getType()==RULE_REF ||
      #rewrite_atom.getType()==LABEL ||
      #rewrite_atom.getType()==TOKEN_REF ||
      #rewrite_atom.getType()==CHAR_LITERAL ||
      #rewrite_atom.getType()==STRING_LITERAL) )
{
    // track per block and for entire rewrite rule
    if ( currentRewriteBlock!=null ) {
        currentRewriteBlock.rewriteRefsShallow.add(#rewrite_atom);
        currentRewriteBlock.rewriteRefsDeep.add(#rewrite_atom);
    }
    currentRewriteRule.rewriteRefsDeep.add(#rewrite_atom);
}
}
    :   RULE_REF 
    |   ( #(TOKEN_REF
            (arg:ARG_ACTION)?
           )
        | CHAR_LITERAL
        | STRING_LITERAL
        )
        {
        if ( #arg!=null ) {
            #arg.outerAltNum = this.outerAltNum;
            trackInlineAction(#arg);
        }
        }

    |	LABEL

    |	ACTION
        {
            #ACTION.outerAltNum = this.outerAltNum;
            trackInlineAction(#ACTION);
        }
    ;

rewrite_template
    :	#( ALT EPSILON EOA ) 
   	|	#( TEMPLATE (id:ID|ind:ACTION)
	       #( ARGLIST
                ( #( ARG arg:ID a:ACTION )
                {
                    #a.outerAltNum = this.outerAltNum;
                    trackInlineAction(#a);
                }
                )*
            )
            {
            if ( #ind!=null ) {
                #ind.outerAltNum = this.outerAltNum;
                trackInlineAction(#ind);
            }
            }

		   ( DOUBLE_QUOTE_STRING_LITERAL
		   | DOUBLE_ANGLE_STRING_LITERAL
		   )?
	     )

	|	act:ACTION
        {
        #act.outerAltNum = this.outerAltNum;
        trackInlineAction(#act);
        }
	;
