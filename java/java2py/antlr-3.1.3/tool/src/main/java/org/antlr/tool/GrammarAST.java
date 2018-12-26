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
package org.antlr.tool;

import antlr.BaseAST;
import antlr.Token;
import antlr.TokenWithIndex;
import antlr.collections.AST;
import org.antlr.analysis.DFA;
import org.antlr.analysis.NFAState;
import org.antlr.misc.IntSet;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.grammar.v2.ANTLRParser;

import java.util.*;

/** Grammars are first converted to ASTs using this class and then are
 *  converted to NFAs via a tree walker.
 *
 *  The reader may notice that I have made a very non-OO decision in this
 *  class to track variables for many different kinds of nodes.  It wastes
 *  space for nodes that don't need the values and OO principles cry out
 *  for a new class type for each kind of node in my tree.  I am doing this
 *  on purpose for a variety of reasons.  I don't like using the type
 *  system for different node types; it yields too many damn class files
 *  which I hate.  Perhaps if I put them all in one file.  Most importantly
 *  though I hate all the type casting that would have to go on.  I would
 *  have all sorts of extra work to do.  Ick.  Anyway, I'm doing all this
 *  on purpose, not out of ignorance. ;)
 */
public class GrammarAST extends BaseAST {
	static int count = 0;

	public int ID = ++count;

	/** This AST node was created from what token? */
    public Token token = null;

    public String enclosingRuleName;

	/** If this is a RULE node then track rule's start, stop tokens' index. */
	public int ruleStartTokenIndex;
	public int ruleStopTokenIndex;

    /** If this is a decision node, what is the lookahead DFA? */
    public DFA lookaheadDFA = null;

    /** What NFA start state was built from this node? */
    public NFAState NFAStartState = null;

	/** This is used for TREE_BEGIN nodes to point into
	 *  the NFA.  TREE_BEGINs point at left edge of DOWN for LOOK computation
     *  purposes (Nullable tree child list needs special code gen when matching).
	 */
	public NFAState NFATreeDownState = null;

	/** Rule ref nodes, token refs, set, and NOT set refs need to track their
	 *  location in the generated NFA so that local FOLLOW sets can be
	 *  computed during code gen for automatic error recovery.
	 */
	public NFAState followingNFAState = null;

	/** If this is a SET node, what are the elements? */
    protected IntSet setValue = null;

    /** If this is a BLOCK node, track options here */
    protected Map<String,Object> blockOptions;

	/** If this is a BLOCK node for a rewrite rule, track referenced
	 *  elements here.  Don't track elements in nested subrules.
	 */
	public Set<GrammarAST> rewriteRefsShallow;

	/*	If REWRITE node, track EVERY element and label ref to right of ->
	 *  for this rewrite rule.  There could be multiple of these per
	 *  rule:
	 *
	 *     a : ( ... -> ... | ... -> ... ) -> ... ;
	 *
	 *  We may need a list of all refs to do definitions for whole rewrite
	 *  later.
	 *
	 *  If BLOCK then tracks every element at that level and below.
	 */
	public Set<GrammarAST> rewriteRefsDeep;

	public Map<String,Object> terminalOptions;

	/** if this is an ACTION node, this is the outermost enclosing
	 *  alt num in rule.  For actions, define.g sets these (used to
	 *  be codegen.g).  We need these set so we can examine actions
	 *  early, before code gen, for refs to rule predefined properties
	 *  and rule labels.  For most part define.g sets outerAltNum, but
	 *  codegen.g does the ones for %foo(a={$ID.text}) type refs as
	 *  the {$ID...} is not seen as an action until code gen pulls apart.
	 */
	public int outerAltNum;

	/** if this is a TOKEN_REF or RULE_REF node, this is the code StringTemplate
	 *  generated for this node.  We need to update it later to add
	 *  a label if someone does $tokenref or $ruleref in an action.
	 */
	public StringTemplate code;
    
    /**
     * 
     * @return
     */
    public Map<String, Object> getBlockOptions() {
        return blockOptions;
    }

    /**
     * 
     * @param blockOptions
     */
    public void setBlockOptions(Map<String, Object> blockOptions) {
        this.blockOptions = blockOptions;
    }
        
	public GrammarAST() {;}

	public GrammarAST(int t, String txt) {
		initialize(t,txt);
	}

	public void initialize(int i, String s) {
        token = new TokenWithIndex(i,s);
    }

    public void initialize(AST ast) {
		GrammarAST t = ((GrammarAST)ast);
		this.token = t.token;
		this.enclosingRuleName = t.enclosingRuleName;
		this.ruleStartTokenIndex = t.ruleStartTokenIndex;
		this.ruleStopTokenIndex = t.ruleStopTokenIndex;
		this.setValue = t.setValue;
		this.blockOptions = t.blockOptions;
		this.outerAltNum = t.outerAltNum;
	}

    public void initialize(Token token) {
        this.token = token;
    }

    public DFA getLookaheadDFA() {
        return lookaheadDFA;
    }

    public void setLookaheadDFA(DFA lookaheadDFA) {
        this.lookaheadDFA = lookaheadDFA;
    }

	public Token getToken() {
		return token;
	}

    public NFAState getNFAStartState() {
        return NFAStartState;
    }

    public void setNFAStartState(NFAState nfaStartState) {
		this.NFAStartState = nfaStartState;
	}

	/** Save the option key/value pair and process it; return the key
	 *  or null if invalid option.
	 */
	public String setBlockOption(Grammar grammar, String key, Object value) {
		if ( blockOptions == null ) {
			blockOptions = new HashMap();
		}
		return setOption(blockOptions, Grammar.legalBlockOptions, grammar, key, value);
	}

	public String setTerminalOption(Grammar grammar, String key, Object value) {
		if ( terminalOptions == null ) {
			terminalOptions = new HashMap<String,Object>();
		}
		return setOption(terminalOptions, Grammar.legalTokenOptions, grammar, key, value);
	}

	public String setOption(Map options, Set legalOptions, Grammar grammar, String key, Object value) {
		if ( !legalOptions.contains(key) ) {
			ErrorManager.grammarError(ErrorManager.MSG_ILLEGAL_OPTION,
									  grammar,
									  token,
									  key);
			return null;
		}
		if ( value instanceof String ) {
			String vs = (String)value;
			if ( vs.charAt(0)=='"' ) {
				value = vs.substring(1,vs.length()-1); // strip quotes
            }
        }
		if ( key.equals("k") ) {
			grammar.numberOfManualLookaheadOptions++;
		}
        if ( key.equals("backtrack") && value.toString().equals("true") ) {
            grammar.composite.getRootGrammar().atLeastOneBacktrackOption = true;
        }        
        options.put(key, value);
		return key;
    }

    public Object getBlockOption(String key) {
		Object value = null;
		if ( blockOptions != null ) {
			value = blockOptions.get(key);
		}
		return value;
	}

    public void setOptions(Grammar grammar, Map options) {
		if ( options==null ) {
			this.blockOptions = null;
			return;
		}
		Set keys = options.keySet();
		for (Iterator it = keys.iterator(); it.hasNext();) {
			String optionName = (String) it.next();
			String stored= setBlockOption(grammar, optionName, options.get(optionName));
			if ( stored==null ) {
				it.remove();
			}
		}
    }

    public String getText() {
        if ( token!=null ) {
            return token.getText();
        }
        return "";
    }

	public void setType(int type) {
		token.setType(type);
	}

	public void setText(String text) {
		token.setText(text);
	}

    public int getType() {
        if ( token!=null ) {
            return token.getType();
        }
        return -1;
    }

    public int getLine() {
		int line=0;
        if ( token!=null ) {
            line = token.getLine();
        }
		if ( line==0 ) {
			AST child = getFirstChild();
			if ( child!=null ) {
				line = child.getLine();
			}
		}
        return line;
    }

    public int getColumn() {
		int col=0;
        if ( token!=null ) {
            col = token.getColumn();
        }
		if ( col==0 ) {
			AST child = getFirstChild();
			if ( child!=null ) {
				col = child.getColumn();
			}
		}
        return col;
    }

    public void setLine(int line) {
        token.setLine(line);
    }

    public void setColumn(int col) {
        token.setColumn(col);
    }

 	public IntSet getSetValue() {
        return setValue;
    }

    public void setSetValue(IntSet setValue) {
        this.setValue = setValue;
    }

    public GrammarAST getLastChild() {
        return ((GrammarAST)getFirstChild()).getLastSibling();
    }

    public GrammarAST getLastSibling() {
        GrammarAST t = this;
        GrammarAST last = null;
        while ( t!=null ) {
            last = t;
            t = (GrammarAST)t.getNextSibling();
        }
        return last;
    }

    /** Get the ith child from 0 */
	public GrammarAST getChild(int i) {
		int n = 0;
		AST t = getFirstChild();
		while ( t!=null ) {
			if ( n==i ) {
				return (GrammarAST)t;
			}
			n++;
			t = (GrammarAST)t.getNextSibling();
		}
		return null;
	}

	public GrammarAST getFirstChildWithType(int ttype) {
		AST t = getFirstChild();
		while ( t!=null ) {
			if ( t.getType()==ttype ) {
				return (GrammarAST)t;
			}
			t = (GrammarAST)t.getNextSibling();
		}
		return null;
	}

    public GrammarAST[] getChildrenAsArray() {
        AST t = getFirstChild();
        GrammarAST[] array = new GrammarAST[getNumberOfChildren()];
        int i = 0;
        while ( t!=null ) {
            array[i] = (GrammarAST)t;
            t = t.getNextSibling();
            i++;
        }
        return array;
    }

	/** Return a reference to the first node (depth-first) that has
	 *  token type ttype.  Assume 'this' is a root node; don't visit siblings
	 *  of root.  Return null if no node found with ttype.
	 */
	public GrammarAST findFirstType(int ttype) {
		// check this node (the root) first
		if ( this.getType()==ttype ) {
			return this;
		}
		// else check children
		GrammarAST child = (GrammarAST)this.getFirstChild();
		while ( child!=null ) {
			GrammarAST result = child.findFirstType(ttype);
			if ( result!=null ) {
				return result;
			}
			child = (GrammarAST)child.getNextSibling();
		}
		return null;
	}

    public int getNumberOfChildrenWithType(int ttype) {
        AST p = this.getFirstChild();
        int n = 0;
        while ( p!=null ) {
            if ( p.getType()==ttype ) n++;
            p = p.getNextSibling();
        }
        return n;
    }

    /** Make nodes unique based upon Token so we can add them to a Set; if
	 *  not a GrammarAST, check type.
	 */
	public boolean equals(Object ast) {
		if ( this == ast ) {
			return true;
		}
		if ( !(ast instanceof GrammarAST) ) {
			return this.getType() == ((AST)ast).getType();
		}
		GrammarAST t = (GrammarAST)ast;
		return token.getLine() == t.getLine() &&
			   token.getColumn() == t.getColumn();
	}

	/** See if tree has exact token types and structure; no text */
	public boolean hasSameTreeStructure(AST t) {
		// check roots first.
		if (this.getType() != t.getType()) return false;
		// if roots match, do full list match test on children.
		if (this.getFirstChild() != null) {
			if (!(((GrammarAST)this.getFirstChild()).hasSameListStructure(t.getFirstChild()))) return false;
		}
		// sibling has no kids, make sure t doesn't either
		else if (t.getFirstChild() != null) {
			return false;
		}
		return true;
	}

	public boolean hasSameListStructure(AST t) {
		AST sibling;

		// the empty tree is not a match of any non-null tree.
		if (t == null) {
			return false;
		}

		// Otherwise, start walking sibling lists.  First mismatch, return false.
		for (sibling = this;
			 sibling != null && t != null;
			 sibling = sibling.getNextSibling(), t = t.getNextSibling())
		{
			// as a quick optimization, check roots first.
			if (sibling.getType()!=t.getType()) {
				return false;
			}
			// if roots match, do full list match test on children.
			if (sibling.getFirstChild() != null) {
				if (!((GrammarAST)sibling.getFirstChild()).hasSameListStructure(t.getFirstChild())) {
					return false;
				}
			}
			// sibling has no kids, make sure t doesn't either
			else if (t.getFirstChild() != null) {
				return false;
			}
		}
		if (sibling == null && t == null) {
			return true;
		}
		// one sibling list has more than the other
		return false;
	}

	public static GrammarAST dup(AST t) {
		if ( t==null ) {
			return null;
		}
		GrammarAST dup_t = new GrammarAST();
		dup_t.initialize(t);
		return dup_t;
	}

	/** Duplicate tree including siblings of root. */
	public static GrammarAST dupListNoActions(GrammarAST t, GrammarAST parent) {
		GrammarAST result = dupTreeNoActions(t, parent);            // if t == null, then result==null
		GrammarAST nt = result;
		while (t != null) {						// for each sibling of the root
			t = (GrammarAST)t.getNextSibling();
			if ( t!=null && t.getType()==ANTLRParser.ACTION ) {
				continue;
			}
			GrammarAST d = dupTreeNoActions(t, parent);
			if ( d!=null ) {
				if ( nt!=null ) {
					nt.setNextSibling(d);	// dup each subtree, building new tree
				}
				nt = d;
			}
		}
		return result;
	}

	/**Duplicate a tree, assuming this is a root node of a tree--
	 * duplicate that node and what's below; ignore siblings of root node.
	 */
	public static GrammarAST dupTreeNoActions(GrammarAST t, GrammarAST parent) {
		if ( t==null ) {
			return null;
		}
		int ttype = t.getType();
		if ( ttype==ANTLRParser.REWRITE ) {
			return null;
		}
		if ( ttype==ANTLRParser.BANG || ttype==ANTLRParser.ROOT ) {
			// return x from ^(ROOT x)
			return (GrammarAST)dupListNoActions((GrammarAST)t.getFirstChild(), t);
		}
        /* DOH!  Must allow labels for sem preds
        if ( (ttype==ANTLRParser.ASSIGN||ttype==ANTLRParser.PLUS_ASSIGN) &&
			 (parent==null||parent.getType()!=ANTLRParser.OPTIONS) )
		{
			return dupTreeNoActions(t.getChild(1), t); // return x from ^(ASSIGN label x)
		}
		*/
		GrammarAST result = dup(t);		// make copy of root
		// copy all children of root.
		GrammarAST kids = dupListNoActions((GrammarAST)t.getFirstChild(), t);
		result.setFirstChild(kids);
		return result;
	}

	public void setTreeEnclosingRuleNameDeeply(String rname) {
		GrammarAST t = this;
		t.enclosingRuleName = rname;
		t = t.getChild(0);
		while (t != null) {						// for each sibling of the root
			t.setTreeEnclosingRuleNameDeeply(rname);
			t = (GrammarAST)t.getNextSibling();
		}
	}

}
