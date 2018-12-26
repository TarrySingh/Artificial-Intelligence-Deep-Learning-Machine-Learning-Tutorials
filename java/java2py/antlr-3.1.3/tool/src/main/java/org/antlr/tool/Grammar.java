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

import antlr.RecognitionException;
import antlr.Token;
import antlr.TokenStreamException;
import antlr.TokenStreamRewriteEngine;
import antlr.TokenWithIndex;
import org.antlr.grammar.v2.*;
import org.antlr.grammar.v3.*;

import org.antlr.misc.*;
import org.antlr.misc.Utils;

import antlr.collections.AST;
import org.antlr.Tool;
import org.antlr.analysis.*;
import org.antlr.codegen.CodeGenerator;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;

import java.io.*;
import java.util.*;

/** Represents a grammar in memory. */
public class Grammar {
	public static final String SYNPRED_RULE_PREFIX = "synpred";

	public static final String GRAMMAR_FILE_EXTENSION = ".g";

	/** used for generating lexer temp files */
	public static final String LEXER_GRAMMAR_FILE_EXTENSION = ".g";

	public static final int INITIAL_DECISION_LIST_SIZE = 300;
	public static final int INVALID_RULE_INDEX = -1;

	// the various kinds of labels. t=type, id=ID, types+=type ids+=ID
	public static final int RULE_LABEL = 1;
	public static final int TOKEN_LABEL = 2;
	public static final int RULE_LIST_LABEL = 3;
	public static final int TOKEN_LIST_LABEL = 4;
    public static final int CHAR_LABEL = 5; // used in lexer for x='a'
    public static final int WILDCARD_TREE_LABEL = 6; // Used in tree grammar x=.
    public static final int WILDCARD_TREE_LIST_LABEL = 7; // Used in tree grammar x+=.


    public static String[] LabelTypeToString =
		{"<invalid>", "rule", "token", "rule-list", "token-list", "wildcard-tree", "wildcard-tree-list"};

	public static final String ARTIFICIAL_TOKENS_RULENAME = "Tokens";
	public static final String FRAGMENT_RULE_MODIFIER = "fragment";

	public static final String SYNPREDGATE_ACTION_NAME = "synpredgate";

	/** When converting ANTLR char and string literals, here is the
	 *  value set of escape chars.
	 */
	public static int ANTLRLiteralEscapedCharValue[] = new int[255];

	/** Given a char, we need to be able to show as an ANTLR literal.
	 */
	public static String ANTLRLiteralCharValueEscape[] = new String[255];

	static {
		ANTLRLiteralEscapedCharValue['n'] = '\n';
		ANTLRLiteralEscapedCharValue['r'] = '\r';
		ANTLRLiteralEscapedCharValue['t'] = '\t';
		ANTLRLiteralEscapedCharValue['b'] = '\b';
		ANTLRLiteralEscapedCharValue['f'] = '\f';
		ANTLRLiteralEscapedCharValue['\\'] = '\\';
		ANTLRLiteralEscapedCharValue['\''] = '\'';
		ANTLRLiteralEscapedCharValue['"'] = '"';
		ANTLRLiteralCharValueEscape['\n'] = "\\n";
		ANTLRLiteralCharValueEscape['\r'] = "\\r";
		ANTLRLiteralCharValueEscape['\t'] = "\\t";
		ANTLRLiteralCharValueEscape['\b'] = "\\b";
		ANTLRLiteralCharValueEscape['\f'] = "\\f";
		ANTLRLiteralCharValueEscape['\\'] = "\\\\";
		ANTLRLiteralCharValueEscape['\''] = "\\'";
	}

	public static final int LEXER = 1;
	public static final int PARSER = 2;
	public static final int TREE_PARSER = 3;
	public static final int COMBINED = 4;
	public static final String[] grammarTypeToString = new String[] {
		"<invalid>",
		"lexer",
		"parser",
		"tree",
		"combined"
	};

	public static final String[] grammarTypeToFileNameSuffix = new String[] {
		"<invalid>",
		"Lexer",
		"Parser",
		"", // no suffix for tree grammars
		"Parser" // if combined grammar, gen Parser and Lexer will be done later
	};

	/** Set of valid imports.  E.g., can only import a tree parser into
	 *  another tree parser.  Maps delegate to set of delegator grammar types.
	 *  validDelegations.get(LEXER) gives list of the kinds of delegators
	 *  that can import lexers.
	 */
	public static MultiMap<Integer,Integer> validDelegations =
		new MultiMap<Integer,Integer>() {
			{
				map(LEXER, LEXER);
				map(LEXER, PARSER);
				map(LEXER, COMBINED);

				map(PARSER, PARSER);
				map(PARSER, COMBINED);

				map(TREE_PARSER, TREE_PARSER);

				// TODO: allow COMBINED
				// map(COMBINED, COMBINED);
			}
		};

	/** This is the buffer of *all* tokens found in the grammar file
	 *  including whitespace tokens etc...  I use this to extract
	 *  lexer rules from combined grammars.
	 */
	protected TokenStreamRewriteEngine tokenBuffer;
	public static final String IGNORE_STRING_IN_GRAMMAR_FILE_NAME = "__";
	public static final String AUTO_GENERATED_TOKEN_NAME_PREFIX = "T__";

	public static class Decision {
		public int decision;
		public NFAState startState;
		public GrammarAST blockAST;
		public DFA dfa;
	}

	public class LabelElementPair {
		public antlr.Token label;
		public GrammarAST elementRef;
		public String referencedRuleName;
		/** Has an action referenced the label?  Set by ActionAnalysis.g
		 *  Currently only set for rule labels.
		 */
		public boolean actionReferencesLabel;
		public int type; // in {RULE_LABEL,TOKEN_LABEL,RULE_LIST_LABEL,TOKEN_LIST_LABEL}
		public LabelElementPair(antlr.Token label, GrammarAST elementRef) {
			this.label = label;
			this.elementRef = elementRef;
			this.referencedRuleName = elementRef.getText();
		}
		public Rule getReferencedRule() {
			return getRule(referencedRuleName);
		}
		public String toString() {
			return elementRef.toString();
		}
	}

	/** What name did the user provide for this grammar? */
	public String name;

	/** What type of grammar is this: lexer, parser, tree walker */
	public int type;

	/** A list of options specified at the grammar level such as language=Java.
	 *  The value can be an AST for complicated values such as character sets.
	 *  There may be code generator specific options in here.  I do no
	 *  interpretation of the key/value pairs...they are simply available for
	 *  who wants them.
	 */
	protected Map options;

	public static final Set legalLexerOptions =
			new HashSet() {
				{
				add("language"); add("tokenVocab");
				add("TokenLabelType");
				add("superClass");
				add("filter");
				add("k");
				add("backtrack");
				add("memoize");
				}
			};

	public static final Set legalParserOptions =
			new HashSet() {
				{
				add("language"); add("tokenVocab");
				add("output"); add("rewrite"); add("ASTLabelType");
				add("TokenLabelType");
				add("superClass");
				add("k");
				add("backtrack");
				add("memoize");
				}
			};

    public static final Set legalTreeParserOptions =
        new HashSet() {
            {
                add("language"); add("tokenVocab");
                add("output"); add("rewrite"); add("ASTLabelType");
                add("TokenLabelType");
                add("superClass");
                add("k");
                add("backtrack");
                add("memoize");
                add("filter");
            }
        };

	public static final Set doNotCopyOptionsToLexer =
		new HashSet() {
			{
				add("output"); add("ASTLabelType"); add("superClass");
				add("k"); add("backtrack"); add("memoize"); add("rewrite");
			}
		};

	public static final Map defaultOptions =
			new HashMap() {
				{
					put("language","Java");
				}
			};

	public static final Set legalBlockOptions =
			new HashSet() {{add("k"); add("greedy"); add("backtrack"); add("memoize");}};

	/** What are the default options for a subrule? */
	public static final Map defaultBlockOptions =
			new HashMap() {{put("greedy","true");}};

	public static final Map defaultLexerBlockOptions =
			new HashMap() {{put("greedy","true");}};

	// Token options are here to avoid contaminating Token object in runtime
	
	/** Legal options for terminal refs like ID<node=MyVarNode> */
	public static final Set legalTokenOptions =
			new HashSet() {
				{
				add(defaultTokenOption);
                add("associativity");
				}
			};
	
	public static final String defaultTokenOption = "node";

	/** Is there a global fixed lookahead set for this grammar?
	 *  If 0, nothing specified.  -1 implies we have not looked at
	 *  the options table yet to set k.
	 */
	protected int global_k = -1;

	/** Map a scope to a map of name:action pairs.
	 *  Map<String, Map<String,GrammarAST>>
	 *  The code generator will use this to fill holes in the output files.
	 *  I track the AST node for the action in case I need the line number
	 *  for errors.
	 */
	protected Map actions = new HashMap();

	/** The NFA that represents the grammar with edges labelled with tokens
	 *  or epsilon.  It is more suitable to analysis than an AST representation.
	 */
	public NFA nfa;

	protected NFAFactory factory;

	/** If this grammar is part of a larger composite grammar via delegate
	 *  statement, then this points at the composite.  The composite holds
	 *  a global list of rules, token types, decision numbers, etc...
	 */
	public CompositeGrammar composite;

	/** A pointer back into grammar tree.  Needed so we can add delegates. */
	public CompositeGrammarTree compositeTreeNode;

	/** If this is a delegate of another grammar, this is the label used
	 *  as an instance var by that grammar to point at this grammar. null
	 *  if no label was specified in the delegate statement.
	 */
	public String label;

	/** TODO: hook this to the charVocabulary option */
	protected IntSet charVocabulary = null;

	/** For ANTLRWorks, we want to be able to map a line:col to a specific
	 *  decision DFA so it can display DFA.
	 */
	Map lineColumnToLookaheadDFAMap = new HashMap();

	public Tool tool;

	/** The unique set of all rule references in any rule; set of tree node
	 *  objects so two refs to same rule can exist but at different line/position.
	 */
	protected Set<GrammarAST> ruleRefs = new HashSet<GrammarAST>();

	protected Set<GrammarAST> scopedRuleRefs = new HashSet();

	/** The unique set of all token ID references in any rule */
	protected Set<antlr.Token> tokenIDRefs = new HashSet<antlr.Token>();

	/** Be able to assign a number to every decision in grammar;
	 *  decisions in 1..n
	 */
	protected int decisionCount = 0;

	/** A list of all rules that are in any left-recursive cycle.  There
	 *  could be multiple cycles, but this is a flat list of all problematic
	 *  rules.
	 */
	protected Set<Rule> leftRecursiveRules;

	/** An external tool requests that DFA analysis abort prematurely.  Stops
	 *  at DFA granularity, which are limited to a DFA size and time computation
	 *  as failsafe.
	 */
	protected boolean externalAnalysisAbort;

	/** When we read in a grammar, we track the list of syntactic predicates
	 *  and build faux rules for them later.  See my blog entry Dec 2, 2005:
	 *  http://www.antlr.org/blog/antlr3/lookahead.tml
	 *  This maps the name (we make up) for a pred to the AST grammar fragment.
	 */
	protected LinkedHashMap nameToSynpredASTMap;

    /** At least one rule has memoize=true */
    public boolean atLeastOneRuleMemoizes;

    /** At least one backtrack=true in rule or decision or grammar. */
    public boolean atLeastOneBacktrackOption;

	/** Was this created from a COMBINED grammar? */
	public boolean implicitLexer;

	/** Map a rule to it's Rule object */
	protected LinkedHashMap<String,Rule> nameToRuleMap = new LinkedHashMap<String,Rule>();

	/** If this rule is a delegate, some rules might be overridden; don't
	 *  want to gen code for them.
	 */
	public Set<String> overriddenRules = new HashSet<String>();

	/** The list of all rules referenced in this grammar, not defined here,
	 *  and defined in a delegate grammar.  Not all of these will be generated
	 *  in the recognizer for this file; only those that are affected by rule
	 *  definitions in this grammar.  I am not sure the Java target will need
	 *  this but I'm leaving in case other targets need it.
	 *  @see NameSpaceChecker.lookForReferencesToUndefinedSymbols()
	 */
	protected Set<Rule> delegatedRuleReferences = new HashSet();

	/** The ANTLRParser tracks lexer rules when reading combined grammars
	 *  so we can build the Tokens rule.
	 */
	public List<String> lexerRuleNamesInCombined = new ArrayList<String>();

	/** Track the scopes defined outside of rules and the scopes associated
	 *  with all rules (even if empty).
	 */
	protected Map scopes = new HashMap();

	/** An AST that records entire input grammar with all rules.  A simple
	 *  grammar with one rule, "grammar t; a : A | B ;", looks like:
	 * ( grammar t ( rule a ( BLOCK ( ALT A ) ( ALT B ) ) <end-of-rule> ) )
	 */
	protected GrammarAST grammarTree = null;

	/** Each subrule/rule is a decision point and we must track them so we
	 *  can go back later and build DFA predictors for them.  This includes
	 *  all the rules, subrules, optional blocks, ()+, ()* etc...
	 */
	protected Vector<Decision> indexToDecision =
		new Vector<Decision>(INITIAL_DECISION_LIST_SIZE);

	/** If non-null, this is the code generator we will use to generate
	 *  recognizers in the target language.
	 */
	protected CodeGenerator generator;

	public NameSpaceChecker nameSpaceChecker = new NameSpaceChecker(this);

	public LL1Analyzer ll1Analyzer = new LL1Analyzer(this);

	/** For merged lexer/parsers, we must construct a separate lexer spec.
	 *  This is the template for lexer; put the literals first then the
	 *  regular rules.  We don't need to specify a token vocab import as
	 *  I make the new grammar import from the old all in memory; don't want
	 *  to force it to read from the disk.  Lexer grammar will have same
	 *  name as original grammar but will be in different filename.  Foo.g
	 *  with combined grammar will have FooParser.java generated and
	 *  Foo__.g with again Foo inside.  It will however generate FooLexer.java
	 *  as it's a lexer grammar.  A bit odd, but autogenerated.  Can tweak
	 *  later if we want.
	 */
	protected StringTemplate lexerGrammarST =
		new StringTemplate(
			"lexer grammar <name>;\n" +
			"<if(options)>" +
			"options {\n" +
			"  <options:{<it.name>=<it.value>;<\\n>}>\n" +
			"}<\\n>\n" +
			"<endif>\n" +
			"<if(imports)>import <imports; separator=\", \">;<endif>\n" +
			"<actionNames,actions:{n,a|@<n> {<a>}\n}>\n" +
			"<literals:{<it.ruleName> : <it.literal> ;\n}>\n" +
			"<rules>",
			AngleBracketTemplateLexer.class
		);

	/** What file name holds this grammar? */
	protected String fileName;

	/** How long in ms did it take to build DFAs for this grammar?
	 *  If this grammar is a combined grammar, it only records time for
	 *  the parser grammar component.  This only records the time to
	 *  do the LL(*) work; NFA->DFA conversion.
	 */
	public long DFACreationWallClockTimeInMS;

	public int numberOfSemanticPredicates = 0;
	public int numberOfManualLookaheadOptions = 0;
	public Set<Integer> setOfNondeterministicDecisionNumbers = new HashSet<Integer>();
	public Set<Integer> setOfNondeterministicDecisionNumbersResolvedWithPredicates =
		new HashSet<Integer>();
	public Set setOfDFAWhoseAnalysisTimedOut = new HashSet();

	/** Track decisions with syn preds specified for reporting.
	 *  This is the a set of BLOCK type AST nodes.
	 */
	public Set<GrammarAST> blocksWithSynPreds = new HashSet();

	/** Track decisions that actually use the syn preds in the DFA.
	 *  Computed during NFA to DFA conversion.
	 */
	public Set<DFA> decisionsWhoseDFAsUsesSynPreds = new HashSet<DFA>();

	/** Track names of preds so we can avoid generating preds that aren't used
	 *  Computed during NFA to DFA conversion.  Just walk accept states
	 *  and look for synpreds because that is the only state target whose
	 *  incident edges can have synpreds.  Same is try for
	 *  decisionsWhoseDFAsUsesSynPreds.
	 */
	public Set<String> synPredNamesUsedInDFA = new HashSet();

	/** Track decisions with syn preds specified for reporting.
	 *  This is the a set of BLOCK type AST nodes.
	 */
	public Set<GrammarAST> blocksWithSemPreds = new HashSet();

	/** Track decisions that actually use the syn preds in the DFA. */
	public Set<DFA> decisionsWhoseDFAsUsesSemPreds = new HashSet();

	protected boolean allDecisionDFACreated = false;

	/** We need a way to detect when a lexer grammar is autogenerated from
	 *  another grammar or we are just sending in a string representing a
	 *  grammar.  We don't want to generate a .tokens file, for example,
	 *  in such cases.
	 */
	protected boolean builtFromString = false;

	/** Factored out the sanity checking code; delegate to it. */
	GrammarSanity sanity = new GrammarSanity(this);

	/** Create a grammar from file name.  */
	public Grammar(Tool tool, String fileName, CompositeGrammar composite) {
		this.composite = composite;
		setTool(tool);
		setFileName(fileName);
		// ensure we have the composite set to something
		if ( composite.delegateGrammarTreeRoot==null ) {
			composite.setDelegationRoot(this);
		}		
	}

	/** Useful for when you are sure that you are not part of a composite
	 *  already.  Used in Interp/RandomPhrase and testing.
	 */
	public Grammar() {
		builtFromString = true;
		composite = new CompositeGrammar(this);
	}

	/** Used for testing; only useful on noncomposite grammars.*/
	public Grammar(String grammarString)
			throws antlr.RecognitionException, antlr.TokenStreamException
	{
		this(null, grammarString);
	}

	/** Used for testing and Interp/RandomPhrase.  Only useful on
	 *  noncomposite grammars.
	 */
	public Grammar(Tool tool, String grammarString)
		throws antlr.RecognitionException
	{
		this();
		setTool(tool);
		setFileName("<string>");
		StringReader r = new StringReader(grammarString);
		parseAndBuildAST(r);
		composite.assignTokenTypes();
		defineGrammarSymbols();
		checkNameSpaceAndActions();
	}

	public void setFileName(String fileName) {
		this.fileName = fileName;
	}

	public String getFileName() {
		return fileName;
	}

	public void setName(String name) {
		if ( name==null ) {
			return;
		}
		// don't error check autogenerated files (those with '__' in them)
		String saneFile = fileName.replace('\\', '/');
		int lastSlash = saneFile.lastIndexOf('/');
		String onlyFileName = saneFile.substring(lastSlash+1, fileName.length());
		if ( !builtFromString ) {
			int lastDot = onlyFileName.lastIndexOf('.');
			String onlyFileNameNoSuffix = null;
			if ( lastDot < 0 ) {
				ErrorManager.error(ErrorManager.MSG_FILENAME_EXTENSION_ERROR, fileName);
				onlyFileNameNoSuffix = onlyFileName+GRAMMAR_FILE_EXTENSION;
			}
			else {
				onlyFileNameNoSuffix = onlyFileName.substring(0,lastDot);
			}
			if ( !name.equals(onlyFileNameNoSuffix) ) {
				ErrorManager.error(ErrorManager.MSG_FILE_AND_GRAMMAR_NAME_DIFFER,
								   name,
								   fileName);
			}
		}
		this.name = name;
	}

	public void setGrammarContent(String grammarString) throws RecognitionException {
		StringReader r = new StringReader(grammarString);
		parseAndBuildAST(r);
		composite.assignTokenTypes();
		composite.defineGrammarSymbols();
	}

	public void parseAndBuildAST()
		throws IOException
	{
		FileReader fr = null;
		BufferedReader br = null;
		try {
			fr = new FileReader(fileName);
			br = new BufferedReader(fr);
			parseAndBuildAST(br);
			br.close();
			br = null;
		}
		finally {
			if ( br!=null ) {
				br.close();
			}
		}
	}

	public void parseAndBuildAST(Reader r) {
		// BUILD AST FROM GRAMMAR
		ANTLRLexer lexer = new ANTLRLexer(r);
		lexer.setFilename(this.getFileName());
		// use the rewrite engine because we want to buffer up all tokens
		// in case they have a merged lexer/parser, send lexer rules to
		// new grammar.
		lexer.setTokenObjectClass("antlr.TokenWithIndex");
		tokenBuffer = new TokenStreamRewriteEngine(lexer);
		tokenBuffer.discard(ANTLRParser.WS);
		tokenBuffer.discard(ANTLRParser.ML_COMMENT);
		tokenBuffer.discard(ANTLRParser.COMMENT);
		tokenBuffer.discard(ANTLRParser.SL_COMMENT);
		ANTLRParser parser = new ANTLRParser(tokenBuffer);
		parser.setFilename(this.getFileName());
		try {
			parser.grammar(this);
		}
		catch (TokenStreamException tse) {
			ErrorManager.internalError("unexpected stream error from parsing "+fileName, tse);
		}
		catch (RecognitionException re) {
			ErrorManager.internalError("unexpected parser recognition error from "+fileName, re);
		}

        dealWithTreeFilterMode(); // tree grammar and filter=true?

        if ( lexer.hasASTOperator && !buildAST() ) {
			Object value = getOption("output");
			if ( value == null ) {
				ErrorManager.grammarWarning(ErrorManager.MSG_REWRITE_OR_OP_WITH_NO_OUTPUT_OPTION,
										    this, null);
				setOption("output", "AST", null);
			}
			else {
				ErrorManager.grammarError(ErrorManager.MSG_AST_OP_WITH_NON_AST_OUTPUT_OPTION,
										  this, null, value);
			}
		}

		grammarTree = (GrammarAST)parser.getAST();
		setFileName(lexer.getFilename()); // the lexer #src might change name
		if ( grammarTree==null || grammarTree.findFirstType(ANTLRParser.RULE)==null ) {
			ErrorManager.error(ErrorManager.MSG_NO_RULES, getFileName());
			return;
		}

		// Get syn pred rules and add to existing tree
		List synpredRules =
			getArtificialRulesForSyntacticPredicates(parser,
													 nameToSynpredASTMap);
		for (int i = 0; i < synpredRules.size(); i++) {
			GrammarAST rAST = (GrammarAST) synpredRules.get(i);
			grammarTree.addChild(rAST);
		}
	}

    protected void dealWithTreeFilterMode() {
        Object filterMode = (String)getOption("filter");
        if ( type==TREE_PARSER && filterMode!=null && filterMode.toString().equals("true") ) {
            // check for conflicting options
            // filter => backtrack=true
            // filter&&output=AST => rewrite=true
            // filter&&output!=AST => error
            // any deviation from valid option set is an error
            Object backtrack = (String)getOption("backtrack");
            Object output = getOption("output");
            Object rewrite = getOption("rewrite");
            if ( backtrack!=null && !backtrack.toString().equals("true") ) {
                ErrorManager.error(ErrorManager.MSG_CONFLICTING_OPTION_IN_TREE_FILTER,
                                   "backtrack", backtrack);
            }
            if ( output!=null && !output.toString().equals("AST") ) {
                ErrorManager.error(ErrorManager.MSG_CONFLICTING_OPTION_IN_TREE_FILTER,
                                   "output", output);
                setOption("output", "", null);
            }
            if ( rewrite!=null && !rewrite.toString().equals("true") ) {
                ErrorManager.error(ErrorManager.MSG_CONFLICTING_OPTION_IN_TREE_FILTER,
                                   "rewrite", rewrite);
            }
            // set options properly
            setOption("backtrack", "true", null);
            if ( output!=null && output.toString().equals("AST") ) {
                setOption("rewrite", "true", null);
            }
            // @synpredgate set to state.backtracking==1 by code gen when filter=true
            // superClass set in template target::treeParser
        }
    }

	public void defineGrammarSymbols() {
		if ( Tool.internalOption_PrintGrammarTree ) {
			System.out.println(grammarTree.toStringList());
		}

		// DEFINE RULES
		//System.out.println("### define "+name+" rules");
		DefineGrammarItemsWalker defineItemsWalker = new DefineGrammarItemsWalker();
		defineItemsWalker.setASTNodeClass("org.antlr.tool.GrammarAST");
		try {
			defineItemsWalker.grammar(grammarTree, this);
		}
		catch (RecognitionException re) {
			ErrorManager.error(ErrorManager.MSG_BAD_AST_STRUCTURE,
							   re);
		}
	}

	/** ANALYZE ACTIONS, LOOKING FOR LABEL AND ATTR REFS, sanity check */
	public void checkNameSpaceAndActions() {
		examineAllExecutableActions();
		checkAllRulesForUselessLabels();

		nameSpaceChecker.checkConflicts();
	}

	/** Many imports are illegal such as lexer into a tree grammar */
	public boolean validImport(Grammar delegate) {
		List<Integer> validDelegators = validDelegations.get(delegate.type);
		return validDelegators!=null && validDelegators.contains(this.type);
	}

	/** If the grammar is a combined grammar, return the text of the implicit
	 *  lexer grammar.
	 */
	public String getLexerGrammar() {
		if ( lexerGrammarST.getAttribute("literals")==null &&
			 lexerGrammarST.getAttribute("rules")==null )
		{
			// if no rules, return nothing
			return null;
		}
		lexerGrammarST.setAttribute("name", name);
		// if there are any actions set for lexer, pass them in
		if ( actions.get("lexer")!=null ) {
			lexerGrammarST.setAttribute("actionNames",
										((Map)actions.get("lexer")).keySet());
			lexerGrammarST.setAttribute("actions",
										((Map)actions.get("lexer")).values());
		}
		// make sure generated grammar has the same options
		if ( options!=null ) {
			Iterator optionNames = options.keySet().iterator();
			while (optionNames.hasNext()) {
				String optionName = (String) optionNames.next();
				if ( !doNotCopyOptionsToLexer.contains(optionName) ) {
					Object value = options.get(optionName);
					lexerGrammarST.setAttribute("options.{name,value}", optionName, value);
				}
			}
		}
		return lexerGrammarST.toString();
	}

	public String getImplicitlyGeneratedLexerFileName() {
		return name+
			   IGNORE_STRING_IN_GRAMMAR_FILE_NAME +
			   LEXER_GRAMMAR_FILE_EXTENSION;
	}

	/** Get the name of the generated recognizer; may or may not be same
	 *  as grammar name.
	 *  Recognizer is TParser and TLexer from T if combined, else
	 *  just use T regardless of grammar type.
	 */
	public String getRecognizerName() {
		String suffix = "";
		List<Grammar> grammarsFromRootToMe = composite.getDelegators(this);
		//System.out.println("grammarsFromRootToMe="+grammarsFromRootToMe);
		String qualifiedName = name;
		if ( grammarsFromRootToMe!=null ) {
			StringBuffer buf = new StringBuffer();
			for (Grammar g : grammarsFromRootToMe) {
				buf.append(g.name);
				buf.append('_');
			}
			buf.append(name);
			qualifiedName = buf.toString();
		}
		if ( type==Grammar.COMBINED ||
			 (type==Grammar.LEXER && implicitLexer) )
		{
			suffix = Grammar.grammarTypeToFileNameSuffix[type];
		}
		return qualifiedName+suffix;
	}

	/** Parse a rule we add artificially that is a list of the other lexer
	 *  rules like this: "Tokens : ID | INT | SEMI ;"  nextToken() will invoke
	 *  this to set the current token.  Add char literals before
	 *  the rule references.
	 *
	 *  If in filter mode, we want every alt to backtrack and we need to
	 *  do k=1 to force the "first token def wins" rule.  Otherwise, the
	 *  longest-match rule comes into play with LL(*).
	 *
	 *  The ANTLRParser antlr.g file now invokes this when parsing a lexer
	 *  grammar, which I think is proper even though it peeks at the info
	 *  that later phases will (re)compute.  It gets a list of lexer rules
	 *  and builds a string representing the rule; then it creates a parser
	 *  and adds the resulting tree to the grammar's tree.
	 */
	public GrammarAST addArtificialMatchTokensRule(GrammarAST grammarAST,
												   List<String> ruleNames,
												   List<String> delegateNames,
												   boolean filterMode) {
		StringTemplate matchTokenRuleST = null;
		if ( filterMode ) {
			matchTokenRuleST = new StringTemplate(
					ARTIFICIAL_TOKENS_RULENAME+
					" options {k=1; backtrack=true;} : <rules; separator=\"|\">;",
					AngleBracketTemplateLexer.class);
		}
		else {
			matchTokenRuleST = new StringTemplate(
					ARTIFICIAL_TOKENS_RULENAME+" : <rules; separator=\"|\">;",
					AngleBracketTemplateLexer.class);
		}

		// Now add token rule references
		for (int i = 0; i < ruleNames.size(); i++) {
			String rname = (String) ruleNames.get(i);
			matchTokenRuleST.setAttribute("rules", rname);
		}
		for (int i = 0; i < delegateNames.size(); i++) {
			String dname = (String) delegateNames.get(i);
			matchTokenRuleST.setAttribute("rules", dname+".Tokens");
		}
		//System.out.println("tokens rule: "+matchTokenRuleST.toString());

		ANTLRLexer lexer = new ANTLRLexer(new StringReader(matchTokenRuleST.toString()));
		lexer.setTokenObjectClass("antlr.TokenWithIndex");
		TokenStreamRewriteEngine tokbuf =
			new TokenStreamRewriteEngine(lexer);
		tokbuf.discard(ANTLRParser.WS);
		tokbuf.discard(ANTLRParser.ML_COMMENT);
		tokbuf.discard(ANTLRParser.COMMENT);
		tokbuf.discard(ANTLRParser.SL_COMMENT);
		ANTLRParser parser = new ANTLRParser(tokbuf);
		parser.setGrammar(this);
		parser.setGtype(ANTLRParser.LEXER_GRAMMAR);
		parser.setASTNodeClass("org.antlr.tool.GrammarAST");
		try {
			parser.rule();
			if ( Tool.internalOption_PrintGrammarTree ) {
				System.out.println("Tokens rule: "+parser.getAST().toStringTree());
			}
			GrammarAST p = grammarAST;
			while ( p.getType()!=ANTLRParser.LEXER_GRAMMAR ) {
				p = (GrammarAST)p.getNextSibling();
			}
			p.addChild(parser.getAST());
		}
		catch (Exception e) {
			ErrorManager.error(ErrorManager.MSG_ERROR_CREATING_ARTIFICIAL_RULE,
							   e);
		}
		return (GrammarAST)parser.getAST();
	}

	/** for any syntactic predicates, we need to define rules for them; they will get
	 *  defined automatically like any other rule. :)
	 */
	protected List getArtificialRulesForSyntacticPredicates(ANTLRParser parser,
															LinkedHashMap nameToSynpredASTMap)
	{
		List rules = new ArrayList();
		if ( nameToSynpredASTMap==null ) {
			return rules;
		}
		Set predNames = nameToSynpredASTMap.keySet();
		boolean isLexer = grammarTree.getType()==ANTLRParser.LEXER_GRAMMAR;
		for (Iterator it = predNames.iterator(); it.hasNext();) {
			String synpredName = (String)it.next();
			GrammarAST fragmentAST =
				(GrammarAST) nameToSynpredASTMap.get(synpredName);
			GrammarAST ruleAST =
				parser.createSimpleRuleAST(synpredName,
										   fragmentAST,
										   isLexer);
			rules.add(ruleAST);
		}
		return rules;
	}

	/** Walk the list of options, altering this Grammar object according
	 *  to any I recognize.
	protected void processOptions() {
		Iterator optionNames = options.keySet().iterator();
		while (optionNames.hasNext()) {
			String optionName = (String) optionNames.next();
			Object value = options.get(optionName);
			if ( optionName.equals("tokenVocab") ) {

			}
		}
	}
	 */

	/** Define all the rule begin/end NFAStates to solve forward reference
	 *  issues.  Critical for composite grammars too.
	 *  This is normally called on all root/delegates manually and then
	 *  buildNFA() is called afterwards because the NFA construction needs
	 *  to see rule start/stop states from potentially every grammar. Has
	 *  to be have these created a priori.  Testing routines will often
	 *  just call buildNFA(), which forces a call to this method if not
	 *  done already. Works ONLY for single noncomposite grammars.
	 */
	public void createRuleStartAndStopNFAStates() {
		//System.out.println("### createRuleStartAndStopNFAStates "+getGrammarTypeString()+" grammar "+name+" NFAs");
		if ( nfa!=null ) {
			return;
		}
		nfa = new NFA(this);
		factory = new NFAFactory(nfa);

		Collection rules = getRules();
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

	public void buildNFA() {
		if ( nfa==null ) {
			createRuleStartAndStopNFAStates();
		}
		if ( nfa.complete ) {
			// don't let it create more than once; has side-effects
			return;
		}
		//System.out.println("### build "+getGrammarTypeString()+" grammar "+name+" NFAs");
		if ( getRules().size()==0 ) {
			return;
		}

		TreeToNFAConverter nfaBuilder = new TreeToNFAConverter(this, nfa, factory);
		try {
			nfaBuilder.grammar(grammarTree);
		}
		catch (RecognitionException re) {
			ErrorManager.error(ErrorManager.MSG_BAD_AST_STRUCTURE,
							   name,
							   re);
		}
		nfa.complete = true;
	}

	/** For each decision in this grammar, compute a single DFA using the
	 *  NFA states associated with the decision.  The DFA construction
	 *  determines whether or not the alternatives in the decision are
	 *  separable using a regular lookahead language.
	 *
	 *  Store the lookahead DFAs in the AST created from the user's grammar
	 *  so the code generator or whoever can easily access it.
	 *
	 *  This is a separate method because you might want to create a
	 *  Grammar without doing the expensive analysis.
	 */
	public void createLookaheadDFAs() {
		createLookaheadDFAs(true);
	}

	public void createLookaheadDFAs(boolean wackTempStructures) {
		if ( nfa==null ) {
			buildNFA();
		}

		// CHECK FOR LEFT RECURSION; Make sure we can actually do analysis
		checkAllRulesForLeftRecursion();

		/*
		// was there a severe problem while sniffing the grammar?
		if ( ErrorManager.doNotAttemptAnalysis() ) {
			return;
		}
		*/

		long start = System.currentTimeMillis();

		//System.out.println("### create DFAs");
		int numDecisions = getNumberOfDecisions();
		if ( NFAToDFAConverter.SINGLE_THREADED_NFA_CONVERSION ) {
			for (int decision=1; decision<=numDecisions; decision++) {
				NFAState decisionStartState = getDecisionNFAStartState(decision);
				if ( leftRecursiveRules.contains(decisionStartState.enclosingRule) ) {
					// don't bother to process decisions within left recursive rules.
					if ( composite.watchNFAConversion ) {
						System.out.println("ignoring decision "+decision+
										   " within left-recursive rule "+decisionStartState.enclosingRule.name);
					}
					continue;
				}
				if ( !externalAnalysisAbort && decisionStartState.getNumberOfTransitions()>1 ) {
					Rule r = decisionStartState.enclosingRule;
					if ( r.isSynPred && !synPredNamesUsedInDFA.contains(r.name) ) {
						continue;
					}
					DFA dfa = null;
					// if k=* or k=1, try LL(1)
					if ( getUserMaxLookahead(decision)==0 ||
						 getUserMaxLookahead(decision)==1 )
					{
						dfa = createLL_1_LookaheadDFA(decision);
					}
					if ( dfa==null ) {
						if ( composite.watchNFAConversion ) {
							System.out.println("decision "+decision+
											   " not suitable for LL(1)-optimized DFA analysis");
						}
						dfa = createLookaheadDFA(decision, wackTempStructures);
					}
					if ( dfa.startState==null ) {
						// something went wrong; wipe out DFA
						setLookaheadDFA(decision, null);
					}
					if ( Tool.internalOption_PrintDFA ) {
						System.out.println("DFA d="+decision);
						FASerializer serializer = new FASerializer(nfa.grammar);
						String result = serializer.serialize(dfa.startState);
						System.out.println(result);
					}
				}
			}
		}
		else {
			ErrorManager.info("two-threaded DFA conversion");
			// create a barrier expecting n DFA and this main creation thread
			Barrier barrier = new Barrier(3);
			// assume 2 CPU for now
			int midpoint = numDecisions/2;
			NFAConversionThread t1 =
				new NFAConversionThread(this, barrier, 1, midpoint);
			new Thread(t1).start();
			if ( midpoint == (numDecisions/2) ) {
				midpoint++;
			}
			NFAConversionThread t2 =
				new NFAConversionThread(this, barrier, midpoint, numDecisions);
			new Thread(t2).start();
			// wait for these two threads to finish
			try {
				barrier.waitForRelease();
			}
			catch(InterruptedException e) {
				ErrorManager.internalError("what the hell? DFA interruptus", e);
			}
		}

		long stop = System.currentTimeMillis();
		DFACreationWallClockTimeInMS = stop - start;

		// indicate that we've finished building DFA (even if #decisions==0)
		allDecisionDFACreated = true;
	}

	public DFA createLL_1_LookaheadDFA(int decision) {
		Decision d = getDecision(decision);
		String enclosingRule = d.startState.enclosingRule.name;
		Rule r = d.startState.enclosingRule;
		NFAState decisionStartState = getDecisionNFAStartState(decision);

		if ( composite.watchNFAConversion ) {
			System.out.println("--------------------\nattempting LL(1) DFA (d="
							   +decisionStartState.getDecisionNumber()+") for "+
							   decisionStartState.getDescription());
		}

		if ( r.isSynPred && !synPredNamesUsedInDFA.contains(enclosingRule) ) {
			return null;
		}

		// compute lookahead for each alt
		int numAlts = getNumberOfAltsForDecisionNFA(decisionStartState);
		LookaheadSet[] altLook = new LookaheadSet[numAlts+1];
		for (int alt = 1; alt <= numAlts; alt++) {
			int walkAlt =
				decisionStartState.translateDisplayAltToWalkAlt(alt);
			NFAState altLeftEdge = getNFAStateForAltOfDecision(decisionStartState, walkAlt);
			NFAState altStartState = (NFAState)altLeftEdge.transition[0].target;
			//System.out.println("alt "+alt+" start state = "+altStartState.stateNumber);
			altLook[alt] = ll1Analyzer.LOOK(altStartState);
			//System.out.println("alt "+alt+": "+altLook[alt].toString(this));
		}

		// compare alt i with alt j for disjointness
		boolean decisionIsLL_1 = true;
outer:
		for (int i = 1; i <= numAlts; i++) {
			for (int j = i+1; j <= numAlts; j++) {
				/*
				System.out.println("compare "+i+", "+j+": "+
								   altLook[i].toString(this)+" with "+
								   altLook[j].toString(this));
				*/
				LookaheadSet collision = altLook[i].intersection(altLook[j]);
				if ( !collision.isNil() ) {
					//System.out.println("collision (non-LL(1)): "+collision.toString(this));
					decisionIsLL_1 = false;
					break outer;
				}
			}
		}

		boolean foundConfoundingPredicate =
			ll1Analyzer.detectConfoundingPredicates(decisionStartState);
		if ( decisionIsLL_1 && !foundConfoundingPredicate ) {
			// build an LL(1) optimized DFA with edge for each altLook[i]
			if ( NFAToDFAConverter.debug ) {
				System.out.println("decision "+decision+" is simple LL(1)");
			}
			DFA lookaheadDFA = new LL1DFA(decision, decisionStartState, altLook);
			setLookaheadDFA(decision, lookaheadDFA);
			updateLineColumnToLookaheadDFAMap(lookaheadDFA);
			return lookaheadDFA;
		}

		// not LL(1) but perhaps we can solve with simplified predicate search
		// even if k=1 set manually, only resolve here if we have preds; i.e.,
		// don't resolve etc...

		/*
		SemanticContext visiblePredicates =
			ll1Analyzer.getPredicates(decisionStartState);
		boolean foundConfoundingPredicate =
			ll1Analyzer.detectConfoundingPredicates(decisionStartState);
			*/

		// exit if not forced k=1 or we found a predicate situation we
		// can't handle: predicates in rules invoked from this decision.
		if ( getUserMaxLookahead(decision)!=1 || // not manually set to k=1
			 !getAutoBacktrackMode(decision) ||
			 foundConfoundingPredicate )
		{
			//System.out.println("trying LL(*)");
			return null;
		}

		List<IntervalSet> edges = new ArrayList<IntervalSet>();
		for (int i = 1; i < altLook.length; i++) {
			LookaheadSet s = altLook[i];
			edges.add((IntervalSet)s.tokenTypeSet);
		}
		List<IntervalSet> disjoint = makeEdgeSetsDisjoint(edges);
		//System.out.println("disjoint="+disjoint);

		MultiMap<IntervalSet, Integer> edgeMap = new MultiMap<IntervalSet, Integer>();
		for (int i = 0; i < disjoint.size(); i++) {
			IntervalSet ds = (IntervalSet) disjoint.get(i);
			for (int alt = 1; alt < altLook.length; alt++) {
				LookaheadSet look = altLook[alt];
				if ( !ds.and(look.tokenTypeSet).isNil() ) {
					edgeMap.map(ds, alt);
				}
			}
		}
		//System.out.println("edge map: "+edgeMap);

		// TODO: how do we know we covered stuff?

		// build an LL(1) optimized DFA with edge for each altLook[i]
		DFA lookaheadDFA = new LL1DFA(decision, decisionStartState, edgeMap);
		setLookaheadDFA(decision, lookaheadDFA);

		// create map from line:col to decision DFA (for ANTLRWorks)
		updateLineColumnToLookaheadDFAMap(lookaheadDFA);

		return lookaheadDFA;
	}

	private void updateLineColumnToLookaheadDFAMap(DFA lookaheadDFA) {
		GrammarAST decisionAST = nfa.grammar.getDecisionBlockAST(lookaheadDFA.decisionNumber);
		int line = decisionAST.getLine();
		int col = decisionAST.getColumn();
		lineColumnToLookaheadDFAMap.put(new StringBuffer().append(line + ":")
										.append(col).toString(), lookaheadDFA);
	}

	protected List<IntervalSet> makeEdgeSetsDisjoint(List<IntervalSet> edges) {
		OrderedHashSet<IntervalSet> disjointSets = new OrderedHashSet<IntervalSet>();
		// walk each incoming edge label/set and add to disjoint set
		int numEdges = edges.size();
		for (int e = 0; e < numEdges; e++) {
			IntervalSet t = (IntervalSet) edges.get(e);
			if ( disjointSets.contains(t) ) { // exact set present
				continue;
			}

			// compare t with set i for disjointness
			IntervalSet remainder = t; // remainder starts out as whole set to add
			int numDisjointElements = disjointSets.size();
			for (int i = 0; i < numDisjointElements; i++) {
				IntervalSet s_i = (IntervalSet)disjointSets.get(i);

				if ( t.and(s_i).isNil() ) { // nothing in common
					continue;
				}
				//System.out.println(label+" collides with "+rl);

				// For any (s_i, t) with s_i&t!=nil replace with (s_i-t, s_i&t)
				// (ignoring s_i-t if nil; don't put in list)

				// Replace existing s_i with intersection since we
				// know that will always be a non nil character class
				IntervalSet intersection = (IntervalSet)s_i.and(t);
				disjointSets.set(i, intersection);

				// Compute s_i-t to see what is in current set and not in incoming
				IntSet existingMinusNewElements = s_i.subtract(t);
				//System.out.println(s_i+"-"+t+"="+existingMinusNewElements);
				if ( !existingMinusNewElements.isNil() ) {
					// found a new character class, add to the end (doesn't affect
					// outer loop duration due to n computation a priori.
					disjointSets.add(existingMinusNewElements);
				}

				// anything left to add to the reachableLabels?
				remainder = (IntervalSet)t.subtract(s_i);
				if ( remainder.isNil() ) {
					break; // nothing left to add to set.  done!
				}

				t = remainder;
			}
			if ( !remainder.isNil() ) {
				disjointSets.add(remainder);
			}
		}
		return disjointSets.elements();
	}

	public DFA createLookaheadDFA(int decision, boolean wackTempStructures) {
		Decision d = getDecision(decision);
		String enclosingRule = d.startState.enclosingRule.name;
		Rule r = d.startState.enclosingRule;

		//System.out.println("createLookaheadDFA(): "+enclosingRule+" dec "+decision+"; synprednames prev used "+synPredNamesUsedInDFA);
		NFAState decisionStartState = getDecisionNFAStartState(decision);
		long startDFA=0,stopDFA=0;
		if ( composite.watchNFAConversion ) {
			System.out.println("--------------------\nbuilding lookahead DFA (d="
							   +decisionStartState.getDecisionNumber()+") for "+
							   decisionStartState.getDescription());
			startDFA = System.currentTimeMillis();
		}

		DFA lookaheadDFA = new DFA(decision, decisionStartState);
		// Retry to create a simpler DFA if analysis failed (non-LL(*),
		// recursion overflow, or time out).
		boolean failed =
			lookaheadDFA.analysisTimedOut() ||
			lookaheadDFA.probe.isNonLLStarDecision() ||
			lookaheadDFA.probe.analysisOverflowed();
		if ( failed && lookaheadDFA.okToRetryDFAWithK1() ) {
			// set k=1 option and try again.
			// First, clean up tracking stuff
			decisionsWhoseDFAsUsesSynPreds.remove(lookaheadDFA);
			// TODO: clean up synPredNamesUsedInDFA also (harder)
			d.blockAST.setBlockOption(this, "k", Utils.integer(1));
			if ( composite.watchNFAConversion ) {
				System.out.print("trying decision "+decision+
								 " again with k=1; reason: "+
								 lookaheadDFA.getReasonForFailure());
			}
			lookaheadDFA = null; // make sure other memory is "free" before redoing
			lookaheadDFA = new DFA(decision, decisionStartState);
		}
		if ( lookaheadDFA.analysisTimedOut() ) { // did analysis bug out?
			ErrorManager.internalError("could not even do k=1 for decision "+
									   decision+"; reason: "+
									   lookaheadDFA.getReasonForFailure());
		}


		setLookaheadDFA(decision, lookaheadDFA);

		if ( wackTempStructures ) {
			for (DFAState s : lookaheadDFA.getUniqueStates().values()) {
				s.reset();
			}
		}

		// create map from line:col to decision DFA (for ANTLRWorks)
		updateLineColumnToLookaheadDFAMap(lookaheadDFA);

		if ( composite.watchNFAConversion ) {
			stopDFA = System.currentTimeMillis();
			System.out.println("cost: "+lookaheadDFA.getNumberOfStates()+
							   " states, "+(int)(stopDFA-startDFA)+" ms");
		}
		//System.out.println("after create DFA; synPredNamesUsedInDFA="+synPredNamesUsedInDFA);
		return lookaheadDFA;
	}

	/** Terminate DFA creation (grammar analysis).
	 */
	public void externallyAbortNFAToDFAConversion() {
		externalAnalysisAbort = true;
	}

	public boolean NFAToDFAConversionExternallyAborted() {
		return externalAnalysisAbort;
	}

	/** Return a new unique integer in the token type space */
	public int getNewTokenType() {
		composite.maxTokenType++;
		return composite.maxTokenType;
	}

	/** Define a token at a particular token type value.  Blast an
	 *  old value with a new one.  This is called normal grammar processsing
	 *  and during import vocab operations to set tokens with specific values.
	 */
	public void defineToken(String text, int tokenType) {
		//System.out.println("defineToken("+text+", "+tokenType+")");
		if ( composite.tokenIDToTypeMap.get(text)!=null ) {
			// already defined?  Must be predefined one like EOF;
			// do nothing
			return;
		}
		// the index in the typeToTokenList table is actually shifted to
		// hold faux labels as you cannot have negative indices.
		if ( text.charAt(0)=='\'' ) {
			composite.stringLiteralToTypeMap.put(text, Utils.integer(tokenType));
			// track in reverse index too
			if ( tokenType>=composite.typeToStringLiteralList.size() ) {
				composite.typeToStringLiteralList.setSize(tokenType+1);
			}
			composite.typeToStringLiteralList.set(tokenType, text);
		}
		else { // must be a label like ID
			composite.tokenIDToTypeMap.put(text, Utils.integer(tokenType));
		}
		int index = Label.NUM_FAUX_LABELS+tokenType-1;
		//System.out.println("defining "+name+" token "+text+" at type="+tokenType+", index="+index);
		composite.maxTokenType = Math.max(composite.maxTokenType, tokenType);
		if ( index>=composite.typeToTokenList.size() ) {
			composite.typeToTokenList.setSize(index+1);
		}
		String prevToken = (String)composite.typeToTokenList.get(index);
		if ( prevToken==null || prevToken.charAt(0)=='\'' ) {
			// only record if nothing there before or if thing before was a literal
			composite.typeToTokenList.set(index, text);
		}
	}

	/** Define a new rule.  A new rule index is created by incrementing
	 *  ruleIndex.
	 */
	public void defineRule(antlr.Token ruleToken,
						   String modifier,
						   Map options,
						   GrammarAST tree,
						   GrammarAST argActionAST,
						   int numAlts)
	{
		String ruleName = ruleToken.getText();
		if ( getLocallyDefinedRule(ruleName)!=null ) {
			ErrorManager.grammarError(ErrorManager.MSG_RULE_REDEFINITION,
									  this, ruleToken, ruleName);
			return;
		}

		if ( (type==Grammar.PARSER||type==Grammar.TREE_PARSER) &&
			 Character.isUpperCase(ruleName.charAt(0)) )
		{
			ErrorManager.grammarError(ErrorManager.MSG_LEXER_RULES_NOT_ALLOWED,
									  this, ruleToken, ruleName);
			return;
		}

		Rule r = new Rule(this, ruleName, composite.ruleIndex, numAlts);
		/*
		System.out.println("defineRule("+ruleName+",modifier="+modifier+
						   "): index="+r.index+", nalts="+numAlts);
		*/
		r.modifier = modifier;
		nameToRuleMap.put(ruleName, r);
		setRuleAST(ruleName, tree);
		r.setOptions(options, ruleToken);
		r.argActionAST = argActionAST;
		composite.ruleIndexToRuleList.setSize(composite.ruleIndex+1);
		composite.ruleIndexToRuleList.set(composite.ruleIndex, r);
		composite.ruleIndex++;
		if ( ruleName.startsWith(SYNPRED_RULE_PREFIX) ) {
			r.isSynPred = true;
		}
	}

	/** Define a new predicate and get back its name for use in building
	 *  a semantic predicate reference to the syn pred.
	 */
	public String defineSyntacticPredicate(GrammarAST blockAST,
										   String currentRuleName)
	{
		if ( nameToSynpredASTMap==null ) {
			nameToSynpredASTMap = new LinkedHashMap();
		}
		String predName =
			SYNPRED_RULE_PREFIX+(nameToSynpredASTMap.size() + 1)+"_"+name;
		blockAST.setTreeEnclosingRuleNameDeeply(predName);
		nameToSynpredASTMap.put(predName, blockAST);
		return predName;
	}

	public LinkedHashMap getSyntacticPredicates() {
		return nameToSynpredASTMap;
	}

	public GrammarAST getSyntacticPredicate(String name) {
		if ( nameToSynpredASTMap==null ) {
			return null;
		}
		return (GrammarAST)nameToSynpredASTMap.get(name);
	}

	public void synPredUsedInDFA(DFA dfa, SemanticContext semCtx) {
		decisionsWhoseDFAsUsesSynPreds.add(dfa);
		semCtx.trackUseOfSyntacticPredicates(this); // walk ctx looking for preds
	}

	/*
	public Set<Rule> getRuleNamesVisitedDuringLOOK() {
		return rulesSensitiveToOtherRules;
	}
	*/

	/** Given @scope::name {action} define it for this grammar.  Later,
	 *  the code generator will ask for the actions table.  For composite
     *  grammars, make sure header action propogates down to all delegates.
	 */
	public void defineNamedAction(GrammarAST ampersandAST,
								  String scope,
								  GrammarAST nameAST,
								  GrammarAST actionAST)
	{
		if ( scope==null ) {
			scope = getDefaultActionScope(type);
		}
		//System.out.println("@"+scope+"::"+nameAST.getText()+"{"+actionAST.getText()+"}");
		String actionName = nameAST.getText();
		Map scopeActions = (Map)actions.get(scope);
		if ( scopeActions==null ) {
			scopeActions = new HashMap();
			actions.put(scope, scopeActions);
		}
		GrammarAST a = (GrammarAST)scopeActions.get(actionName);
		if ( a!=null ) {
			ErrorManager.grammarError(
				ErrorManager.MSG_ACTION_REDEFINITION,this,
				nameAST.getToken(),nameAST.getText());
		}
		else {
			scopeActions.put(actionName,actionAST);
		}
        // propogate header (regardless of scope (lexer, parser, ...) ?
        if ( this==composite.getRootGrammar() && actionName.equals("header") ) {
            List<Grammar> allgrammars = composite.getRootGrammar().getDelegates();
            for (Grammar g : allgrammars) {
                g.defineNamedAction(ampersandAST, scope, nameAST, actionAST);
            }
        }
    }

    public void setSynPredGateIfNotAlready(StringTemplate gateST) {
        String scope = getDefaultActionScope(type);
        Map actionsForGrammarScope = (Map)actions.get(scope);
        // if no synpredgate action set by user then set
        if ( (actionsForGrammarScope==null ||
             !actionsForGrammarScope.containsKey(Grammar.SYNPREDGATE_ACTION_NAME)) )
        {
            if ( actionsForGrammarScope==null ) {
                actionsForGrammarScope=new HashMap();
                actions.put(scope, actionsForGrammarScope);
            }
            actionsForGrammarScope.put(Grammar.SYNPREDGATE_ACTION_NAME,
                                       gateST);
        }
    }

	public Map getActions() {
		return actions;
	}

	/** Given a grammar type, what should be the default action scope?
	 *  If I say @members in a COMBINED grammar, for example, the
	 *  default scope should be "parser".
	 */
	public String getDefaultActionScope(int grammarType) {
		switch (grammarType) {
			case Grammar.LEXER :
				return "lexer";
			case Grammar.PARSER :
			case Grammar.COMBINED :
				return "parser";
			case Grammar.TREE_PARSER :
				return "treeparser";
		}
		return null;
	}

	public void defineLexerRuleFoundInParser(antlr.Token ruleToken,
											 GrammarAST ruleAST)
	{
		//System.out.println("rule tree is:\n"+ruleAST.toStringTree());
		/*
		String ruleText = tokenBuffer.toOriginalString(ruleAST.ruleStartTokenIndex,
											   ruleAST.ruleStopTokenIndex);
		*/
		// first, create the text of the rule
		StringBuffer buf = new StringBuffer();
		buf.append("// $ANTLR src \"");
		buf.append(getFileName());
		buf.append("\" ");
		buf.append(ruleAST.getLine());
		buf.append("\n");
		for (int i=ruleAST.ruleStartTokenIndex;
			 i<=ruleAST.ruleStopTokenIndex && i<tokenBuffer.size();
			 i++)
		{
			TokenWithIndex t = (TokenWithIndex)tokenBuffer.getToken(i);
			// undo the text deletions done by the lexer (ugh)
			if ( t.getType()==ANTLRParser.BLOCK ) {
				buf.append("(");
			}
			else if ( t.getType()==ANTLRParser.ACTION ) {
				buf.append("{");
				buf.append(t.getText());
				buf.append("}");
			}
			else if ( t.getType()==ANTLRParser.SEMPRED ||
					  t.getType()==ANTLRParser.SYN_SEMPRED ||
					  t.getType()==ANTLRParser.GATED_SEMPRED ||
					  t.getType()==ANTLRParser.BACKTRACK_SEMPRED )
			{
				buf.append("{");
				buf.append(t.getText());
				buf.append("}?");
			}
			else if ( t.getType()==ANTLRParser.ARG_ACTION ) {
				buf.append("[");
				buf.append(t.getText());
				buf.append("]");
			}
			else {
				buf.append(t.getText());
			}
		}
		String ruleText = buf.toString();
		//System.out.println("[["+ruleText+"]]");
		// now put the rule into the lexer grammar template
		if ( getGrammarIsRoot() ) { // don't build lexers for delegates
			lexerGrammarST.setAttribute("rules", ruleText);
		}
		// track this lexer rule's name
		composite.lexerRules.add(ruleToken.getText());
	}

	/** If someone does PLUS='+' in the parser, must make sure we get
	 *  "PLUS : '+' ;" in lexer not "T73 : '+';"
	 */
	public void defineLexerRuleForAliasedStringLiteral(String tokenID,
													   String literal,
													   int tokenType)
	{
		if ( getGrammarIsRoot() ) { // don't build lexers for delegates
			//System.out.println("defineLexerRuleForAliasedStringLiteral: "+literal+" "+tokenType);
			lexerGrammarST.setAttribute("literals.{ruleName,type,literal}",
										tokenID,
										Utils.integer(tokenType),
										literal);
		}
		// track this lexer rule's name
		composite.lexerRules.add(tokenID);
	}

	public void defineLexerRuleForStringLiteral(String literal, int tokenType) {
		//System.out.println("defineLexerRuleForStringLiteral: "+literal+" "+tokenType);
		// compute new token name like T237 and define it as having tokenType
		String tokenID = computeTokenNameFromLiteral(tokenType,literal);
		defineToken(tokenID, tokenType);
		// tell implicit lexer to define a rule to match the literal
		if ( getGrammarIsRoot() ) { // don't build lexers for delegates
			lexerGrammarST.setAttribute("literals.{ruleName,type,literal}",
										tokenID,
										Utils.integer(tokenType),
										literal);
		}
	}

	public Rule getLocallyDefinedRule(String ruleName) {
		Rule r = nameToRuleMap.get(ruleName);
		return r;
	}

	public Rule getRule(String ruleName) {
		Rule r = composite.getRule(ruleName);
		/*
		if ( r!=null && r.grammar != this ) {
			System.out.println(name+".getRule("+ruleName+")="+r);
		}
		*/
		return r;
	}

	public Rule getRule(String scopeName, String ruleName) {
		if ( scopeName!=null ) { // scope override
			Grammar scope = composite.getGrammar(scopeName);
			if ( scope==null ) {
				return null;
			}
			return scope.getLocallyDefinedRule(ruleName);
		}
		return getRule(ruleName);
	}

	public int getRuleIndex(String scopeName, String ruleName) {
		Rule r = getRule(scopeName, ruleName);
		if ( r!=null ) {
			return r.index;
		}
		return INVALID_RULE_INDEX;
	}

	public int getRuleIndex(String ruleName) {
		return getRuleIndex(null, ruleName);
	}

	public String getRuleName(int ruleIndex) {
		Rule r = composite.ruleIndexToRuleList.get(ruleIndex);
		if ( r!=null ) {
			return r.name;
		}
		return null;
	}

	/** Should codegen.g gen rule for ruleName?
	 * 	If synpred, only gen if used in a DFA.
	 *  If regular rule, only gen if not overridden in delegator
	 *  Always gen Tokens rule though.
	 */
	public boolean generateMethodForRule(String ruleName) {
		if ( ruleName.equals(ARTIFICIAL_TOKENS_RULENAME) ) {
			// always generate Tokens rule to satisfy lexer interface
			// but it may have no alternatives.
			return true;
		}
		if ( overriddenRules.contains(ruleName) ) {
			// don't generate any overridden rules
			return false;
		}
		// generate if non-synpred or synpred used in a DFA
		Rule r = getLocallyDefinedRule(ruleName);
		return !r.isSynPred ||
			   (r.isSynPred&&synPredNamesUsedInDFA.contains(ruleName));
	}

	public AttributeScope defineGlobalScope(String name, Token scopeAction) {
		AttributeScope scope = new AttributeScope(this, name, scopeAction);
		scopes.put(name,scope);
		return scope;
	}

	public AttributeScope createReturnScope(String ruleName, Token retAction) {
		AttributeScope scope = new AttributeScope(this, ruleName, retAction);
		scope.isReturnScope = true;
		return scope;
	}

	public AttributeScope createRuleScope(String ruleName, Token scopeAction) {
		AttributeScope scope = new AttributeScope(this, ruleName, scopeAction);
		scope.isDynamicRuleScope = true;
		return scope;
	}

	public AttributeScope createParameterScope(String ruleName, Token argAction) {
		AttributeScope scope = new AttributeScope(this, ruleName, argAction);
		scope.isParameterScope = true;
		return scope;
	}

	/** Get a global scope */
	public AttributeScope getGlobalScope(String name) {
		return (AttributeScope)scopes.get(name);
	}

	public Map getGlobalScopes() {
		return scopes;
	}

	/** Define a label defined in a rule r; check the validity then ask the
	 *  Rule object to actually define it.
	 */
	protected void defineLabel(Rule r, antlr.Token label, GrammarAST element, int type) {
		boolean err = nameSpaceChecker.checkForLabelTypeMismatch(r, label, type);
		if ( err ) {
			return;
		}
		r.defineLabel(label, element, type);
	}

	public void defineTokenRefLabel(String ruleName,
									antlr.Token label,
									GrammarAST tokenRef)
	{
		Rule r = getLocallyDefinedRule(ruleName);
		if ( r!=null ) {
			if ( type==LEXER &&
				 (tokenRef.getType()==ANTLRParser.CHAR_LITERAL||
				  tokenRef.getType()==ANTLRParser.BLOCK||
				  tokenRef.getType()==ANTLRParser.NOT||
				  tokenRef.getType()==ANTLRParser.CHAR_RANGE||
				  tokenRef.getType()==ANTLRParser.WILDCARD))
			{
				defineLabel(r, label, tokenRef, CHAR_LABEL);
			}
            else {
				defineLabel(r, label, tokenRef, TOKEN_LABEL);
			}
		}
	}

    public void defineWildcardTreeLabel(String ruleName,
                                           antlr.Token label,
                                           GrammarAST tokenRef)
    {
        Rule r = getLocallyDefinedRule(ruleName);
        if ( r!=null ) {
            defineLabel(r, label, tokenRef, WILDCARD_TREE_LABEL);
        }
    }

    public void defineWildcardTreeListLabel(String ruleName,
                                           antlr.Token label,
                                           GrammarAST tokenRef)
    {
        Rule r = getLocallyDefinedRule(ruleName);
        if ( r!=null ) {
            defineLabel(r, label, tokenRef, WILDCARD_TREE_LIST_LABEL);
        }
    }

    public void defineRuleRefLabel(String ruleName,
								   antlr.Token label,
								   GrammarAST ruleRef)
	{
		Rule r = getLocallyDefinedRule(ruleName);
		if ( r!=null ) {
			defineLabel(r, label, ruleRef, RULE_LABEL);
		}
	}

	public void defineTokenListLabel(String ruleName,
									 antlr.Token label,
									 GrammarAST element)
	{
		Rule r = getLocallyDefinedRule(ruleName);
		if ( r!=null ) {
			defineLabel(r, label, element, TOKEN_LIST_LABEL);
		}
	}

	public void defineRuleListLabel(String ruleName,
									antlr.Token label,
									GrammarAST element)
	{
		Rule r = getLocallyDefinedRule(ruleName);
		if ( r!=null ) {
			if ( !r.getHasMultipleReturnValues() ) {
				ErrorManager.grammarError(
					ErrorManager.MSG_LIST_LABEL_INVALID_UNLESS_RETVAL_STRUCT,this,
					label,label.getText());
			}
			defineLabel(r, label, element, RULE_LIST_LABEL);
		}
	}

	/** Given a set of all rewrite elements on right of ->, filter for
	 *  label types such as Grammar.TOKEN_LABEL, Grammar.TOKEN_LIST_LABEL, ...
	 *  Return a displayable token type name computed from the GrammarAST.
	 */
	public Set<String> getLabels(Set<GrammarAST> rewriteElements, int labelType) {
		Set<String> labels = new HashSet<String>();
		for (Iterator it = rewriteElements.iterator(); it.hasNext();) {
			GrammarAST el = (GrammarAST) it.next();
			if ( el.getType()==ANTLRParser.LABEL ) {
				String labelName = el.getText();
				Rule enclosingRule = getLocallyDefinedRule(el.enclosingRuleName);
				LabelElementPair pair = enclosingRule.getLabel(labelName);
                /*
                // if tree grammar and we have a wildcard, only notice it
                // when looking for rule labels not token label. x=. should
                // look like a rule ref since could be subtree.
                if ( type==TREE_PARSER && pair!=null &&
                     pair.elementRef.getType()==ANTLRParser.WILDCARD )
                {
                    if ( labelType==WILDCARD_TREE_LABEL ) {
                        labels.add(labelName);
                        continue;
                    }
                    else continue;
                }
                 */
                // if valid label and type is what we're looking for
				// and not ref to old value val $rule, add to list
				if ( pair!=null && pair.type==labelType &&
					 !labelName.equals(el.enclosingRuleName) )
				{
					labels.add(labelName);
				}
			}
		}
		return labels;
	}

	/** Before generating code, we examine all actions that can have
	 *  $x.y and $y stuff in them because some code generation depends on
	 *  Rule.referencedPredefinedRuleAttributes.  I need to remove unused
	 *  rule labels for example.
	 */
	protected void examineAllExecutableActions() {
		Collection rules = getRules();
		for (Iterator it = rules.iterator(); it.hasNext();) {
			Rule r = (Rule) it.next();
			// walk all actions within the rule elements, args, and exceptions
			List<GrammarAST> actions = r.getInlineActions();
			for (int i = 0; i < actions.size(); i++) {
				GrammarAST actionAST = (GrammarAST) actions.get(i);
				ActionAnalysis sniffer =
					new ActionAnalysis(this, r.name, actionAST);
				sniffer.analyze();
			}
			// walk any named actions like @init, @after
			Collection<GrammarAST> namedActions = r.getActions().values();
			for (Iterator it2 = namedActions.iterator(); it2.hasNext();) {
				GrammarAST actionAST = (GrammarAST) it2.next();
				ActionAnalysis sniffer =
					new ActionAnalysis(this, r.name, actionAST);
				sniffer.analyze();
			}
		}
	}

	/** Remove all labels on rule refs whose target rules have no return value.
	 *  Do this for all rules in grammar.
	 */
	public void checkAllRulesForUselessLabels() {
		if ( type==LEXER ) {
			return;
		}
		Set rules = nameToRuleMap.keySet();
		for (Iterator it = rules.iterator(); it.hasNext();) {
			String ruleName = (String) it.next();
			Rule r = getRule(ruleName);
			removeUselessLabels(r.getRuleLabels());
			removeUselessLabels(r.getRuleListLabels());
		}
	}

	/** A label on a rule is useless if the rule has no return value, no
	 *  tree or template output, and it is not referenced in an action.
	 */
	protected void removeUselessLabels(Map ruleToElementLabelPairMap) {
		if ( ruleToElementLabelPairMap==null ) {
			return;
		}
		Collection labels = ruleToElementLabelPairMap.values();
		List kill = new ArrayList();
		for (Iterator labelit = labels.iterator(); labelit.hasNext();) {
			LabelElementPair pair = (LabelElementPair) labelit.next();
			Rule refdRule = getRule(pair.elementRef.getText());
			if ( refdRule!=null && !refdRule.getHasReturnValue() && !pair.actionReferencesLabel ) {
				//System.out.println(pair.label.getText()+" is useless");
				kill.add(pair.label.getText());
			}
		}
		for (int i = 0; i < kill.size(); i++) {
			String labelToKill = (String) kill.get(i);
			// System.out.println("kill "+labelToKill);
			ruleToElementLabelPairMap.remove(labelToKill);
		}
	}

	/** Track a rule reference within an outermost alt of a rule.  Used
	 *  at the moment to decide if $ruleref refers to a unique rule ref in
	 *  the alt.  Rewrite rules force tracking of all rule AST results.
	 *
	 *  This data is also used to verify that all rules have been defined.
	 */
	public void altReferencesRule(String enclosingRuleName,
								  GrammarAST refScopeAST,
								  GrammarAST refAST,
								  int outerAltNum)
	{
		/* Do nothing for now; not sure need; track S.x as x
		String scope = null;
		Grammar scopeG = null;
		if ( refScopeAST!=null ) {
			if ( !scopedRuleRefs.contains(refScopeAST) ) {
				scopedRuleRefs.add(refScopeAST);
			}
			scope = refScopeAST.getText();
		}
		*/
		Rule r = getRule(enclosingRuleName);
		if ( r==null ) {
			return; // no error here; see NameSpaceChecker
		}
		r.trackRuleReferenceInAlt(refAST, outerAltNum);
		antlr.Token refToken = refAST.getToken();
		if ( !ruleRefs.contains(refAST) ) {
			ruleRefs.add(refAST);
		}
	}

	/** Track a token reference within an outermost alt of a rule.  Used
	 *  to decide if $tokenref refers to a unique token ref in
	 *  the alt. Does not track literals!
	 *
	 *  Rewrite rules force tracking of all tokens.
	 */
	public void altReferencesTokenID(String ruleName, GrammarAST refAST, int outerAltNum) {
		Rule r = getLocallyDefinedRule(ruleName);
		if ( r==null ) {
			return;
		}
		r.trackTokenReferenceInAlt(refAST, outerAltNum);
		if ( !tokenIDRefs.contains(refAST.getToken()) ) {
			tokenIDRefs.add(refAST.getToken());
		}
	}

	/** To yield smaller, more readable code, track which rules have their
	 *  predefined attributes accessed.  If the rule has no user-defined
	 *  return values, then don't generate the return value scope classes
	 *  etc...  Make the rule have void return value.  Don't track for lexer
	 *  rules.
	 */
	public void referenceRuleLabelPredefinedAttribute(String ruleName) {
		Rule r = getRule(ruleName);
		if ( r!=null && type!=LEXER ) {
			// indicate that an action ref'd an attr unless it's in a lexer
			// so that $ID.text refs don't force lexer rules to define
			// return values...Token objects are created by the caller instead.
			r.referencedPredefinedRuleAttributes = true;
		}
	}

	public List checkAllRulesForLeftRecursion() {
		return sanity.checkAllRulesForLeftRecursion();
	}

	/** Return a list of left-recursive rules; no analysis can be done
	 *  successfully on these.  Useful to skip these rules then and also
	 *  for ANTLRWorks to highlight them.
	 */
	public Set<Rule> getLeftRecursiveRules() {
		if ( nfa==null ) {
			buildNFA();
		}
		if ( leftRecursiveRules!=null ) {
			return leftRecursiveRules;
		}
		sanity.checkAllRulesForLeftRecursion();
		return leftRecursiveRules;
	}

	public void checkRuleReference(GrammarAST scopeAST,
								   GrammarAST refAST,
								   GrammarAST argsAST,
								   String currentRuleName)
	{
		sanity.checkRuleReference(scopeAST, refAST, argsAST, currentRuleName);
	}

	/** Rules like "a : ;" and "a : {...} ;" should not generate
	 *  try/catch blocks for RecognitionException.  To detect this
	 *  it's probably ok to just look for any reference to an atom
	 *  that can match some input.  W/o that, the rule is unlikey to have
	 *  any else.
	 */
	public boolean isEmptyRule(GrammarAST block) {
		GrammarAST aTokenRefNode =
			block.findFirstType(ANTLRParser.TOKEN_REF);
		GrammarAST aStringLiteralRefNode =
			block.findFirstType(ANTLRParser.STRING_LITERAL);
		GrammarAST aCharLiteralRefNode =
			block.findFirstType(ANTLRParser.CHAR_LITERAL);
		GrammarAST aWildcardRefNode =
			block.findFirstType(ANTLRParser.WILDCARD);
		GrammarAST aRuleRefNode =
			block.findFirstType(ANTLRParser.RULE_REF);
		if ( aTokenRefNode==null&&
			 aStringLiteralRefNode==null&&
			 aCharLiteralRefNode==null&&
			 aWildcardRefNode==null&&
			 aRuleRefNode==null )
		{
			return true;
		}
		return false;
	}

	public boolean isAtomTokenType(int ttype) {
		return ttype == ANTLRParser.WILDCARD||
			   ttype == ANTLRParser.CHAR_LITERAL||
			   ttype == ANTLRParser.CHAR_RANGE||
			   ttype == ANTLRParser.STRING_LITERAL||
			   ttype == ANTLRParser.NOT||
			   (type != LEXER && ttype == ANTLRParser.TOKEN_REF);
	}

	public int getTokenType(String tokenName) {
		Integer I = null;
		if ( tokenName.charAt(0)=='\'') {
			I = (Integer)composite.stringLiteralToTypeMap.get(tokenName);
		}
		else { // must be a label like ID
			I = (Integer)composite.tokenIDToTypeMap.get(tokenName);
		}
		int i = (I!=null)?I.intValue():Label.INVALID;
		//System.out.println("grammar type "+type+" "+tokenName+"->"+i);
		return i;
	}

	/** Get the list of tokens that are IDs like BLOCK and LPAREN */
	public Set getTokenIDs() {
		return composite.tokenIDToTypeMap.keySet();
	}

	/** Return an ordered integer list of token types that have no
	 *  corresponding token ID like INT or KEYWORD_BEGIN; for stuff
	 *  like 'begin'.
	 */
	public Collection getTokenTypesWithoutID() {
		List types = new ArrayList();
		for (int t =Label.MIN_TOKEN_TYPE; t<=getMaxTokenType(); t++) {
			String name = getTokenDisplayName(t);
			if ( name.charAt(0)=='\'' ) {
				types.add(Utils.integer(t));
			}
		}
		return types;
	}

	/** Get a list of all token IDs and literals that have an associated
	 *  token type.
	 */
	public Set<String> getTokenDisplayNames() {
		Set<String> names = new HashSet<String>();
		for (int t =Label.MIN_TOKEN_TYPE; t <=getMaxTokenType(); t++) {
			names.add(getTokenDisplayName(t));
		}
		return names;
	}

	/** Given a literal like (the 3 char sequence with single quotes) 'a',
	 *  return the int value of 'a'. Convert escape sequences here also.
	 *  ANTLR's antlr.g parser does not convert escape sequences.
	 *
	 *  11/26/2005: I changed literals to always be '...' even for strings.
	 *  This routine still works though.
	 */
	public static int getCharValueFromGrammarCharLiteral(String literal) {
		switch ( literal.length() ) {
			case 3 :
				// 'x'
				return literal.charAt(1); // no escape char
			case 4 :
				// '\x'  (antlr lexer will catch invalid char)
				if ( Character.isDigit(literal.charAt(2)) ) {
					ErrorManager.error(ErrorManager.MSG_SYNTAX_ERROR,
									   "invalid char literal: "+literal);
					return -1;
				}
				int escChar = literal.charAt(2);
				int charVal = ANTLRLiteralEscapedCharValue[escChar];
				if ( charVal==0 ) {
					// Unnecessary escapes like '\{' should just yield {
					return escChar;
				}
				return charVal;
			case 8 :
				// '\u1234'
				String unicodeChars = literal.substring(3,literal.length()-1);
				return Integer.parseInt(unicodeChars, 16);
			default :
				ErrorManager.error(ErrorManager.MSG_SYNTAX_ERROR,
								   "invalid char literal: "+literal);
				return -1;
		}
	}

	/** ANTLR does not convert escape sequences during the parse phase because
	 *  it could not know how to print String/char literals back out when
	 *  printing grammars etc...  Someone in China might use the real unicode
	 *  char in a literal as it will display on their screen; when printing
	 *  back out, I could not know whether to display or use a unicode escape.
	 *
	 *  This routine converts a string literal with possible escape sequences
	 *  into a pure string of 16-bit char values.  Escapes and unicode \u0000
	 *  specs are converted to pure chars.  return in a buffer; people may
	 *  want to walk/manipulate further.
	 *
	 *  The NFA construction routine must know the actual char values.
	 */
	public static StringBuffer getUnescapedStringFromGrammarStringLiteral(String literal) {
		//System.out.println("escape: ["+literal+"]");
		StringBuffer buf = new StringBuffer();
		int last = literal.length()-1; // skip quotes on outside
		for (int i=1; i<last; i++) {
			char c = literal.charAt(i);
			if ( c=='\\' ) {
				i++;
				c = literal.charAt(i);
				if ( Character.toUpperCase(c)=='U' ) {
					// \u0000
					i++;
					String unicodeChars = literal.substring(i,i+4);
					// parse the unicode 16 bit hex value
					int val = Integer.parseInt(unicodeChars, 16);
					i+=4-1; // loop will inc by 1; only jump 3 then
					buf.append((char)val);
				}
				else if ( Character.isDigit(c) ) {
					ErrorManager.error(ErrorManager.MSG_SYNTAX_ERROR,
									   "invalid char literal: "+literal);
					buf.append("\\"+(char)c);
				}
				else {
					buf.append((char)ANTLRLiteralEscapedCharValue[c]); // normal \x escape
				}
			}
			else {
				buf.append(c); // simple char x
			}
		}
		//System.out.println("string: ["+buf.toString()+"]");
		return buf;
	}

	/** Pull your token definitions from an existing grammar in memory.
	 *  You must use Grammar() ctor then this method then setGrammarContent()
	 *  to make this work.  This was useful primarily for testing and
	 *  interpreting grammars until I added import grammar functionality.
	 *  When you import a grammar you implicitly import its vocabulary as well
	 *  and keep the same token type values.
	 *
	 *  Returns the max token type found.
	 */
	public int importTokenVocabulary(Grammar importFromGr) {
		Set importedTokenIDs = importFromGr.getTokenIDs();
		for (Iterator it = importedTokenIDs.iterator(); it.hasNext();) {
			String tokenID = (String) it.next();
			int tokenType = importFromGr.getTokenType(tokenID);
			composite.maxTokenType = Math.max(composite.maxTokenType,tokenType);
			if ( tokenType>=Label.MIN_TOKEN_TYPE ) {
				//System.out.println("import token from grammar "+tokenID+"="+tokenType);
				defineToken(tokenID, tokenType);
			}
		}
		return composite.maxTokenType; // return max found
	}

	/** Import the rules/tokens of a delegate grammar. All delegate grammars are
	 *  read during the ctor of first Grammar created.
	 *
	 *  Do not create NFA here because NFA construction needs to hook up with
	 *  overridden rules in delegation root grammar.
	 */
	public void importGrammar(GrammarAST grammarNameAST, String label) {
		String grammarName = grammarNameAST.getText();
		//System.out.println("import "+gfile.getName());
		String gname = grammarName + GRAMMAR_FILE_EXTENSION;
		BufferedReader br = null;
		try {
			String fullName = tool.getLibraryFile(gname);
			FileReader fr = new FileReader(fullName);
			br = new BufferedReader(fr);
			Grammar delegateGrammar = null;
			delegateGrammar = new Grammar(tool, gname, composite);
			delegateGrammar.label = label;

			addDelegateGrammar(delegateGrammar);

			delegateGrammar.parseAndBuildAST(br);
			if ( !validImport(delegateGrammar) ) {
				ErrorManager.grammarError(ErrorManager.MSG_INVALID_IMPORT,
										  this,
										  grammarNameAST.token,
										  this,
										  delegateGrammar);
				return;
			}
			if ( this.type==COMBINED &&
				 (delegateGrammar.name.equals(this.name+grammarTypeToFileNameSuffix[LEXER])||
				  delegateGrammar.name.equals(this.name+grammarTypeToFileNameSuffix[PARSER])) )
			{
				ErrorManager.grammarError(ErrorManager.MSG_IMPORT_NAME_CLASH,
										  this,
										  grammarNameAST.token,
										  this,
										  delegateGrammar);
				return;
			}
			if ( delegateGrammar.grammarTree!=null ) {
				// we have a valid grammar
				// deal with combined grammars
				if ( delegateGrammar.type == LEXER && this.type == COMBINED ) {
					// ooops, we wasted some effort; tell lexer to read it in
					// later
					lexerGrammarST.setAttribute("imports", grammarName);
					// but, this parser grammar will need the vocab
					// so add to composite anyway so we suck in the tokens later
				}
			}
			//System.out.println("Got grammar:\n"+delegateGrammar);
		}
		catch (IOException ioe) {
			ErrorManager.error(ErrorManager.MSG_CANNOT_OPEN_FILE,
							   gname,
							   ioe);
		}
		finally {
			if ( br!=null ) {
				try {
					br.close();
				}
				catch (IOException ioe) {
					ErrorManager.error(ErrorManager.MSG_CANNOT_CLOSE_FILE,
									   gname,
									   ioe);
				}
			}
		}
	}

	/** add new delegate to composite tree */
	protected void addDelegateGrammar(Grammar delegateGrammar) {
		CompositeGrammarTree t = composite.delegateGrammarTreeRoot.findNode(this);
		t.addChild(new CompositeGrammarTree(delegateGrammar));
		// make sure new grammar shares this composite
		delegateGrammar.composite = this.composite;
	}

	/** Load a vocab file <vocabName>.tokens and return max token type found. */
	public int importTokenVocabulary(GrammarAST tokenVocabOptionAST,
									 String vocabName)
	{
		if ( !getGrammarIsRoot() ) {
			ErrorManager.grammarWarning(ErrorManager.MSG_TOKEN_VOCAB_IN_DELEGATE,
										this,
										tokenVocabOptionAST.token,
										name);
			return composite.maxTokenType;
		}

		File fullFile = tool.getImportedVocabFile(vocabName);
		try {
			FileReader fr = new FileReader(fullFile);
			BufferedReader br = new BufferedReader(fr);
			StreamTokenizer tokenizer = new StreamTokenizer(br);
			tokenizer.parseNumbers();
			tokenizer.wordChars('_', '_');
			tokenizer.eolIsSignificant(true);
			tokenizer.slashSlashComments(true);
			tokenizer.slashStarComments(true);
			tokenizer.ordinaryChar('=');
			tokenizer.quoteChar('\'');
			tokenizer.whitespaceChars(' ',' ');
			tokenizer.whitespaceChars('\t','\t');
			int lineNum = 1;
			int token = tokenizer.nextToken();
			while (token != StreamTokenizer.TT_EOF) {
				String tokenID;
				if ( token == StreamTokenizer.TT_WORD ) {
					tokenID = tokenizer.sval;
				}
				else if ( token == '\'' ) {
					tokenID = "'"+tokenizer.sval+"'";
				}
				else {
					ErrorManager.error(ErrorManager.MSG_TOKENS_FILE_SYNTAX_ERROR,
									   vocabName+CodeGenerator.VOCAB_FILE_EXTENSION,
									   Utils.integer(lineNum));
					while ( tokenizer.nextToken() != StreamTokenizer.TT_EOL ) {;}
					token = tokenizer.nextToken();
					continue;
				}
				token = tokenizer.nextToken();
				if ( token != '=' ) {
					ErrorManager.error(ErrorManager.MSG_TOKENS_FILE_SYNTAX_ERROR,
									   vocabName+CodeGenerator.VOCAB_FILE_EXTENSION,
									   Utils.integer(lineNum));
					while ( tokenizer.nextToken() != StreamTokenizer.TT_EOL ) {;}
					token = tokenizer.nextToken();
					continue;
				}
				token = tokenizer.nextToken(); // skip '='
				if ( token != StreamTokenizer.TT_NUMBER ) {
					ErrorManager.error(ErrorManager.MSG_TOKENS_FILE_SYNTAX_ERROR,
									   vocabName+CodeGenerator.VOCAB_FILE_EXTENSION,
									   Utils.integer(lineNum));
					while ( tokenizer.nextToken() != StreamTokenizer.TT_EOL ) {;}
					token = tokenizer.nextToken();
					continue;
				}
				int tokenType = (int)tokenizer.nval;
				token = tokenizer.nextToken();
				//System.out.println("import "+tokenID+"="+tokenType);
				composite.maxTokenType = Math.max(composite.maxTokenType,tokenType);
				defineToken(tokenID, tokenType);
				lineNum++;
				if ( token != StreamTokenizer.TT_EOL ) {
					ErrorManager.error(ErrorManager.MSG_TOKENS_FILE_SYNTAX_ERROR,
									   vocabName+CodeGenerator.VOCAB_FILE_EXTENSION,
									   Utils.integer(lineNum));
					while ( tokenizer.nextToken() != StreamTokenizer.TT_EOL ) {;}
					token = tokenizer.nextToken();
					continue;
				}
				token = tokenizer.nextToken(); // skip newline
			}
			br.close();
		}
		catch (FileNotFoundException fnfe) {
			ErrorManager.error(ErrorManager.MSG_CANNOT_FIND_TOKENS_FILE,
							   fullFile);
		}
		catch (IOException ioe) {
			ErrorManager.error(ErrorManager.MSG_ERROR_READING_TOKENS_FILE,
							   fullFile,
							   ioe);
		}
		catch (Exception e) {
			ErrorManager.error(ErrorManager.MSG_ERROR_READING_TOKENS_FILE,
							   fullFile,
							   e);
		}
		return composite.maxTokenType;
	}

	/** Given a token type, get a meaningful name for it such as the ID
	 *  or string literal.  If this is a lexer and the ttype is in the
	 *  char vocabulary, compute an ANTLR-valid (possibly escaped) char literal.
	 */
	public String getTokenDisplayName(int ttype) {
		String tokenName = null;
		int index=0;
		// inside any target's char range and is lexer grammar?
		if ( this.type==LEXER &&
			 ttype >= Label.MIN_CHAR_VALUE && ttype <= Label.MAX_CHAR_VALUE )
		{
			return getANTLRCharLiteralForChar(ttype);
		}
		// faux label?
		else if ( ttype<0 ) {
			tokenName = (String)composite.typeToTokenList.get(Label.NUM_FAUX_LABELS+ttype);
		}
		else {
			// compute index in typeToTokenList for ttype
			index = ttype-1; // normalize to 0..n-1
			index += Label.NUM_FAUX_LABELS;     // jump over faux tokens

			if ( index<composite.typeToTokenList.size() ) {
				tokenName = (String)composite.typeToTokenList.get(index);
				if ( tokenName!=null &&
					 tokenName.startsWith(AUTO_GENERATED_TOKEN_NAME_PREFIX) )
				{
					tokenName = composite.typeToStringLiteralList.get(ttype);
				}
			}
			else {
				tokenName = String.valueOf(ttype);
			}
		}
		//System.out.println("getTokenDisplayName ttype="+ttype+", index="+index+", name="+tokenName);
		return tokenName;
	}

	/** Get the list of ANTLR String literals */
	public Set<String> getStringLiterals() {
		return composite.stringLiteralToTypeMap.keySet();
	}

	public String getGrammarTypeString() {
		return grammarTypeToString[type];
	}

	public int getGrammarMaxLookahead() {
		if ( global_k>=0 ) {
			return global_k;
		}
		Object k = getOption("k");
		if ( k==null ) {
			global_k = 0;
		}
		else if (k instanceof Integer) {
			Integer kI = (Integer)k;
			global_k = kI.intValue();
		}
		else {
			// must be String "*"
			if ( k.equals("*") ) {  // this the default anyway
				global_k = 0;
			}
		}
		return global_k;
	}

	/** Save the option key/value pair and process it; return the key
	 *  or null if invalid option.
	 */
	public String setOption(String key, Object value, antlr.Token optionsStartToken) {
		if ( legalOption(key) ) {
			ErrorManager.grammarError(ErrorManager.MSG_ILLEGAL_OPTION,
									  this,
									  optionsStartToken,
									  key);
			return null;
		}
		if ( !optionIsValid(key, value) ) {
			return null;
		}
        if ( key.equals("backtrack") && value.toString().equals("true") ) {
            composite.getRootGrammar().atLeastOneBacktrackOption = true;
        }
        if ( options==null ) {
			options = new HashMap();
		}
		options.put(key, value);
		return key;
	}

	public boolean legalOption(String key) {
		switch ( type ) {
			case LEXER :
				return !legalLexerOptions.contains(key);
			case PARSER :
				return !legalParserOptions.contains(key);
			case TREE_PARSER :
				return !legalTreeParserOptions.contains(key);
			default :
				return !legalParserOptions.contains(key);
		}
	}

	public void setOptions(Map options, antlr.Token optionsStartToken) {
		if ( options==null ) {
			this.options = null;
			return;
		}
		Set keys = options.keySet();
		for (Iterator it = keys.iterator(); it.hasNext();) {
			String optionName = (String) it.next();
			Object optionValue = options.get(optionName);
			String stored=setOption(optionName, optionValue, optionsStartToken);
			if ( stored==null ) {
				it.remove();
			}
		}
	}

	public Object getOption(String key) {
		return composite.getOption(key);
	}

	public Object getLocallyDefinedOption(String key) {
		Object value = null;
		if ( options!=null ) {
			value = options.get(key);
		}
		if ( value==null ) {
			value = defaultOptions.get(key);
		}
		return value;
	}

	public Object getBlockOption(GrammarAST blockAST, String key) {
		String v = (String)blockAST.getBlockOption(key);
		if ( v!=null ) {
			return v;
		}
		if ( type==Grammar.LEXER ) {
			return defaultLexerBlockOptions.get(key);
		}
		return defaultBlockOptions.get(key);
	}

	public int getUserMaxLookahead(int decision) {
		int user_k = 0;
		GrammarAST blockAST = nfa.grammar.getDecisionBlockAST(decision);
		Object k = blockAST.getBlockOption("k");
		if ( k==null ) {
			user_k = nfa.grammar.getGrammarMaxLookahead();
			return user_k;
		}
		if (k instanceof Integer) {
			Integer kI = (Integer)k;
			user_k = kI.intValue();
		}
		else {
			// must be String "*"
			if ( k.equals("*") ) {
				user_k = 0;
			}
		}
		return user_k;
	}

	public boolean getAutoBacktrackMode(int decision) {
		NFAState decisionNFAStartState = getDecisionNFAStartState(decision);
		String autoBacktrack =
			(String)getBlockOption(decisionNFAStartState.associatedASTNode, "backtrack");
		
		if ( autoBacktrack==null ) {
			autoBacktrack = (String)nfa.grammar.getOption("backtrack");
		}
		return autoBacktrack!=null&&autoBacktrack.equals("true");
	}

	public boolean optionIsValid(String key, Object value) {
		return true;
	}

	public boolean buildAST() {
		String outputType = (String)getOption("output");
		if ( outputType!=null ) {
			return outputType.toString().equals("AST");
		}
		return false;
	}

	public boolean rewriteMode() {
		Object outputType = getOption("rewrite");
		if ( outputType!=null ) {
			return outputType.toString().equals("true");
		}
		return false;
	}

	public boolean isBuiltFromString() {
		return builtFromString;
	}

	public boolean buildTemplate() {
		String outputType = (String)getOption("output");
		if ( outputType!=null ) {
			return outputType.toString().equals("template");
		}
		return false;
	}

	public Collection<Rule> getRules() {
		return nameToRuleMap.values();
	}

	/** Get the set of Rules that need to have manual delegations
	 *  like "void rule() { importedGrammar.rule(); }"
	 *
	 *  If this grammar is master, get list of all rule definitions from all
	 *  delegate grammars.  Only master has complete interface from combined
	 *  grammars...we will generated delegates as helper objects.
	 *
	 *  Composite grammars that are not the root/master do not have complete
	 *  interfaces.  It is not my intention that people use subcomposites.
	 *  Only the outermost grammar should be used from outside code.  The
	 *  other grammar components are specifically generated to work only
	 *  with the master/root. 
	 *
	 *  delegatedRules = imported - overridden
	 */
	public Set<Rule> getDelegatedRules() {
		return composite.getDelegatedRules(this);
	}

	/** Get set of all rules imported from all delegate grammars even if
	 *  indirectly delegated.
	 */
	public Set<Rule> getAllImportedRules() {
		return composite.getAllImportedRules(this);
	}

	/** Get list of all delegates from all grammars directly or indirectly
	 *  imported into this grammar.
	 */
	public List<Grammar> getDelegates() {
		return composite.getDelegates(this);
	}

	public List<String> getDelegateNames() {
		// compute delegates:{Grammar g | return g.name;}
		List<String> names = new ArrayList<String>();
		List<Grammar> delegates = composite.getDelegates(this);
		if ( delegates!=null ) {
			for (Grammar g : delegates) {
				names.add(g.name);
			}
		}
		return names;
	}

	public List<Grammar> getDirectDelegates() {
		return composite.getDirectDelegates(this);
	}
	
	/** Get delegates below direct delegates */
	public List<Grammar> getIndirectDelegates() {
		return composite.getIndirectDelegates(this);
	}

	/** Get list of all delegators.  This amounts to the grammars on the path
	 *  to the root of the delegation tree.
	 */
	public List<Grammar> getDelegators() {
		return composite.getDelegators(this);
	}

	/** Who's my direct parent grammar? */
	public Grammar getDelegator() {
		return composite.getDelegator(this);
	}

	public Set<Rule> getDelegatedRuleReferences() {
		return delegatedRuleReferences;
	}

	public boolean getGrammarIsRoot() {
		return composite.delegateGrammarTreeRoot.grammar == this;
	}

	public void setRuleAST(String ruleName, GrammarAST t) {
		Rule r = getLocallyDefinedRule(ruleName);
		if ( r!=null ) {
			r.tree = t;
			r.EORNode = t.getLastChild();
		}
	}

	public NFAState getRuleStartState(String ruleName) {
		return getRuleStartState(null, ruleName);
	}

	public NFAState getRuleStartState(String scopeName, String ruleName) {
		Rule r = getRule(scopeName, ruleName);
		if ( r!=null ) {
			//System.out.println("getRuleStartState("+scopeName+", "+ruleName+")="+r.startState);
			return r.startState;
		}
		//System.out.println("getRuleStartState("+scopeName+", "+ruleName+")=null");
		return null;
	}

	public String getRuleModifier(String ruleName) {
		Rule r = getRule(ruleName);
		if ( r!=null ) {
			return r.modifier;
		}
		return null;
	}

	public NFAState getRuleStopState(String ruleName) {
		Rule r = getRule(ruleName);
		if ( r!=null ) {
			return r.stopState;
		}
		return null;
	}

	public int assignDecisionNumber(NFAState state) {
		decisionCount++;
		state.setDecisionNumber(decisionCount);
		return decisionCount;
	}

	protected Decision getDecision(int decision) {
		int index = decision-1;
		if ( index >= indexToDecision.size() ) {
			return null;
		}
		Decision d = (Decision)indexToDecision.get(index);
		return d;
	}

	protected Decision createDecision(int decision) {
		int index = decision-1;
		if ( index < indexToDecision.size() ) {
			return getDecision(decision); // don't recreate
		}
		Decision d = new Decision();
		d.decision = decision;
		indexToDecision.setSize(getNumberOfDecisions());
		indexToDecision.set(index, d);
		return d;
	}

	public List getDecisionNFAStartStateList() {
		List states = new ArrayList(100);
		for (int d = 0; d < indexToDecision.size(); d++) {
			Decision dec = (Decision) indexToDecision.get(d);
			states.add(dec.startState);
		}
		return states;
	}

	public NFAState getDecisionNFAStartState(int decision) {
		Decision d = getDecision(decision);
		if ( d==null ) {
			return null;
		}
		return d.startState;
	}

	public DFA getLookaheadDFA(int decision) {
		Decision d = getDecision(decision);
		if ( d==null ) {
			return null;
		}
		return d.dfa;
	}

	public GrammarAST getDecisionBlockAST(int decision) {
		Decision d = getDecision(decision);
		if ( d==null ) {
			return null;
		}
		return d.blockAST;
	}

	/** returns a list of column numbers for all decisions
	 *  on a particular line so ANTLRWorks choose the decision
	 *  depending on the location of the cursor (otherwise,
	 *  ANTLRWorks has to give the *exact* location which
	 *  is not easy from the user point of view).
	 *
	 *  This is not particularly fast as it walks entire line:col->DFA map
	 *  looking for a prefix of "line:".
	 */
	public List getLookaheadDFAColumnsForLineInFile(int line) {
		String prefix = line+":";
		List columns = new ArrayList();
		for(Iterator iter = lineColumnToLookaheadDFAMap.keySet().iterator();
			iter.hasNext(); ) {
			String key = (String)iter.next();
			if(key.startsWith(prefix)) {
				columns.add(Integer.valueOf(key.substring(prefix.length())));
			}
		}
		return columns;
	}

	/** Useful for ANTLRWorks to map position in file to the DFA for display */
	public DFA getLookaheadDFAFromPositionInFile(int line, int col) {
		return (DFA)lineColumnToLookaheadDFAMap.get(
			new StringBuffer().append(line + ":").append(col).toString());
	}

	public Map getLineColumnToLookaheadDFAMap() {
		return lineColumnToLookaheadDFAMap;
	}

	/*
	public void setDecisionOptions(int decision, Map options) {
		Decision d = createDecision(decision);
		d.options = options;
	}

	public void setDecisionOption(int decision, String name, Object value) {
		Decision d = getDecision(decision);
		if ( d!=null ) {
			if ( d.options==null ) {
				d.options = new HashMap();
			}
			d.options.put(name,value);
		}
	}

	public Map getDecisionOptions(int decision) {
		Decision d = getDecision(decision);
		if ( d==null ) {
			return null;
		}
		return d.options;
    }
    */

	public int getNumberOfDecisions() {
		return decisionCount;
	}

	public int getNumberOfCyclicDecisions() {
		int n = 0;
		for (int i=1; i<=getNumberOfDecisions(); i++) {
			Decision d = getDecision(i);
			if ( d.dfa!=null && d.dfa.isCyclic() ) {
				n++;
			}
		}
		return n;
	}

	/** Set the lookahead DFA for a particular decision.  This means
	 *  that the appropriate AST node must updated to have the new lookahead
	 *  DFA.  This method could be used to properly set the DFAs without
	 *  using the createLookaheadDFAs() method.  You could do this
	 *
	 *    Grammar g = new Grammar("...");
	 *    g.setLookahead(1, dfa1);
	 *    g.setLookahead(2, dfa2);
	 *    ...
	 */
	public void setLookaheadDFA(int decision, DFA lookaheadDFA) {
		Decision d = createDecision(decision);
		d.dfa = lookaheadDFA;
		GrammarAST ast = d.startState.associatedASTNode;
		ast.setLookaheadDFA(lookaheadDFA);
	}

	public void setDecisionNFA(int decision, NFAState state) {
		Decision d = createDecision(decision);
		d.startState = state;
	}

	public void setDecisionBlockAST(int decision, GrammarAST blockAST) {
		//System.out.println("setDecisionBlockAST("+decision+", "+blockAST.token);
		Decision d = createDecision(decision);
		d.blockAST = blockAST;
	}

	public boolean allDecisionDFAHaveBeenCreated() {
		return allDecisionDFACreated;
	}

	/** How many token types have been allocated so far? */
	public int getMaxTokenType() {
		return composite.maxTokenType;
	}

	/** What is the max char value possible for this grammar's target?  Use
	 *  unicode max if no target defined.
	 */
	public int getMaxCharValue() {
		if ( generator!=null ) {
			return generator.target.getMaxCharValue(generator);
		}
		else {
			return Label.MAX_CHAR_VALUE;
		}
	}

	/** Return a set of all possible token or char types for this grammar */
	public IntSet getTokenTypes() {
		if ( type==LEXER ) {
			return getAllCharValues();
		}
		return IntervalSet.of(Label.MIN_TOKEN_TYPE, getMaxTokenType());
	}

	/** If there is a char vocabulary, use it; else return min to max char
	 *  as defined by the target.  If no target, use max unicode char value.
	 */
	public IntSet getAllCharValues() {
		if ( charVocabulary!=null ) {
			return charVocabulary;
		}
		IntSet allChar = IntervalSet.of(Label.MIN_CHAR_VALUE, getMaxCharValue());
		return allChar;
	}

	/** Return a string representing the escaped char for code c.  E.g., If c
	 *  has value 0x100, you will get "\u0100".  ASCII gets the usual
	 *  char (non-hex) representation.  Control characters are spit out
	 *  as unicode.  While this is specially set up for returning Java strings,
	 *  it can be used by any language target that has the same syntax. :)
	 *
	 *  11/26/2005: I changed this to use double quotes, consistent with antlr.g
	 *  12/09/2005: I changed so everything is single quotes
	 */
	public static String getANTLRCharLiteralForChar(int c) {
		if ( c<Label.MIN_CHAR_VALUE ) {
			ErrorManager.internalError("invalid char value "+c);
			return "'<INVALID>'";
		}
		if ( c<ANTLRLiteralCharValueEscape.length && ANTLRLiteralCharValueEscape[c]!=null ) {
			return '\''+ANTLRLiteralCharValueEscape[c]+'\'';
		}
		if ( Character.UnicodeBlock.of((char)c)==Character.UnicodeBlock.BASIC_LATIN &&
			 !Character.isISOControl((char)c) ) {
			if ( c=='\\' ) {
				return "'\\\\'";
			}
			if ( c=='\'') {
				return "'\\''";
			}
			return '\''+Character.toString((char)c)+'\'';
		}
		// turn on the bit above max "\uFFFF" value so that we pad with zeros
		// then only take last 4 digits
		String hex = Integer.toHexString(c|0x10000).toUpperCase().substring(1,5);
		String unicodeStr = "'\\u"+hex+"'";
		return unicodeStr;
	}

	/** For lexer grammars, return everything in unicode not in set.
	 *  For parser and tree grammars, return everything in token space
	 *  from MIN_TOKEN_TYPE to last valid token type or char value.
	 */
	public IntSet complement(IntSet set) {
		//System.out.println("complement "+set.toString(this));
		//System.out.println("vocabulary "+getTokenTypes().toString(this));
		IntSet c = set.complement(getTokenTypes());
		//System.out.println("result="+c.toString(this));
		return c;
	}

	public IntSet complement(int atom) {
		return complement(IntervalSet.of(atom));
	}

	/** Given set tree like ( SET A B ), check that A and B
	 *  are both valid sets themselves, else we must tree like a BLOCK
	 */
	public boolean isValidSet(TreeToNFAConverter nfabuilder, GrammarAST t) {
		boolean valid = true;
		try {
			//System.out.println("parse BLOCK as set tree: "+t.toStringTree());
			nfabuilder.testBlockAsSet(t);
		}
		catch (RecognitionException re) {
			// The rule did not parse as a set, return null; ignore exception
			valid = false;
		}
		//System.out.println("valid? "+valid);
		return valid;
	}

	/** Get the set equivalent (if any) of the indicated rule from this
	 *  grammar.  Mostly used in the lexer to do ~T for some fragment rule
	 *  T.  If the rule AST has a SET use that.  If the rule is a single char
	 *  convert it to a set and return.  If rule is not a simple set (w/o actions)
	 *  then return null.
	 *  Rules have AST form:
	 *
	 *		^( RULE ID modifier ARG RET SCOPE block EOR )
	 */
	public IntSet getSetFromRule(TreeToNFAConverter nfabuilder, String ruleName)
		throws RecognitionException
	{
		Rule r = getRule(ruleName);
		if ( r==null ) {
			return null;
		}
		IntSet elements = null;
		//System.out.println("parsed tree: "+r.tree.toStringTree());
		elements = nfabuilder.setRule(r.tree);
		//System.out.println("elements="+elements);
		return elements;
	}

	/** Decisions are linked together with transition(1).  Count how
	 *  many there are.  This is here rather than in NFAState because
	 *  a grammar decides how NFAs are put together to form a decision.
	 */
	public int getNumberOfAltsForDecisionNFA(NFAState decisionState) {
		if ( decisionState==null ) {
			return 0;
		}
		int n = 1;
		NFAState p = decisionState;
		while ( p.transition[1] !=null ) {
			n++;
			p = (NFAState)p.transition[1].target;
		}
		return n;
	}

	/** Get the ith alternative (1..n) from a decision; return null when
	 *  an invalid alt is requested.  I must count in to find the right
	 *  alternative number.  For (A|B), you get NFA structure (roughly):
	 *
	 *  o->o-A->o
	 *  |
	 *  o->o-B->o
	 *
	 *  This routine returns the leftmost state for each alt.  So alt=1, returns
	 *  the upperleft most state in this structure.
	 */
	public NFAState getNFAStateForAltOfDecision(NFAState decisionState, int alt) {
		if ( decisionState==null || alt<=0 ) {
			return null;
		}
		int n = 1;
		NFAState p = decisionState;
		while ( p!=null ) {
			if ( n==alt ) {
				return p;
			}
			n++;
			Transition next = p.transition[1];
			p = null;
			if ( next!=null ) {
				p = (NFAState)next.target;
			}
		}
		return null;
	}

	/*
	public void computeRuleFOLLOWSets() {
		if ( getNumberOfDecisions()==0 ) {
			createNFAs();
		}
		for (Iterator it = getRules().iterator(); it.hasNext();) {
			Rule r = (Rule)it.next();
			if ( r.isSynPred ) {
				continue;
			}
			LookaheadSet s = ll1Analyzer.FOLLOW(r);
			System.out.println("FOLLOW("+r.name+")="+s);
		}
	}
	*/

	public LookaheadSet FIRST(NFAState s) {
		return ll1Analyzer.FIRST(s);
	}

	public LookaheadSet LOOK(NFAState s) {
		return ll1Analyzer.LOOK(s);
	}

	public void setCodeGenerator(CodeGenerator generator) {
		this.generator = generator;
	}

	public CodeGenerator getCodeGenerator() {
		return generator;
	}

	public GrammarAST getGrammarTree() {
		return grammarTree;
	}

	public Tool getTool() {
		return tool;
	}

	public void setTool(Tool tool) {
		this.tool = tool;
	}

	/** given a token type and the text of the literal, come up with a
	 *  decent token type label.  For now it's just T<type>.  Actually,
	 *  if there is an aliased name from tokens like PLUS='+', use it.
	 */
	public String computeTokenNameFromLiteral(int tokenType, String literal) {
		return AUTO_GENERATED_TOKEN_NAME_PREFIX +tokenType;
	}

	public String toString() {
		return grammarTreeToString(grammarTree);
	}

	public String grammarTreeToString(GrammarAST t) {
		return grammarTreeToString(t, true);
	}

	public String grammarTreeToString(GrammarAST t, boolean showActions) {
		String s = null;
		try {
			s = t.getLine()+":"+t.getColumn()+": ";
			s += new ANTLRTreePrinter().toString((AST)t, this, showActions);
		}
		catch (Exception e) {
			s = "<invalid or missing tree structure>";
		}
		return s;
	}

	public void printGrammar(PrintStream output) {
		ANTLRTreePrinter printer = new ANTLRTreePrinter();
		printer.setASTNodeClass("org.antlr.tool.GrammarAST");
		try {
			String g = printer.toString(grammarTree, this, false);
			output.println(g);
		}
		catch (RecognitionException re) {
			ErrorManager.error(ErrorManager.MSG_SYNTAX_ERROR,re);
		}
	}

}
