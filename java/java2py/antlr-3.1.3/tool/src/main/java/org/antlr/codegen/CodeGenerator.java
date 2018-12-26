/*
[The "BSD licence"]
Copyright (c) 2005-2007 Terence Parr
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
package org.antlr.codegen;


import antlr.ANTLRLexer;
import antlr.RecognitionException;
import antlr.TokenStreamRewriteEngine;
import antlr.collections.AST;
import org.antlr.Tool;
import org.antlr.analysis.*;
import org.antlr.misc.*;
import org.antlr.stringtemplate.*;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;
import org.antlr.tool.*;

import java.io.IOException;
import java.io.StringReader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.antlr.grammar.v2.*;
import org.antlr.grammar.v3.ActionTranslator;

/** ANTLR's code generator.
 *
 *  Generate recognizers derived from grammars.  Language independence
 *  achieved through the use of StringTemplateGroup objects.  All output
 *  strings are completely encapsulated in the group files such as Java.stg.
 *  Some computations are done that are unused by a particular language.
 *  This generator just computes and sets the values into the templates;
 *  the templates are free to use or not use the information.
 *
 *  To make a new code generation target, define X.stg for language X
 *  by copying from existing Y.stg most closely releated to your language;
 *  e.g., to do CSharp.stg copy Java.stg.  The template group file has a
 *  bunch of templates that are needed by the code generator.  You can add
 *  a new target w/o even recompiling ANTLR itself.  The language=X option
 *  in a grammar file dictates which templates get loaded/used.
 *
 *  Some language like C need both parser files and header files.  Java needs
 *  to have a separate file for the cyclic DFA as ANTLR generates bytecodes
 *  directly (which cannot be in the generated parser Java file).  To facilitate
 *  this,
 *
 * cyclic can be in same file, but header, output must be searpate.  recognizer
 *  is in outptufile.
 */
public class CodeGenerator {
	/** When generating SWITCH statements, some targets might need to limit
	 *  the size (based upon the number of case labels).  Generally, this
	 *  limit will be hit only for lexers where wildcard in a UNICODE
	 *  vocabulary environment would generate a SWITCH with 65000 labels.
	 */
	public int MAX_SWITCH_CASE_LABELS = 300;
	public int MIN_SWITCH_ALTS = 3;
	public boolean GENERATE_SWITCHES_WHEN_POSSIBLE = true;
	//public static boolean GEN_ACYCLIC_DFA_INLINE = true;
	public static boolean EMIT_TEMPLATE_DELIMITERS = false;
	public static int MAX_ACYCLIC_DFA_STATES_INLINE = 10;

	public String classpathTemplateRootDirectoryName =
		"org/antlr/codegen/templates";

	/** Which grammar are we generating code for?  Each generator
	 *  is attached to a specific grammar.
	 */
	public Grammar grammar;

	/** What language are we generating? */
	protected String language;

	/** The target specifies how to write out files and do other language
	 *  specific actions.
	 */
	public Target target = null;

	/** Where are the templates this generator should use to generate code? */
	protected StringTemplateGroup templates;

	/** The basic output templates without AST or templates stuff; this will be
	 *  the templates loaded for the language such as Java.stg *and* the Dbg
	 *  stuff if turned on.  This is used for generating syntactic predicates.
	 */
	protected StringTemplateGroup baseTemplates;

	protected StringTemplate recognizerST;
	protected StringTemplate outputFileST;
	protected StringTemplate headerFileST;

	/** Used to create unique labels */
	protected int uniqueLabelNumber = 1;

	/** A reference to the ANTLR tool so we can learn about output directories
	 *  and such.
	 */
	protected Tool tool;

	/** Generate debugging event method calls */
	protected boolean debug;

	/** Create a Tracer object and make the recognizer invoke this. */
	protected boolean trace;

	/** Track runtime parsing information about decisions etc...
	 *  This requires the debugging event mechanism to work.
	 */
	protected boolean profile;

	protected int lineWidth = 72;

	/** I have factored out the generation of acyclic DFAs to separate class */
	public ACyclicDFACodeGenerator acyclicDFAGenerator =
		new ACyclicDFACodeGenerator(this);

	/** I have factored out the generation of cyclic DFAs to separate class */
	/*
	public CyclicDFACodeGenerator cyclicDFAGenerator =
		new CyclicDFACodeGenerator(this);
		*/

	public static final String VOCAB_FILE_EXTENSION = ".tokens";
	protected final static String vocabFilePattern =
		"<tokens:{<attr.name>=<attr.type>\n}>" +
		"<literals:{<attr.name>=<attr.type>\n}>";

	public CodeGenerator(Tool tool, Grammar grammar, String language) {
		this.tool = tool;
		this.grammar = grammar;
		this.language = language;
		loadLanguageTarget(language);
	}

	protected void loadLanguageTarget(String language) {
		String targetName = "org.antlr.codegen."+language+"Target";
		try {
			Class c = Class.forName(targetName);
			target = (Target)c.newInstance();
		}
		catch (ClassNotFoundException cnfe) {
			target = new Target(); // use default
		}
		catch (InstantiationException ie) {
			ErrorManager.error(ErrorManager.MSG_CANNOT_CREATE_TARGET_GENERATOR,
							   targetName,
							   ie);
		}
		catch (IllegalAccessException cnfe) {
			ErrorManager.error(ErrorManager.MSG_CANNOT_CREATE_TARGET_GENERATOR,
							   targetName,
							   cnfe);
		}
	}

	/** load the main language.stg template group file */
	public void loadTemplates(String language) {
		// get a group loader containing main templates dir and target subdir
		String templateDirs =
			classpathTemplateRootDirectoryName+":"+
			classpathTemplateRootDirectoryName+"/"+language;
		//System.out.println("targets="+templateDirs.toString());
		StringTemplateGroupLoader loader =
			new CommonGroupLoader(templateDirs,
								  ErrorManager.getStringTemplateErrorListener());
		StringTemplateGroup.registerGroupLoader(loader);
		StringTemplateGroup.registerDefaultLexer(AngleBracketTemplateLexer.class);

		// first load main language template
		StringTemplateGroup coreTemplates =
			StringTemplateGroup.loadGroup(language);
		baseTemplates = coreTemplates;
		if ( coreTemplates ==null ) {
			ErrorManager.error(ErrorManager.MSG_MISSING_CODE_GEN_TEMPLATES,
							   language);
			return;
		}

		// dynamically add subgroups that act like filters to apply to
		// their supergroup.  E.g., Java:Dbg:AST:ASTParser::ASTDbg.
		String outputOption = (String)grammar.getOption("output");
		if ( outputOption!=null && outputOption.equals("AST") ) {
			if ( debug && grammar.type!=Grammar.LEXER ) {
				StringTemplateGroup dbgTemplates =
					StringTemplateGroup.loadGroup("Dbg", coreTemplates);
				baseTemplates = dbgTemplates;
				StringTemplateGroup astTemplates =
					StringTemplateGroup.loadGroup("AST",dbgTemplates);
				StringTemplateGroup astParserTemplates = astTemplates;
				//if ( !grammar.rewriteMode() ) {
					if ( grammar.type==Grammar.TREE_PARSER ) {
						astParserTemplates =
							StringTemplateGroup.loadGroup("ASTTreeParser", astTemplates);
					}
					else {
						astParserTemplates =
							StringTemplateGroup.loadGroup("ASTParser", astTemplates);
					}
				//}
				StringTemplateGroup astDbgTemplates =
					StringTemplateGroup.loadGroup("ASTDbg", astParserTemplates);
				templates = astDbgTemplates;
			}
			else {
				StringTemplateGroup astTemplates =
					StringTemplateGroup.loadGroup("AST", coreTemplates);
				StringTemplateGroup astParserTemplates = astTemplates;
				//if ( !grammar.rewriteMode() ) {
					if ( grammar.type==Grammar.TREE_PARSER ) {
						astParserTemplates =
							StringTemplateGroup.loadGroup("ASTTreeParser", astTemplates);
					}
					else {
						astParserTemplates =
							StringTemplateGroup.loadGroup("ASTParser", astTemplates);
					}
				//}
				templates = astParserTemplates;
			}
		}
		else if ( outputOption!=null && outputOption.equals("template") ) {
			if ( debug && grammar.type!=Grammar.LEXER ) {
				StringTemplateGroup dbgTemplates =
					StringTemplateGroup.loadGroup("Dbg", coreTemplates);
				baseTemplates = dbgTemplates;
				StringTemplateGroup stTemplates =
					StringTemplateGroup.loadGroup("ST",dbgTemplates);
				templates = stTemplates;
			}
			else {
				templates = StringTemplateGroup.loadGroup("ST", coreTemplates);
			}
		}
		else if ( debug && grammar.type!=Grammar.LEXER ) {
			templates = StringTemplateGroup.loadGroup("Dbg", coreTemplates);
			baseTemplates = templates;
		}
		else {
			templates = coreTemplates;
		}

		if ( EMIT_TEMPLATE_DELIMITERS ) {
			templates.emitDebugStartStopStrings(true);
			templates.doNotEmitDebugStringsForTemplate("codeFileExtension");
			templates.doNotEmitDebugStringsForTemplate("headerFileExtension");
		}
	}

	/** Given the grammar to which we are attached, walk the AST associated
	 *  with that grammar to create NFAs.  Then create the DFAs for all
	 *  decision points in the grammar by converting the NFAs to DFAs.
	 *  Finally, walk the AST again to generate code.
	 *
	 *  Either 1 or 2 files are written:
	 *
	 * 		recognizer: the main parser/lexer/treewalker item
	 * 		header file: language like C/C++ need extern definitions
	 *
	 *  The target, such as JavaTarget, dictates which files get written.
	 */
	public StringTemplate genRecognizer() {
		//System.out.println("### generate "+grammar.name+" recognizer");
		// LOAD OUTPUT TEMPLATES
		loadTemplates(language);
		if ( templates==null ) {
			return null;
		}

		// CREATE NFA FROM GRAMMAR, CREATE DFA FROM NFA
		if ( ErrorManager.doNotAttemptAnalysis() ) {
			return null;
		}
		target.performGrammarAnalysis(this, grammar);


		// some grammar analysis errors will not yield reliable DFA
		if ( ErrorManager.doNotAttemptCodeGen() ) {
			return null;
		}

		// OPTIMIZE DFA
		DFAOptimizer optimizer = new DFAOptimizer(grammar);
		optimizer.optimize();

		// OUTPUT FILE (contains recognizerST)
		outputFileST = templates.getInstanceOf("outputFile");

		// HEADER FILE
		if ( templates.isDefined("headerFile") ) {
			headerFileST = templates.getInstanceOf("headerFile");
		}
		else {
			// create a dummy to avoid null-checks all over code generator
			headerFileST = new StringTemplate(templates,"");
			headerFileST.setName("dummy-header-file");
		}

		boolean filterMode = grammar.getOption("filter")!=null &&
							  grammar.getOption("filter").equals("true");
        boolean canBacktrack = grammar.getSyntacticPredicates()!=null ||
                               grammar.composite.getRootGrammar().atLeastOneBacktrackOption ||
                               filterMode;

        // TODO: move this down further because generating the recognizer
		// alters the model with info on who uses predefined properties etc...
		// The actions here might refer to something.

		// The only two possible output files are available at this point.
		// Verify action scopes are ok for target and dump actions into output
		// Templates can say <actions.parser.header> for example.
		Map actions = grammar.getActions();
		verifyActionScopesOkForTarget(actions);
		// translate $x::y references
		translateActionAttributeReferences(actions);

        StringTemplate gateST = templates.getInstanceOf("actionGate");
        if ( filterMode ) {
            // if filtering, we need to set actions to execute at backtracking
            // level 1 not 0.
            gateST = templates.getInstanceOf("filteringActionGate");
        }
        grammar.setSynPredGateIfNotAlready(gateST);

        headerFileST.setAttribute("actions", actions);
		outputFileST.setAttribute("actions", actions);

		headerFileST.setAttribute("buildTemplate", new Boolean(grammar.buildTemplate()));
		outputFileST.setAttribute("buildTemplate", new Boolean(grammar.buildTemplate()));
		headerFileST.setAttribute("buildAST", new Boolean(grammar.buildAST()));
		outputFileST.setAttribute("buildAST", new Boolean(grammar.buildAST()));

		outputFileST.setAttribute("rewriteMode", Boolean.valueOf(grammar.rewriteMode()));
		headerFileST.setAttribute("rewriteMode", Boolean.valueOf(grammar.rewriteMode()));

		outputFileST.setAttribute("backtracking", Boolean.valueOf(canBacktrack));
		headerFileST.setAttribute("backtracking", Boolean.valueOf(canBacktrack));
		// turn on memoize attribute at grammar level so we can create ruleMemo.
		// each rule has memoize attr that hides this one, indicating whether
		// it needs to save results
		String memoize = (String)grammar.getOption("memoize");
		outputFileST.setAttribute("memoize",
								  (grammar.atLeastOneRuleMemoizes||
								  Boolean.valueOf(memoize!=null&&memoize.equals("true"))&&
									          canBacktrack));
		headerFileST.setAttribute("memoize",
								  (grammar.atLeastOneRuleMemoizes||
								  Boolean.valueOf(memoize!=null&&memoize.equals("true"))&&
									          canBacktrack));


		outputFileST.setAttribute("trace", Boolean.valueOf(trace));
		headerFileST.setAttribute("trace", Boolean.valueOf(trace));

		outputFileST.setAttribute("profile", Boolean.valueOf(profile));
		headerFileST.setAttribute("profile", Boolean.valueOf(profile));

		// RECOGNIZER
		if ( grammar.type==Grammar.LEXER ) {
			recognizerST = templates.getInstanceOf("lexer");
			outputFileST.setAttribute("LEXER", Boolean.valueOf(true));
			headerFileST.setAttribute("LEXER", Boolean.valueOf(true));
			recognizerST.setAttribute("filterMode",
									  Boolean.valueOf(filterMode));
		}
		else if ( grammar.type==Grammar.PARSER ||
			grammar.type==Grammar.COMBINED )
		{
			recognizerST = templates.getInstanceOf("parser");
			outputFileST.setAttribute("PARSER", Boolean.valueOf(true));
			headerFileST.setAttribute("PARSER", Boolean.valueOf(true));
		}
		else {
			recognizerST = templates.getInstanceOf("treeParser");
			outputFileST.setAttribute("TREE_PARSER", Boolean.valueOf(true));
			headerFileST.setAttribute("TREE_PARSER", Boolean.valueOf(true));
            recognizerST.setAttribute("filterMode",
                                      Boolean.valueOf(filterMode));
		}
		outputFileST.setAttribute("recognizer", recognizerST);
		headerFileST.setAttribute("recognizer", recognizerST);
		outputFileST.setAttribute("actionScope",
								  grammar.getDefaultActionScope(grammar.type));
		headerFileST.setAttribute("actionScope",
								  grammar.getDefaultActionScope(grammar.type));

		String targetAppropriateFileNameString =
			target.getTargetStringLiteralFromString(grammar.getFileName());
		outputFileST.setAttribute("fileName", targetAppropriateFileNameString);
		headerFileST.setAttribute("fileName", targetAppropriateFileNameString);
		outputFileST.setAttribute("ANTLRVersion", tool.VERSION);
		headerFileST.setAttribute("ANTLRVersion", tool.VERSION);
		outputFileST.setAttribute("generatedTimestamp", Tool.getCurrentTimeStamp());
		headerFileST.setAttribute("generatedTimestamp", Tool.getCurrentTimeStamp());

		// GENERATE RECOGNIZER
		// Walk the AST holding the input grammar, this time generating code
		// Decisions are generated by using the precomputed DFAs
		// Fill in the various templates with data
		CodeGenTreeWalker gen = new CodeGenTreeWalker();
		try {
			gen.grammar((AST)grammar.getGrammarTree(),
						grammar,
						recognizerST,
						outputFileST,
						headerFileST);
		}
		catch (RecognitionException re) {
			ErrorManager.error(ErrorManager.MSG_BAD_AST_STRUCTURE,
							   re);
		}

		genTokenTypeConstants(recognizerST);
		genTokenTypeConstants(outputFileST);
		genTokenTypeConstants(headerFileST);

		if ( grammar.type!=Grammar.LEXER ) {
			genTokenTypeNames(recognizerST);
			genTokenTypeNames(outputFileST);
			genTokenTypeNames(headerFileST);
		}

		// Now that we know what synpreds are used, we can set into template
		Set synpredNames = null;
		if ( grammar.synPredNamesUsedInDFA.size()>0 ) {
			synpredNames = grammar.synPredNamesUsedInDFA;
		}
		outputFileST.setAttribute("synpreds", synpredNames);
		headerFileST.setAttribute("synpreds", synpredNames);
		
		// all recognizers can see Grammar object
		recognizerST.setAttribute("grammar", grammar);

		// WRITE FILES
		try {
			target.genRecognizerFile(tool,this,grammar,outputFileST);
			if ( templates.isDefined("headerFile") ) {
				StringTemplate extST = templates.getInstanceOf("headerFileExtension");
				target.genRecognizerHeaderFile(tool,this,grammar,headerFileST,extST.toString());
			}
			// write out the vocab interchange file; used by antlr,
			// does not change per target
			StringTemplate tokenVocabSerialization = genTokenVocabOutput();
			String vocabFileName = getVocabFileName();
			if ( vocabFileName!=null ) {
				write(tokenVocabSerialization, vocabFileName);
			}
			//System.out.println(outputFileST.getDOTForDependencyGraph(false));
		}
		catch (IOException ioe) {
			ErrorManager.error(ErrorManager.MSG_CANNOT_WRITE_FILE,
							   getVocabFileName(),
							   ioe);
		}
		/*
		System.out.println("num obj.prop refs: "+ ASTExpr.totalObjPropRefs);
		System.out.println("num reflection lookups: "+ ASTExpr.totalReflectionLookups);
		*/

		return outputFileST;
	}

	/** Some targets will have some extra scopes like C++ may have
	 *  '@headerfile:name {action}' or something.  Make sure the
	 *  target likes the scopes in action table.
	 */
	protected void verifyActionScopesOkForTarget(Map actions) {
		Set actionScopeKeySet = actions.keySet();
		for (Iterator it = actionScopeKeySet.iterator(); it.hasNext();) {
			String scope = (String)it.next();
			if ( !target.isValidActionScope(grammar.type, scope) ) {
				// get any action from the scope to get error location
				Map scopeActions = (Map)actions.get(scope);
				GrammarAST actionAST =
					(GrammarAST)scopeActions.values().iterator().next();
				ErrorManager.grammarError(
					ErrorManager.MSG_INVALID_ACTION_SCOPE,grammar,
					actionAST.getToken(),scope,
					grammar.getGrammarTypeString());
			}
		}
	}

	/** Actions may reference $x::y attributes, call translateAction on
	 *  each action and replace that action in the Map.
	 */
	protected void translateActionAttributeReferences(Map actions) {
		Set actionScopeKeySet = actions.keySet();
		for (Iterator it = actionScopeKeySet.iterator(); it.hasNext();) {
			String scope = (String)it.next();
			Map scopeActions = (Map)actions.get(scope);
			translateActionAttributeReferencesForSingleScope(null,scopeActions);
		}
	}

	/** Use for translating rule @init{...} actions that have no scope */
	public void translateActionAttributeReferencesForSingleScope(
		Rule r,
		Map scopeActions)
	{
		String ruleName=null;
		if ( r!=null ) {
			ruleName = r.name;
		}
		Set actionNameSet = scopeActions.keySet();
		for (Iterator nameIT = actionNameSet.iterator(); nameIT.hasNext();) {
			String name = (String) nameIT.next();
			GrammarAST actionAST = (GrammarAST)scopeActions.get(name);
			List chunks = translateAction(ruleName,actionAST);
			scopeActions.put(name, chunks); // replace with translation
		}
	}

	/** Error recovery in ANTLR recognizers.
	 *
	 *  Based upon original ideas:
	 *
	 *  Algorithms + Data Structures = Programs by Niklaus Wirth
	 *
	 *  and
	 *
	 *  A note on error recovery in recursive descent parsers:
	 *  http://portal.acm.org/citation.cfm?id=947902.947905
	 *
	 *  Later, Josef Grosch had some good ideas:
	 *  Efficient and Comfortable Error Recovery in Recursive Descent Parsers:
	 *  ftp://www.cocolab.com/products/cocktail/doca4.ps/ell.ps.zip
	 *
	 *  Like Grosch I implemented local FOLLOW sets that are combined at run-time
	 *  upon error to avoid parsing overhead.
	 */
	public void generateLocalFOLLOW(GrammarAST referencedElementNode,
									String referencedElementName,
									String enclosingRuleName,
									int elementIndex)
	{
		/*
		System.out.println("compute FOLLOW "+grammar.name+"."+referencedElementNode.toString()+
						 " for "+referencedElementName+"#"+elementIndex +" in "+
						 enclosingRuleName+
						 " line="+referencedElementNode.getLine());
						 */
		NFAState followingNFAState = referencedElementNode.followingNFAState;
		LookaheadSet follow = null;
		if ( followingNFAState!=null ) {
			// compute follow for this element and, as side-effect, track
			// the rule LOOK sensitivity.
			follow = grammar.FIRST(followingNFAState);
		}

		if ( follow==null ) {
			ErrorManager.internalError("no follow state or cannot compute follow");
			follow = new LookaheadSet();
		}
		if ( follow.member(Label.EOF) ) {
			// TODO: can we just remove?  Seems needed here:
			// compilation_unit : global_statement* EOF
			// Actually i guess we resync to EOF regardless
			follow.remove(Label.EOF);
		}
		//System.out.println(" "+follow);

        List tokenTypeList = null;
        long[] words = null;
		if ( follow.tokenTypeSet==null ) {
			words = new long[1];
            tokenTypeList = new ArrayList();
        }
		else {
			BitSet bits = BitSet.of(follow.tokenTypeSet);
			words = bits.toPackedArray();
            tokenTypeList = follow.tokenTypeSet.toList();
        }
		// use the target to convert to hex strings (typically)
		String[] wordStrings = new String[words.length];
		for (int j = 0; j < words.length; j++) {
			long w = words[j];
			wordStrings[j] = target.getTarget64BitStringFromValue(w);
		}
        recognizerST.setAttribute("bitsets.{name,inName,bits,tokenTypes,tokenIndex}",
                referencedElementName,
                enclosingRuleName,
                wordStrings,
                tokenTypeList,
                Utils.integer(elementIndex));
        outputFileST.setAttribute("bitsets.{name,inName,bits,tokenTypes,tokenIndex}",
                referencedElementName,
                enclosingRuleName,
                wordStrings,
                tokenTypeList,
                Utils.integer(elementIndex));
        headerFileST.setAttribute("bitsets.{name,inName,bits,tokenTypes,tokenIndex}",
                referencedElementName,
                enclosingRuleName,
                wordStrings,
                tokenTypeList,
                Utils.integer(elementIndex));
	}

	// L O O K A H E A D  D E C I S I O N  G E N E R A T I O N

	/** Generate code that computes the predicted alt given a DFA.  The
	 *  recognizerST can be either the main generated recognizerTemplate
	 *  for storage in the main parser file or a separate file.  It's up to
	 *  the code that ultimately invokes the codegen.g grammar rule.
	 *
	 *  Regardless, the output file and header file get a copy of the DFAs.
	 */
	public StringTemplate genLookaheadDecision(StringTemplate recognizerST,
											   DFA dfa)
	{
		StringTemplate decisionST;
		// If we are doing inline DFA and this one is acyclic and LL(*)
		// I have to check for is-non-LL(*) because if non-LL(*) the cyclic
		// check is not done by DFA.verify(); that is, verify() avoids
		// doesStateReachAcceptState() if non-LL(*)
		if ( dfa.canInlineDecision() ) {
			decisionST =
				acyclicDFAGenerator.genFixedLookaheadDecision(getTemplates(), dfa);
		}
		else {
			// generate any kind of DFA here (cyclic or acyclic)
			dfa.createStateTables(this);
			outputFileST.setAttribute("cyclicDFAs", dfa);
			headerFileST.setAttribute("cyclicDFAs", dfa);
			decisionST = templates.getInstanceOf("dfaDecision");
			String description = dfa.getNFADecisionStartState().getDescription();
			description = target.getTargetStringLiteralFromString(description);
			if ( description!=null ) {
				decisionST.setAttribute("description", description);
			}
			decisionST.setAttribute("decisionNumber",
									Utils.integer(dfa.getDecisionNumber()));
		}
		return decisionST;
	}

	/** A special state is huge (too big for state tables) or has a predicated
	 *  edge.  Generate a simple if-then-else.  Cannot be an accept state as
	 *  they have no emanating edges.  Don't worry about switch vs if-then-else
	 *  because if you get here, the state is super complicated and needs an
	 *  if-then-else.  This is used by the new DFA scheme created June 2006.
	 */
	public StringTemplate generateSpecialState(DFAState s) {
		StringTemplate stateST;
		stateST = templates.getInstanceOf("cyclicDFAState");
		stateST.setAttribute("needErrorClause", Boolean.valueOf(true));
		stateST.setAttribute("semPredState",
							 Boolean.valueOf(s.isResolvedWithPredicates()));
		stateST.setAttribute("stateNumber", s.stateNumber);
		stateST.setAttribute("decisionNumber", s.dfa.decisionNumber);

		boolean foundGatedPred = false;
		StringTemplate eotST = null;
		for (int i = 0; i < s.getNumberOfTransitions(); i++) {
			Transition edge = (Transition) s.transition(i);
			StringTemplate edgeST;
			if ( edge.label.getAtom()==Label.EOT ) {
				// this is the default clause; has to held until last
				edgeST = templates.getInstanceOf("eotDFAEdge");
				stateST.removeAttribute("needErrorClause");
				eotST = edgeST;
			}
			else {
				edgeST = templates.getInstanceOf("cyclicDFAEdge");
				StringTemplate exprST =
					genLabelExpr(templates,edge,1);
				edgeST.setAttribute("labelExpr", exprST);
			}
			edgeST.setAttribute("edgeNumber", Utils.integer(i+1));
			edgeST.setAttribute("targetStateNumber",
								 Utils.integer(edge.target.stateNumber));
			// stick in any gated predicates for any edge if not already a pred
			if ( !edge.label.isSemanticPredicate() ) {
				DFAState t = (DFAState)edge.target;
				SemanticContext preds =	t.getGatedPredicatesInNFAConfigurations();
				if ( preds!=null ) {
					foundGatedPred = true;
					StringTemplate predST = preds.genExpr(this,
														  getTemplates(),
														  t.dfa);
					edgeST.setAttribute("predicates", predST.toString());
				}
			}
			if ( edge.label.getAtom()!=Label.EOT ) {
				stateST.setAttribute("edges", edgeST);
			}
		}
		if ( foundGatedPred ) {
			// state has >= 1 edge with a gated pred (syn or sem)
			// must rewind input first, set flag.
			stateST.setAttribute("semPredState", new Boolean(foundGatedPred));
		}
		if ( eotST!=null ) {
			stateST.setAttribute("edges", eotST);
		}
		return stateST;
	}

	/** Generate an expression for traversing an edge. */
	protected StringTemplate genLabelExpr(StringTemplateGroup templates,
										  Transition edge,
										  int k)
	{
		Label label = edge.label;
		if ( label.isSemanticPredicate() ) {
			return genSemanticPredicateExpr(templates, edge);
		}
		if ( label.isSet() ) {
			return genSetExpr(templates, label.getSet(), k, true);
		}
		// must be simple label
		StringTemplate eST = templates.getInstanceOf("lookaheadTest");
		eST.setAttribute("atom", getTokenTypeAsTargetLabel(label.getAtom()));
		eST.setAttribute("atomAsInt", Utils.integer(label.getAtom()));
		eST.setAttribute("k", Utils.integer(k));
		return eST;
	}

	protected StringTemplate genSemanticPredicateExpr(StringTemplateGroup templates,
													  Transition edge)
	{
		DFA dfa = ((DFAState)edge.target).dfa; // which DFA are we in
		Label label = edge.label;
		SemanticContext semCtx = label.getSemanticContext();
		return semCtx.genExpr(this,templates,dfa);
	}

	/** For intervals such as [3..3, 30..35], generate an expression that
	 *  tests the lookahead similar to LA(1)==3 || (LA(1)>=30&&LA(1)<=35)
	 */
	public StringTemplate genSetExpr(StringTemplateGroup templates,
									 IntSet set,
									 int k,
									 boolean partOfDFA)
	{
		if ( !(set instanceof IntervalSet) ) {
			throw new IllegalArgumentException("unable to generate expressions for non IntervalSet objects");
		}
		IntervalSet iset = (IntervalSet)set;
		if ( iset.getIntervals()==null || iset.getIntervals().size()==0 ) {
			StringTemplate emptyST = new StringTemplate(templates, "");
			emptyST.setName("empty-set-expr");
			return emptyST;
		}
		String testSTName = "lookaheadTest";
		String testRangeSTName = "lookaheadRangeTest";
		if ( !partOfDFA ) {
			testSTName = "isolatedLookaheadTest";
			testRangeSTName = "isolatedLookaheadRangeTest";
		}
		StringTemplate setST = templates.getInstanceOf("setTest");
		Iterator iter = iset.getIntervals().iterator();
		int rangeNumber = 1;
		while (iter.hasNext()) {
			Interval I = (Interval) iter.next();
			int a = I.a;
			int b = I.b;
			StringTemplate eST;
			if ( a==b ) {
				eST = templates.getInstanceOf(testSTName);
				eST.setAttribute("atom", getTokenTypeAsTargetLabel(a));
				eST.setAttribute("atomAsInt", Utils.integer(a));
				//eST.setAttribute("k",Utils.integer(k));
			}
			else {
				eST = templates.getInstanceOf(testRangeSTName);
				eST.setAttribute("lower",getTokenTypeAsTargetLabel(a));
				eST.setAttribute("lowerAsInt", Utils.integer(a));
				eST.setAttribute("upper",getTokenTypeAsTargetLabel(b));
				eST.setAttribute("upperAsInt", Utils.integer(b));
				eST.setAttribute("rangeNumber",Utils.integer(rangeNumber));
			}
			eST.setAttribute("k",Utils.integer(k));
			setST.setAttribute("ranges", eST);
			rangeNumber++;
		}
		return setST;
	}

	// T O K E N  D E F I N I T I O N  G E N E R A T I O N

	/** Set attributes tokens and literals attributes in the incoming
	 *  code template.  This is not the token vocab interchange file, but
	 *  rather a list of token type ID needed by the recognizer.
	 */
	protected void genTokenTypeConstants(StringTemplate code) {
		// make constants for the token types
		Iterator tokenIDs = grammar.getTokenIDs().iterator();
		while (tokenIDs.hasNext()) {
			String tokenID = (String) tokenIDs.next();
			int tokenType = grammar.getTokenType(tokenID);
			if ( tokenType==Label.EOF ||
				 tokenType>=Label.MIN_TOKEN_TYPE )
			{
				// don't do FAUX labels 'cept EOF
				code.setAttribute("tokens.{name,type}", tokenID, Utils.integer(tokenType));
			}
		}
	}

	/** Generate a token names table that maps token type to a printable
	 *  name: either the label like INT or the literal like "begin".
	 */
	protected void genTokenTypeNames(StringTemplate code) {
		for (int t=Label.MIN_TOKEN_TYPE; t<=grammar.getMaxTokenType(); t++) {
			String tokenName = grammar.getTokenDisplayName(t);
			if ( tokenName!=null ) {
				tokenName=target.getTargetStringLiteralFromString(tokenName, true);
				code.setAttribute("tokenNames", tokenName);
			}
		}
	}

	/** Get a meaningful name for a token type useful during code generation.
	 *  Literals without associated names are converted to the string equivalent
	 *  of their integer values. Used to generate x==ID and x==34 type comparisons
	 *  etc...  Essentially we are looking for the most obvious way to refer
	 *  to a token type in the generated code.  If in the lexer, return the
	 *  char literal translated to the target language.  For example, ttype=10
	 *  will yield '\n' from the getTokenDisplayName method.  That must
	 *  be converted to the target languages literals.  For most C-derived
	 *  languages no translation is needed.
	 */
	public String getTokenTypeAsTargetLabel(int ttype) {
		if ( grammar.type==Grammar.LEXER ) {
			String name = grammar.getTokenDisplayName(ttype);
			return target.getTargetCharLiteralFromANTLRCharLiteral(this,name);
		}
		return target.getTokenTypeAsTargetLabel(this,ttype);
	}

	/** Generate a token vocab file with all the token names/types.  For example:
	 *  ID=7
	 *  FOR=8
	 *  'for'=8
	 *
	 *  This is independent of the target language; used by antlr internally
	 */
	protected StringTemplate genTokenVocabOutput() {
		StringTemplate vocabFileST =
			new StringTemplate(vocabFilePattern,
							   AngleBracketTemplateLexer.class);
		vocabFileST.setName("vocab-file");
		// make constants for the token names
		Iterator tokenIDs = grammar.getTokenIDs().iterator();
		while (tokenIDs.hasNext()) {
			String tokenID = (String) tokenIDs.next();
			int tokenType = grammar.getTokenType(tokenID);
			if ( tokenType>=Label.MIN_TOKEN_TYPE ) {
				vocabFileST.setAttribute("tokens.{name,type}", tokenID, Utils.integer(tokenType));
			}
		}

		// now dump the strings
		Iterator literals = grammar.getStringLiterals().iterator();
		while (literals.hasNext()) {
			String literal = (String) literals.next();
			int tokenType = grammar.getTokenType(literal);
			if ( tokenType>=Label.MIN_TOKEN_TYPE ) {
				vocabFileST.setAttribute("tokens.{name,type}", literal, Utils.integer(tokenType));
			}
		}

		return vocabFileST;
	}

	public List translateAction(String ruleName,
								GrammarAST actionTree)
	{
		if ( actionTree.getType()==ANTLRParser.ARG_ACTION ) {
			return translateArgAction(ruleName, actionTree);
		}
		ActionTranslator translator = new ActionTranslator(this,ruleName,actionTree);
		List chunks = translator.translateToChunks();
		chunks = target.postProcessAction(chunks, actionTree.token);
		return chunks;
	}

	/** Translate an action like [3,"foo",a[3]] and return a List of the
	 *  translated actions.  Because actions are themselves translated to a list
	 *  of chunks, must cat together into a StringTemplate>.  Don't translate
	 *  to strings early as we need to eval templates in context.
	 */
	public List<StringTemplate> translateArgAction(String ruleName,
										   GrammarAST actionTree)
	{
		String actionText = actionTree.token.getText();
		List<String> args = getListOfArgumentsFromAction(actionText,',');
		List<StringTemplate> translatedArgs = new ArrayList<StringTemplate>();
		for (String arg : args) {
			if ( arg!=null ) {
				antlr.Token actionToken =
					new antlr.CommonToken(ANTLRParser.ACTION,arg);
				ActionTranslator translator =
					new ActionTranslator(this,ruleName,
											  actionToken,
											  actionTree.outerAltNum);
				List chunks = translator.translateToChunks();
				chunks = target.postProcessAction(chunks, actionToken);
				StringTemplate catST = new StringTemplate(templates, "<chunks>");
				catST.setAttribute("chunks", chunks);
				templates.createStringTemplate();
				translatedArgs.add(catST);
			}
		}
		if ( translatedArgs.size()==0 ) {
			return null;
		}
		return translatedArgs;
	}

	public static List<String> getListOfArgumentsFromAction(String actionText,
															int separatorChar)
	{
		List<String> args = new ArrayList<String>();
		getListOfArgumentsFromAction(actionText, 0, -1, separatorChar, args);
		return args;
	}

	/** Given an arg action like
	 *
	 *  [x, (*a).foo(21,33), 3.2+1, '\n',
	 *  "a,oo\nick", {bl, "fdkj"eck}, ["cat\n,", x, 43]]
	 *
	 *  convert to a list of arguments.  Allow nested square brackets etc...
	 *  Set separatorChar to ';' or ',' or whatever you want.
	 */
	public static int getListOfArgumentsFromAction(String actionText,
												   int start,
												   int targetChar,
												   int separatorChar,
												   List<String> args)
	{
		if ( actionText==null ) {
			return -1;
		}
		actionText = actionText.replaceAll("//.*\n", "");
		int n = actionText.length();
		//System.out.println("actionText@"+start+"->"+(char)targetChar+"="+actionText.substring(start,n));
		int p = start;
		int last = p;
		while ( p<n && actionText.charAt(p)!=targetChar ) {
			int c = actionText.charAt(p);
			switch ( c ) {
				case '\'' :
					p++;
					while ( p<n && actionText.charAt(p)!='\'' ) {
						if ( actionText.charAt(p)=='\\' && (p+1)<n &&
							 actionText.charAt(p+1)=='\'' )
						{
							p++; // skip escaped quote
						}
						p++;
					}
					p++;
					break;
				case '"' :
					p++;
					while ( p<n && actionText.charAt(p)!='\"' ) {
						if ( actionText.charAt(p)=='\\' && (p+1)<n &&
							 actionText.charAt(p+1)=='\"' )
						{
							p++; // skip escaped quote
						}
						p++;
					}
					p++;
					break;
				case '(' :
					p = getListOfArgumentsFromAction(actionText,p+1,')',separatorChar,args);
					break;
				case '{' :
					p = getListOfArgumentsFromAction(actionText,p+1,'}',separatorChar,args);
					break;
				case '<' :
					if ( actionText.indexOf('>',p+1)>=p ) {
						// do we see a matching '>' ahead?  if so, hope it's a generic
						// and not less followed by expr with greater than
						p = getListOfArgumentsFromAction(actionText,p+1,'>',separatorChar,args);
					}
					else {
						p++; // treat as normal char
					}
					break;
				case '[' :
					p = getListOfArgumentsFromAction(actionText,p+1,']',separatorChar,args);
					break;
				default :
					if ( c==separatorChar && targetChar==-1 ) {
						String arg = actionText.substring(last, p);
						//System.out.println("arg="+arg);
						args.add(arg.trim());
						last = p+1;
					}
					p++;
					break;
			}
		}
		if ( targetChar==-1 && p<=n ) {
			String arg = actionText.substring(last, p).trim();
			//System.out.println("arg="+arg);
			if ( arg.length()>0 ) {
				args.add(arg.trim());
			}
		}
		p++;
		return p;
	}

	/** Given a template constructor action like %foo(a={...}) in
	 *  an action, translate it to the appropriate template constructor
	 *  from the templateLib. This translates a *piece* of the action.
	 */
	public StringTemplate translateTemplateConstructor(String ruleName,
													   int outerAltNum,
													   antlr.Token actionToken,
													   String templateActionText)
	{
		// first, parse with antlr.g
		//System.out.println("translate template: "+templateActionText);
		ANTLRLexer lexer = new ANTLRLexer(new StringReader(templateActionText));
		lexer.setFilename(grammar.getFileName());
		lexer.setTokenObjectClass("antlr.TokenWithIndex");
		TokenStreamRewriteEngine tokenBuffer = new TokenStreamRewriteEngine(lexer);
		tokenBuffer.discard(ANTLRParser.WS);
		tokenBuffer.discard(ANTLRParser.ML_COMMENT);
		tokenBuffer.discard(ANTLRParser.COMMENT);
		tokenBuffer.discard(ANTLRParser.SL_COMMENT);
		ANTLRParser parser = new ANTLRParser(tokenBuffer);
		parser.setFilename(grammar.getFileName());
		parser.setASTNodeClass("org.antlr.tool.GrammarAST");
		try {
			parser.rewrite_template();
		}
		catch (RecognitionException re) {
			ErrorManager.grammarError(ErrorManager.MSG_INVALID_TEMPLATE_ACTION,
										  grammar,
										  actionToken,
										  templateActionText);
		}
		catch (Exception tse) {
			ErrorManager.internalError("can't parse template action",tse);
		}
		GrammarAST rewriteTree = (GrammarAST)parser.getAST();

		// then translate via codegen.g
		CodeGenTreeWalker gen = new CodeGenTreeWalker();
		gen.init(grammar);
		gen.setCurrentRuleName(ruleName);
		gen.setOuterAltNum(outerAltNum);
		StringTemplate st = null;
		try {
			st = gen.rewrite_template((AST)rewriteTree);
		}
		catch (RecognitionException re) {
			ErrorManager.error(ErrorManager.MSG_BAD_AST_STRUCTURE,
							   re);
		}
		return st;
	}


	public void issueInvalidScopeError(String x,
									   String y,
									   Rule enclosingRule,
									   antlr.Token actionToken,
									   int outerAltNum)
	{
		//System.out.println("error $"+x+"::"+y);
		Rule r = grammar.getRule(x);
		AttributeScope scope = grammar.getGlobalScope(x);
		if ( scope==null ) {
			if ( r!=null ) {
				scope = r.ruleScope; // if not global, might be rule scope
			}
		}
		if ( scope==null ) {
			ErrorManager.grammarError(ErrorManager.MSG_UNKNOWN_DYNAMIC_SCOPE,
										  grammar,
										  actionToken,
										  x);
		}
		else if ( scope.getAttribute(y)==null ) {
			ErrorManager.grammarError(ErrorManager.MSG_UNKNOWN_DYNAMIC_SCOPE_ATTRIBUTE,
										  grammar,
										  actionToken,
										  x,
										  y);
		}
	}

	public void issueInvalidAttributeError(String x,
										   String y,
										   Rule enclosingRule,
										   antlr.Token actionToken,
										   int outerAltNum)
	{
		//System.out.println("error $"+x+"."+y);
		if ( enclosingRule==null ) {
			// action not in a rule
			ErrorManager.grammarError(ErrorManager.MSG_ATTRIBUTE_REF_NOT_IN_RULE,
										  grammar,
										  actionToken,
										  x,
										  y);
			return;
		}

		// action is in a rule
		Grammar.LabelElementPair label = enclosingRule.getRuleLabel(x);

		if ( label!=null || enclosingRule.getRuleRefsInAlt(x, outerAltNum)!=null ) {
			// $rulelabel.attr or $ruleref.attr; must be unknown attr
			String refdRuleName = x;
			if ( label!=null ) {
				refdRuleName = enclosingRule.getRuleLabel(x).referencedRuleName;
			}
			Rule refdRule = grammar.getRule(refdRuleName);
			AttributeScope scope = refdRule.getAttributeScope(y);
			if ( scope==null ) {
				ErrorManager.grammarError(ErrorManager.MSG_UNKNOWN_RULE_ATTRIBUTE,
										  grammar,
										  actionToken,
										  refdRuleName,
										  y);
			}
			else if ( scope.isParameterScope ) {
				ErrorManager.grammarError(ErrorManager.MSG_INVALID_RULE_PARAMETER_REF,
										  grammar,
										  actionToken,
										  refdRuleName,
										  y);
			}
			else if ( scope.isDynamicRuleScope ) {
				ErrorManager.grammarError(ErrorManager.MSG_INVALID_RULE_SCOPE_ATTRIBUTE_REF,
										  grammar,
										  actionToken,
										  refdRuleName,
										  y);
			}
		}

	}

	public void issueInvalidAttributeError(String x,
										   Rule enclosingRule,
										   antlr.Token actionToken,
										   int outerAltNum)
	{
		//System.out.println("error $"+x);
		if ( enclosingRule==null ) {
			// action not in a rule
			ErrorManager.grammarError(ErrorManager.MSG_ATTRIBUTE_REF_NOT_IN_RULE,
										  grammar,
										  actionToken,
										  x);
			return;
		}

		// action is in a rule
		Grammar.LabelElementPair label = enclosingRule.getRuleLabel(x);
		AttributeScope scope = enclosingRule.getAttributeScope(x);

		if ( label!=null ||
			 enclosingRule.getRuleRefsInAlt(x, outerAltNum)!=null ||
			 enclosingRule.name.equals(x) )
		{
			ErrorManager.grammarError(ErrorManager.MSG_ISOLATED_RULE_SCOPE,
										  grammar,
										  actionToken,
										  x);
		}
		else if ( scope!=null && scope.isDynamicRuleScope ) {
			ErrorManager.grammarError(ErrorManager.MSG_ISOLATED_RULE_ATTRIBUTE,
										  grammar,
										  actionToken,
										  x);
		}
		else {
			ErrorManager.grammarError(ErrorManager.MSG_UNKNOWN_SIMPLE_ATTRIBUTE,
									  grammar,
									  actionToken,
									  x);
		}
	}

	// M I S C

	public StringTemplateGroup getTemplates() {
		return templates;
	}

	public StringTemplateGroup getBaseTemplates() {
		return baseTemplates;
	}

	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	public void setTrace(boolean trace) {
		this.trace = trace;
	}

	public void setProfile(boolean profile) {
		this.profile = profile;
		if ( profile ) {
			setDebug(true); // requires debug events
		}
	}

	public StringTemplate getRecognizerST() {
		return outputFileST;
	}

	/** Generate TParser.java and TLexer.java from T.g if combined, else
	 *  just use T.java as output regardless of type.
	 */
	public String getRecognizerFileName(String name, int type) {
		StringTemplate extST = templates.getInstanceOf("codeFileExtension");
		String recognizerName = grammar.getRecognizerName();
		return recognizerName+extST.toString();
		/*
		String suffix = "";
		if ( type==Grammar.COMBINED ||
			 (type==Grammar.LEXER && !grammar.implicitLexer) )
		{
			suffix = Grammar.grammarTypeToFileNameSuffix[type];
		}
		return name+suffix+extST.toString();
		*/
	}

	/** What is the name of the vocab file generated for this grammar?
	 *  Returns null if no .tokens file should be generated.
	 */
	public String getVocabFileName() {
		if ( grammar.isBuiltFromString() ) {
			return null;
		}
		return grammar.name+VOCAB_FILE_EXTENSION;
	}

	public void write(StringTemplate code, String fileName) throws IOException {
		long start = System.currentTimeMillis();
		Writer w = tool.getOutputFile(grammar, fileName);
		// Write the output to a StringWriter
		StringTemplateWriter wr = templates.getStringTemplateWriter(w);
		wr.setLineWidth(lineWidth);
		code.write(wr);
		w.close();
		long stop = System.currentTimeMillis();
		//System.out.println("render time for "+fileName+": "+(int)(stop-start)+"ms");
	}

	/** You can generate a switch rather than if-then-else for a DFA state
	 *  if there are no semantic predicates and the number of edge label
	 *  values is small enough; e.g., don't generate a switch for a state
	 *  containing an edge label such as 20..52330 (the resulting byte codes
	 *  would overflow the method 65k limit probably).
	 */
	protected boolean canGenerateSwitch(DFAState s) {
		if ( !GENERATE_SWITCHES_WHEN_POSSIBLE ) {
			return false;
		}
		int size = 0;
		for (int i = 0; i < s.getNumberOfTransitions(); i++) {
			Transition edge = (Transition) s.transition(i);
			if ( edge.label.isSemanticPredicate() ) {
				return false;
			}
			// can't do a switch if the edges are going to require predicates
			if ( edge.label.getAtom()==Label.EOT ) {
				int EOTPredicts = ((DFAState)edge.target).getUniquelyPredictedAlt();
				if ( EOTPredicts==NFA.INVALID_ALT_NUMBER ) {
					// EOT target has to be a predicate then; no unique alt
					return false;
				}
			}
			// if target is a state with gated preds, we need to use preds on
			// this edge then to reach it.
			if ( ((DFAState)edge.target).getGatedPredicatesInNFAConfigurations()!=null ) {
				return false;
			}
			size += edge.label.getSet().size();
		}
		if ( s.getNumberOfTransitions()<MIN_SWITCH_ALTS ||
			 size>MAX_SWITCH_CASE_LABELS ) {
			return false;
		}
		return true;
	}

	/** Create a label to track a token / rule reference's result.
	 *  Technically, this is a place where I break model-view separation
	 *  as I am creating a variable name that could be invalid in a
	 *  target language, however, label ::= <ID><INT> is probably ok in
	 *  all languages we care about.
	 */
	public String createUniqueLabel(String name) {
		return new StringBuffer()
			.append(name).append(uniqueLabelNumber++).toString();
	}
}
