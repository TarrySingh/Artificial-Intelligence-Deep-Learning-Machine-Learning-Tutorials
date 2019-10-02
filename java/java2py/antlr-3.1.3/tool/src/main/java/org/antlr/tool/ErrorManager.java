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

import antlr.Token;
import org.antlr.Tool;
import org.antlr.misc.BitSet;
import org.antlr.analysis.DFAState;
import org.antlr.analysis.DecisionProbe;
import org.antlr.analysis.Label;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.StringTemplateErrorListener;
import org.antlr.stringtemplate.StringTemplateGroup;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.util.*;

/** Defines all the errors ANTLR can generator for both the tool and for
 *  issues with a grammar.
 *
 *  Here is a list of language names:
 *
 *  http://ftp.ics.uci.edu/pub/ietf/http/related/iso639.txt
 *
 *  Here is a list of country names:
 *
 *  http://www.chemie.fu-berlin.de/diverse/doc/ISO_3166.html
 *
 *  I use constants not strings to identify messages as the compiler will
 *  find any errors/mismatches rather than leaving a mistyped string in
 *  the code to be found randomly in the future.  Further, Intellij can
 *  do field name expansion to save me some typing.  I have to map
 *  int constants to template names, however, which could introduce a mismatch.
 *  Someone could provide a .stg file that had a template name wrong.  When
 *  I load the group, then, I must verify that all messages are there.
 *
 *  This is essentially the functionality of the resource bundle stuff Java
 *  has, but I don't want to load a property file--I want to load a template
 *  group file and this is so simple, why mess with their junk.
 *
 *  I use the default Locale as defined by java to compute a group file name
 *  in the org/antlr/tool/templates/messages dir called en_US.stg and so on.
 *
 *  Normally we want to use the default locale, but often a message file will
 *  not exist for it so we must fall back on the US local.
 *
 *  During initialization of this class, all errors go straight to System.err.
 *  There is no way around this.  If I have not set up the error system, how
 *  can I do errors properly?  For example, if the string template group file
 *  full of messages has an error, how could I print to anything but System.err?
 *
 *  TODO: how to map locale to a file encoding for the stringtemplate group file?
 *  StringTemplate knows how to pay attention to the default encoding so it
 *  should probably just work unless a GUI sets the local to some chinese
 *  variation but System.getProperty("file.encoding") is US.  Hmm...
 *
 *  TODO: get antlr.g etc.. parsing errors to come here.
 */
public class ErrorManager {
	// TOOL ERRORS
	// file errors
	public static final int MSG_CANNOT_WRITE_FILE = 1;
	public static final int MSG_CANNOT_CLOSE_FILE = 2;
	public static final int MSG_CANNOT_FIND_TOKENS_FILE = 3;
	public static final int MSG_ERROR_READING_TOKENS_FILE = 4;
	public static final int MSG_DIR_NOT_FOUND = 5;
	public static final int MSG_OUTPUT_DIR_IS_FILE = 6;
	public static final int MSG_CANNOT_OPEN_FILE = 7;
	public static final int MSG_FILE_AND_GRAMMAR_NAME_DIFFER = 8;
	public static final int MSG_FILENAME_EXTENSION_ERROR = 9;

	public static final int MSG_INTERNAL_ERROR = 10;
	public static final int MSG_INTERNAL_WARNING = 11;
	public static final int MSG_ERROR_CREATING_ARTIFICIAL_RULE = 12;
	public static final int MSG_TOKENS_FILE_SYNTAX_ERROR = 13;
	public static final int MSG_CANNOT_GEN_DOT_FILE = 14;
	public static final int MSG_BAD_AST_STRUCTURE = 15;
	public static final int MSG_BAD_ACTION_AST_STRUCTURE = 16;

	// code gen errors
	public static final int MSG_MISSING_CODE_GEN_TEMPLATES = 20;
	public static final int MSG_MISSING_CYCLIC_DFA_CODE_GEN_TEMPLATES = 21;
	public static final int MSG_CODE_GEN_TEMPLATES_INCOMPLETE = 22;
	public static final int MSG_CANNOT_CREATE_TARGET_GENERATOR = 23;
	//public static final int MSG_CANNOT_COMPUTE_SAMPLE_INPUT_SEQ = 24;

	// GRAMMAR ERRORS
	public static final int MSG_SYNTAX_ERROR = 100;
	public static final int MSG_RULE_REDEFINITION = 101;
	public static final int MSG_LEXER_RULES_NOT_ALLOWED = 102;
	public static final int MSG_PARSER_RULES_NOT_ALLOWED = 103;
	public static final int MSG_CANNOT_FIND_ATTRIBUTE_NAME_IN_DECL = 104;
	public static final int MSG_NO_TOKEN_DEFINITION = 105;
	public static final int MSG_UNDEFINED_RULE_REF = 106;
	public static final int MSG_LITERAL_NOT_ASSOCIATED_WITH_LEXER_RULE = 107;
	public static final int MSG_CANNOT_ALIAS_TOKENS_IN_LEXER = 108;
	public static final int MSG_ATTRIBUTE_REF_NOT_IN_RULE = 111;
	public static final int MSG_INVALID_RULE_SCOPE_ATTRIBUTE_REF = 112;
	public static final int MSG_UNKNOWN_ATTRIBUTE_IN_SCOPE = 113;
	public static final int MSG_UNKNOWN_SIMPLE_ATTRIBUTE = 114;
	public static final int MSG_INVALID_RULE_PARAMETER_REF = 115;
	public static final int MSG_UNKNOWN_RULE_ATTRIBUTE = 116;
	public static final int MSG_ISOLATED_RULE_SCOPE = 117;
	public static final int MSG_SYMBOL_CONFLICTS_WITH_GLOBAL_SCOPE = 118;
	public static final int MSG_LABEL_CONFLICTS_WITH_RULE = 119;
	public static final int MSG_LABEL_CONFLICTS_WITH_TOKEN = 120;
	public static final int MSG_LABEL_CONFLICTS_WITH_RULE_SCOPE_ATTRIBUTE = 121;
	public static final int MSG_LABEL_CONFLICTS_WITH_RULE_ARG_RETVAL = 122;
	public static final int MSG_ATTRIBUTE_CONFLICTS_WITH_RULE = 123;
	public static final int MSG_ATTRIBUTE_CONFLICTS_WITH_RULE_ARG_RETVAL = 124;
	public static final int MSG_LABEL_TYPE_CONFLICT = 125;
	public static final int MSG_ARG_RETVAL_CONFLICT = 126;
	public static final int MSG_NONUNIQUE_REF = 127;
	public static final int MSG_FORWARD_ELEMENT_REF = 128;
	public static final int MSG_MISSING_RULE_ARGS = 129;
	public static final int MSG_RULE_HAS_NO_ARGS = 130;
	public static final int MSG_ARGS_ON_TOKEN_REF = 131;
	public static final int MSG_RULE_REF_AMBIG_WITH_RULE_IN_ALT = 132;
	public static final int MSG_ILLEGAL_OPTION = 133;
	public static final int MSG_LIST_LABEL_INVALID_UNLESS_RETVAL_STRUCT = 134;
	public static final int MSG_UNDEFINED_TOKEN_REF_IN_REWRITE = 135;
	public static final int MSG_REWRITE_ELEMENT_NOT_PRESENT_ON_LHS = 136;
	public static final int MSG_UNDEFINED_LABEL_REF_IN_REWRITE = 137;
	public static final int MSG_NO_GRAMMAR_START_RULE = 138;
	public static final int MSG_EMPTY_COMPLEMENT = 139;
	public static final int MSG_UNKNOWN_DYNAMIC_SCOPE = 140;
	public static final int MSG_UNKNOWN_DYNAMIC_SCOPE_ATTRIBUTE = 141;
	public static final int MSG_ISOLATED_RULE_ATTRIBUTE = 142;
	public static final int MSG_INVALID_ACTION_SCOPE = 143;
	public static final int MSG_ACTION_REDEFINITION = 144;
	public static final int MSG_DOUBLE_QUOTES_ILLEGAL = 145;
	public static final int MSG_INVALID_TEMPLATE_ACTION = 146;
	public static final int MSG_MISSING_ATTRIBUTE_NAME = 147;
	public static final int MSG_ARG_INIT_VALUES_ILLEGAL = 148;
	public static final int MSG_REWRITE_OR_OP_WITH_NO_OUTPUT_OPTION = 149;
	public static final int MSG_NO_RULES = 150;
	public static final int MSG_WRITE_TO_READONLY_ATTR = 151;
	public static final int MSG_MISSING_AST_TYPE_IN_TREE_GRAMMAR = 152;
	public static final int MSG_REWRITE_FOR_MULTI_ELEMENT_ALT = 153;
	public static final int MSG_RULE_INVALID_SET = 154;
	public static final int MSG_HETERO_ILLEGAL_IN_REWRITE_ALT = 155;
	public static final int MSG_NO_SUCH_GRAMMAR_SCOPE = 156;
	public static final int MSG_NO_SUCH_RULE_IN_SCOPE = 157;
	public static final int MSG_TOKEN_ALIAS_CONFLICT = 158;
	public static final int MSG_TOKEN_ALIAS_REASSIGNMENT = 159;
	public static final int MSG_TOKEN_VOCAB_IN_DELEGATE = 160;
	public static final int MSG_INVALID_IMPORT = 161;
	public static final int MSG_IMPORTED_TOKENS_RULE_EMPTY = 162;
	public static final int MSG_IMPORT_NAME_CLASH = 163;
	public static final int MSG_AST_OP_WITH_NON_AST_OUTPUT_OPTION = 164;
	public static final int MSG_AST_OP_IN_ALT_WITH_REWRITE = 165;
    public static final int MSG_WILDCARD_AS_ROOT = 166;
    public static final int MSG_CONFLICTING_OPTION_IN_TREE_FILTER = 167;


	// GRAMMAR WARNINGS
	public static final int MSG_GRAMMAR_NONDETERMINISM = 200; // A predicts alts 1,2
	public static final int MSG_UNREACHABLE_ALTS = 201;       // nothing predicts alt i
	public static final int MSG_DANGLING_STATE = 202;         // no edges out of state
	public static final int MSG_INSUFFICIENT_PREDICATES = 203;
	public static final int MSG_DUPLICATE_SET_ENTRY = 204;    // (A|A)
	public static final int MSG_ANALYSIS_ABORTED = 205;
	public static final int MSG_RECURSION_OVERLOW = 206;
	public static final int MSG_LEFT_RECURSION = 207;
	public static final int MSG_UNREACHABLE_TOKENS = 208; // nothing predicts token
	public static final int MSG_TOKEN_NONDETERMINISM = 209; // alts of Tokens rule
	public static final int MSG_LEFT_RECURSION_CYCLES = 210;
	public static final int MSG_NONREGULAR_DECISION = 211;


    // Dependency sorting errors
    //
    public static final int MSG_CIRCULAR_DEPENDENCY = 212; // t1.g -> t2.g -> t3.g ->t1.g

	public static final int MAX_MESSAGE_NUMBER = 212;

	/** Do not do perform analysis if one of these happens */
	public static final BitSet ERRORS_FORCING_NO_ANALYSIS = new BitSet() {
		{
			add(MSG_RULE_REDEFINITION);
			add(MSG_UNDEFINED_RULE_REF);
			add(MSG_LEFT_RECURSION_CYCLES);
			add(MSG_REWRITE_OR_OP_WITH_NO_OUTPUT_OPTION);
			add(MSG_NO_RULES);
			add(MSG_NO_SUCH_GRAMMAR_SCOPE);
			add(MSG_NO_SUCH_RULE_IN_SCOPE);
			add(MSG_LEXER_RULES_NOT_ALLOWED);
            add(MSG_WILDCARD_AS_ROOT);
            add(MSG_CIRCULAR_DEPENDENCY);
            // TODO: ...
		}
	};

	/** Do not do code gen if one of these happens */
	public static final BitSet ERRORS_FORCING_NO_CODEGEN = new BitSet() {
		{
			add(MSG_NONREGULAR_DECISION);
			add(MSG_RECURSION_OVERLOW);
			add(MSG_UNREACHABLE_ALTS);
			add(MSG_FILE_AND_GRAMMAR_NAME_DIFFER);
			add(MSG_INVALID_IMPORT);
			add(MSG_AST_OP_WITH_NON_AST_OUTPUT_OPTION);
            add(MSG_CIRCULAR_DEPENDENCY);
			// TODO: ...
		}
	};

	/** Only one error can be emitted for any entry in this table.
	 *  Map<String,Set> where the key is a method name like danglingState.
	 *  The set is whatever that method accepts or derives like a DFA.
	 */
	public static final Map emitSingleError = new HashMap() {
		{
			put("danglingState", new HashSet());
		}
	};


	/** Messages should be sensitive to the locale. */
	private static Locale locale;
	private static String formatName;

	/** Each thread might need it's own error listener; e.g., a GUI with
	 *  multiple window frames holding multiple grammars.
	 */
	private static Map threadToListenerMap = new HashMap();

	static class ErrorState {
		public int errors;
		public int warnings;
		public int infos;
		/** Track all msgIDs; we use to abort later if necessary
		 *  also used in Message to find out what type of message it is via getMessageType()
		 */
		public BitSet errorMsgIDs = new BitSet();
		public BitSet warningMsgIDs = new BitSet();
		// TODO: figure out how to do info messages. these do not have IDs...kr
		//public BitSet infoMsgIDs = new BitSet();
	}

	/** Track the number of errors regardless of the listener but track
	 *  per thread.
	 */
	private static Map threadToErrorStateMap = new HashMap();

	/** Each thread has its own ptr to a Tool object, which knows how
	 *  to panic, for example.  In a GUI, the thread might just throw an Error
	 *  to exit rather than the suicide System.exit.
	 */
	private static Map threadToToolMap = new HashMap();

	/** The group of templates that represent all possible ANTLR errors. */
	private static StringTemplateGroup messages;
	/** The group of templates that represent the current message format. */
	private static StringTemplateGroup format;

	/** From a msgID how can I get the name of the template that describes
	 *  the error or warning?
	 */
	private static String[] idToMessageTemplateName = new String[MAX_MESSAGE_NUMBER+1];

	static ANTLRErrorListener theDefaultErrorListener = new ANTLRErrorListener() {
		public void info(String msg) {
			if (formatWantsSingleLineMessage()) {
				msg = msg.replaceAll("\n", " ");
			}
			System.err.println(msg);
		}

		public void error(Message msg) {
			String outputMsg = msg.toString();
			if (formatWantsSingleLineMessage()) {
				outputMsg = outputMsg.replaceAll("\n", " ");
			}
			System.err.println(outputMsg);
		}

		public void warning(Message msg) {
			String outputMsg = msg.toString();
			if (formatWantsSingleLineMessage()) {
				outputMsg = outputMsg.replaceAll("\n", " ");
			}
			System.err.println(outputMsg);
		}

		public void error(ToolMessage msg) {
			String outputMsg = msg.toString();
			if (formatWantsSingleLineMessage()) {
				outputMsg = outputMsg.replaceAll("\n", " ");
			}
			System.err.println(outputMsg);
		}
	};

	/** Handle all ST error listeners here (code gen, Grammar, and this class
	 *  use templates.
	 */
	static StringTemplateErrorListener initSTListener =
		new StringTemplateErrorListener() {
			public void error(String s, Throwable e) {
				System.err.println("ErrorManager init error: "+s);
				if ( e!=null ) {
					System.err.println("exception: "+e);
				}
				/*
				if ( e!=null ) {
					e.printStackTrace(System.err);
				}
				*/
			}
			public void warning(String s) {
				System.err.println("ErrorManager init warning: "+s);
			}
			public void debug(String s) {}
		};

	/** During verification of the messages group file, don't gen errors.
	 *  I'll handle them here.  This is used only after file has loaded ok
	 *  and only for the messages STG.
	 */
	static StringTemplateErrorListener blankSTListener =
		new StringTemplateErrorListener() {
			public void error(String s, Throwable e) {}
			public void warning(String s) {}
			public void debug(String s) {}
		};

	/** Errors during initialization related to ST must all go to System.err.
	 */
	static StringTemplateErrorListener theDefaultSTListener =
		new StringTemplateErrorListener() {
		public void error(String s, Throwable e) {
			if ( e instanceof InvocationTargetException ) {
				e = ((InvocationTargetException)e).getTargetException();
			}
			ErrorManager.error(ErrorManager.MSG_INTERNAL_ERROR, s, e);
		}
		public void warning(String s) {
			ErrorManager.warning(ErrorManager.MSG_INTERNAL_WARNING, s);
		}
		public void debug(String s) {
		}
	};

	// make sure that this class is ready to use after loading
	static {
		initIdToMessageNameMapping();
		// it is inefficient to set the default locale here if another
		// piece of code is going to set the locale, but that would
		// require that a user call an init() function or something.  I prefer
		// that this class be ready to go when loaded as I'm absentminded ;)
		setLocale(Locale.getDefault());
		// try to load the message format group
		// the user might have specified one on the command line
		// if not, or if the user has given an illegal value, we will fall back to "antlr"
		setFormat("antlr");
	}

    public static StringTemplateErrorListener getStringTemplateErrorListener() {
		return theDefaultSTListener;
	}

	/** We really only need a single locale for entire running ANTLR code
	 *  in a single VM.  Only pay attention to the language, not the country
	 *  so that French Canadians and French Frenchies all get the same
	 *  template file, fr.stg.  Just easier this way.
	 */
	public static void setLocale(Locale locale) {
		ErrorManager.locale = locale;
		String language = locale.getLanguage();
		String fileName = "org/antlr/tool/templates/messages/languages/"+language+".stg";
		ClassLoader cl = Thread.currentThread().getContextClassLoader();
		InputStream is = cl.getResourceAsStream(fileName);
		if ( is==null ) {
			cl = ErrorManager.class.getClassLoader();
			is = cl.getResourceAsStream(fileName);
		}
		if ( is==null && language.equals(Locale.US.getLanguage()) ) {
			rawError("ANTLR installation corrupted; cannot find English messages file "+fileName);
			panic();
		}
		else if ( is==null ) {
			//rawError("no such locale file "+fileName+" retrying with English locale");
			setLocale(Locale.US); // recurse on this rule, trying the US locale
			return;
		}
		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(is));
			messages = new StringTemplateGroup(br,
											   AngleBracketTemplateLexer.class,
											   initSTListener);
			br.close();
		}
		catch (IOException ioe) {
			rawError("error reading message file "+fileName, ioe);
		}
		finally {
			if ( br!=null ) {
				try {
					br.close();
				}
				catch (IOException ioe) {
					rawError("cannot close message file "+fileName, ioe);
				}
			}
		}

		messages.setErrorListener(blankSTListener);
		boolean messagesOK = verifyMessages();
		if ( !messagesOK && language.equals(Locale.US.getLanguage()) ) {
			rawError("ANTLR installation corrupted; English messages file "+language+".stg incomplete");
			panic();
		}
		else if ( !messagesOK ) {
			setLocale(Locale.US); // try US to see if that will work
		}
	}

	/** The format gets reset either from the Tool if the user supplied a command line option to that effect
	 *  Otherwise we just use the default "antlr".
	 */
	public static void setFormat(String formatName) {
		ErrorManager.formatName = formatName;
		String fileName = "org/antlr/tool/templates/messages/formats/"+formatName+".stg";
		ClassLoader cl = Thread.currentThread().getContextClassLoader();
		InputStream is = cl.getResourceAsStream(fileName);
		if ( is==null ) {
			cl = ErrorManager.class.getClassLoader();
			is = cl.getResourceAsStream(fileName);
		}
		if ( is==null && formatName.equals("antlr") ) {
			rawError("ANTLR installation corrupted; cannot find ANTLR messages format file "+fileName);
			panic();
		}
		else if ( is==null ) {
			rawError("no such message format file "+fileName+" retrying with default ANTLR format");
			setFormat("antlr"); // recurse on this rule, trying the default message format
			return;
		}
		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(is));
			format = new StringTemplateGroup(br,
											   AngleBracketTemplateLexer.class,
											   initSTListener);
		}
		finally {
			try {
				if ( br!=null ) {
					br.close();
				}
			}
			catch (IOException ioe) {
				rawError("cannot close message format file "+fileName, ioe);
			}
		}

		format.setErrorListener(blankSTListener);
		boolean formatOK = verifyFormat();
		if ( !formatOK && formatName.equals("antlr") ) {
			rawError("ANTLR installation corrupted; ANTLR messages format file "+formatName+".stg incomplete");
			panic();
		}
		else if ( !formatOK ) {
			setFormat("antlr"); // recurse on this rule, trying the default message format
		}
	}

	/** Encodes the error handling found in setLocale, but does not trigger
	 *  panics, which would make GUI tools die if ANTLR's installation was
	 *  a bit screwy.  Duplicated code...ick.
	public static Locale getLocaleForValidMessages(Locale locale) {
		ErrorManager.locale = locale;
		String language = locale.getLanguage();
		String fileName = "org/antlr/tool/templates/messages/"+language+".stg";
		ClassLoader cl = Thread.currentThread().getContextClassLoader();
		InputStream is = cl.getResourceAsStream(fileName);
		if ( is==null && language.equals(Locale.US.getLanguage()) ) {
			return null;
		}
		else if ( is==null ) {
			return getLocaleForValidMessages(Locale.US); // recurse on this rule, trying the US locale
		}

		boolean messagesOK = verifyMessages();
		if ( !messagesOK && language.equals(Locale.US.getLanguage()) ) {
			return null;
		}
		else if ( !messagesOK ) {
			return getLocaleForValidMessages(Locale.US); // try US to see if that will work
		}
		return true;
	}
	 */

	/** In general, you'll want all errors to go to a single spot.
	 *  However, in a GUI, you might have two frames up with two
	 *  different grammars.  Two threads might launch to process the
	 *  grammars--you would want errors to go to different objects
	 *  depending on the thread.  I store a single listener per
	 *  thread.
	 */
	public static void setErrorListener(ANTLRErrorListener listener) {
		threadToListenerMap.put(Thread.currentThread(), listener);
	}

    public static void removeErrorListener() {
        threadToListenerMap.remove(Thread.currentThread());
    }

	public static void setTool(Tool tool) {
		threadToToolMap.put(Thread.currentThread(), tool);
	}

	/** Given a message ID, return a StringTemplate that somebody can fill
	 *  with data.  We need to convert the int ID to the name of a template
	 *  in the messages ST group.
	 */
	public static StringTemplate getMessage(int msgID) {
        String msgName = idToMessageTemplateName[msgID];
		return messages.getInstanceOf(msgName);
	}
	public static String getMessageType(int msgID) {
		if (getErrorState().warningMsgIDs.member(msgID)) {
			return messages.getInstanceOf("warning").toString();
		}
		else if (getErrorState().errorMsgIDs.member(msgID)) {
			return messages.getInstanceOf("error").toString();
		}
		assertTrue(false, "Assertion failed! Message ID " + msgID + " created but is not present in errorMsgIDs or warningMsgIDs.");
		return "";
	}

	/** Return a StringTemplate that refers to the current format used for
	 * emitting messages.
	 */
	public static StringTemplate getLocationFormat() {
		return format.getInstanceOf("location");
	}
	public static StringTemplate getReportFormat() {
		return format.getInstanceOf("report");
	}
	public static StringTemplate getMessageFormat() {
		return format.getInstanceOf("message");
	}
	public static boolean formatWantsSingleLineMessage() {
		return format.getInstanceOf("wantsSingleLineMessage").toString().equals("true");
	}

	public static ANTLRErrorListener getErrorListener() {
		ANTLRErrorListener el =
			(ANTLRErrorListener)threadToListenerMap.get(Thread.currentThread());
		if ( el==null ) {
			return theDefaultErrorListener;
		}
		return el;
	}

	public static ErrorState getErrorState() {
		ErrorState ec =
			(ErrorState)threadToErrorStateMap.get(Thread.currentThread());
		if ( ec==null ) {
			ec = new ErrorState();
			threadToErrorStateMap.put(Thread.currentThread(), ec);
		}
		return ec;
	}

	public static int getNumErrors() {
		return getErrorState().errors;
	}

	public static void resetErrorState() {
        threadToListenerMap = new HashMap();        
        ErrorState ec = new ErrorState();
		threadToErrorStateMap.put(Thread.currentThread(), ec);
	}

	public static void info(String msg) {
		getErrorState().infos++;
		getErrorListener().info(msg);
	}

	public static void error(int msgID) {
		getErrorState().errors++;
		getErrorState().errorMsgIDs.add(msgID);
		getErrorListener().error(new ToolMessage(msgID));
	}

	public static void error(int msgID, Throwable e) {
		getErrorState().errors++;
		getErrorState().errorMsgIDs.add(msgID);
		getErrorListener().error(new ToolMessage(msgID,e));
	}

	public static void error(int msgID, Object arg) {
		getErrorState().errors++;
		getErrorState().errorMsgIDs.add(msgID);
		getErrorListener().error(new ToolMessage(msgID, arg));
	}

	public static void error(int msgID, Object arg, Object arg2) {
		getErrorState().errors++;
		getErrorState().errorMsgIDs.add(msgID);
		getErrorListener().error(new ToolMessage(msgID, arg, arg2));
	}

	public static void error(int msgID, Object arg, Throwable e) {
		getErrorState().errors++;
		getErrorState().errorMsgIDs.add(msgID);
		getErrorListener().error(new ToolMessage(msgID, arg, e));
	}

	public static void warning(int msgID, Object arg) {
		getErrorState().warnings++;
		getErrorState().warningMsgIDs.add(msgID);
		getErrorListener().warning(new ToolMessage(msgID, arg));
	}

	public static void nondeterminism(DecisionProbe probe,
									  DFAState d)
	{
		getErrorState().warnings++;
		Message msg = new GrammarNonDeterminismMessage(probe,d);
		getErrorState().warningMsgIDs.add(msg.msgID);
		getErrorListener().warning(msg);
	}

	public static void danglingState(DecisionProbe probe,
									 DFAState d)
	{
		getErrorState().errors++;
		Message msg = new GrammarDanglingStateMessage(probe,d);
		getErrorState().errorMsgIDs.add(msg.msgID);
		Set seen = (Set)emitSingleError.get("danglingState");
		if ( !seen.contains(d.dfa.decisionNumber+"|"+d.getAltSet()) ) {
			getErrorListener().error(msg);
			// we've seen this decision and this alt set; never again
			seen.add(d.dfa.decisionNumber+"|"+d.getAltSet());
		}
	}

	public static void analysisAborted(DecisionProbe probe)
	{
		getErrorState().warnings++;
		Message msg = new GrammarAnalysisAbortedMessage(probe);
		getErrorState().warningMsgIDs.add(msg.msgID);
		getErrorListener().warning(msg);
	}

	public static void unreachableAlts(DecisionProbe probe,
									   List alts)
	{
		getErrorState().errors++;
		Message msg = new GrammarUnreachableAltsMessage(probe,alts);
		getErrorState().errorMsgIDs.add(msg.msgID);
		getErrorListener().error(msg);
	}

	public static void insufficientPredicates(DecisionProbe probe,
											  DFAState d,
											  Map<Integer, Set<Token>> altToUncoveredLocations)
	{
		getErrorState().warnings++;
		Message msg = new GrammarInsufficientPredicatesMessage(probe,d,altToUncoveredLocations);
		getErrorState().warningMsgIDs.add(msg.msgID);
		getErrorListener().warning(msg);
	}

	public static void nonLLStarDecision(DecisionProbe probe) {
		getErrorState().errors++;
		Message msg = new NonRegularDecisionMessage(probe, probe.getNonDeterministicAlts());
		getErrorState().errorMsgIDs.add(msg.msgID);
		getErrorListener().error(msg);
	}

	public static void recursionOverflow(DecisionProbe probe,
										 DFAState sampleBadState,
										 int alt,
										 Collection targetRules,
										 Collection callSiteStates)
	{
		getErrorState().errors++;
		Message msg = new RecursionOverflowMessage(probe,sampleBadState, alt,
										 targetRules, callSiteStates);
		getErrorState().errorMsgIDs.add(msg.msgID);
		getErrorListener().error(msg);
	}

	/*
	// TODO: we can remove I think.  All detected now with cycles check.
	public static void leftRecursion(DecisionProbe probe,
									 int alt,
									 Collection targetRules,
									 Collection callSiteStates)
	{
		getErrorState().warnings++;
		Message msg = new LeftRecursionMessage(probe, alt, targetRules, callSiteStates);
		getErrorState().warningMsgIDs.add(msg.msgID);
		getErrorListener().warning(msg);
	}
	*/

	public static void leftRecursionCycles(Collection cycles) {
		getErrorState().errors++;
		Message msg = new LeftRecursionCyclesMessage(cycles);
		getErrorState().errorMsgIDs.add(msg.msgID);
		getErrorListener().warning(msg);
	}

	public static void grammarError(int msgID,
									Grammar g,
									Token token,
									Object arg,
									Object arg2)
	{
		getErrorState().errors++;
		Message msg = new GrammarSemanticsMessage(msgID,g,token,arg,arg2);
		getErrorState().errorMsgIDs.add(msgID);
		getErrorListener().error(msg);
	}

	public static void grammarError(int msgID,
									Grammar g,
									Token token,
									Object arg)
	{
		grammarError(msgID,g,token,arg,null);
	}

	public static void grammarError(int msgID,
									Grammar g,
									Token token)
	{
		grammarError(msgID,g,token,null,null);
	}

	public static void grammarWarning(int msgID,
									  Grammar g,
									  Token token,
									  Object arg,
									  Object arg2)
	{
		getErrorState().warnings++;
		Message msg = new GrammarSemanticsMessage(msgID,g,token,arg,arg2);
		getErrorState().warningMsgIDs.add(msgID);
		getErrorListener().warning(msg);
	}

	public static void grammarWarning(int msgID,
									  Grammar g,
									  Token token,
									  Object arg)
	{
		grammarWarning(msgID,g,token,arg,null);
	}

	public static void grammarWarning(int msgID,
									  Grammar g,
									  Token token)
	{
		grammarWarning(msgID,g,token,null,null);
	}

	public static void syntaxError(int msgID,
								   Grammar grammar,
								   Token token,
								   Object arg,
								   antlr.RecognitionException re)
	{
		getErrorState().errors++;
		getErrorState().errorMsgIDs.add(msgID);
		getErrorListener().error(
			new GrammarSyntaxMessage(msgID,grammar,token,arg,re)
		);
	}

	public static void internalError(Object error, Throwable e) {
		StackTraceElement location = getLastNonErrorManagerCodeLocation(e);
		String msg = "Exception "+e+"@"+location+": "+error;
		error(MSG_INTERNAL_ERROR, msg);
	}

	public static void internalError(Object error) {
		StackTraceElement location =
			getLastNonErrorManagerCodeLocation(new Exception());
		String msg = location+": "+error;
		error(MSG_INTERNAL_ERROR, msg);
	}

	public static boolean doNotAttemptAnalysis() {
		return !getErrorState().errorMsgIDs.and(ERRORS_FORCING_NO_ANALYSIS).isNil();
	}

	public static boolean doNotAttemptCodeGen() {
		return doNotAttemptAnalysis() ||
			   !getErrorState().errorMsgIDs.and(ERRORS_FORCING_NO_CODEGEN).isNil();
	}

	/** Return first non ErrorManager code location for generating messages */
	private static StackTraceElement getLastNonErrorManagerCodeLocation(Throwable e) {
		StackTraceElement[] stack = e.getStackTrace();
		int i = 0;
		for (; i < stack.length; i++) {
			StackTraceElement t = stack[i];
			if ( t.toString().indexOf("ErrorManager")<0 ) {
				break;
			}
		}
		StackTraceElement location = stack[i];
		return location;
	}

	// A S S E R T I O N  C O D E

	public static void assertTrue(boolean condition, String message) {
		if ( !condition ) {
			internalError(message);
		}
	}

	// S U P P O R T  C O D E

	protected static boolean initIdToMessageNameMapping() {
		// make sure a message exists, even if it's just to indicate a problem
		for (int i = 0; i < idToMessageTemplateName.length; i++) {
			idToMessageTemplateName[i] = "INVALID MESSAGE ID: "+i;
		}
		// get list of fields and use it to fill in idToMessageTemplateName mapping
		Field[] fields = ErrorManager.class.getFields();
		for (int i = 0; i < fields.length; i++) {
			Field f = fields[i];
			String fieldName = f.getName();
			if ( !fieldName.startsWith("MSG_") ) {
				continue;
			}
			String templateName =
				fieldName.substring("MSG_".length(),fieldName.length());
			int msgID = 0;
			try {
				// get the constant value from this class object
				msgID = f.getInt(ErrorManager.class);
			}
			catch (IllegalAccessException iae) {
				System.err.println("cannot get const value for "+f.getName());
				continue;
			}
			if ( fieldName.startsWith("MSG_") ) {
                idToMessageTemplateName[msgID] = templateName;
			}
		}
		return true;
	}

	/** Use reflection to find list of MSG_ fields and then verify a
	 *  template exists for each one from the locale's group.
	 */
	protected static boolean verifyMessages() {
		boolean ok = true;
		Field[] fields = ErrorManager.class.getFields();
		for (int i = 0; i < fields.length; i++) {
			Field f = fields[i];
			String fieldName = f.getName();
			String templateName =
				fieldName.substring("MSG_".length(),fieldName.length());
			if ( fieldName.startsWith("MSG_") ) {
				if ( !messages.isDefined(templateName) ) {
					System.err.println("Message "+templateName+" in locale "+
									   locale+" not found");
					ok = false;
				}
			}
		}
		// check for special templates
		if (!messages.isDefined("warning")) {
			System.err.println("Message template 'warning' not found in locale "+ locale);
			ok = false;
		}
		if (!messages.isDefined("error")) {
			System.err.println("Message template 'error' not found in locale "+ locale);
			ok = false;
		}
		return ok;
	}

	/** Verify the message format template group */
	protected static boolean verifyFormat() {
		boolean ok = true;
		if (!format.isDefined("location")) {
			System.err.println("Format template 'location' not found in " + formatName);
			ok = false;
		}
		if (!format.isDefined("message")) {
			System.err.println("Format template 'message' not found in " + formatName);
			ok = false;
		}
		if (!format.isDefined("report")) {
			System.err.println("Format template 'report' not found in " + formatName);
			ok = false;
		}
		return ok;
	}

	/** If there are errors during ErrorManager init, we have no choice
	 *  but to go to System.err.
	 */
	static void rawError(String msg) {
		System.err.println(msg);
	}

	static void rawError(String msg, Throwable e) {
		rawError(msg);
		e.printStackTrace(System.err);
	}

	/** I *think* this will allow Tool subclasses to exit gracefully
	 *  for GUIs etc...
	 */
	public static void panic() {
		Tool tool = (Tool)threadToToolMap.get(Thread.currentThread());
		if ( tool==null ) {
			// no tool registered, exit
			throw new Error("ANTLR ErrorManager panic");
		}
		else {
			tool.panic();
		}
	}
}
