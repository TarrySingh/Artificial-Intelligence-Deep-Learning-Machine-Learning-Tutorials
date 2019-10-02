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
package org.antlr;

import antlr.TokenStreamException;
import antlr.RecognitionException;
import antlr.ANTLRException;
import org.antlr.analysis.*;
import org.antlr.codegen.CodeGenerator;
import org.antlr.runtime.misc.Stats;
import org.antlr.tool.*;
import org.antlr.misc.Graph;

import java.io.*;
import java.util.*;

/** The main ANTLR entry point.  Read a grammar and generate a parser. */
public class Tool {

    public final Properties antlrSettings = new Properties();
    public String VERSION = "!Unknown version!";
    //public static final String VERSION = "${project.version}";
    public static final String UNINITIALIZED_DIR = "<unset-dir>";
    private List<String> grammarFileNames = new ArrayList<String>();
    private boolean generate_NFA_dot = false;
    private boolean generate_DFA_dot = false;
    private String outputDirectory = ".";
    private boolean haveOutputDir = false;
    private String inputDirectory = null;
    private String parentGrammarDirectory;
    private String grammarOutputDirectory;
    private boolean haveInputDir = false;
    private String libDirectory = ".";
    private boolean debug = false;
    private boolean trace = false;
    private boolean profile = false;
    private boolean report = false;
    private boolean printGrammar = false;
    private boolean depend = false;
    private boolean forceAllFilesToOutputDir = false;
    private boolean forceRelativeOutput = false;
    protected boolean deleteTempLexer = true;
    private boolean verbose = false;
    /** Don't process grammar file if generated files are newer than grammar */
    private boolean make = false;
    private boolean showBanner = true;
    private static boolean exitNow = false;

    // The internal options are for my use on the command line during dev
    //
    public static boolean internalOption_PrintGrammarTree = false;
    public static boolean internalOption_PrintDFA = false;
    public static boolean internalOption_ShowNFAConfigsInDFA = false;
    public static boolean internalOption_watchNFAConversion = false;

    /**
     * A list of dependency generators that are accumulated aaaas (and if) the
     * tool is required to sort the provided grammars into build dependency order.
    protected Map<String, BuildDependencyGenerator> buildDependencyGenerators;
     */

    public static void main(String[] args) {
        Tool antlr = new Tool(args);

        if (!exitNow) {
            antlr.process();
            if (ErrorManager.getNumErrors() > 0) {
                System.exit(1);
            }
            System.exit(0);
        }
    }

    /**
     * Load the properties file org/antlr/antlr.properties and populate any
     * variables that must be initialized from it, such as the version of ANTLR.
     */
    private void loadResources() {
        InputStream in = null;
        in = this.getClass().getResourceAsStream("antlr.properties");

        // If we found the resource, then load it, otherwise revert to the
        // defaults.
        //
        if (in != null) {
            try {
                // Load the resources into the map
                //
                antlrSettings.load(in);

                // Set any variables that we need to populate from the resources
                //
                VERSION = antlrSettings.getProperty("antlr.version");

            } catch (Exception e) {
                // Do nothing, just leave the defaults in place
            }
        }
    }

    public Tool() {
        loadResources();
    }

    public Tool(String[] args) {

        loadResources();

        // Set all the options and pick up all the named grammar files
        //
        processArgs(args);


    }

    public void processArgs(String[] args) {

        if (isVerbose()) {
            ErrorManager.info("ANTLR Parser Generator  Version " + VERSION);
            showBanner = false;
        }

        if (args == null || args.length == 0) {
            help();
            return;
        }
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-o") || args[i].equals("-fo")) {
                if (i + 1 >= args.length) {
                    System.err.println("missing output directory with -fo/-o option; ignoring");
                }
                else {
                    if (args[i].equals("-fo")) { // force output into dir
                        setForceAllFilesToOutputDir(true);
                    }
                    i++;
                    outputDirectory = args[i];
                    if (outputDirectory.endsWith("/") ||
                        outputDirectory.endsWith("\\")) {
                        outputDirectory =
                            outputDirectory.substring(0, getOutputDirectory().length() - 1);
                    }
                    File outDir = new File(outputDirectory);
                    haveOutputDir = true;
                    if (outDir.exists() && !outDir.isDirectory()) {
                        ErrorManager.error(ErrorManager.MSG_OUTPUT_DIR_IS_FILE, outputDirectory);
                        setLibDirectory(".");
                    }
                }
            }
            else if (args[i].equals("-lib")) {
                if (i + 1 >= args.length) {
                    System.err.println("missing library directory with -lib option; ignoring");
                }
                else {
                    i++;
                    setLibDirectory(args[i]);
                    if (getLibraryDirectory().endsWith("/") ||
                        getLibraryDirectory().endsWith("\\")) {
                        setLibDirectory(getLibraryDirectory().substring(0, getLibraryDirectory().length() - 1));
                    }
                    File outDir = new File(getLibraryDirectory());
                    if (!outDir.exists()) {
                        ErrorManager.error(ErrorManager.MSG_DIR_NOT_FOUND, getLibraryDirectory());
                        setLibDirectory(".");
                    }
                }
            }
            else if (args[i].equals("-nfa")) {
                setGenerate_NFA_dot(true);
            }
            else if (args[i].equals("-dfa")) {
                setGenerate_DFA_dot(true);
            }
            else if (args[i].equals("-debug")) {
                setDebug(true);
            }
            else if (args[i].equals("-trace")) {
                setTrace(true);
            }
            else if (args[i].equals("-report")) {
                setReport(true);
            }
            else if (args[i].equals("-profile")) {
                setProfile(true);
            }
            else if (args[i].equals("-print")) {
                setPrintGrammar(true);
            }
            else if (args[i].equals("-depend")) {
                setDepend(true);
            }
            else if (args[i].equals("-verbose")) {
                setVerbose(true);
            }
            else if (args[i].equals("-version")) {
                version();
                exitNow = true;
            }
            else if (args[i].equals("-make")) {
                setMake(true);
            }
            else if (args[i].equals("-message-format")) {
                if (i + 1 >= args.length) {
                    System.err.println("missing output format with -message-format option; using default");
                }
                else {
                    i++;
                    ErrorManager.setFormat(args[i]);
                }
            }
            else if (args[i].equals("-Xgrtree")) {
                internalOption_PrintGrammarTree = true; // print grammar tree
            }
            else if (args[i].equals("-Xdfa")) {
                internalOption_PrintDFA = true;
            }
            else if (args[i].equals("-Xnoprune")) {
                DFAOptimizer.PRUNE_EBNF_EXIT_BRANCHES = false;
            }
            else if (args[i].equals("-Xnocollapse")) {
                DFAOptimizer.COLLAPSE_ALL_PARALLEL_EDGES = false;
            }
            else if (args[i].equals("-Xdbgconversion")) {
                NFAToDFAConverter.debug = true;
            }
            else if (args[i].equals("-Xmultithreaded")) {
                NFAToDFAConverter.SINGLE_THREADED_NFA_CONVERSION = false;
            }
            else if (args[i].equals("-Xnomergestopstates")) {
                DFAOptimizer.MERGE_STOP_STATES = false;
            }
            else if (args[i].equals("-Xdfaverbose")) {
                internalOption_ShowNFAConfigsInDFA = true;
            }
            else if (args[i].equals("-Xwatchconversion")) {
                internalOption_watchNFAConversion = true;
            }
            else if (args[i].equals("-XdbgST")) {
                CodeGenerator.EMIT_TEMPLATE_DELIMITERS = true;
            }
            else if (args[i].equals("-Xmaxinlinedfastates")) {
                if (i + 1 >= args.length) {
                    System.err.println("missing max inline dfa states -Xmaxinlinedfastates option; ignoring");
                }
                else {
                    i++;
                    CodeGenerator.MAX_ACYCLIC_DFA_STATES_INLINE = Integer.parseInt(args[i]);
                }
            }
            else if (args[i].equals("-Xm")) {
                if (i + 1 >= args.length) {
                    System.err.println("missing max recursion with -Xm option; ignoring");
                }
                else {
                    i++;
                    NFAContext.MAX_SAME_RULE_INVOCATIONS_PER_NFA_CONFIG_STACK = Integer.parseInt(args[i]);
                }
            }
            else if (args[i].equals("-Xmaxdfaedges")) {
                if (i + 1 >= args.length) {
                    System.err.println("missing max number of edges with -Xmaxdfaedges option; ignoring");
                }
                else {
                    i++;
                    DFA.MAX_STATE_TRANSITIONS_FOR_TABLE = Integer.parseInt(args[i]);
                }
            }
            else if (args[i].equals("-Xconversiontimeout")) {
                if (i + 1 >= args.length) {
                    System.err.println("missing max time in ms -Xconversiontimeout option; ignoring");
                }
                else {
                    i++;
                    DFA.MAX_TIME_PER_DFA_CREATION = Integer.parseInt(args[i]);
                }
            }
            else if (args[i].equals("-Xnfastates")) {
                DecisionProbe.verbose = true;
            }
            else if (args[i].equals("-X")) {
                Xhelp();
            }
            else {
                if (args[i].charAt(0) != '-') {
                    // Must be the grammar file
                    addGrammarFile(args[i]);
                }
            }
        }
    }

    /*
    protected void checkForInvalidArguments(String[] args, BitSet cmdLineArgValid) {
    // check for invalid command line args
    for (int a = 0; a < args.length; a++) {
    if (!cmdLineArgValid.member(a)) {
    System.err.println("invalid command-line argument: " + args[a] + "; ignored");
    }
    }
    }
     */
    
    /**
     * Checks to see if the list of outputFiles all exist, and have
     * last-modified timestamps which are later than the last-modified
     * timestamp of all the grammar files involved in build the output
     * (imports must be checked). If these conditions hold, the method
     * returns false, otherwise, it returns true.
     *
     * @param grammarFileName The grammar file we are checking
     */
    public boolean buildRequired(String grammarFileName)
        throws IOException, ANTLRException
    {
        BuildDependencyGenerator bd =
            new BuildDependencyGenerator(this, grammarFileName);

        List<File> outputFiles = bd.getGeneratedFileList();
        List<File> inputFiles = bd.getDependenciesFileList();
        File grammarFile = new File(grammarFileName);
        long grammarLastModified = grammarFile.lastModified();
        for (File outputFile : outputFiles) {
            if (!outputFile.exists() || grammarLastModified > outputFile.lastModified()) {
                // One of the output files does not exist or is out of date, so we must build it
                return true;
            }

            // Check all of the imported grammars and see if any of these are younger
            // than any of the output files.
            if (inputFiles != null) {
                for (File inputFile : inputFiles) {
                    if (inputFile.lastModified() > outputFile.lastModified()) {
                        // One of the imported grammar files has been updated so we must build
                        return true;
                    }
                }
            }
        }
        if (isVerbose()) {
            System.out.println("Grammar " + grammarFileName + " is up to date - build skipped");
        }
        return false;
    }

    public void process() {
        boolean exceptionWhenWritingLexerFile = false;
        String lexerGrammarFileName = null;		// necessary at this scope to have access in the catch below

        // Have to be tricky here when Maven or build tools call in and must new Tool()
        // before setting options. The banner won't display that way!
        if (isVerbose() && showBanner) {
            ErrorManager.info("ANTLR Parser Generator  Version " + VERSION);
            showBanner = false;
        }

        try {
            sortGrammarFiles(); // update grammarFileNames
        }
        catch (Exception e) {
            ErrorManager.error(ErrorManager.MSG_INTERNAL_ERROR,e);
        }
        catch (Error e) {
            ErrorManager.error(ErrorManager.MSG_INTERNAL_ERROR, e);
        }

        for (String grammarFileName : grammarFileNames) {
            // If we are in make mode (to support build tools like Maven) and the
            // file is already up to date, then we do not build it (and in verbose mode
            // we will say so).
            if (make) {
                try {
                    if ( !buildRequired(grammarFileName) ) continue;
                }
                catch (Exception e) {
                    ErrorManager.error(ErrorManager.MSG_INTERNAL_ERROR,e);
                }
            }

            if (isVerbose() && !isDepend()) {
                System.out.println(grammarFileName);
            }
            try {
                if (isDepend()) {
                    BuildDependencyGenerator dep =
                        new BuildDependencyGenerator(this, grammarFileName);
                    /*
                    List outputFiles = dep.getGeneratedFileList();
                    List dependents = dep.getDependenciesFileList();
                    System.out.println("output: "+outputFiles);
                    System.out.println("dependents: "+dependents);
                     */
                    System.out.println(dep.getDependencies());
                    continue;
                }

                Grammar grammar = getRootGrammar(grammarFileName);
                // we now have all grammars read in as ASTs
                // (i.e., root and all delegates)
                grammar.composite.assignTokenTypes();
                grammar.composite.defineGrammarSymbols();
                grammar.composite.createNFAs();

                generateRecognizer(grammar);

                if (isPrintGrammar()) {
                    grammar.printGrammar(System.out);
                }

                if (isReport()) {
                    GrammarReport greport = new GrammarReport(grammar);
                    System.out.println(greport.toString());
                    // print out a backtracking report too (that is not encoded into log)
                    System.out.println(greport.getBacktrackingReport());
                    // same for aborted NFA->DFA conversions
                    System.out.println(greport.getAnalysisTimeoutReport());
                }
                if (isProfile()) {
                    GrammarReport greport = new GrammarReport(grammar);
                    Stats.writeReport(GrammarReport.GRAMMAR_STATS_FILENAME,
                                      greport.toNotifyString());
                }

                // now handle the lexer if one was created for a merged spec
                String lexerGrammarStr = grammar.getLexerGrammar();
                //System.out.println("lexer grammar:\n"+lexerGrammarStr);
                if (grammar.type == Grammar.COMBINED && lexerGrammarStr != null) {
                    lexerGrammarFileName = grammar.getImplicitlyGeneratedLexerFileName();
                    try {
                        Writer w = getOutputFile(grammar, lexerGrammarFileName);
                        w.write(lexerGrammarStr);
                        w.close();
                    }
                    catch (IOException e) {
                        // emit different error message when creating the implicit lexer fails
                        // due to write permission error
                        exceptionWhenWritingLexerFile = true;
                        throw e;
                    }
                    try {
                        StringReader sr = new StringReader(lexerGrammarStr);
                        Grammar lexerGrammar = new Grammar();
                        lexerGrammar.composite.watchNFAConversion = internalOption_watchNFAConversion;
                        lexerGrammar.implicitLexer = true;
                        lexerGrammar.setTool(this);
                        File lexerGrammarFullFile =
                            new File(getFileDirectory(lexerGrammarFileName), lexerGrammarFileName);
                        lexerGrammar.setFileName(lexerGrammarFullFile.toString());

                        lexerGrammar.importTokenVocabulary(grammar);
                        lexerGrammar.parseAndBuildAST(sr);

                        sr.close();

                        lexerGrammar.composite.assignTokenTypes();
                        lexerGrammar.composite.defineGrammarSymbols();
                        lexerGrammar.composite.createNFAs();

                        generateRecognizer(lexerGrammar);
                    }
                    finally {
                        // make sure we clean up
                        if (deleteTempLexer) {
                            File outputDir = getOutputDirectory(lexerGrammarFileName);
                            File outputFile = new File(outputDir, lexerGrammarFileName);
                            outputFile.delete();
                        }
                    }
                }
            }
            catch (IOException e) {
                if (exceptionWhenWritingLexerFile) {
                    ErrorManager.error(ErrorManager.MSG_CANNOT_WRITE_FILE,
                                       lexerGrammarFileName, e);
                }
                else {
                    ErrorManager.error(ErrorManager.MSG_CANNOT_OPEN_FILE,
                                       grammarFileName);
                }
            }
            catch (Exception e) {
                ErrorManager.error(ErrorManager.MSG_INTERNAL_ERROR, grammarFileName, e);
            }
            /*
           finally {
           System.out.println("creates="+ Interval.creates);
           System.out.println("hits="+ Interval.hits);
           System.out.println("misses="+ Interval.misses);
           System.out.println("outOfRange="+ Interval.outOfRange);
           }
            */
        }
    }

    public void sortGrammarFiles() throws IOException {
        //System.out.println("Grammar names "+getGrammarFileNames());
        Graph g = new Graph();
        for (String gfile : getGrammarFileNames()) {
            GrammarSpelunker grammar = new GrammarSpelunker(inputDirectory, gfile);
            grammar.parse();
            String vocabName = grammar.getTokenVocab();
            String grammarName = grammar.getGrammarName();
            // Make all grammars depend on any tokenVocab options
            if ( vocabName!=null ) g.addEdge(gfile, vocabName+CodeGenerator.VOCAB_FILE_EXTENSION);
            // Make all generated tokens files depend on their grammars
            g.addEdge(grammarName+CodeGenerator.VOCAB_FILE_EXTENSION, gfile);
        }
        List<Object> sorted = g.sort();
        //System.out.println("sorted="+sorted);
        grammarFileNames.clear(); // wipe so we can give new ordered list
        for (int i = 0; i < sorted.size(); i++) {
            String f = (String)sorted.get(i);
            if ( f.endsWith(".g") ) grammarFileNames.add(f);
        }
        //System.out.println("new grammars="+grammarFileNames);
    }

    /** Get a grammar mentioned on the command-line and any delegates */
    public Grammar getRootGrammar(String grammarFileName)
        throws IOException
    {
        //StringTemplate.setLintMode(true);
        // grammars mentioned on command line are either roots or single grammars.
        // create the necessary composite in case it's got delegates; even
        // single grammar needs it to get token types.
        CompositeGrammar composite = new CompositeGrammar();
        Grammar grammar = new Grammar(this, grammarFileName, composite);
        composite.setDelegationRoot(grammar);
        FileReader fr = null;
        File f = null;

        if (haveInputDir) {
            f = new File(inputDirectory, grammarFileName);
        }
        else {
            f = new File(grammarFileName);
        }

        // Store the location of this grammar as if we import files, we can then
        // search for imports in the same location as the original grammar as well as in
        // the lib directory.
        //
        parentGrammarDirectory = f.getParent();

        if (grammarFileName.lastIndexOf(File.separatorChar) == -1) {
            grammarOutputDirectory = ".";
        }
        else {
            grammarOutputDirectory = grammarFileName.substring(0, grammarFileName.lastIndexOf(File.separatorChar));
        }
        fr = new FileReader(f);
        BufferedReader br = new BufferedReader(fr);
        grammar.parseAndBuildAST(br);
        composite.watchNFAConversion = internalOption_watchNFAConversion;
        br.close();
        fr.close();
        return grammar;
    }

    /** Create NFA, DFA and generate code for grammar.
     *  Create NFA for any delegates first.  Once all NFA are created,
     *  it's ok to create DFA, which must check for left-recursion.  That check
     *  is done by walking the full NFA, which therefore must be complete.
     *  After all NFA, comes DFA conversion for root grammar then code gen for
     *  root grammar.  DFA and code gen for delegates comes next.
     */
    protected void generateRecognizer(Grammar grammar) {
        String language = (String) grammar.getOption("language");
        if (language != null) {
            CodeGenerator generator = new CodeGenerator(this, grammar, language);
            grammar.setCodeGenerator(generator);
            generator.setDebug(isDebug());
            generator.setProfile(isProfile());
            generator.setTrace(isTrace());

            // generate NFA early in case of crash later (for debugging)
            if (isGenerate_NFA_dot()) {
                generateNFAs(grammar);
            }

            // GENERATE CODE
            generator.genRecognizer();

            if (isGenerate_DFA_dot()) {
                generateDFAs(grammar);
            }

            List<Grammar> delegates = grammar.getDirectDelegates();
            for (int i = 0; delegates != null && i < delegates.size(); i++) {
                Grammar delegate = (Grammar) delegates.get(i);
                if (delegate != grammar) { // already processing this one
                    generateRecognizer(delegate);
                }
            }
        }
    }

    public void generateDFAs(Grammar g) {
        for (int d = 1; d <= g.getNumberOfDecisions(); d++) {
            DFA dfa = g.getLookaheadDFA(d);
            if (dfa == null) {
                continue; // not there for some reason, ignore
            }
            DOTGenerator dotGenerator = new DOTGenerator(g);
            String dot = dotGenerator.getDOT(dfa.startState);
            String dotFileName = g.name + "." + "dec-" + d;
            if (g.implicitLexer) {
                dotFileName = g.name + Grammar.grammarTypeToFileNameSuffix[g.type] + "." + "dec-" + d;
            }
            try {
                writeDOTFile(g, dotFileName, dot);
            } catch (IOException ioe) {
                ErrorManager.error(ErrorManager.MSG_CANNOT_GEN_DOT_FILE,
                                   dotFileName,
                                   ioe);
            }
        }
    }

    protected void generateNFAs(Grammar g) {
        DOTGenerator dotGenerator = new DOTGenerator(g);
        Collection rules = g.getAllImportedRules();
        rules.addAll(g.getRules());

        for (Iterator itr = rules.iterator(); itr.hasNext();) {
            Rule r = (Rule) itr.next();
            try {
                String dot = dotGenerator.getDOT(r.startState);
                if (dot != null) {
                    writeDOTFile(g, r, dot);
                }
            } catch (IOException ioe) {
                ErrorManager.error(ErrorManager.MSG_CANNOT_WRITE_FILE, ioe);
            }
        }
    }

    protected void writeDOTFile(Grammar g, Rule r, String dot) throws IOException {
        writeDOTFile(g, r.grammar.name + "." + r.name, dot);
    }

    protected void writeDOTFile(Grammar g, String name, String dot) throws IOException {
        Writer fw = getOutputFile(g, name + ".dot");
        fw.write(dot);
        fw.close();
    }

    private static void version() {
        ErrorManager.info("ANTLR Parser Generator  Version " + new Tool().VERSION);
    }

    private static void help() {
        ErrorManager.info("ANTLR Parser Generator  Version " + new Tool().VERSION);
        System.err.println("usage: java org.antlr.Tool [args] file.g [file2.g file3.g ...]");
        System.err.println("  -o outputDir          specify output directory where all output is generated");
        System.err.println("  -fo outputDir         same as -o but force even files with relative paths to dir");
        System.err.println("  -lib dir              specify location of token files");
        System.err.println("  -depend               generate file dependencies");
        System.err.println("  -report               print out a report about the grammar(s) processed");
        System.err.println("  -print                print out the grammar without actions");
        System.err.println("  -debug                generate a parser that emits debugging events");
        System.err.println("  -profile              generate a parser that computes profiling information");
        System.err.println("  -nfa                  generate an NFA for each rule");
        System.err.println("  -dfa                  generate a DFA for each decision point");
        System.err.println("  -message-format name  specify output style for messages");
        System.err.println("  -verbose              generate ANTLR version and other information");
        System.err.println("  -make                 only build if generated files older than grammar");
        System.err.println("  -version              print the version of ANTLR and exit.");
        System.err.println("  -X                    display extended argument list");
    }

    private static void Xhelp() {
        ErrorManager.info("ANTLR Parser Generator  Version " + new Tool().VERSION);
        System.err.println("  -Xgrtree               print the grammar AST");
        System.err.println("  -Xdfa                  print DFA as text ");
        System.err.println("  -Xnoprune              test lookahead against EBNF block exit branches");
        System.err.println("  -Xnocollapse           collapse incident edges into DFA states");
        System.err.println("  -Xdbgconversion        dump lots of info during NFA conversion");
        System.err.println("  -Xmultithreaded        run the analysis in 2 threads");
        System.err.println("  -Xnomergestopstates    do not merge stop states");
        System.err.println("  -Xdfaverbose           generate DFA states in DOT with NFA configs");
        System.err.println("  -Xwatchconversion      print a message for each NFA before converting");
        System.err.println("  -XdbgST                put tags at start/stop of all templates in output");
        System.err.println("  -Xm m                  max number of rule invocations during conversion");
        System.err.println("  -Xmaxdfaedges m        max \"comfortable\" number of edges for single DFA state");
        System.err.println("  -Xconversiontimeout t  set NFA conversion timeout for each decision");
        System.err.println("  -Xmaxinlinedfastates m max DFA states before table used rather than inlining");
        System.err.println("  -Xnfastates            for nondeterminisms, list NFA states for each path");
    }

    /**
     * Set the location (base directory) where output files should be produced
     * by the ANTLR tool.
     * @param outputDirectory
     */
    public void setOutputDirectory(String outputDirectory) {
        haveOutputDir = true;
        this.outputDirectory = outputDirectory;
    }

    /**
     * Used by build tools to force the output files to always be
     * relative to the base output directory, even though the tool
     * had to set the output directory to an absolute path as it
     * cannot rely on the workign directory like command line invocation
     * can.
     *
     * @param forceRelativeOutput true if output files hould always be relative to base output directory
     */
    public void setForceRelativeOutput(boolean forceRelativeOutput) {
        this.forceRelativeOutput = forceRelativeOutput;
    }

    /**
     * Set the base location of input files. Normally (when the tool is
     * invoked from the command line), the inputDirectory is not set, but
     * for build tools such as Maven, we need to be able to locate the input
     * files relative to the base, as the working directory could be anywhere and
     * changing workig directories is not a valid concept for JVMs because of threading and
     * so on. Setting the directory just means that the getFileDirectory() method will
     * try to open files relative to this input directory.
     *
     * @param inputDirectory Input source base directory
     */
    public void setInputDirectory(String inputDirectory) {
        this.inputDirectory = inputDirectory;
        haveInputDir = true;
    }

    /** This method is used by all code generators to create new output
     *  files. If the outputDir set by -o is not present it will be created.
     *  The final filename is sensitive to the output directory and
     *  the directory where the grammar file was found.  If -o is /tmp
     *  and the original grammar file was foo/t.g then output files
     *  go in /tmp/foo.
     *
     *  The output dir -o spec takes precedence if it's absolute.
     *  E.g., if the grammar file dir is absolute the output dir is given
     *  precendence. "-o /tmp /usr/lib/t.g" results in "/tmp/T.java" as
     *  output (assuming t.g holds T.java).
     *
     *  If no -o is specified, then just write to the directory where the
     *  grammar file was found.
     *
     *  If outputDirectory==null then write a String.
     */
    public Writer getOutputFile(Grammar g, String fileName) throws IOException {
        if (getOutputDirectory() == null) {
            return new StringWriter();
        }
        // output directory is a function of where the grammar file lives
        // for subdir/T.g, you get subdir here.  Well, depends on -o etc...
        // But, if this is a .tokens file, then we force the output to
        // be the base output directory (or current directory if there is not a -o)
        //
        File outputDir;
        if (fileName.endsWith(CodeGenerator.VOCAB_FILE_EXTENSION)) {
            if (haveOutputDir) {
                outputDir = new File(getOutputDirectory());
            }
            else {
                outputDir = new File(".");
            }
        }
        else {
            outputDir = getOutputDirectory(g.getFileName());
        }
        File outputFile = new File(outputDir, fileName);

        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        FileWriter fw = new FileWriter(outputFile);
        return new BufferedWriter(fw);
    }

    /**
     * Return the location where ANTLR will generate output files for a given file. This is a
     * base directory and output files will be relative to here in some cases
     * such as when -o option is used and input files are given relative
     * to the input directory.
     *
     * @param fileNameWithPath path to input source
     * @return
     */
    public File getOutputDirectory(String fileNameWithPath) {

        File outputDir = new File(getOutputDirectory());
        String fileDirectory;

        // Some files are given to us without a PATH but should should
        // still be written to the output directory in the relative path of
        // the output directory. The file directory is either the set of sub directories
        // or just or the relative path recorded for the parent grammar. This means
        // that when we write the tokens files, or the .java files for imported grammars
        // taht we will write them in the correct place.
        //
        if (fileNameWithPath.lastIndexOf(File.separatorChar) == -1) {

            // No path is included in the file name, so make the file
            // directory the same as the parent grammar (which might sitll be just ""
            // but when it is not, we will write the file in the correct place.
            //
            fileDirectory = grammarOutputDirectory;

        }
        else {
            fileDirectory = fileNameWithPath.substring(0, fileNameWithPath.lastIndexOf(File.separatorChar));
        }
        if (haveOutputDir) {
            // -o /tmp /var/lib/t.g => /tmp/T.java
            // -o subdir/output /usr/lib/t.g => subdir/output/T.java
            // -o . /usr/lib/t.g => ./T.java
            if ((fileDirectory != null && !forceRelativeOutput) &&
                (new File(fileDirectory).isAbsolute() ||
                 fileDirectory.startsWith("~")) || // isAbsolute doesn't count this :(
                isForceAllFilesToOutputDir()) {
                // somebody set the dir, it takes precendence; write new file there
                outputDir = new File(getOutputDirectory());
            }
            else {
                // -o /tmp subdir/t.g => /tmp/subdir/t.g
                if (fileDirectory != null) {
                    outputDir = new File(getOutputDirectory(), fileDirectory);
                }
                else {
                    outputDir = new File(getOutputDirectory());
                }
            }
        }
        else {
            // they didn't specify a -o dir so just write to location
            // where grammar is, absolute or relative, this will only happen
            // with command line invocation as build tools will always
            // supply an output directory.
            //
            outputDir = new File(fileDirectory);
        }
        return outputDir;
    }

    /**
     * Name a file from the -lib dir.  Imported grammars and .tokens files
     *
     * If we do not locate the file in the library directory, then we try
     * the location of the originating grammar.
     *
     * @param fileName input name we are looking for
     * @return Path to file that we think shuold be the import file
     *
     * @throws java.io.IOException
     */
    public String getLibraryFile(String fileName) throws IOException {

        // First, see if we can find the file in the library directory
        //
        File f = new File(getLibraryDirectory() + File.separator + fileName);

        if (f.exists()) {

            // Found in the library directory
            //
            return f.getAbsolutePath();
        }

        // Need to assume it is in the same location as the input file. Note that
        // this is only relevant for external build tools and when the input grammar
        // was specified relative to the source directory (working directory if using
        // the command line.
        //
        return parentGrammarDirectory + File.separator + fileName;
    }

    /** Return the directory containing the grammar file for this grammar.
     *  normally this is a relative path from current directory.  People will
     *  often do "java org.antlr.Tool grammars/*.g3"  So the file will be
     *  "grammars/foo.g3" etc...  This method returns "grammars".
     *
     *  If we have been given a specific input directory as a base, then
     *  we must find the directory relative to this directory, unless the
     *  file name is given to us in absolute terms.
     */
    public String getFileDirectory(String fileName) {

        File f;
        if (haveInputDir && !fileName.startsWith(File.separator)) {
            f = new File(inputDirectory, fileName);
        }
        else {
            f = new File(fileName);
        }
        // And ask Java what the base directory of this location is
        //
        return f.getParent();
    }

    /** Return a File descriptor for vocab file.  Look in library or
     *  in -o output path.  antlr -o foo T.g U.g where U needs T.tokens
     *  won't work unless we look in foo too. If we do not find the
     *  file in the lib directory then must assume that the .tokens file
     *  is going to be generated as part of this build and we have defined
     *  .tokens files so that they ALWAYS are generated in the base output
     *  directory, which means the current directory for the command line tool if there
     *  was no output directory specified.
     */
    public File getImportedVocabFile(String vocabName) {

        File f = new File(getLibraryDirectory(),
                          File.separator +
                          vocabName +
                          CodeGenerator.VOCAB_FILE_EXTENSION);
        if (f.exists()) {
            return f;
        }

        // We did not find the vocab file in the lib directory, so we need
        // to look for it in the output directory which is where .tokens
        // files are generated (in the base, not relative to the input
        // location.)
        //
        if (haveOutputDir) {
            f = new File(getOutputDirectory(), vocabName + CodeGenerator.VOCAB_FILE_EXTENSION);
        }
        else {
            f = new File(vocabName + CodeGenerator.VOCAB_FILE_EXTENSION);
        }
        return f;
    }

    /** If the tool needs to panic/exit, how do we do that?
     */
    public void panic() {
        throw new Error("ANTLR panic");
    }

    /** Return a time stamp string accurate to sec: yyyy-mm-dd hh:mm:ss
     */
    public static String getCurrentTimeStamp() {
        GregorianCalendar calendar = new java.util.GregorianCalendar();
        int y = calendar.get(Calendar.YEAR);
        int m = calendar.get(Calendar.MONTH) + 1; // zero-based for months
        int d = calendar.get(Calendar.DAY_OF_MONTH);
        int h = calendar.get(Calendar.HOUR_OF_DAY);
        int min = calendar.get(Calendar.MINUTE);
        int sec = calendar.get(Calendar.SECOND);
        String sy = String.valueOf(y);
        String sm = m < 10 ? "0" + m : String.valueOf(m);
        String sd = d < 10 ? "0" + d : String.valueOf(d);
        String sh = h < 10 ? "0" + h : String.valueOf(h);
        String smin = min < 10 ? "0" + min : String.valueOf(min);
        String ssec = sec < 10 ? "0" + sec : String.valueOf(sec);
        return new StringBuffer().append(sy).append("-").append(sm).append("-").append(sd).append(" ").append(sh).append(":").append(smin).append(":").append(ssec).toString();
    }

    /**
     * Provide the List of all grammar file names that the ANTLR tool will
     * process or has processed.
     *
     * @return the grammarFileNames
     */
    public List<String> getGrammarFileNames() {
        return grammarFileNames;
    }

    /**
     * Indicates whether ANTLR has gnerated or will generate a description of
     * all the NFAs in <a href="http://www.graphviz.org">Dot format</a>
     *
     * @return the generate_NFA_dot
     */
    public boolean isGenerate_NFA_dot() {
        return generate_NFA_dot;
    }

    /**
     * Indicates whether ANTLR has generated or will generate a description of
     * all the NFAs in <a href="http://www.graphviz.org">Dot format</a>
     *
     * @return the generate_DFA_dot
     */
    public boolean isGenerate_DFA_dot() {
        return generate_DFA_dot;
    }

    /**
     * Return the Path to the base output directory, where ANTLR
     * will generate all the output files for the current language target as
     * well as any ancillary files such as .tokens vocab files.
     *
     * @return the output Directory
     */
    public String getOutputDirectory() {
        return outputDirectory;
    }

    /**
     * Return the Path to the directory in which ANTLR will search for ancillary
     * files such as .tokens vocab files and imported grammar files.
     *
     * @return the lib Directory
     */
    public String getLibraryDirectory() {
        return libDirectory;
    }

    /**
     * Indicate if ANTLR has generated, or will generate a debug version of the
     * recognizer. Debug versions of a parser communicate with a debugger such
     * as that contained in ANTLRWorks and at start up will 'hang' waiting for
     * a connection on an IP port (49100 by default).
     *
     * @return the debug flag
     */
    public boolean isDebug() {
        return debug;
    }

    /**
     * Indicate whether ANTLR has generated, or will generate a version of the
     * recognizer that prints trace messages on entry and exit of each rule.
     *
     * @return the trace flag
     */
    public boolean isTrace() {
        return trace;
    }

    /**
     * Indicates whether ANTLR has generated or will generate a version of the
     * recognizer that gathers statistics about its execution, which it prints when
     * it terminates.
     *
     * @return the profile
     */
    public boolean isProfile() {
        return profile;
    }

    /**
     * Indicates whether ANTLR has generated or will generate a report of various
     * elements of the grammar analysis, once it it has finished analyzing a grammar
     * file.
     *
     * @return the report flag
     */
    public boolean isReport() {
        return report;
    }

    /**
     * Indicates whether ANTLR has printed, or will print, a version of the input grammar
     * file(s) that is stripped of any action code embedded within.
     *
     * @return the printGrammar flag
     */
    public boolean isPrintGrammar() {
        return printGrammar;
    }

    /**
     * Indicates whether ANTLR has supplied, or will supply, a list of all the things
     * that the input grammar depends upon and all the things that will be generated
     * when that grammar is successfully analyzed.
     *
     * @return the depend flag
     */
    public boolean isDepend() {
        return depend;
    }

    /**
     * Indicates whether ANTLR will force all files to the output directory, even
     * if the input files have relative paths from the input directory.
     *
     * @return the forceAllFilesToOutputDir flag
     */
    public boolean isForceAllFilesToOutputDir() {
        return forceAllFilesToOutputDir;
    }

    /**
     * Indicates whether ANTLR will be verbose when analyzing grammar files, such as
     * displaying the names of the files it is generating and similar information.
     *
     * @return the verbose flag
     */
    public boolean isVerbose() {
        return verbose;
    }

    /**
     * Provide the current setting of the conversion timeout on DFA creation.
     *
     * @return DFA creation timeout value in milliseconds
     */
    public int getConversionTimeout() {
        return DFA.MAX_TIME_PER_DFA_CREATION;
    }

    /**
     * Returns the current setting of the message format descriptor
     * @return Current message format
     */
    public String getMessageFormat() {
        return ErrorManager.getMessageFormat().toString();
    }

    /**
     * Returns the number of errors that the analysis/processing threw up.
     * @return Error count
     */
    public int getNumErrors() {
        return ErrorManager.getNumErrors();
    }

    /**
     * Indicate whether the tool will analyze the dependencies of the provided grammar
     * file list and ensure that grammars with dependencies are built
     * after any of the other gramamrs in the list that they are dependent on. Setting
     * this option also has the side effect that any grammars that are includes for other
     * grammars in the list are excluded from individual analysis, which allows the caller
     * to invoke the tool via org.antlr.tool -make *.g and not worry about the inclusion
     * of grammars that are just includes for other grammars or what order the grammars
     * appear on the command line.
     *
     * This option was coded to make life easier for tool integration (such as Maven) but
     * may also be useful at the command line.
     *
     * @return true if the tool is currently configured to analyze and sort grammar files.
     */
    public boolean getMake() {
        return make;
    }

    /**
     * Set the message format to one of ANTLR, gnu, vs2005
     *
     * @param format
     */
    public void setMessageFormat(String format) {
        ErrorManager.setFormat(format);
    }

    /**
     * Set the timeout value (in milliseconds) after which DFA creation stops
     *
     * @param timeout value in milliseconds
     */
    public void setConversionTimeout(int timeout) {
        DFA.MAX_TIME_PER_DFA_CREATION = timeout;
    }

    /** Provide the List of all grammar file names that the ANTLR tool should process.
     *
     * @param grammarFileNames The list of grammar files to process
     */
    public void setGrammarFileNames(List<String> grammarFileNames) {
        this.grammarFileNames = grammarFileNames;
    }

    public void addGrammarFile(String grammarFileName) {
        if (!grammarFileNames.contains(grammarFileName)) {
            grammarFileNames.add(grammarFileName);
        }
    }

    /**
     * Indicate whether ANTLR should generate a description of
     * all the NFAs in <a href="http://www.graphviz.org">Dot format</a>
     *
     * @param generate_NFA_dot True to generate dot descriptions
     */
    public void setGenerate_NFA_dot(boolean generate_NFA_dot) {
        this.generate_NFA_dot = generate_NFA_dot;
    }

    /**
     * Indicates whether ANTLR should generate a description of
     * all the NFAs in <a href="http://www.graphviz.org">Dot format</a>
     *
     * @param generate_DFA_dot True to generate dot descriptions
     */
    public void setGenerate_DFA_dot(boolean generate_DFA_dot) {
        this.generate_DFA_dot = generate_DFA_dot;
    }

    /**
     * Set the Path to the directory in which ANTLR will search for ancillary
     * files such as .tokens vocab files and imported grammar files.
     *
     * @param libDirectory the libDirectory to set
     */
    public void setLibDirectory(String libDirectory) {
        this.libDirectory = libDirectory;
    }

    /**
     * Indicate whether ANTLR should generate a debug version of the
     * recognizer. Debug versions of a parser communicate with a debugger such
     * as that contained in ANTLRWorks and at start up will 'hang' waiting for
     * a connection on an IP port (49100 by default).
     *
     * @param debug true to generate a debug mode parser
     */
    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    /**
     * Indicate whether ANTLR should generate a version of the
     * recognizer that prints trace messages on entry and exit of each rule
     *
     * @param trace true to generate a tracing parser
     */
    public void setTrace(boolean trace) {
        this.trace = trace;
    }

    /**
     * Indicate whether ANTLR should generate a version of the
     * recognizer that gathers statistics about its execution, which it prints when
     * it terminates.
     *
     * @param profile true to generate a profiling parser
     */
    public void setProfile(boolean profile) {
        this.profile = profile;
    }

    /**
     * Indicate whether ANTLR should generate a report of various
     * elements of the grammar analysis, once it it has finished analyzing a grammar
     * file.
     *
     * @param report true to generate the analysis report
     */
    public void setReport(boolean report) {
        this.report = report;
    }

    /**
     * Indicate whether ANTLR should print a version of the input grammar
     * file(s) that is stripped of any action code embedded within.
     *
     * @param printGrammar true to generate a stripped file
     */
    public void setPrintGrammar(boolean printGrammar) {
        this.printGrammar = printGrammar;
    }

    /**
     * Indicate whether ANTLR should supply a list of all the things
     * that the input grammar depends upon and all the things that will be generated
     * when that gramamr is successfully analyzed.
     *
     * @param depend true to get depends set rather than process the grammar
     */
    public void setDepend(boolean depend) {
        this.depend = depend;
    }

    /**
     * Indicates whether ANTLR will force all files to the output directory, even
     * if the input files have relative paths from the input directory.
     *
     * @param forceAllFilesToOutputDir true to force files to output directory
     */
    public void setForceAllFilesToOutputDir(boolean forceAllFilesToOutputDir) {
        this.forceAllFilesToOutputDir = forceAllFilesToOutputDir;
    }

    /**
     * Indicate whether ANTLR should be verbose when analyzing grammar files, such as
     * displaying the names of the files it is generating and similar information.
     *
     * @param verbose true to be verbose
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * Indicate whether the tool should analyze the dependencies of the provided grammar
     * file list and ensure that the grammars with dependencies are built
     * after any of the other gramamrs in the list that they are dependent on. Setting
     * this option also has the side effect that any grammars that are includes for other
     * grammars in the list are excluded from individual analysis, which allows the caller
     * to invoke the tool via org.antlr.tool -make *.g and not worry about the inclusion
     * of grammars that are just includes for other grammars or what order the grammars
     * appear on the command line.
     *
     * This option was coded to make life easier for tool integration (such as Maven) but
     * may also be useful at the command line.
     *
     * @param make
     */
    public void setMake(boolean make) {
        this.make = make;
    }

}
