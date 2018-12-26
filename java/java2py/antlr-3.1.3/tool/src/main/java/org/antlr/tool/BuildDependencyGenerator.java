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

import org.antlr.Tool;
import org.antlr.misc.Utils;
import org.antlr.codegen.CodeGenerator;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.StringTemplateGroup;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;

import java.util.List;
import java.util.ArrayList;
import java.io.*;

/** Given a grammar file, show the dependencies on .tokens etc...
 *  Using ST, emit a simple "make compatible" list of dependencies.
 *  For example, combined grammar T.g (no token import) generates:
 *
 *		TParser.java : T.g
 * 		T.tokens : T.g
 * 		T__g : T.g
 *
 *  For tree grammar TP with import of T.tokens:
 *
 * 		TP.g : T.tokens
 * 		TP.java : TP.g
 *
 *  If "-lib libdir" is used on command-line with -depend, then include the
 *  path like
 *
 * 		TP.g : libdir/T.tokens
 *
 *  Pay attention to -o as well:
 *
 * 		outputdir/TParser.java : T.g
 *
 *  So this output shows what the grammar depends on *and* what it generates.
 *
 *  Operate on one grammar file at a time.  If given a list of .g on the
 *  command-line with -depend, just emit the dependencies.  The grammars
 *  may depend on each other, but the order doesn't matter.  Build tools,
 *  reading in this output, will know how to organize it.
 *
 *  This is a wee bit slow probably because the code generator has to load
 *  all of its template files in order to figure out the file extension
 *  for the generated recognizer.
 *
 *  This code was obvious until I removed redundant "./" on front of files
 *  and had to escape spaces in filenames :(
 */
public class BuildDependencyGenerator {
    protected String grammarFileName;
    protected String tokenVocab;
    protected Tool tool;
    protected Grammar grammar;
    protected CodeGenerator generator;
    protected StringTemplateGroup templates;

    public BuildDependencyGenerator(Tool tool, String grammarFileName)
            throws IOException, antlr.TokenStreamException, antlr.RecognitionException {
        this.tool = tool;
        this.grammarFileName = grammarFileName;
        grammar = tool.getRootGrammar(grammarFileName);
        String language = (String) grammar.getOption("language");
        generator = new CodeGenerator(tool, grammar, language);
        generator.loadTemplates(language);
    }

    /** From T.g return a list of File objects that
     *  name files ANTLR will emit from T.g.
     */
    public List<File> getGeneratedFileList() {
        List<File> files = new ArrayList<File>();
        File outputDir = tool.getOutputDirectory(grammarFileName);
        if (outputDir.getName().equals(".")) {
            outputDir = null;
        } else if (outputDir.getName().indexOf(' ') >= 0) { // has spaces?
            String escSpaces = Utils.replace(outputDir.toString(),
                    " ",
                    "\\ ");
            outputDir = new File(escSpaces);
        }
        // add generated recognizer; e.g., TParser.java
        String recognizer =
                generator.getRecognizerFileName(grammar.name, grammar.type);
        files.add(new File(outputDir, recognizer));
        // add output vocab file; e.g., T.tokens. This is always generated to
        // the base output directory, which will be just . if there is no -o option
        //
        files.add(new File(tool.getOutputDirectory(), generator.getVocabFileName()));
        // are we generating a .h file?
        StringTemplate headerExtST = null;
        StringTemplate extST = generator.getTemplates().getInstanceOf("codeFileExtension");
        if (generator.getTemplates().isDefined("headerFile")) {
            headerExtST = generator.getTemplates().getInstanceOf("headerFileExtension");
            String suffix = Grammar.grammarTypeToFileNameSuffix[grammar.type];
            String fileName = grammar.name + suffix + headerExtST.toString();
            files.add(new File(outputDir, fileName));
        }
        if (grammar.type == Grammar.COMBINED) {
            // add autogenerated lexer; e.g., TLexer.java TLexer.h TLexer.tokens
            // don't add T__.g (just a temp file)
            
            String suffix = Grammar.grammarTypeToFileNameSuffix[Grammar.LEXER];
            String lexer = grammar.name + suffix + extST.toString();
            files.add(new File(outputDir, lexer));

            // TLexer.h
            if (headerExtST != null) {
                String header = grammar.name + suffix + headerExtST.toString();
                files.add(new File(outputDir, header));
            }
        // for combined, don't generate TLexer.tokens
        }

        // handle generated files for imported grammars
        List<Grammar> imports =
                grammar.composite.getDelegates(grammar.composite.getRootGrammar());
        for (Grammar g : imports) {
            outputDir = tool.getOutputDirectory(g.getFileName());
            String fname = groomQualifiedFileName(outputDir.toString(), g.getRecognizerName() + extST.toString());
            files.add(new File(fname));
        }

        if (files.size() == 0) {
            return null;
        }
        return files;
    }

    /**
     * Return a list of File objects that name files ANTLR will read
     * to process T.g; This can be .tokens files if the grammar uses the tokenVocab option
     * as well as any imported grammar files.
     */
    public List<File> getDependenciesFileList() {
        // Find all the things other than imported grammars
        List<File> files = getNonImportDependenciesFileList();

        // Handle imported grammars
        List<Grammar> imports =
                grammar.composite.getDelegates(grammar.composite.getRootGrammar());
        for (Grammar g : imports) {
            String libdir = tool.getLibraryDirectory();
            String fileName = groomQualifiedFileName(libdir, g.fileName);
            files.add(new File(fileName));
        }

        if (files.size() == 0) {
            return null;
        }
        return files;
    }

    /**
     * Return a list of File objects that name files ANTLR will read
     * to process T.g; This can only be .tokens files and only
     * if they use the tokenVocab option.
     *
     * @return List of dependencies other than imported grammars
     */
    public List<File> getNonImportDependenciesFileList() {
        List<File> files = new ArrayList<File>();

        // handle token vocabulary loads
        tokenVocab = (String) grammar.getOption("tokenVocab");
        if (tokenVocab != null) {

            File vocabFile = tool.getImportedVocabFile(tokenVocab);
            files.add(vocabFile);
        }

        return files;
    }

    public StringTemplate getDependencies() {
        loadDependencyTemplates();
        StringTemplate dependenciesST = templates.getInstanceOf("dependencies");
        dependenciesST.setAttribute("in", getDependenciesFileList());
        dependenciesST.setAttribute("out", getGeneratedFileList());
        dependenciesST.setAttribute("grammarFileName", grammar.fileName);
        return dependenciesST;
    }

    public void loadDependencyTemplates() {
        if (templates != null) {
            return;
        }
        String fileName = "org/antlr/tool/templates/depend.stg";
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        InputStream is = cl.getResourceAsStream(fileName);
        if (is == null) {
            cl = ErrorManager.class.getClassLoader();
            is = cl.getResourceAsStream(fileName);
        }
        if (is == null) {
            ErrorManager.internalError("Can't load dependency templates: " + fileName);
            return;
        }
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(is));
            templates = new StringTemplateGroup(br,
                    AngleBracketTemplateLexer.class);
            br.close();
        } catch (IOException ioe) {
            ErrorManager.internalError("error reading dependency templates file " + fileName, ioe);
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException ioe) {
                    ErrorManager.internalError("cannot close dependency templates file " + fileName, ioe);
                }
            }
        }
    }

    public String getTokenVocab() {
        return tokenVocab;
    }

    public CodeGenerator getGenerator() {
        return generator;
    }    

    public String groomQualifiedFileName(String outputDir, String fileName) {
        if (outputDir.equals(".")) {
            return fileName;
        } else if (outputDir.indexOf(' ') >= 0) { // has spaces?
            String escSpaces = Utils.replace(outputDir.toString(),
                    " ",
                    "\\ ");
            return escSpaces + File.separator + fileName;
        } else {
            return outputDir + File.separator + fileName;
        }
    }
}
