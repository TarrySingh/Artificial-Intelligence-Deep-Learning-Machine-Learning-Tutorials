package org.antlr.gunit.swingui.runner;

import java.io.*;

public class ParserLoader extends ClassLoader {

    private String ParserClassFile;
    private String LexerClassFile;
    private String ParserClassName;
    private String LexerClassName;
    
    private byte[] parserBytes = null;
    private byte[] lexerBytes = null;

    private Class ParserClass = null;
    private Class LexerClass = null;

    public ParserLoader(String grammarName, String classDir) {
        ParserClassName = grammarName + "Parser";
        LexerClassName = grammarName + "Lexer";
        ParserClassFile = classDir + File.separator + ParserClassName + ".class";
        LexerClassFile = classDir + File.separator + LexerClassName + ".class";

        prepareClasses();
    }

    private void prepareClasses() {
        try {
            final InputStream inLexer = new FileInputStream(LexerClassFile);
            lexerBytes = new byte[inLexer.available()];
            inLexer.read(lexerBytes);
            inLexer.close();

            final InputStream inParser = new FileInputStream(ParserClassFile);
            parserBytes = new byte[inParser.available()];
            inParser.read(parserBytes);
            inParser.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new Error(e);
        }
    }

    @Override
    public synchronized Class loadClass(String name, boolean resolve) throws ClassNotFoundException {

        if(name.endsWith(LexerClassName)) {

            // load lexer class
            
            if(LexerClass != null) return LexerClass;

            LexerClass = defineClass(null, lexerBytes, 0, lexerBytes.length);
            resolveClass(LexerClass);
            return LexerClass;

        } else if(name.endsWith(ParserClassName)) {

            // load parser class

            if(ParserClass != null) return ParserClass;

            ParserClass = defineClass(null, parserBytes, 0, parserBytes.length);
            resolveClass(ParserClass);
            return ParserClass;

        } else {

            // load system class
            
            return findSystemClass(name);
            
        }
    }

    private String stripPackage(String className) {
        return className.substring(className.lastIndexOf("/"));
    }
}
