package org.antlr.runtime;

import java.io.IOException;
import java.io.FileInputStream;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.util.List;
import java.util.ArrayList;

public class SerializedGrammar {
    public static final String COOKIE = "$ANTLR";
    public static final int FORMAT_VERSION = 1;
    //public static org.antlr.tool.Grammar gr; // TESTING ONLY; remove later

    public String name;
    public char type; // in {l, p, t, c}
    public List rules;

    class Rule {
        String name;
        Block block;
        public Rule(String name, Block block) {
            this.name = name;
            this.block = block;
        }
        public String toString() {
            return name+":"+block;
        }
    }

    class Block {
        List[] alts;
        public Block(List[] alts) {
            this.alts = alts;
        }
        public String toString() {
            StringBuffer buf = new StringBuffer();
            buf.append("(");
            for (int i = 0; i < alts.length; i++) {
                List alt = alts[i];
                if ( i>0 ) buf.append("|");
                buf.append(alt.toString());
            }
            buf.append(")");
            return buf.toString();
        }
    }

    class TokenRef {
        int ttype;
        public TokenRef(int ttype) { this.ttype = ttype; }
        public String toString() { return String.valueOf(ttype); }
    }

    class RuleRef {
        int ruleIndex;
        public RuleRef(int ruleIndex) { this.ruleIndex = ruleIndex; }
        public String toString() { return String.valueOf(ruleIndex); }
    }

    public SerializedGrammar(String filename) throws IOException {
        System.out.println("loading "+filename);
        FileInputStream fis = new FileInputStream(filename);
        BufferedInputStream bos = new BufferedInputStream(fis);
        DataInputStream in = new DataInputStream(bos);
        readFile(in);
        in.close();
    }

    protected void readFile(DataInputStream in) throws IOException {
        String cookie = readString(in); // get $ANTLR
        if ( !cookie.equals(COOKIE) ) throw new IOException("not a serialized grammar file");
        int version = in.readByte();
        char grammarType = (char)in.readByte();
        this.type = grammarType;
        String grammarName = readString(in);
        this.name = grammarName;
        System.out.println(grammarType+" grammar "+grammarName);
        int numRules = in.readShort();
        System.out.println("num rules = "+numRules);
        rules = readRules(in, numRules);
    }

    protected List readRules(DataInputStream in, int numRules) throws IOException {
        List rules = new ArrayList();
        for (int i=0; i<numRules; i++) {
            Rule r = readRule(in);
            rules.add(r);
        }
        return rules;
    }

    protected Rule readRule(DataInputStream in) throws IOException {
        byte R = in.readByte();
        if ( R!='R' ) throw new IOException("missing R on start of rule");
        String name = readString(in);
        System.out.println("rule: "+name);
        byte B = in.readByte();
        Block b = readBlock(in);
        byte period = in.readByte();
        if ( period!='.' ) throw new IOException("missing . on end of rule");
        return new Rule(name, b);
    }

    protected Block readBlock(DataInputStream in) throws IOException {
        int nalts = in.readShort();
        List[] alts = new List[nalts];
        //System.out.println("enter block n="+nalts);
        for (int i=0; i<nalts; i++) {
            List alt = readAlt(in);
            alts[i] = alt;
        }
        //System.out.println("exit block");
        return new Block(alts);
    }

    protected List readAlt(DataInputStream in) throws IOException {
        List alt = new ArrayList();
        byte A = in.readByte();
        if ( A!='A' ) throw new IOException("missing A on start of alt");
        byte cmd = in.readByte();
        while ( cmd!=';' ) {
            switch (cmd) {
                case 't' :
                    int ttype = in.readShort();
                    alt.add(new TokenRef(ttype));
                    //System.out.println("read token "+gr.getTokenDisplayName(ttype));
                    break;
                case 'r' :
                    int ruleIndex = in.readShort();
                    alt.add(new RuleRef(ruleIndex));
                    //System.out.println("read rule "+gr.getRuleName(ruleIndex));
                    break;
                case '.' : // wildcard
                    break;
                case '-' : // range
                    int from = in.readChar();
                    int to = in.readChar();
                    break;
                case '~' : // not
                    int notThisTokenType = in.readShort();
                    break;
                case 'B' : // nested block
                    Block b = readBlock(in);
                    alt.add(b);
                    break;
            }
            cmd = in.readByte();
        }
        //System.out.println("exit alt");
        return alt;
    }

    protected String readString(DataInputStream in) throws IOException {
        byte c = in.readByte();
        StringBuffer buf = new StringBuffer();
        while ( c!=';' ) {
            buf.append((char)c);
            c = in.readByte();
        }
        return buf.toString();
    }

    public String toString() {
        StringBuffer buf = new StringBuffer();
        buf.append(type+" grammar "+name);
        buf.append(rules);
        return buf.toString();
    }
}
