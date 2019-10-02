import org.antlr.runtime.*;

public class Main {
	public static void main(String[] args) throws Exception {
	    for (int i=0; i<args.length; i++) {
		CharStream input = new ANTLRFileStream(args[i]);
		FuzzyJava lex = new FuzzyJava(input);
		TokenStream tokens = new CommonTokenStream(lex);
		tokens.toString();
		//System.out.println(tokens);
	    }
	}
}
