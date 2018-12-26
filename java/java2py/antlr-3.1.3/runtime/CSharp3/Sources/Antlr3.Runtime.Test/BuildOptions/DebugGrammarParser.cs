// $ANTLR 3.1.2 BuildOptions\\DebugGrammar.g3 2009-03-16 13:19:16

// The variable 'variable' is assigned but its value is never used.
#pragma warning disable 219
// Unreachable code detected.
#pragma warning disable 162


using System.Collections.Generic;
using Antlr.Runtime;
using Stack = System.Collections.Generic.Stack<object>;
using List = System.Collections.IList;
using ArrayList = System.Collections.Generic.List<object>;

using Antlr.Runtime.Debug;
using IOException = System.IO.IOException;

using Antlr.Runtime.Tree;
using RewriteRuleITokenStream = Antlr.Runtime.Tree.RewriteRuleTokenStream;

public partial class DebugGrammarParser : DebugParser
{
	public static readonly string[] tokenNames = new string[] {
		"<invalid>", "<EOR>", "<DOWN>", "<UP>", "CALL", "FUNC", "ID", "INT", "NEWLINE", "WS", "'-'", "'%'", "'('", "')'", "'*'", "'/'", "'+'", "'='"
	};
	public const int EOF=-1;
	public const int T__10=10;
	public const int T__11=11;
	public const int T__12=12;
	public const int T__13=13;
	public const int T__14=14;
	public const int T__15=15;
	public const int T__16=16;
	public const int T__17=17;
	public const int CALL=4;
	public const int FUNC=5;
	public const int ID=6;
	public const int INT=7;
	public const int NEWLINE=8;
	public const int WS=9;

	// delegates
	// delegators

	public static readonly string[] ruleNames =
		new string[]
		{
			"invalidRule", "atom", "expr", "formalPar", "func", "multExpr", "prog", 
		"stat"
		};

		int ruleLevel = 0;
		public virtual int RuleLevel { get { return ruleLevel; } }
		public virtual void IncRuleLevel() { ruleLevel++; }
		public virtual void DecRuleLevel() { ruleLevel--; }
		public DebugGrammarParser( ITokenStream input )
			: this( input, DebugEventSocketProxy.DEFAULT_DEBUGGER_PORT, new RecognizerSharedState() )
		{
		}
		public DebugGrammarParser( ITokenStream input, int port, RecognizerSharedState state )
			: base( input, state )
		{
			DebugEventSocketProxy proxy = new DebugEventSocketProxy( this, port, adaptor );
			DebugListener = proxy;
            // TODO: I had to manually correct this line from ITokenStream
			TokenStream = new DebugTokenStream( input, proxy );
			try
			{
				proxy.Handshake();
			}
			catch ( IOException ioe )
			{
				ReportError( ioe );
			}
			ITreeAdaptor adap = new CommonTreeAdaptor();
			TreeAdaptor = adap;
			proxy.TreeAdaptor = adap;
		}
	public DebugGrammarParser( ITokenStream input, IDebugEventListener dbg )
		: base( input, dbg )
	{


		ITreeAdaptor adap = new CommonTreeAdaptor();
		TreeAdaptor = adap;

	}
	protected virtual bool EvalPredicate( bool result, string predicate )
	{
		dbg.SemanticPredicate( result, predicate );
		return result;
	}

	protected DebugTreeAdaptor adaptor;
	public ITreeAdaptor TreeAdaptor
	{
		get
		{
			return adaptor;
		}
		set
		{
			this.adaptor = new DebugTreeAdaptor(dbg,adaptor);

		}
	}


	public override string[] GetTokenNames() { return DebugGrammarParser.tokenNames; }
	public override string GrammarFileName { get { return "BuildOptions\\DebugGrammar.g3"; } }


	#region Rules
	public class prog_return : ParserRuleReturnScope
	{
		public CommonTree tree;
		public override object Tree { get { return tree; } }
	}

	// $ANTLR start "prog"
	// BuildOptions\\DebugGrammar.g3:50:0: prog : ( stat )* ;
	private DebugGrammarParser.prog_return prog(  )
	{
		DebugGrammarParser.prog_return retval = new DebugGrammarParser.prog_return();
		retval.start = input.LT(1);

		CommonTree root_0 = null;

		DebugGrammarParser.stat_return stat1 = default(DebugGrammarParser.stat_return);


		try
		{
			dbg.EnterRule( GrammarFileName, "prog" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 50, -1 );

		try
		{
			// BuildOptions\\DebugGrammar.g3:50:7: ( ( stat )* )
			dbg.EnterAlt( 1 );

			// BuildOptions\\DebugGrammar.g3:50:7: ( stat )*
			{
			root_0 = (CommonTree)adaptor.Nil();

			dbg.Location( 50, 6 );
			// BuildOptions\\DebugGrammar.g3:50:7: ( stat )*
			try
			{
				dbg.EnterSubRule( 1 );

			for ( ; ; )
			{
				int alt1=2;
				try
				{
					dbg.EnterDecision( 1 );

				int LA1_0 = input.LA(1);

				if ( ((LA1_0>=ID && LA1_0<=NEWLINE)||LA1_0==12) )
				{
					alt1=1;
				}


				}
				finally
				{
					dbg.ExitDecision( 1 );
				}

				switch ( alt1 )
				{
				case 1:
					dbg.EnterAlt( 1 );

					// BuildOptions\\DebugGrammar.g3:50:9: stat
					{
					dbg.Location( 50, 8 );
					PushFollow(Follow._stat_in_prog53);
					stat1=stat();

					state._fsp--;

					adaptor.AddChild(root_0, stat1.Tree);

					}
					break;

				default:
					goto loop1;
				}
			}

			loop1:
				;

			}
			finally
			{
				dbg.ExitSubRule( 1 );
			}


			}

			retval.stop = input.LT(-1);

			retval.tree = (CommonTree)adaptor.RulePostProcessing(root_0);
			adaptor.SetTokenBoundaries(retval.tree, retval.start, retval.stop);

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		retval.tree = (CommonTree)adaptor.ErrorNode(input, retval.start, input.LT(-1), re);

		}
		finally
		{
		}
		dbg.Location(51, 4);

		}
		finally
		{
			dbg.ExitRule( GrammarFileName, "prog" );
			DecRuleLevel();
			if ( RuleLevel == 0 )
			{
				dbg.Terminate();
			}
		}

		return retval;
	}
	// $ANTLR end "prog"

	public class stat_return : ParserRuleReturnScope
	{
		public CommonTree tree;
		public override object Tree { get { return tree; } }
	}

	// $ANTLR start "stat"
	// BuildOptions\\DebugGrammar.g3:53:0: stat : ( expr NEWLINE -> expr | ID '=' expr NEWLINE -> ^( '=' ID expr ) | func NEWLINE -> func | NEWLINE ->);
	private DebugGrammarParser.stat_return stat(  )
	{
		DebugGrammarParser.stat_return retval = new DebugGrammarParser.stat_return();
		retval.start = input.LT(1);

		CommonTree root_0 = null;

		IToken NEWLINE3=null;
		IToken ID4=null;
		IToken char_literal5=null;
		IToken NEWLINE7=null;
		IToken NEWLINE9=null;
		IToken NEWLINE10=null;
		DebugGrammarParser.expr_return expr2 = default(DebugGrammarParser.expr_return);
		DebugGrammarParser.expr_return expr6 = default(DebugGrammarParser.expr_return);
		DebugGrammarParser.func_return func8 = default(DebugGrammarParser.func_return);

		CommonTree NEWLINE3_tree=null;
		CommonTree ID4_tree=null;
		CommonTree char_literal5_tree=null;
		CommonTree NEWLINE7_tree=null;
		CommonTree NEWLINE9_tree=null;
		CommonTree NEWLINE10_tree=null;
		RewriteRuleITokenStream stream_NEWLINE=new RewriteRuleITokenStream(adaptor,"token NEWLINE");
		RewriteRuleITokenStream stream_ID=new RewriteRuleITokenStream(adaptor,"token ID");
		RewriteRuleITokenStream stream_17=new RewriteRuleITokenStream(adaptor,"token 17");
		RewriteRuleSubtreeStream stream_expr=new RewriteRuleSubtreeStream(adaptor,"rule expr");
		RewriteRuleSubtreeStream stream_func=new RewriteRuleSubtreeStream(adaptor,"rule func");
		try
		{
			dbg.EnterRule( GrammarFileName, "stat" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 53, -1 );

		try
		{
			// BuildOptions\\DebugGrammar.g3:53:9: ( expr NEWLINE -> expr | ID '=' expr NEWLINE -> ^( '=' ID expr ) | func NEWLINE -> func | NEWLINE ->)
			int alt2=4;
			try
			{
				dbg.EnterDecision( 2 );

			try
			{
				isCyclicDecision = true;
				alt2 = dfa2.Predict(input);
			}
			catch ( NoViableAltException nvae )
			{
				dbg.RecognitionException( nvae );
				throw nvae;
			}
			}
			finally
			{
				dbg.ExitDecision( 2 );
			}

			switch ( alt2 )
			{
			case 1:
				dbg.EnterAlt( 1 );

				// BuildOptions\\DebugGrammar.g3:53:9: expr NEWLINE
				{
				dbg.Location( 53, 8 );
				PushFollow(Follow._expr_in_stat70);
				expr2=expr();

				state._fsp--;

				stream_expr.Add(expr2.Tree);
				dbg.Location( 53, 13 );
				NEWLINE3=(IToken)Match(input,NEWLINE,Follow._NEWLINE_in_stat72);  
				stream_NEWLINE.Add(NEWLINE3);



				{
				// AST REWRITE
				// elements: expr
				// token labels: 
				// rule labels: retval
				// token list labels: 
				// rule list labels: 
				// wildcard labels: 
				retval.tree = root_0;
				RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"rule retval",retval!=null?retval.tree:null);

				root_0 = (CommonTree)adaptor.Nil();
				// 53:41: -> expr
				{
					dbg.Location( 53, 43 );
					adaptor.AddChild(root_0, stream_expr.NextTree());

				}

				retval.tree = root_0;
				}

				}
				break;
			case 2:
				dbg.EnterAlt( 2 );

				// BuildOptions\\DebugGrammar.g3:54:9: ID '=' expr NEWLINE
				{
				dbg.Location( 54, 8 );
				ID4=(IToken)Match(input,ID,Follow._ID_in_stat105);  
				stream_ID.Add(ID4);

				dbg.Location( 54, 11 );
				char_literal5=(IToken)Match(input,17,Follow._17_in_stat107);  
				stream_17.Add(char_literal5);

				dbg.Location( 54, 15 );
				PushFollow(Follow._expr_in_stat109);
				expr6=expr();

				state._fsp--;

				stream_expr.Add(expr6.Tree);
				dbg.Location( 54, 20 );
				NEWLINE7=(IToken)Match(input,NEWLINE,Follow._NEWLINE_in_stat111);  
				stream_NEWLINE.Add(NEWLINE7);



				{
				// AST REWRITE
				// elements: 17, ID, expr
				// token labels: 
				// rule labels: retval
				// token list labels: 
				// rule list labels: 
				// wildcard labels: 
				retval.tree = root_0;
				RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"rule retval",retval!=null?retval.tree:null);

				root_0 = (CommonTree)adaptor.Nil();
				// 54:41: -> ^( '=' ID expr )
				{
					dbg.Location( 54, 43 );
					// BuildOptions\\DebugGrammar.g3:54:44: ^( '=' ID expr )
					{
					CommonTree root_1 = (CommonTree)adaptor.Nil();
					dbg.Location( 54, 45 );
					root_1 = (CommonTree)adaptor.BecomeRoot(stream_17.NextNode(), root_1);

					dbg.Location( 54, 49 );
					adaptor.AddChild(root_1, stream_ID.NextNode());
					dbg.Location( 54, 52 );
					adaptor.AddChild(root_1, stream_expr.NextTree());

					adaptor.AddChild(root_0, root_1);
					}

				}

				retval.tree = root_0;
				}

				}
				break;
			case 3:
				dbg.EnterAlt( 3 );

				// BuildOptions\\DebugGrammar.g3:55:9: func NEWLINE
				{
				dbg.Location( 55, 8 );
				PushFollow(Follow._func_in_stat143);
				func8=func();

				state._fsp--;

				stream_func.Add(func8.Tree);
				dbg.Location( 55, 13 );
				NEWLINE9=(IToken)Match(input,NEWLINE,Follow._NEWLINE_in_stat145);  
				stream_NEWLINE.Add(NEWLINE9);



				{
				// AST REWRITE
				// elements: func
				// token labels: 
				// rule labels: retval
				// token list labels: 
				// rule list labels: 
				// wildcard labels: 
				retval.tree = root_0;
				RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"rule retval",retval!=null?retval.tree:null);

				root_0 = (CommonTree)adaptor.Nil();
				// 55:41: -> func
				{
					dbg.Location( 55, 43 );
					adaptor.AddChild(root_0, stream_func.NextTree());

				}

				retval.tree = root_0;
				}

				}
				break;
			case 4:
				dbg.EnterAlt( 4 );

				// BuildOptions\\DebugGrammar.g3:56:9: NEWLINE
				{
				dbg.Location( 56, 8 );
				NEWLINE10=(IToken)Match(input,NEWLINE,Follow._NEWLINE_in_stat178);  
				stream_NEWLINE.Add(NEWLINE10);



				{
				// AST REWRITE
				// elements: 
				// token labels: 
				// rule labels: retval
				// token list labels: 
				// rule list labels: 
				// wildcard labels: 
				retval.tree = root_0;
				RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"rule retval",retval!=null?retval.tree:null);

				root_0 = (CommonTree)adaptor.Nil();
				// 56:41: ->
				{
					dbg.Location( 57, 4 );
					root_0 = null;
				}

				retval.tree = root_0;
				}

				}
				break;

			}
			retval.stop = input.LT(-1);

			retval.tree = (CommonTree)adaptor.RulePostProcessing(root_0);
			adaptor.SetTokenBoundaries(retval.tree, retval.start, retval.stop);

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		retval.tree = (CommonTree)adaptor.ErrorNode(input, retval.start, input.LT(-1), re);

		}
		finally
		{
		}
		dbg.Location(57, 4);

		}
		finally
		{
			dbg.ExitRule( GrammarFileName, "stat" );
			DecRuleLevel();
			if ( RuleLevel == 0 )
			{
				dbg.Terminate();
			}
		}

		return retval;
	}
	// $ANTLR end "stat"

	public class func_return : ParserRuleReturnScope
	{
		public CommonTree tree;
		public override object Tree { get { return tree; } }
	}

	// $ANTLR start "func"
	// BuildOptions\\DebugGrammar.g3:59:0: func : ID '(' formalPar ')' '=' expr -> ^( FUNC ID formalPar expr ) ;
	private DebugGrammarParser.func_return func(  )
	{
		DebugGrammarParser.func_return retval = new DebugGrammarParser.func_return();
		retval.start = input.LT(1);

		CommonTree root_0 = null;

		IToken ID11=null;
		IToken char_literal12=null;
		IToken char_literal14=null;
		IToken char_literal15=null;
		DebugGrammarParser.formalPar_return formalPar13 = default(DebugGrammarParser.formalPar_return);
		DebugGrammarParser.expr_return expr16 = default(DebugGrammarParser.expr_return);

		CommonTree ID11_tree=null;
		CommonTree char_literal12_tree=null;
		CommonTree char_literal14_tree=null;
		CommonTree char_literal15_tree=null;
		RewriteRuleITokenStream stream_ID=new RewriteRuleITokenStream(adaptor,"token ID");
		RewriteRuleITokenStream stream_12=new RewriteRuleITokenStream(adaptor,"token 12");
		RewriteRuleITokenStream stream_13=new RewriteRuleITokenStream(adaptor,"token 13");
		RewriteRuleITokenStream stream_17=new RewriteRuleITokenStream(adaptor,"token 17");
		RewriteRuleSubtreeStream stream_formalPar=new RewriteRuleSubtreeStream(adaptor,"rule formalPar");
		RewriteRuleSubtreeStream stream_expr=new RewriteRuleSubtreeStream(adaptor,"rule expr");
		try
		{
			dbg.EnterRule( GrammarFileName, "func" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 59, -1 );

		try
		{
			// BuildOptions\\DebugGrammar.g3:59:9: ( ID '(' formalPar ')' '=' expr -> ^( FUNC ID formalPar expr ) )
			dbg.EnterAlt( 1 );

			// BuildOptions\\DebugGrammar.g3:59:9: ID '(' formalPar ')' '=' expr
			{
			dbg.Location( 59, 8 );
			ID11=(IToken)Match(input,ID,Follow._ID_in_func219);  
			stream_ID.Add(ID11);

			dbg.Location( 59, 12 );
			char_literal12=(IToken)Match(input,12,Follow._12_in_func222);  
			stream_12.Add(char_literal12);

			dbg.Location( 59, 16 );
			PushFollow(Follow._formalPar_in_func224);
			formalPar13=formalPar();

			state._fsp--;

			stream_formalPar.Add(formalPar13.Tree);
			dbg.Location( 59, 26 );
			char_literal14=(IToken)Match(input,13,Follow._13_in_func226);  
			stream_13.Add(char_literal14);

			dbg.Location( 59, 30 );
			char_literal15=(IToken)Match(input,17,Follow._17_in_func228);  
			stream_17.Add(char_literal15);

			dbg.Location( 59, 34 );
			PushFollow(Follow._expr_in_func230);
			expr16=expr();

			state._fsp--;

			stream_expr.Add(expr16.Tree);


			{
			// AST REWRITE
			// elements: ID, formalPar, expr
			// token labels: 
			// rule labels: retval
			// token list labels: 
			// rule list labels: 
			// wildcard labels: 
			retval.tree = root_0;
			RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"rule retval",retval!=null?retval.tree:null);

			root_0 = (CommonTree)adaptor.Nil();
			// 59:41: -> ^( FUNC ID formalPar expr )
			{
				dbg.Location( 59, 43 );
				// BuildOptions\\DebugGrammar.g3:59:44: ^( FUNC ID formalPar expr )
				{
				CommonTree root_1 = (CommonTree)adaptor.Nil();
				dbg.Location( 59, 45 );
				root_1 = (CommonTree)adaptor.BecomeRoot((CommonTree)adaptor.Create(FUNC, "FUNC"), root_1);

				dbg.Location( 59, 50 );
				adaptor.AddChild(root_1, stream_ID.NextNode());
				dbg.Location( 59, 53 );
				adaptor.AddChild(root_1, stream_formalPar.NextTree());
				dbg.Location( 59, 63 );
				adaptor.AddChild(root_1, stream_expr.NextTree());

				adaptor.AddChild(root_0, root_1);
				}

			}

			retval.tree = root_0;
			}

			}

			retval.stop = input.LT(-1);

			retval.tree = (CommonTree)adaptor.RulePostProcessing(root_0);
			adaptor.SetTokenBoundaries(retval.tree, retval.start, retval.stop);

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		retval.tree = (CommonTree)adaptor.ErrorNode(input, retval.start, input.LT(-1), re);

		}
		finally
		{

				  functionDefinitions.Add(((CommonTree)retval.tree));
				
		}
		dbg.Location(60, 4);

		}
		finally
		{
			dbg.ExitRule( GrammarFileName, "func" );
			DecRuleLevel();
			if ( RuleLevel == 0 )
			{
				dbg.Terminate();
			}
		}

		return retval;
	}
	// $ANTLR end "func"

	public class formalPar_return : ParserRuleReturnScope
	{
		public CommonTree tree;
		public override object Tree { get { return tree; } }
	}

	// $ANTLR start "formalPar"
	// BuildOptions\\DebugGrammar.g3:65:0: formalPar : ( ID | INT );
	private DebugGrammarParser.formalPar_return formalPar(  )
	{
		DebugGrammarParser.formalPar_return retval = new DebugGrammarParser.formalPar_return();
		retval.start = input.LT(1);

		CommonTree root_0 = null;

		IToken set17=null;

		CommonTree set17_tree=null;

		try
		{
			dbg.EnterRule( GrammarFileName, "formalPar" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 65, -1 );

		try
		{
			// BuildOptions\\DebugGrammar.g3:66:9: ( ID | INT )
			dbg.EnterAlt( 1 );

			// BuildOptions\\DebugGrammar.g3:
			{
			root_0 = (CommonTree)adaptor.Nil();

			dbg.Location( 66, 8 );
			set17=(IToken)input.LT(1);
			if ( (input.LA(1)>=ID && input.LA(1)<=INT) )
			{
				input.Consume();
				adaptor.AddChild(root_0, (CommonTree)adaptor.Create(set17));
				state.errorRecovery=false;
			}
			else
			{
				MismatchedSetException mse = new MismatchedSetException(null,input);
				dbg.RecognitionException( mse );
				throw mse;
			}


			}

			retval.stop = input.LT(-1);

			retval.tree = (CommonTree)adaptor.RulePostProcessing(root_0);
			adaptor.SetTokenBoundaries(retval.tree, retval.start, retval.stop);

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		retval.tree = (CommonTree)adaptor.ErrorNode(input, retval.start, input.LT(-1), re);

		}
		finally
		{
		}
		dbg.Location(68, 1);

		}
		finally
		{
			dbg.ExitRule( GrammarFileName, "formalPar" );
			DecRuleLevel();
			if ( RuleLevel == 0 )
			{
				dbg.Terminate();
			}
		}

		return retval;
	}
	// $ANTLR end "formalPar"

	public class expr_return : ParserRuleReturnScope
	{
		public CommonTree tree;
		public override object Tree { get { return tree; } }
	}

	// $ANTLR start "expr"
	// BuildOptions\\DebugGrammar.g3:73:0: expr : multExpr ( ( '+' | '-' ) multExpr )* ;
	private DebugGrammarParser.expr_return expr(  )
	{
		DebugGrammarParser.expr_return retval = new DebugGrammarParser.expr_return();
		retval.start = input.LT(1);

		CommonTree root_0 = null;

		IToken char_literal19=null;
		IToken char_literal20=null;
		DebugGrammarParser.multExpr_return multExpr18 = default(DebugGrammarParser.multExpr_return);
		DebugGrammarParser.multExpr_return multExpr21 = default(DebugGrammarParser.multExpr_return);

		CommonTree char_literal19_tree=null;
		CommonTree char_literal20_tree=null;

		try
		{
			dbg.EnterRule( GrammarFileName, "expr" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 73, -1 );

		try
		{
			// BuildOptions\\DebugGrammar.g3:73:9: ( multExpr ( ( '+' | '-' ) multExpr )* )
			dbg.EnterAlt( 1 );

			// BuildOptions\\DebugGrammar.g3:73:9: multExpr ( ( '+' | '-' ) multExpr )*
			{
			root_0 = (CommonTree)adaptor.Nil();

			dbg.Location( 73, 8 );
			PushFollow(Follow._multExpr_in_expr288);
			multExpr18=multExpr();

			state._fsp--;

			adaptor.AddChild(root_0, multExpr18.Tree);
			dbg.Location( 73, 17 );
			// BuildOptions\\DebugGrammar.g3:73:18: ( ( '+' | '-' ) multExpr )*
			try
			{
				dbg.EnterSubRule( 4 );

			for ( ; ; )
			{
				int alt4=2;
				try
				{
					dbg.EnterDecision( 4 );

				int LA4_0 = input.LA(1);

				if ( (LA4_0==10||LA4_0==16) )
				{
					alt4=1;
				}


				}
				finally
				{
					dbg.ExitDecision( 4 );
				}

				switch ( alt4 )
				{
				case 1:
					dbg.EnterAlt( 1 );

					// BuildOptions\\DebugGrammar.g3:73:19: ( '+' | '-' ) multExpr
					{
					dbg.Location( 73, 18 );
					// BuildOptions\\DebugGrammar.g3:73:19: ( '+' | '-' )
					int alt3=2;
					try
					{
						dbg.EnterSubRule( 3 );
					try
					{
						dbg.EnterDecision( 3 );

					int LA3_0 = input.LA(1);

					if ( (LA3_0==16) )
					{
						alt3=1;
					}
					else if ( (LA3_0==10) )
					{
						alt3=2;
					}
					else
					{
						NoViableAltException nvae = new NoViableAltException("", 3, 0, input);

						dbg.RecognitionException( nvae );
						throw nvae;
					}
					}
					finally
					{
						dbg.ExitDecision( 3 );
					}

					switch ( alt3 )
					{
					case 1:
						dbg.EnterAlt( 1 );

						// BuildOptions\\DebugGrammar.g3:73:20: '+'
						{
						dbg.Location( 73, 22 );
						char_literal19=(IToken)Match(input,16,Follow._16_in_expr292); 
						char_literal19_tree = (CommonTree)adaptor.Create(char_literal19);
						root_0 = (CommonTree)adaptor.BecomeRoot(char_literal19_tree, root_0);


						}
						break;
					case 2:
						dbg.EnterAlt( 2 );

						// BuildOptions\\DebugGrammar.g3:73:25: '-'
						{
						dbg.Location( 73, 27 );
						char_literal20=(IToken)Match(input,10,Follow._10_in_expr295); 
						char_literal20_tree = (CommonTree)adaptor.Create(char_literal20);
						root_0 = (CommonTree)adaptor.BecomeRoot(char_literal20_tree, root_0);


						}
						break;

					}
					}
					finally
					{
						dbg.ExitSubRule( 3 );
					}

					dbg.Location( 73, 30 );
					PushFollow(Follow._multExpr_in_expr299);
					multExpr21=multExpr();

					state._fsp--;

					adaptor.AddChild(root_0, multExpr21.Tree);

					}
					break;

				default:
					goto loop4;
				}
			}

			loop4:
				;

			}
			finally
			{
				dbg.ExitSubRule( 4 );
			}


			}

			retval.stop = input.LT(-1);

			retval.tree = (CommonTree)adaptor.RulePostProcessing(root_0);
			adaptor.SetTokenBoundaries(retval.tree, retval.start, retval.stop);

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		retval.tree = (CommonTree)adaptor.ErrorNode(input, retval.start, input.LT(-1), re);

		}
		finally
		{
		}
		dbg.Location(74, 4);

		}
		finally
		{
			dbg.ExitRule( GrammarFileName, "expr" );
			DecRuleLevel();
			if ( RuleLevel == 0 )
			{
				dbg.Terminate();
			}
		}

		return retval;
	}
	// $ANTLR end "expr"

	public class multExpr_return : ParserRuleReturnScope
	{
		public CommonTree tree;
		public override object Tree { get { return tree; } }
	}

	// $ANTLR start "multExpr"
	// BuildOptions\\DebugGrammar.g3:76:0: multExpr : atom ( ( '*' | '/' | '%' ) atom )* ;
	private DebugGrammarParser.multExpr_return multExpr(  )
	{
		DebugGrammarParser.multExpr_return retval = new DebugGrammarParser.multExpr_return();
		retval.start = input.LT(1);

		CommonTree root_0 = null;

		IToken set23=null;
		DebugGrammarParser.atom_return atom22 = default(DebugGrammarParser.atom_return);
		DebugGrammarParser.atom_return atom24 = default(DebugGrammarParser.atom_return);

		CommonTree set23_tree=null;

		try
		{
			dbg.EnterRule( GrammarFileName, "multExpr" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 76, -1 );

		try
		{
			// BuildOptions\\DebugGrammar.g3:77:9: ( atom ( ( '*' | '/' | '%' ) atom )* )
			dbg.EnterAlt( 1 );

			// BuildOptions\\DebugGrammar.g3:77:9: atom ( ( '*' | '/' | '%' ) atom )*
			{
			root_0 = (CommonTree)adaptor.Nil();

			dbg.Location( 77, 8 );
			PushFollow(Follow._atom_in_multExpr320);
			atom22=atom();

			state._fsp--;

			adaptor.AddChild(root_0, atom22.Tree);
			dbg.Location( 77, 13 );
			// BuildOptions\\DebugGrammar.g3:77:14: ( ( '*' | '/' | '%' ) atom )*
			try
			{
				dbg.EnterSubRule( 5 );

			for ( ; ; )
			{
				int alt5=2;
				try
				{
					dbg.EnterDecision( 5 );

				int LA5_0 = input.LA(1);

				if ( (LA5_0==11||(LA5_0>=14 && LA5_0<=15)) )
				{
					alt5=1;
				}


				}
				finally
				{
					dbg.ExitDecision( 5 );
				}

				switch ( alt5 )
				{
				case 1:
					dbg.EnterAlt( 1 );

					// BuildOptions\\DebugGrammar.g3:77:15: ( '*' | '/' | '%' ) atom
					{
					dbg.Location( 77, 27 );
					set23=(IToken)input.LT(1);
					set23=(IToken)input.LT(1);
					if ( input.LA(1)==11||(input.LA(1)>=14 && input.LA(1)<=15) )
					{
						input.Consume();
						root_0 = (CommonTree)adaptor.BecomeRoot((CommonTree)adaptor.Create(set23), root_0);
						state.errorRecovery=false;
					}
					else
					{
						MismatchedSetException mse = new MismatchedSetException(null,input);
						dbg.RecognitionException( mse );
						throw mse;
					}

					dbg.Location( 77, 29 );
					PushFollow(Follow._atom_in_multExpr332);
					atom24=atom();

					state._fsp--;

					adaptor.AddChild(root_0, atom24.Tree);

					}
					break;

				default:
					goto loop5;
				}
			}

			loop5:
				;

			}
			finally
			{
				dbg.ExitSubRule( 5 );
			}


			}

			retval.stop = input.LT(-1);

			retval.tree = (CommonTree)adaptor.RulePostProcessing(root_0);
			adaptor.SetTokenBoundaries(retval.tree, retval.start, retval.stop);

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		retval.tree = (CommonTree)adaptor.ErrorNode(input, retval.start, input.LT(-1), re);

		}
		finally
		{
		}
		dbg.Location(78, 4);

		}
		finally
		{
			dbg.ExitRule( GrammarFileName, "multExpr" );
			DecRuleLevel();
			if ( RuleLevel == 0 )
			{
				dbg.Terminate();
			}
		}

		return retval;
	}
	// $ANTLR end "multExpr"

	public class atom_return : ParserRuleReturnScope
	{
		public CommonTree tree;
		public override object Tree { get { return tree; } }
	}

	// $ANTLR start "atom"
	// BuildOptions\\DebugGrammar.g3:80:0: atom : ( INT | ID | '(' expr ')' -> expr | ID '(' expr ')' -> ^( CALL ID expr ) );
	private DebugGrammarParser.atom_return atom(  )
	{
		DebugGrammarParser.atom_return retval = new DebugGrammarParser.atom_return();
		retval.start = input.LT(1);

		CommonTree root_0 = null;

		IToken INT25=null;
		IToken ID26=null;
		IToken char_literal27=null;
		IToken char_literal29=null;
		IToken ID30=null;
		IToken char_literal31=null;
		IToken char_literal33=null;
		DebugGrammarParser.expr_return expr28 = default(DebugGrammarParser.expr_return);
		DebugGrammarParser.expr_return expr32 = default(DebugGrammarParser.expr_return);

		CommonTree INT25_tree=null;
		CommonTree ID26_tree=null;
		CommonTree char_literal27_tree=null;
		CommonTree char_literal29_tree=null;
		CommonTree ID30_tree=null;
		CommonTree char_literal31_tree=null;
		CommonTree char_literal33_tree=null;
		RewriteRuleITokenStream stream_12=new RewriteRuleITokenStream(adaptor,"token 12");
		RewriteRuleITokenStream stream_13=new RewriteRuleITokenStream(adaptor,"token 13");
		RewriteRuleITokenStream stream_ID=new RewriteRuleITokenStream(adaptor,"token ID");
		RewriteRuleSubtreeStream stream_expr=new RewriteRuleSubtreeStream(adaptor,"rule expr");
		try
		{
			dbg.EnterRule( GrammarFileName, "atom" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 80, -1 );

		try
		{
			// BuildOptions\\DebugGrammar.g3:80:9: ( INT | ID | '(' expr ')' -> expr | ID '(' expr ')' -> ^( CALL ID expr ) )
			int alt6=4;
			try
			{
				dbg.EnterDecision( 6 );

			switch ( input.LA(1) )
			{
			case INT:
				{
				alt6=1;
				}
				break;
			case ID:
				{
				int LA6_2 = input.LA(2);

				if ( (LA6_2==12) )
				{
					alt6=4;
				}
				else if ( (LA6_2==NEWLINE||(LA6_2>=10 && LA6_2<=11)||(LA6_2>=13 && LA6_2<=16)) )
				{
					alt6=2;
				}
				else
				{
					NoViableAltException nvae = new NoViableAltException("", 6, 2, input);

					dbg.RecognitionException( nvae );
					throw nvae;
				}
				}
				break;
			case 12:
				{
				alt6=3;
				}
				break;
			default:
				{
					NoViableAltException nvae = new NoViableAltException("", 6, 0, input);

					dbg.RecognitionException( nvae );
					throw nvae;
				}
			}

			}
			finally
			{
				dbg.ExitDecision( 6 );
			}

			switch ( alt6 )
			{
			case 1:
				dbg.EnterAlt( 1 );

				// BuildOptions\\DebugGrammar.g3:80:9: INT
				{
				root_0 = (CommonTree)adaptor.Nil();

				dbg.Location( 80, 8 );
				INT25=(IToken)Match(input,INT,Follow._INT_in_atom348); 
				INT25_tree = (CommonTree)adaptor.Create(INT25);
				adaptor.AddChild(root_0, INT25_tree);


				}
				break;
			case 2:
				dbg.EnterAlt( 2 );

				// BuildOptions\\DebugGrammar.g3:81:9: ID
				{
				root_0 = (CommonTree)adaptor.Nil();

				dbg.Location( 81, 8 );
				ID26=(IToken)Match(input,ID,Follow._ID_in_atom358); 
				ID26_tree = (CommonTree)adaptor.Create(ID26);
				adaptor.AddChild(root_0, ID26_tree);


				}
				break;
			case 3:
				dbg.EnterAlt( 3 );

				// BuildOptions\\DebugGrammar.g3:82:9: '(' expr ')'
				{
				dbg.Location( 82, 8 );
				char_literal27=(IToken)Match(input,12,Follow._12_in_atom368);  
				stream_12.Add(char_literal27);

				dbg.Location( 82, 12 );
				PushFollow(Follow._expr_in_atom370);
				expr28=expr();

				state._fsp--;

				stream_expr.Add(expr28.Tree);
				dbg.Location( 82, 17 );
				char_literal29=(IToken)Match(input,13,Follow._13_in_atom372);  
				stream_13.Add(char_literal29);



				{
				// AST REWRITE
				// elements: expr
				// token labels: 
				// rule labels: retval
				// token list labels: 
				// rule list labels: 
				// wildcard labels: 
				retval.tree = root_0;
				RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"rule retval",retval!=null?retval.tree:null);

				root_0 = (CommonTree)adaptor.Nil();
				// 82:25: -> expr
				{
					dbg.Location( 82, 27 );
					adaptor.AddChild(root_0, stream_expr.NextTree());

				}

				retval.tree = root_0;
				}

				}
				break;
			case 4:
				dbg.EnterAlt( 4 );

				// BuildOptions\\DebugGrammar.g3:83:9: ID '(' expr ')'
				{
				dbg.Location( 83, 8 );
				ID30=(IToken)Match(input,ID,Follow._ID_in_atom389);  
				stream_ID.Add(ID30);

				dbg.Location( 83, 11 );
				char_literal31=(IToken)Match(input,12,Follow._12_in_atom391);  
				stream_12.Add(char_literal31);

				dbg.Location( 83, 15 );
				PushFollow(Follow._expr_in_atom393);
				expr32=expr();

				state._fsp--;

				stream_expr.Add(expr32.Tree);
				dbg.Location( 83, 20 );
				char_literal33=(IToken)Match(input,13,Follow._13_in_atom395);  
				stream_13.Add(char_literal33);



				{
				// AST REWRITE
				// elements: ID, expr
				// token labels: 
				// rule labels: retval
				// token list labels: 
				// rule list labels: 
				// wildcard labels: 
				retval.tree = root_0;
				RewriteRuleSubtreeStream stream_retval=new RewriteRuleSubtreeStream(adaptor,"rule retval",retval!=null?retval.tree:null);

				root_0 = (CommonTree)adaptor.Nil();
				// 83:25: -> ^( CALL ID expr )
				{
					dbg.Location( 83, 27 );
					// BuildOptions\\DebugGrammar.g3:83:28: ^( CALL ID expr )
					{
					CommonTree root_1 = (CommonTree)adaptor.Nil();
					dbg.Location( 83, 29 );
					root_1 = (CommonTree)adaptor.BecomeRoot((CommonTree)adaptor.Create(CALL, "CALL"), root_1);

					dbg.Location( 83, 34 );
					adaptor.AddChild(root_1, stream_ID.NextNode());
					dbg.Location( 83, 37 );
					adaptor.AddChild(root_1, stream_expr.NextTree());

					adaptor.AddChild(root_0, root_1);
					}

				}

				retval.tree = root_0;
				}

				}
				break;

			}
			retval.stop = input.LT(-1);

			retval.tree = (CommonTree)adaptor.RulePostProcessing(root_0);
			adaptor.SetTokenBoundaries(retval.tree, retval.start, retval.stop);

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		retval.tree = (CommonTree)adaptor.ErrorNode(input, retval.start, input.LT(-1), re);

		}
		finally
		{
		}
		dbg.Location(84, 4);

		}
		finally
		{
			dbg.ExitRule( GrammarFileName, "atom" );
			DecRuleLevel();
			if ( RuleLevel == 0 )
			{
				dbg.Terminate();
			}
		}

		return retval;
	}
	// $ANTLR end "atom"
	#endregion

	// Delegated rules

	#region Synpreds
	#endregion

	#region DFA
	DFA2 dfa2;

	protected override void InitDFAs()
	{
		base.InitDFAs();
		dfa2 = new DFA2( this );
	}

	class DFA2 : DFA
	{

		const string DFA2_eotS =
			"\xA\xFFFF";
		const string DFA2_eofS =
			"\xA\xFFFF";
		const string DFA2_minS =
			"\x1\x6\x1\xFFFF\x1\x8\x1\xFFFF\x1\x6\x1\xFFFF\x2\xA\x1\x8\x1\xFFFF";
		const string DFA2_maxS =
			"\x1\xC\x1\xFFFF\x1\x11\x1\xFFFF\x1\xC\x1\xFFFF\x2\x10\x1\x11\x1\xFFFF";
		const string DFA2_acceptS =
			"\x1\xFFFF\x1\x1\x1\xFFFF\x1\x4\x1\xFFFF\x1\x2\x3\xFFFF\x1\x3";
		const string DFA2_specialS =
			"\xA\xFFFF}>";
		static readonly string[] DFA2_transitionS =
			{
				"\x1\x2\x1\x1\x1\x3\x3\xFFFF\x1\x1",
				"",
				"\x1\x1\x1\xFFFF\x2\x1\x1\x4\x1\xFFFF\x3\x1\x1\x5",
				"",
				"\x1\x7\x1\x6\x4\xFFFF\x1\x1",
				"",
				"\x2\x1\x1\xFFFF\x1\x8\x3\x1",
				"\x3\x1\x1\x8\x3\x1",
				"\x1\x1\x1\xFFFF\x2\x1\x2\xFFFF\x3\x1\x1\x9",
				""
			};

		static readonly short[] DFA2_eot = DFA.UnpackEncodedString(DFA2_eotS);
		static readonly short[] DFA2_eof = DFA.UnpackEncodedString(DFA2_eofS);
		static readonly char[] DFA2_min = DFA.UnpackEncodedStringToUnsignedChars(DFA2_minS);
		static readonly char[] DFA2_max = DFA.UnpackEncodedStringToUnsignedChars(DFA2_maxS);
		static readonly short[] DFA2_accept = DFA.UnpackEncodedString(DFA2_acceptS);
		static readonly short[] DFA2_special = DFA.UnpackEncodedString(DFA2_specialS);
		static readonly short[][] DFA2_transition;

		static DFA2()
		{
			int numStates = DFA2_transitionS.Length;
			DFA2_transition = new short[numStates][];
			for ( int i=0; i < numStates; i++ )
			{
				DFA2_transition[i] = DFA.UnpackEncodedString(DFA2_transitionS[i]);
			}
		}

		public DFA2( BaseRecognizer recognizer )
		{
			this.recognizer = recognizer;
			this.decisionNumber = 2;
			this.eot = DFA2_eot;
			this.eof = DFA2_eof;
			this.min = DFA2_min;
			this.max = DFA2_max;
			this.accept = DFA2_accept;
			this.special = DFA2_special;
			this.transition = DFA2_transition;
		}
		public override string GetDescription()
		{
			return "53:0: stat : ( expr NEWLINE -> expr | ID '=' expr NEWLINE -> ^( '=' ID expr ) | func NEWLINE -> func | NEWLINE ->);";
		}
		public override void Error( NoViableAltException nvae )
		{
			((DebugParser)recognizer).dbg.RecognitionException( nvae );
		}
	}


	#endregion

	#region Follow Sets
	public static class Follow
	{
		public static readonly BitSet _stat_in_prog53 = new BitSet(new ulong[]{0x11C2UL});
		public static readonly BitSet _expr_in_stat70 = new BitSet(new ulong[]{0x100UL});
		public static readonly BitSet _NEWLINE_in_stat72 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _ID_in_stat105 = new BitSet(new ulong[]{0x20000UL});
		public static readonly BitSet _17_in_stat107 = new BitSet(new ulong[]{0x10C0UL});
		public static readonly BitSet _expr_in_stat109 = new BitSet(new ulong[]{0x100UL});
		public static readonly BitSet _NEWLINE_in_stat111 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _func_in_stat143 = new BitSet(new ulong[]{0x100UL});
		public static readonly BitSet _NEWLINE_in_stat145 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _NEWLINE_in_stat178 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _ID_in_func219 = new BitSet(new ulong[]{0x1000UL});
		public static readonly BitSet _12_in_func222 = new BitSet(new ulong[]{0xC0UL});
		public static readonly BitSet _formalPar_in_func224 = new BitSet(new ulong[]{0x2000UL});
		public static readonly BitSet _13_in_func226 = new BitSet(new ulong[]{0x20000UL});
		public static readonly BitSet _17_in_func228 = new BitSet(new ulong[]{0x10C0UL});
		public static readonly BitSet _expr_in_func230 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _set_in_formalPar267 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _multExpr_in_expr288 = new BitSet(new ulong[]{0x10402UL});
		public static readonly BitSet _16_in_expr292 = new BitSet(new ulong[]{0x10C0UL});
		public static readonly BitSet _10_in_expr295 = new BitSet(new ulong[]{0x10C0UL});
		public static readonly BitSet _multExpr_in_expr299 = new BitSet(new ulong[]{0x10402UL});
		public static readonly BitSet _atom_in_multExpr320 = new BitSet(new ulong[]{0xC802UL});
		public static readonly BitSet _set_in_multExpr323 = new BitSet(new ulong[]{0x10C0UL});
		public static readonly BitSet _atom_in_multExpr332 = new BitSet(new ulong[]{0xC802UL});
		public static readonly BitSet _INT_in_atom348 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _ID_in_atom358 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _12_in_atom368 = new BitSet(new ulong[]{0x10C0UL});
		public static readonly BitSet _expr_in_atom370 = new BitSet(new ulong[]{0x2000UL});
		public static readonly BitSet _13_in_atom372 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _ID_in_atom389 = new BitSet(new ulong[]{0x1000UL});
		public static readonly BitSet _12_in_atom391 = new BitSet(new ulong[]{0x10C0UL});
		public static readonly BitSet _expr_in_atom393 = new BitSet(new ulong[]{0x2000UL});
		public static readonly BitSet _13_in_atom395 = new BitSet(new ulong[]{0x2UL});

	}
	#endregion
}
