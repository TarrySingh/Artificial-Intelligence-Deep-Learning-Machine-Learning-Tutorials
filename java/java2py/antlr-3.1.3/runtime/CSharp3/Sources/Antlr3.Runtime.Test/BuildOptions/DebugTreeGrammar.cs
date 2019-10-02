// $ANTLR 3.1.2 BuildOptions\\DebugTreeGrammar.g3 2009-03-16 13:19:18

// The variable 'variable' is assigned but its value is never used.
#pragma warning disable 219
// Unreachable code detected.
#pragma warning disable 162


//import java.util.Map;
//import java.util.HashMap;
using BigInteger = java.math.BigInteger;
using Console = System.Console;


using System.Collections.Generic;
using Antlr.Runtime;
using Antlr.Runtime.Tree;
using RewriteRuleITokenStream = Antlr.Runtime.Tree.RewriteRuleTokenStream;using Stack = System.Collections.Generic.Stack<object>;
using List = System.Collections.IList;
using ArrayList = System.Collections.Generic.List<object>;

using Antlr.Runtime.Debug;
using IOException = System.IO.IOException;
public partial class DebugTreeGrammar : DebugTreeParser
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
			"invalidRule", "call", "expr", "prog", "stat"
		};

		int ruleLevel = 0;
		public virtual int RuleLevel { get { return ruleLevel; } }
		public virtual void IncRuleLevel() { ruleLevel++; }
		public virtual void DecRuleLevel() { ruleLevel--; }
		public DebugTreeGrammar( ITreeNodeStream input )
			: this( input, DebugEventSocketProxy.DEFAULT_DEBUGGER_PORT, new RecognizerSharedState() )
		{
		}
		public DebugTreeGrammar( ITreeNodeStream input, int port, RecognizerSharedState state )
			: base( input, state )
		{
			DebugEventSocketProxy proxy = new DebugEventSocketProxy( this, port, input.TreeAdaptor );
			DebugListener = proxy;
			try
			{
				proxy.Handshake();
			}
			catch ( IOException ioe )
			{
				ReportError( ioe );
			}
		}
	public DebugTreeGrammar( ITreeNodeStream input, IDebugEventListener dbg )
		: base( input, dbg, new RecognizerSharedState() )
	{

	}
	protected virtual bool EvalPredicate( bool result, string predicate )
	{
		dbg.SemanticPredicate( result, predicate );
		return result;
	}


	public override string[] GetTokenNames() { return DebugTreeGrammar.tokenNames; }
	public override string GrammarFileName { get { return "BuildOptions\\DebugTreeGrammar.g3"; } }


	#region Rules

	// $ANTLR start "prog"
	// BuildOptions\\DebugTreeGrammar.g3:53:0: prog : ( stat )* ;
	private void prog(  )
	{
		try
		{
			dbg.EnterRule( GrammarFileName, "prog" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 53, -1 );

		try
		{
			// BuildOptions\\DebugTreeGrammar.g3:53:9: ( ( stat )* )
			dbg.EnterAlt( 1 );

			// BuildOptions\\DebugTreeGrammar.g3:53:9: ( stat )*
			{
			dbg.Location( 53, 8 );
			// BuildOptions\\DebugTreeGrammar.g3:53:9: ( stat )*
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

				if ( ((LA1_0>=CALL && LA1_0<=INT)||(LA1_0>=10 && LA1_0<=11)||(LA1_0>=14 && LA1_0<=17)) )
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

					// BuildOptions\\DebugTreeGrammar.g3:53:0: stat
					{
					dbg.Location( 53, 8 );
					PushFollow(Follow._stat_in_prog48);
					stat();

					state._fsp--;


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

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		}
		finally
		{
		}
		dbg.Location(54, 4);

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

		return ;
	}
	// $ANTLR end "prog"


	// $ANTLR start "stat"
	// BuildOptions\\DebugTreeGrammar.g3:56:0: stat : ( expr | ^( '=' ID expr ) | ^( FUNC ( . )+ ) );
	private void stat(  )
	{
		CommonTree ID2=null;
		BigInteger expr1 = default(BigInteger);
		BigInteger expr3 = default(BigInteger);

		try
		{
			dbg.EnterRule( GrammarFileName, "stat" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 56, -1 );

		try
		{
			// BuildOptions\\DebugTreeGrammar.g3:56:9: ( expr | ^( '=' ID expr ) | ^( FUNC ( . )+ ) )
			int alt3=3;
			try
			{
				dbg.EnterDecision( 3 );

			switch ( input.LA(1) )
			{
			case CALL:
			case ID:
			case INT:
			case 10:
			case 11:
			case 14:
			case 15:
			case 16:
				{
				alt3=1;
				}
				break;
			case 17:
				{
				alt3=2;
				}
				break;
			case FUNC:
				{
				alt3=3;
				}
				break;
			default:
				{
					NoViableAltException nvae = new NoViableAltException("", 3, 0, input);

					dbg.RecognitionException( nvae );
					throw nvae;
				}
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

				// BuildOptions\\DebugTreeGrammar.g3:56:9: expr
				{
				dbg.Location( 56, 8 );
				PushFollow(Follow._expr_in_stat63);
				expr1=expr();

				state._fsp--;

				dbg.Location( 56, 35 );
				 string result = expr1.ToString();
				                                     Console.Out.WriteLine(expr1 + " (about " + result[0] + "*10^" + (result.Length-1) + ")");
				                                   

				}
				break;
			case 2:
				dbg.EnterAlt( 2 );

				// BuildOptions\\DebugTreeGrammar.g3:59:9: ^( '=' ID expr )
				{
				dbg.Location( 59, 8 );
				dbg.Location( 59, 10 );
				Match(input,17,Follow._17_in_stat98); 

				Match(input, TokenConstants.DOWN, null); 
				dbg.Location( 59, 14 );
				ID2=(CommonTree)Match(input,ID,Follow._ID_in_stat100); 
				dbg.Location( 59, 17 );
				PushFollow(Follow._expr_in_stat102);
				expr3=expr();

				state._fsp--;


				Match(input, TokenConstants.UP, null); 
				dbg.Location( 59, 35 );
				 globalMemory[(ID2!=null?ID2.Text:null)] = expr3; 

				}
				break;
			case 3:
				dbg.EnterAlt( 3 );

				// BuildOptions\\DebugTreeGrammar.g3:60:9: ^( FUNC ( . )+ )
				{
				dbg.Location( 60, 8 );
				dbg.Location( 60, 10 );
				Match(input,FUNC,Follow._FUNC_in_stat128); 

				Match(input, TokenConstants.DOWN, null); 
				dbg.Location( 60, 15 );
				// BuildOptions\\DebugTreeGrammar.g3:60:16: ( . )+
				int cnt2=0;
				try
				{
					dbg.EnterSubRule( 2 );

				for ( ; ; )
				{
					int alt2=2;
					try
					{
						dbg.EnterDecision( 2 );

					int LA2_0 = input.LA(1);

					if ( ((LA2_0>=CALL && LA2_0<=17)) )
					{
						alt2=1;
					}
					else if ( (LA2_0==UP) )
					{
						alt2=2;
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

						// BuildOptions\\DebugTreeGrammar.g3:60:0: .
						{
						dbg.Location( 60, 15 );
						MatchAny(input); 

						}
						break;

					default:
						if ( cnt2 >= 1 )
							goto loop2;

						EarlyExitException eee2 = new EarlyExitException( 2, input );
						dbg.RecognitionException( eee2 );

						throw eee2;
					}
					cnt2++;
				}
				loop2:
					;

				}
				finally
				{
					dbg.ExitSubRule( 2 );
				}


				Match(input, TokenConstants.UP, null); 

				}
				break;

			}
		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		}
		finally
		{
		}
		dbg.Location(61, 4);

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

		return ;
	}
	// $ANTLR end "stat"


	// $ANTLR start "expr"
	// BuildOptions\\DebugTreeGrammar.g3:63:0: expr returns [BigInteger value] : ( ^( '+' a= expr b= expr ) | ^( '-' a= expr b= expr ) | ^( '*' a= expr b= expr ) | ^( '/' a= expr b= expr ) | ^( '%' a= expr b= expr ) | ID | INT | call );
	private BigInteger expr(  )
	{

		BigInteger value = default(BigInteger);

		CommonTree ID4=null;
		CommonTree INT5=null;
		BigInteger a = default(BigInteger);
		BigInteger b = default(BigInteger);
		BigInteger call6 = default(BigInteger);

		try
		{
			dbg.EnterRule( GrammarFileName, "expr" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 63, -1 );

		try
		{
			// BuildOptions\\DebugTreeGrammar.g3:64:9: ( ^( '+' a= expr b= expr ) | ^( '-' a= expr b= expr ) | ^( '*' a= expr b= expr ) | ^( '/' a= expr b= expr ) | ^( '%' a= expr b= expr ) | ID | INT | call )
			int alt4=8;
			try
			{
				dbg.EnterDecision( 4 );

			switch ( input.LA(1) )
			{
			case 16:
				{
				alt4=1;
				}
				break;
			case 10:
				{
				alt4=2;
				}
				break;
			case 14:
				{
				alt4=3;
				}
				break;
			case 15:
				{
				alt4=4;
				}
				break;
			case 11:
				{
				alt4=5;
				}
				break;
			case ID:
				{
				alt4=6;
				}
				break;
			case INT:
				{
				alt4=7;
				}
				break;
			case CALL:
				{
				alt4=8;
				}
				break;
			default:
				{
					NoViableAltException nvae = new NoViableAltException("", 4, 0, input);

					dbg.RecognitionException( nvae );
					throw nvae;
				}
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

				// BuildOptions\\DebugTreeGrammar.g3:64:9: ^( '+' a= expr b= expr )
				{
				dbg.Location( 64, 8 );
				dbg.Location( 64, 10 );
				Match(input,16,Follow._16_in_expr172); 

				Match(input, TokenConstants.DOWN, null); 
				dbg.Location( 64, 15 );
				PushFollow(Follow._expr_in_expr176);
				a=expr();

				state._fsp--;

				dbg.Location( 64, 22 );
				PushFollow(Follow._expr_in_expr180);
				b=expr();

				state._fsp--;


				Match(input, TokenConstants.UP, null); 
				dbg.Location( 64, 35 );
				 value = a.add(b); 

				}
				break;
			case 2:
				dbg.EnterAlt( 2 );

				// BuildOptions\\DebugTreeGrammar.g3:65:9: ^( '-' a= expr b= expr )
				{
				dbg.Location( 65, 8 );
				dbg.Location( 65, 10 );
				Match(input,10,Follow._10_in_expr200); 

				Match(input, TokenConstants.DOWN, null); 
				dbg.Location( 65, 15 );
				PushFollow(Follow._expr_in_expr204);
				a=expr();

				state._fsp--;

				dbg.Location( 65, 22 );
				PushFollow(Follow._expr_in_expr208);
				b=expr();

				state._fsp--;


				Match(input, TokenConstants.UP, null); 
				dbg.Location( 65, 35 );
				 value = a.subtract(b); 

				}
				break;
			case 3:
				dbg.EnterAlt( 3 );

				// BuildOptions\\DebugTreeGrammar.g3:66:9: ^( '*' a= expr b= expr )
				{
				dbg.Location( 66, 8 );
				dbg.Location( 66, 10 );
				Match(input,14,Follow._14_in_expr228); 

				Match(input, TokenConstants.DOWN, null); 
				dbg.Location( 66, 15 );
				PushFollow(Follow._expr_in_expr232);
				a=expr();

				state._fsp--;

				dbg.Location( 66, 22 );
				PushFollow(Follow._expr_in_expr236);
				b=expr();

				state._fsp--;


				Match(input, TokenConstants.UP, null); 
				dbg.Location( 66, 35 );
				 value = a.multiply(b); 

				}
				break;
			case 4:
				dbg.EnterAlt( 4 );

				// BuildOptions\\DebugTreeGrammar.g3:67:9: ^( '/' a= expr b= expr )
				{
				dbg.Location( 67, 8 );
				dbg.Location( 67, 10 );
				Match(input,15,Follow._15_in_expr256); 

				Match(input, TokenConstants.DOWN, null); 
				dbg.Location( 67, 15 );
				PushFollow(Follow._expr_in_expr260);
				a=expr();

				state._fsp--;

				dbg.Location( 67, 22 );
				PushFollow(Follow._expr_in_expr264);
				b=expr();

				state._fsp--;


				Match(input, TokenConstants.UP, null); 
				dbg.Location( 67, 35 );
				 value = a.divide(b); 

				}
				break;
			case 5:
				dbg.EnterAlt( 5 );

				// BuildOptions\\DebugTreeGrammar.g3:68:9: ^( '%' a= expr b= expr )
				{
				dbg.Location( 68, 8 );
				dbg.Location( 68, 10 );
				Match(input,11,Follow._11_in_expr284); 

				Match(input, TokenConstants.DOWN, null); 
				dbg.Location( 68, 15 );
				PushFollow(Follow._expr_in_expr288);
				a=expr();

				state._fsp--;

				dbg.Location( 68, 22 );
				PushFollow(Follow._expr_in_expr292);
				b=expr();

				state._fsp--;


				Match(input, TokenConstants.UP, null); 
				dbg.Location( 68, 35 );
				 value = a.remainder(b); 

				}
				break;
			case 6:
				dbg.EnterAlt( 6 );

				// BuildOptions\\DebugTreeGrammar.g3:69:9: ID
				{
				dbg.Location( 69, 8 );
				ID4=(CommonTree)Match(input,ID,Follow._ID_in_expr311); 
				dbg.Location( 69, 35 );
				 value = getValue((ID4!=null?ID4.Text:null)); 

				}
				break;
			case 7:
				dbg.EnterAlt( 7 );

				// BuildOptions\\DebugTreeGrammar.g3:70:9: INT
				{
				dbg.Location( 70, 8 );
				INT5=(CommonTree)Match(input,INT,Follow._INT_in_expr347); 
				dbg.Location( 70, 35 );
				 value = new BigInteger((INT5!=null?INT5.Text:null)); 

				}
				break;
			case 8:
				dbg.EnterAlt( 8 );

				// BuildOptions\\DebugTreeGrammar.g3:71:9: call
				{
				dbg.Location( 71, 8 );
				PushFollow(Follow._call_in_expr382);
				call6=call();

				state._fsp--;

				dbg.Location( 71, 35 );
				 value = call6; 

				}
				break;

			}
		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		}
		finally
		{
		}
		dbg.Location(72, 4);

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

		return value;
	}
	// $ANTLR end "expr"


	// $ANTLR start "call"
	// BuildOptions\\DebugTreeGrammar.g3:74:0: call returns [BigInteger value] : ^( CALL ID expr ) ;
	private BigInteger call(  )
	{

		BigInteger value = default(BigInteger);

		CommonTree ID8=null;
		BigInteger expr7 = default(BigInteger);

		try
		{
			dbg.EnterRule( GrammarFileName, "call" );
			if ( RuleLevel == 0 )
			{
				dbg.Commence();
			}
			IncRuleLevel();
			dbg.Location( 74, -1 );

		try
		{
			// BuildOptions\\DebugTreeGrammar.g3:75:9: ( ^( CALL ID expr ) )
			dbg.EnterAlt( 1 );

			// BuildOptions\\DebugTreeGrammar.g3:75:9: ^( CALL ID expr )
			{
			dbg.Location( 75, 8 );
			dbg.Location( 75, 10 );
			Match(input,CALL,Follow._CALL_in_call430); 

			Match(input, TokenConstants.DOWN, null); 
			dbg.Location( 75, 15 );
			ID8=(CommonTree)Match(input,ID,Follow._ID_in_call432); 
			dbg.Location( 75, 18 );
			PushFollow(Follow._expr_in_call434);
			expr7=expr();

			state._fsp--;


			Match(input, TokenConstants.UP, null); 
			dbg.Location( 75, 35 );
			 BigInteger p = expr7;
			                                     CommonTree funcRoot = findFunction((ID8!=null?ID8.Text:null), p);
			                                     if (funcRoot == null) {
			                                         Console.Error.WriteLine("No match found for " + (ID8!=null?ID8.Text:null) + "(" + p + ")");
			                                     } else {
			                                         // Here we set up the local evaluator to run over the
			                                         // function definition with the parameter value.
			                                         // This re-reads a sub-AST of our input AST!
			                                         DebugTreeGrammar e = new DebugTreeGrammar(funcRoot, functionDefinitions, globalMemory, p);
			                                         value = e.expr();
			                                     }
			                                   

			}

		}
		catch ( RecognitionException re )
		{
			ReportError(re);
			Recover(input,re);
		}
		finally
		{
		}
		dbg.Location(87, 4);

		}
		finally
		{
			dbg.ExitRule( GrammarFileName, "call" );
			DecRuleLevel();
			if ( RuleLevel == 0 )
			{
				dbg.Terminate();
			}
		}

		return value;
	}
	// $ANTLR end "call"
	#endregion

	// Delegated rules

	#region Synpreds
	#endregion

	#region DFA

	protected override void InitDFAs()
	{
		base.InitDFAs();
	}

	#endregion

	#region Follow Sets
	public static class Follow
	{
		public static readonly BitSet _stat_in_prog48 = new BitSet(new ulong[]{0x3CCF2UL});
		public static readonly BitSet _expr_in_stat63 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _17_in_stat98 = new BitSet(new ulong[]{0x4UL});
		public static readonly BitSet _ID_in_stat100 = new BitSet(new ulong[]{0x1CCD0UL});
		public static readonly BitSet _expr_in_stat102 = new BitSet(new ulong[]{0x8UL});
		public static readonly BitSet _FUNC_in_stat128 = new BitSet(new ulong[]{0x4UL});
		public static readonly BitSet _16_in_expr172 = new BitSet(new ulong[]{0x4UL});
		public static readonly BitSet _expr_in_expr176 = new BitSet(new ulong[]{0x1CCD0UL});
		public static readonly BitSet _expr_in_expr180 = new BitSet(new ulong[]{0x8UL});
		public static readonly BitSet _10_in_expr200 = new BitSet(new ulong[]{0x4UL});
		public static readonly BitSet _expr_in_expr204 = new BitSet(new ulong[]{0x1CCD0UL});
		public static readonly BitSet _expr_in_expr208 = new BitSet(new ulong[]{0x8UL});
		public static readonly BitSet _14_in_expr228 = new BitSet(new ulong[]{0x4UL});
		public static readonly BitSet _expr_in_expr232 = new BitSet(new ulong[]{0x1CCD0UL});
		public static readonly BitSet _expr_in_expr236 = new BitSet(new ulong[]{0x8UL});
		public static readonly BitSet _15_in_expr256 = new BitSet(new ulong[]{0x4UL});
		public static readonly BitSet _expr_in_expr260 = new BitSet(new ulong[]{0x1CCD0UL});
		public static readonly BitSet _expr_in_expr264 = new BitSet(new ulong[]{0x8UL});
		public static readonly BitSet _11_in_expr284 = new BitSet(new ulong[]{0x4UL});
		public static readonly BitSet _expr_in_expr288 = new BitSet(new ulong[]{0x1CCD0UL});
		public static readonly BitSet _expr_in_expr292 = new BitSet(new ulong[]{0x8UL});
		public static readonly BitSet _ID_in_expr311 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _INT_in_expr347 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _call_in_expr382 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _CALL_in_call430 = new BitSet(new ulong[]{0x4UL});
		public static readonly BitSet _ID_in_call432 = new BitSet(new ulong[]{0x1CCD0UL});
		public static readonly BitSet _expr_in_call434 = new BitSet(new ulong[]{0x8UL});

	}
	#endregion
}
