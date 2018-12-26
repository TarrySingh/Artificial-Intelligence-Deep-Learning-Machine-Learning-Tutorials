// $ANTLR 3.1.2 JavaCompat\\Expr.g3 2009-03-16 13:19:15

// The variable 'variable' is assigned but its value is never used.
#pragma warning disable 219
// Unreachable code detected.
#pragma warning disable 162


// 'member' is obsolete
#pragma warning disable 612

using Antlr.Runtime.JavaExtensions;
using HashMap = System.Collections.Generic.Dictionary<object,object>;
using Integer = java.lang.Integer;


using System.Collections.Generic;
using Antlr.Runtime;
using Stack = System.Collections.Generic.Stack<object>;
using List = System.Collections.IList;
using ArrayList = System.Collections.Generic.List<object>;

public partial class ExprParser : Parser
{
	public static readonly string[] tokenNames = new string[] {
		"<invalid>", "<EOR>", "<DOWN>", "<UP>", "ID", "INT", "NEWLINE", "WS", "'-'", "'('", "')'", "'*'", "'+'", "'='"
	};
	public const int EOF=-1;
	public const int T__8=8;
	public const int T__9=9;
	public const int T__10=10;
	public const int T__11=11;
	public const int T__12=12;
	public const int T__13=13;
	public const int ID=4;
	public const int INT=5;
	public const int NEWLINE=6;
	public const int WS=7;

	// delegates
	// delegators

	public ExprParser( ITokenStream input )
		: this( input, new RecognizerSharedState() )
	{
	}
	public ExprParser( ITokenStream input, RecognizerSharedState state )
		: base( input, state )
	{
	}
		

	public override string[] GetTokenNames() { return ExprParser.tokenNames; }
	public override string GrammarFileName { get { return "JavaCompat\\Expr.g3"; } }


	/** Map variable name to Integer object holding value */
	HashMap memory = new HashMap();


	#region Rules

	// $ANTLR start "prog"
	// JavaCompat\\Expr.g3:77:0: prog : ( stat )+ ;
	private void prog(  )
	{
		try
		{
			// JavaCompat\\Expr.g3:77:9: ( ( stat )+ )
			// JavaCompat\\Expr.g3:77:9: ( stat )+
			{
			// JavaCompat\\Expr.g3:77:9: ( stat )+
			int cnt1=0;
			for ( ; ; )
			{
				int alt1=2;
				int LA1_0 = input.LA(1);

				if ( ((LA1_0>=ID && LA1_0<=NEWLINE)||LA1_0==9) )
				{
					alt1=1;
				}


				switch ( alt1 )
				{
				case 1:
					// JavaCompat\\Expr.g3:77:0: stat
					{
					PushFollow(Follow._stat_in_prog40);
					stat();

					state._fsp--;


					}
					break;

				default:
					if ( cnt1 >= 1 )
						goto loop1;

					EarlyExitException eee1 = new EarlyExitException( 1, input );
					throw eee1;
				}
				cnt1++;
			}
			loop1:
				;



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
		return ;
	}
	// $ANTLR end "prog"


	// $ANTLR start "stat"
	// JavaCompat\\Expr.g3:79:0: stat : ( expr NEWLINE | ID '=' expr NEWLINE | NEWLINE );
	private void stat(  )
	{
		IToken ID2=null;
		int expr1 = default(int);
		int expr3 = default(int);

		try
		{
			// JavaCompat\\Expr.g3:79:9: ( expr NEWLINE | ID '=' expr NEWLINE | NEWLINE )
			int alt2=3;
			switch ( input.LA(1) )
			{
			case INT:
			case 9:
				{
				alt2=1;
				}
				break;
			case ID:
				{
				int LA2_2 = input.LA(2);

				if ( (LA2_2==13) )
				{
					alt2=2;
				}
				else if ( (LA2_2==NEWLINE||LA2_2==8||(LA2_2>=11 && LA2_2<=12)) )
				{
					alt2=1;
				}
				else
				{
					NoViableAltException nvae = new NoViableAltException("", 2, 2, input);

					throw nvae;
				}
				}
				break;
			case NEWLINE:
				{
				alt2=3;
				}
				break;
			default:
				{
					NoViableAltException nvae = new NoViableAltException("", 2, 0, input);

					throw nvae;
				}
			}

			switch ( alt2 )
			{
			case 1:
				// JavaCompat\\Expr.g3:79:9: expr NEWLINE
				{
				PushFollow(Follow._expr_in_stat51);
				expr1=expr();

				state._fsp--;

				Match(input,NEWLINE,Follow._NEWLINE_in_stat53); 
				JSystem.@out.println(expr1);

				}
				break;
			case 2:
				// JavaCompat\\Expr.g3:80:9: ID '=' expr NEWLINE
				{
				ID2=(IToken)Match(input,ID,Follow._ID_in_stat65); 
				Match(input,13,Follow._13_in_stat67); 
				PushFollow(Follow._expr_in_stat69);
				expr3=expr();

				state._fsp--;

				Match(input,NEWLINE,Follow._NEWLINE_in_stat71); 
				memory.put((ID2!=null?ID2.Text:null), new Integer(expr3));

				}
				break;
			case 3:
				// JavaCompat\\Expr.g3:82:9: NEWLINE
				{
				Match(input,NEWLINE,Follow._NEWLINE_in_stat91); 

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
		return ;
	}
	// $ANTLR end "stat"


	// $ANTLR start "expr"
	// JavaCompat\\Expr.g3:85:0: expr returns [int value] : e= multExpr ( '+' e= multExpr | '-' e= multExpr )* ;
	private int expr(  )
	{

		int value = default(int);

		int e = default(int);

		try
		{
			// JavaCompat\\Expr.g3:86:9: (e= multExpr ( '+' e= multExpr | '-' e= multExpr )* )
			// JavaCompat\\Expr.g3:86:9: e= multExpr ( '+' e= multExpr | '-' e= multExpr )*
			{
			PushFollow(Follow._multExpr_in_expr116);
			e=multExpr();

			state._fsp--;

			value = e;
			// JavaCompat\\Expr.g3:87:9: ( '+' e= multExpr | '-' e= multExpr )*
			for ( ; ; )
			{
				int alt3=3;
				int LA3_0 = input.LA(1);

				if ( (LA3_0==12) )
				{
					alt3=1;
				}
				else if ( (LA3_0==8) )
				{
					alt3=2;
				}


				switch ( alt3 )
				{
				case 1:
					// JavaCompat\\Expr.g3:87:13: '+' e= multExpr
					{
					Match(input,12,Follow._12_in_expr132); 
					PushFollow(Follow._multExpr_in_expr136);
					e=multExpr();

					state._fsp--;

					value += e;

					}
					break;
				case 2:
					// JavaCompat\\Expr.g3:88:13: '-' e= multExpr
					{
					Match(input,8,Follow._8_in_expr152); 
					PushFollow(Follow._multExpr_in_expr156);
					e=multExpr();

					state._fsp--;

					value -= e;

					}
					break;

				default:
					goto loop3;
				}
			}

			loop3:
				;



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
		return value;
	}
	// $ANTLR end "expr"


	// $ANTLR start "multExpr"
	// JavaCompat\\Expr.g3:92:0: multExpr returns [int value] : e= atom ( '*' e= atom )* ;
	private int multExpr(  )
	{

		int value = default(int);

		int e = default(int);

		try
		{
			// JavaCompat\\Expr.g3:93:9: (e= atom ( '*' e= atom )* )
			// JavaCompat\\Expr.g3:93:9: e= atom ( '*' e= atom )*
			{
			PushFollow(Follow._atom_in_multExpr194);
			e=atom();

			state._fsp--;

			value = e;
			// JavaCompat\\Expr.g3:93:37: ( '*' e= atom )*
			for ( ; ; )
			{
				int alt4=2;
				int LA4_0 = input.LA(1);

				if ( (LA4_0==11) )
				{
					alt4=1;
				}


				switch ( alt4 )
				{
				case 1:
					// JavaCompat\\Expr.g3:93:38: '*' e= atom
					{
					Match(input,11,Follow._11_in_multExpr199); 
					PushFollow(Follow._atom_in_multExpr203);
					e=atom();

					state._fsp--;

					value *= e;

					}
					break;

				default:
					goto loop4;
				}
			}

			loop4:
				;



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
		return value;
	}
	// $ANTLR end "multExpr"


	// $ANTLR start "atom"
	// JavaCompat\\Expr.g3:96:0: atom returns [int value] : ( INT | ID | '(' expr ')' );
	private int atom(  )
	{

		int value = default(int);

		IToken INT4=null;
		IToken ID5=null;
		int expr6 = default(int);

		try
		{
			// JavaCompat\\Expr.g3:97:9: ( INT | ID | '(' expr ')' )
			int alt5=3;
			switch ( input.LA(1) )
			{
			case INT:
				{
				alt5=1;
				}
				break;
			case ID:
				{
				alt5=2;
				}
				break;
			case 9:
				{
				alt5=3;
				}
				break;
			default:
				{
					NoViableAltException nvae = new NoViableAltException("", 5, 0, input);

					throw nvae;
				}
			}

			switch ( alt5 )
			{
			case 1:
				// JavaCompat\\Expr.g3:97:9: INT
				{
				INT4=(IToken)Match(input,INT,Follow._INT_in_atom231); 
				value = Integer.parseInt((INT4!=null?INT4.Text:null));

				}
				break;
			case 2:
				// JavaCompat\\Expr.g3:98:9: ID
				{
				ID5=(IToken)Match(input,ID,Follow._ID_in_atom243); 

				        Integer v = (Integer)memory.get((ID5!=null?ID5.Text:null));
				        if ( v!=null ) value = v.intValue();
				        else JSystem.err.println("undefined variable "+(ID5!=null?ID5.Text:null));
				        

				}
				break;
			case 3:
				// JavaCompat\\Expr.g3:104:9: '(' expr ')'
				{
				Match(input,9,Follow._9_in_atom263); 
				PushFollow(Follow._expr_in_atom265);
				expr6=expr();

				state._fsp--;

				Match(input,10,Follow._10_in_atom267); 
				value = expr6;

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
		return value;
	}
	// $ANTLR end "atom"
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
		public static readonly BitSet _stat_in_prog40 = new BitSet(new ulong[]{0x272UL});
		public static readonly BitSet _expr_in_stat51 = new BitSet(new ulong[]{0x40UL});
		public static readonly BitSet _NEWLINE_in_stat53 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _ID_in_stat65 = new BitSet(new ulong[]{0x2000UL});
		public static readonly BitSet _13_in_stat67 = new BitSet(new ulong[]{0x230UL});
		public static readonly BitSet _expr_in_stat69 = new BitSet(new ulong[]{0x40UL});
		public static readonly BitSet _NEWLINE_in_stat71 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _NEWLINE_in_stat91 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _multExpr_in_expr116 = new BitSet(new ulong[]{0x1102UL});
		public static readonly BitSet _12_in_expr132 = new BitSet(new ulong[]{0x230UL});
		public static readonly BitSet _multExpr_in_expr136 = new BitSet(new ulong[]{0x1102UL});
		public static readonly BitSet _8_in_expr152 = new BitSet(new ulong[]{0x230UL});
		public static readonly BitSet _multExpr_in_expr156 = new BitSet(new ulong[]{0x1102UL});
		public static readonly BitSet _atom_in_multExpr194 = new BitSet(new ulong[]{0x802UL});
		public static readonly BitSet _11_in_multExpr199 = new BitSet(new ulong[]{0x230UL});
		public static readonly BitSet _atom_in_multExpr203 = new BitSet(new ulong[]{0x802UL});
		public static readonly BitSet _INT_in_atom231 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _ID_in_atom243 = new BitSet(new ulong[]{0x2UL});
		public static readonly BitSet _9_in_atom263 = new BitSet(new ulong[]{0x230UL});
		public static readonly BitSet _expr_in_atom265 = new BitSet(new ulong[]{0x400UL});
		public static readonly BitSet _10_in_atom267 = new BitSet(new ulong[]{0x2UL});

	}
	#endregion
}
