// $ANTLR 3.1.2 JavaCompat\\Expr.g3 2009-03-16 13:19:15

// The variable 'variable' is assigned but its value is never used.
#pragma warning disable 219
// Unreachable code detected.
#pragma warning disable 162


using System.Collections.Generic;
using Antlr.Runtime;
using Stack = System.Collections.Generic.Stack<object>;
using List = System.Collections.IList;
using ArrayList = System.Collections.Generic.List<object>;

public partial class ExprLexer : Lexer
{
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

	public ExprLexer() {}
	public ExprLexer( ICharStream input )
		: this( input, new RecognizerSharedState() )
	{
	}
	public ExprLexer( ICharStream input, RecognizerSharedState state )
		: base( input, state )
	{

	}
	public override string GrammarFileName { get { return "JavaCompat\\Expr.g3"; } }

	// $ANTLR start "T__8"
	private void mT__8()
	{
		try
		{
			int _type = T__8;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:7:8: ( '-' )
			// JavaCompat\\Expr.g3:7:8: '-'
			{
			Match('-'); 

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "T__8"

	// $ANTLR start "T__9"
	private void mT__9()
	{
		try
		{
			int _type = T__9;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:8:8: ( '(' )
			// JavaCompat\\Expr.g3:8:8: '('
			{
			Match('('); 

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "T__9"

	// $ANTLR start "T__10"
	private void mT__10()
	{
		try
		{
			int _type = T__10;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:9:9: ( ')' )
			// JavaCompat\\Expr.g3:9:9: ')'
			{
			Match(')'); 

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "T__10"

	// $ANTLR start "T__11"
	private void mT__11()
	{
		try
		{
			int _type = T__11;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:10:9: ( '*' )
			// JavaCompat\\Expr.g3:10:9: '*'
			{
			Match('*'); 

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "T__11"

	// $ANTLR start "T__12"
	private void mT__12()
	{
		try
		{
			int _type = T__12;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:11:9: ( '+' )
			// JavaCompat\\Expr.g3:11:9: '+'
			{
			Match('+'); 

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "T__12"

	// $ANTLR start "T__13"
	private void mT__13()
	{
		try
		{
			int _type = T__13;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:12:9: ( '=' )
			// JavaCompat\\Expr.g3:12:9: '='
			{
			Match('='); 

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "T__13"

	// $ANTLR start "ID"
	private void mID()
	{
		try
		{
			int _type = ID;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:107:9: ( ( 'a' .. 'z' | 'A' .. 'Z' )+ )
			// JavaCompat\\Expr.g3:107:9: ( 'a' .. 'z' | 'A' .. 'Z' )+
			{
			// JavaCompat\\Expr.g3:107:9: ( 'a' .. 'z' | 'A' .. 'Z' )+
			int cnt1=0;
			for ( ; ; )
			{
				int alt1=2;
				int LA1_0 = input.LA(1);

				if ( ((LA1_0>='A' && LA1_0<='Z')||(LA1_0>='a' && LA1_0<='z')) )
				{
					alt1=1;
				}


				switch ( alt1 )
				{
				case 1:
					// JavaCompat\\Expr.g3:
					{
					input.Consume();


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

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "ID"

	// $ANTLR start "INT"
	private void mINT()
	{
		try
		{
			int _type = INT;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:108:9: ( ( '0' .. '9' )+ )
			// JavaCompat\\Expr.g3:108:9: ( '0' .. '9' )+
			{
			// JavaCompat\\Expr.g3:108:9: ( '0' .. '9' )+
			int cnt2=0;
			for ( ; ; )
			{
				int alt2=2;
				int LA2_0 = input.LA(1);

				if ( ((LA2_0>='0' && LA2_0<='9')) )
				{
					alt2=1;
				}


				switch ( alt2 )
				{
				case 1:
					// JavaCompat\\Expr.g3:
					{
					input.Consume();


					}
					break;

				default:
					if ( cnt2 >= 1 )
						goto loop2;

					EarlyExitException eee2 = new EarlyExitException( 2, input );
					throw eee2;
				}
				cnt2++;
			}
			loop2:
				;



			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "INT"

	// $ANTLR start "NEWLINE"
	private void mNEWLINE()
	{
		try
		{
			int _type = NEWLINE;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:109:9: ( ( '\\r' )? '\\n' )
			// JavaCompat\\Expr.g3:109:9: ( '\\r' )? '\\n'
			{
			// JavaCompat\\Expr.g3:109:9: ( '\\r' )?
			int alt3=2;
			int LA3_0 = input.LA(1);

			if ( (LA3_0=='\r') )
			{
				alt3=1;
			}
			switch ( alt3 )
			{
			case 1:
				// JavaCompat\\Expr.g3:109:0: '\\r'
				{
				Match('\r'); 

				}
				break;

			}

			Match('\n'); 

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "NEWLINE"

	// $ANTLR start "WS"
	private void mWS()
	{
		try
		{
			int _type = WS;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// JavaCompat\\Expr.g3:110:9: ( ( ' ' | '\\t' )+ )
			// JavaCompat\\Expr.g3:110:9: ( ' ' | '\\t' )+
			{
			// JavaCompat\\Expr.g3:110:9: ( ' ' | '\\t' )+
			int cnt4=0;
			for ( ; ; )
			{
				int alt4=2;
				int LA4_0 = input.LA(1);

				if ( (LA4_0=='\t'||LA4_0==' ') )
				{
					alt4=1;
				}


				switch ( alt4 )
				{
				case 1:
					// JavaCompat\\Expr.g3:
					{
					input.Consume();


					}
					break;

				default:
					if ( cnt4 >= 1 )
						goto loop4;

					EarlyExitException eee4 = new EarlyExitException( 4, input );
					throw eee4;
				}
				cnt4++;
			}
			loop4:
				;


			Skip();

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "WS"

	public override void mTokens()
	{
		// JavaCompat\\Expr.g3:1:10: ( T__8 | T__9 | T__10 | T__11 | T__12 | T__13 | ID | INT | NEWLINE | WS )
		int alt5=10;
		switch ( input.LA(1) )
		{
		case '-':
			{
			alt5=1;
			}
			break;
		case '(':
			{
			alt5=2;
			}
			break;
		case ')':
			{
			alt5=3;
			}
			break;
		case '*':
			{
			alt5=4;
			}
			break;
		case '+':
			{
			alt5=5;
			}
			break;
		case '=':
			{
			alt5=6;
			}
			break;
		case 'A':
		case 'B':
		case 'C':
		case 'D':
		case 'E':
		case 'F':
		case 'G':
		case 'H':
		case 'I':
		case 'J':
		case 'K':
		case 'L':
		case 'M':
		case 'N':
		case 'O':
		case 'P':
		case 'Q':
		case 'R':
		case 'S':
		case 'T':
		case 'U':
		case 'V':
		case 'W':
		case 'X':
		case 'Y':
		case 'Z':
		case 'a':
		case 'b':
		case 'c':
		case 'd':
		case 'e':
		case 'f':
		case 'g':
		case 'h':
		case 'i':
		case 'j':
		case 'k':
		case 'l':
		case 'm':
		case 'n':
		case 'o':
		case 'p':
		case 'q':
		case 'r':
		case 's':
		case 't':
		case 'u':
		case 'v':
		case 'w':
		case 'x':
		case 'y':
		case 'z':
			{
			alt5=7;
			}
			break;
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			{
			alt5=8;
			}
			break;
		case '\n':
		case '\r':
			{
			alt5=9;
			}
			break;
		case '\t':
		case ' ':
			{
			alt5=10;
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
			// JavaCompat\\Expr.g3:1:10: T__8
			{
			mT__8(); 

			}
			break;
		case 2:
			// JavaCompat\\Expr.g3:1:15: T__9
			{
			mT__9(); 

			}
			break;
		case 3:
			// JavaCompat\\Expr.g3:1:20: T__10
			{
			mT__10(); 

			}
			break;
		case 4:
			// JavaCompat\\Expr.g3:1:26: T__11
			{
			mT__11(); 

			}
			break;
		case 5:
			// JavaCompat\\Expr.g3:1:32: T__12
			{
			mT__12(); 

			}
			break;
		case 6:
			// JavaCompat\\Expr.g3:1:38: T__13
			{
			mT__13(); 

			}
			break;
		case 7:
			// JavaCompat\\Expr.g3:1:44: ID
			{
			mID(); 

			}
			break;
		case 8:
			// JavaCompat\\Expr.g3:1:47: INT
			{
			mINT(); 

			}
			break;
		case 9:
			// JavaCompat\\Expr.g3:1:51: NEWLINE
			{
			mNEWLINE(); 

			}
			break;
		case 10:
			// JavaCompat\\Expr.g3:1:59: WS
			{
			mWS(); 

			}
			break;

		}

	}


	#region DFA

	protected override void InitDFAs()
	{
		base.InitDFAs();
	}

 
	#endregion

}
