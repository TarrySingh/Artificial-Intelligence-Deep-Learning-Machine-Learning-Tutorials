// $ANTLR 3.1.2 BuildOptions\\ProfileGrammar.g3 2009-03-16 13:19:19

// The variable 'variable' is assigned but its value is never used.
#pragma warning disable 219
// Unreachable code detected.
#pragma warning disable 162


using System.Collections.Generic;
using Antlr.Runtime;
using Stack = System.Collections.Generic.Stack<object>;
using List = System.Collections.IList;
using ArrayList = System.Collections.Generic.List<object>;

public partial class ProfileGrammarLexer : Lexer
{
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

	public ProfileGrammarLexer() {}
	public ProfileGrammarLexer( ICharStream input )
		: this( input, new RecognizerSharedState() )
	{
	}
	public ProfileGrammarLexer( ICharStream input, RecognizerSharedState state )
		: base( input, state )
	{

	}
	public override string GrammarFileName { get { return "BuildOptions\\ProfileGrammar.g3"; } }

	// $ANTLR start "T__10"
	private void mT__10()
	{
		try
		{
			int _type = T__10;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// BuildOptions\\ProfileGrammar.g3:7:9: ( '-' )
			// BuildOptions\\ProfileGrammar.g3:7:9: '-'
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
	// $ANTLR end "T__10"

	// $ANTLR start "T__11"
	private void mT__11()
	{
		try
		{
			int _type = T__11;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// BuildOptions\\ProfileGrammar.g3:8:9: ( '%' )
			// BuildOptions\\ProfileGrammar.g3:8:9: '%'
			{
			Match('%'); 

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
			// BuildOptions\\ProfileGrammar.g3:9:9: ( '(' )
			// BuildOptions\\ProfileGrammar.g3:9:9: '('
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
	// $ANTLR end "T__12"

	// $ANTLR start "T__13"
	private void mT__13()
	{
		try
		{
			int _type = T__13;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// BuildOptions\\ProfileGrammar.g3:10:9: ( ')' )
			// BuildOptions\\ProfileGrammar.g3:10:9: ')'
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
	// $ANTLR end "T__13"

	// $ANTLR start "T__14"
	private void mT__14()
	{
		try
		{
			int _type = T__14;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// BuildOptions\\ProfileGrammar.g3:11:9: ( '*' )
			// BuildOptions\\ProfileGrammar.g3:11:9: '*'
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
	// $ANTLR end "T__14"

	// $ANTLR start "T__15"
	private void mT__15()
	{
		try
		{
			int _type = T__15;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// BuildOptions\\ProfileGrammar.g3:12:9: ( '/' )
			// BuildOptions\\ProfileGrammar.g3:12:9: '/'
			{
			Match('/'); 

			}

			state.type = _type;
			state.channel = _channel;
		}
		finally
		{
		}
	}
	// $ANTLR end "T__15"

	// $ANTLR start "T__16"
	private void mT__16()
	{
		try
		{
			int _type = T__16;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// BuildOptions\\ProfileGrammar.g3:13:9: ( '+' )
			// BuildOptions\\ProfileGrammar.g3:13:9: '+'
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
	// $ANTLR end "T__16"

	// $ANTLR start "T__17"
	private void mT__17()
	{
		try
		{
			int _type = T__17;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// BuildOptions\\ProfileGrammar.g3:14:9: ( '=' )
			// BuildOptions\\ProfileGrammar.g3:14:9: '='
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
	// $ANTLR end "T__17"

	// $ANTLR start "ID"
	private void mID()
	{
		try
		{
			int _type = ID;
			int _channel = DEFAULT_TOKEN_CHANNEL;
			// BuildOptions\\ProfileGrammar.g3:88:9: ( ( 'a' .. 'z' | 'A' .. 'Z' )+ )
			// BuildOptions\\ProfileGrammar.g3:88:9: ( 'a' .. 'z' | 'A' .. 'Z' )+
			{
			// BuildOptions\\ProfileGrammar.g3:88:9: ( 'a' .. 'z' | 'A' .. 'Z' )+
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
					// BuildOptions\\ProfileGrammar.g3:
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
			// BuildOptions\\ProfileGrammar.g3:91:9: ( ( '0' .. '9' )+ )
			// BuildOptions\\ProfileGrammar.g3:91:9: ( '0' .. '9' )+
			{
			// BuildOptions\\ProfileGrammar.g3:91:9: ( '0' .. '9' )+
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
					// BuildOptions\\ProfileGrammar.g3:
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
			// BuildOptions\\ProfileGrammar.g3:95:7: ( ( '\\r' )? '\\n' )
			// BuildOptions\\ProfileGrammar.g3:95:7: ( '\\r' )? '\\n'
			{
			// BuildOptions\\ProfileGrammar.g3:95:7: ( '\\r' )?
			int alt3=2;
			int LA3_0 = input.LA(1);

			if ( (LA3_0=='\r') )
			{
				alt3=1;
			}
			switch ( alt3 )
			{
			case 1:
				// BuildOptions\\ProfileGrammar.g3:95:0: '\\r'
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
			// BuildOptions\\ProfileGrammar.g3:98:9: ( ( ' ' | '\\t' )+ )
			// BuildOptions\\ProfileGrammar.g3:98:9: ( ' ' | '\\t' )+
			{
			// BuildOptions\\ProfileGrammar.g3:98:9: ( ' ' | '\\t' )+
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
					// BuildOptions\\ProfileGrammar.g3:
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
		// BuildOptions\\ProfileGrammar.g3:1:10: ( T__10 | T__11 | T__12 | T__13 | T__14 | T__15 | T__16 | T__17 | ID | INT | NEWLINE | WS )
		int alt5=12;
		switch ( input.LA(1) )
		{
		case '-':
			{
			alt5=1;
			}
			break;
		case '%':
			{
			alt5=2;
			}
			break;
		case '(':
			{
			alt5=3;
			}
			break;
		case ')':
			{
			alt5=4;
			}
			break;
		case '*':
			{
			alt5=5;
			}
			break;
		case '/':
			{
			alt5=6;
			}
			break;
		case '+':
			{
			alt5=7;
			}
			break;
		case '=':
			{
			alt5=8;
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
			alt5=9;
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
			alt5=10;
			}
			break;
		case '\n':
		case '\r':
			{
			alt5=11;
			}
			break;
		case '\t':
		case ' ':
			{
			alt5=12;
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
			// BuildOptions\\ProfileGrammar.g3:1:10: T__10
			{
			mT__10(); 

			}
			break;
		case 2:
			// BuildOptions\\ProfileGrammar.g3:1:16: T__11
			{
			mT__11(); 

			}
			break;
		case 3:
			// BuildOptions\\ProfileGrammar.g3:1:22: T__12
			{
			mT__12(); 

			}
			break;
		case 4:
			// BuildOptions\\ProfileGrammar.g3:1:28: T__13
			{
			mT__13(); 

			}
			break;
		case 5:
			// BuildOptions\\ProfileGrammar.g3:1:34: T__14
			{
			mT__14(); 

			}
			break;
		case 6:
			// BuildOptions\\ProfileGrammar.g3:1:40: T__15
			{
			mT__15(); 

			}
			break;
		case 7:
			// BuildOptions\\ProfileGrammar.g3:1:46: T__16
			{
			mT__16(); 

			}
			break;
		case 8:
			// BuildOptions\\ProfileGrammar.g3:1:52: T__17
			{
			mT__17(); 

			}
			break;
		case 9:
			// BuildOptions\\ProfileGrammar.g3:1:58: ID
			{
			mID(); 

			}
			break;
		case 10:
			// BuildOptions\\ProfileGrammar.g3:1:61: INT
			{
			mINT(); 

			}
			break;
		case 11:
			// BuildOptions\\ProfileGrammar.g3:1:65: NEWLINE
			{
			mNEWLINE(); 

			}
			break;
		case 12:
			// BuildOptions\\ProfileGrammar.g3:1:73: WS
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
