// $ANTLR 3.1b1 SimpleC.g 2007-12-16 20:47:43

#import "SimpleCLexer.h"

/** As per Terence: No returns for lexer rules!
#pragma mark Rule return scopes start
#pragma mark Rule return scopes end
*/
@implementation SimpleCLexer

static NSArray *tokenNames;


+ (void) initialize
{
    // todo: get tokenNames into lexer - requires changes to CodeGenerator.java and ANTLRCore.sti
    tokenNames = [[NSArray alloc] init];
}

- (id) initWithCharStream:(id<ANTLRCharStream>)anInput
{
	if (nil!=(self = [super initWithCharStream:anInput])) {
	}
	return self;
}

- (void) dealloc
{
	[super dealloc];
}

+ (NSString *) tokenNameForType:(int)aTokenType
{
    return nil;
}

+ (NSArray *) tokenNames
{
    return tokenNames;
}

- (NSString *) grammarFileName
{
	return @"SimpleC.g";
}


- (void) mT21
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_T21;
        // SimpleC.g:7:5: ( ';' ) // ruleBlockSingleAlt
        // SimpleC.g:7:7: ';' // alt
        {
        [self matchChar:';'];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end T21


- (void) mT22
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_T22;
        // SimpleC.g:8:5: ( '(' ) // ruleBlockSingleAlt
        // SimpleC.g:8:7: '(' // alt
        {
        [self matchChar:'('];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end T22


- (void) mT23
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_T23;
        // SimpleC.g:9:5: ( ',' ) // ruleBlockSingleAlt
        // SimpleC.g:9:7: ',' // alt
        {
        [self matchChar:','];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end T23


- (void) mT24
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_T24;
        // SimpleC.g:10:5: ( ')' ) // ruleBlockSingleAlt
        // SimpleC.g:10:7: ')' // alt
        {
        [self matchChar:')'];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end T24


- (void) mT25
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_T25;
        // SimpleC.g:11:5: ( '{' ) // ruleBlockSingleAlt
        // SimpleC.g:11:7: '{' // alt
        {
        [self matchChar:'{'];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end T25


- (void) mT26
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_T26;
        // SimpleC.g:12:5: ( '}' ) // ruleBlockSingleAlt
        // SimpleC.g:12:7: '}' // alt
        {
        [self matchChar:'}'];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end T26


- (void) mFOR
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_FOR;
        // SimpleC.g:91:5: ( 'for' ) // ruleBlockSingleAlt
        // SimpleC.g:91:7: 'for' // alt
        {
        [self matchString:@"for"];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end FOR


- (void) mINT_TYPE
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_INT_TYPE;
        // SimpleC.g:92:10: ( 'int' ) // ruleBlockSingleAlt
        // SimpleC.g:92:12: 'int' // alt
        {
        [self matchString:@"int"];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end INT_TYPE


- (void) mCHAR
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_CHAR;
        // SimpleC.g:93:5: ( 'char' ) // ruleBlockSingleAlt
        // SimpleC.g:93:7: 'char' // alt
        {
        [self matchString:@"char"];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end CHAR


- (void) mVOID
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_VOID;
        // SimpleC.g:94:5: ( 'void' ) // ruleBlockSingleAlt
        // SimpleC.g:94:7: 'void' // alt
        {
        [self matchString:@"void"];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end VOID


- (void) mID
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_ID;
        // SimpleC.g:96:5: ( ( 'a' .. 'z' | 'A' .. 'Z' | '_' ) ( 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' )* ) // ruleBlockSingleAlt
        // SimpleC.g:96:9: ( 'a' .. 'z' | 'A' .. 'Z' | '_' ) ( 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' )* // alt
        {
        if (([input LA:1]>='A' && [input LA:1]<='Z')||[input LA:1]=='_'||([input LA:1]>='a' && [input LA:1]<='z')) {
        	[input consume];

        } else {
        	ANTLRMismatchedSetException *mse = [ANTLRMismatchedSetException exceptionWithSet:nil stream:input];
        	[self recover:mse];	@throw mse;
        }

        do {
            int alt1=2;
            {
            	int LA1_0 = [input LA:1];
            	if ( (LA1_0>='0' && LA1_0<='9')||(LA1_0>='A' && LA1_0<='Z')||LA1_0=='_'||(LA1_0>='a' && LA1_0<='z') ) {
            		alt1 = 1;
            	}

            }
            switch (alt1) {
        	case 1 :
        	    // SimpleC.g: // alt
        	    {
        	    if (([input LA:1]>='0' && [input LA:1]<='9')||([input LA:1]>='A' && [input LA:1]<='Z')||[input LA:1]=='_'||([input LA:1]>='a' && [input LA:1]<='z')) {
        	    	[input consume];

        	    } else {
        	    	ANTLRMismatchedSetException *mse = [ANTLRMismatchedSetException exceptionWithSet:nil stream:input];
        	    	[self recover:mse];	@throw mse;
        	    }


        	    }
        	    break;

        	default :
        	    goto loop1;
            }
        } while (YES); loop1: ;


        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end ID


- (void) mINT
{
    ANTLRLexerState *_state = [self state];
    id<ANTLRToken>  _int = nil;
    NSMutableArray *_list_int = nil;

    @try {
        int _type = SimpleCLexer_INT;
        // SimpleC.g:99:5: ( (int+= ( '0' .. '9' ) )+ ) // ruleBlockSingleAlt
        // SimpleC.g:99:7: (int+= ( '0' .. '9' ) )+ // alt
        {
        // SimpleC.g:99:10: (int+= ( '0' .. '9' ) )+	// positiveClosureBlock
        int cnt2=0;

        do {
            int alt2=2;
            {
            	int LA2_0 = [input LA:1];
            	if ( (LA2_0>='0' && LA2_0<='9') ) {
            		alt2 = 1;
            	}

            }
            switch (alt2) {
        	case 1 :
        	    // SimpleC.g:99:10: int+= ( '0' .. '9' ) // alt
        	    {
        	    // SimpleC.g:99:12: ( '0' .. '9' ) // blockSingleAlt
        	    // SimpleC.g:99:13: '0' .. '9' // alt
        	    {
        	    [self matchRangeFromChar:'0' to:'9'];

        	    }


        	    }
        	    break;

        	default :
        	    if ( cnt2 >= 1 )  goto loop2;
        			ANTLREarlyExitException *eee = [ANTLREarlyExitException exceptionWithStream:input decisionNumber:2];
        			@throw eee;
            }
            cnt2++;
        } while (YES); loop2: ;

        NSLog(@"%@", _list_int);

        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels
        [_list_int release];

    }
    return;
}
// $ANTLR end INT


- (void) mEQ
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_EQ;
        // SimpleC.g:102:6: ( '=' ) // ruleBlockSingleAlt
        // SimpleC.g:102:8: '=' // alt
        {
        [self matchChar:'='];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end EQ


- (void) mEQEQ
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_EQEQ;
        // SimpleC.g:103:6: ( '==' ) // ruleBlockSingleAlt
        // SimpleC.g:103:8: '==' // alt
        {
        [self matchString:@"=="];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end EQEQ


- (void) mLT
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_LT;
        // SimpleC.g:104:6: ( '<' ) // ruleBlockSingleAlt
        // SimpleC.g:104:8: '<' // alt
        {
        [self matchChar:'<'];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end LT


- (void) mPLUS
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_PLUS;
        // SimpleC.g:105:6: ( '+' ) // ruleBlockSingleAlt
        // SimpleC.g:105:8: '+' // alt
        {
        [self matchChar:'+'];



        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end PLUS


- (void) mWS
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = SimpleCLexer_WS;
        // SimpleC.g:107:5: ( ( ' ' | '\\t' | '\\r' | '\\n' )+ ) // ruleBlockSingleAlt
        // SimpleC.g:107:9: ( ' ' | '\\t' | '\\r' | '\\n' )+ // alt
        {
        // SimpleC.g:107:9: ( ' ' | '\\t' | '\\r' | '\\n' )+	// positiveClosureBlock
        int cnt3=0;

        do {
            int alt3=2;
            {
            	int LA3_0 = [input LA:1];
            	if ( (LA3_0>='\t' && LA3_0<='\n')||LA3_0=='\r'||LA3_0==' ' ) {
            		alt3 = 1;
            	}

            }
            switch (alt3) {
        	case 1 :
        	    // SimpleC.g: // alt
        	    {
        	    if (([input LA:1]>='\t' && [input LA:1]<='\n')||[input LA:1]=='\r'||[input LA:1]==' ') {
        	    	[input consume];

        	    } else {
        	    	ANTLRMismatchedSetException *mse = [ANTLRMismatchedSetException exceptionWithSet:nil stream:input];
        	    	[self recover:mse];	@throw mse;
        	    }


        	    }
        	    break;

        	default :
        	    if ( cnt3 >= 1 )  goto loop3;
        			ANTLREarlyExitException *eee = [ANTLREarlyExitException exceptionWithStream:input decisionNumber:3];
        			@throw eee;
            }
            cnt3++;
        } while (YES); loop3: ;

         [_state setChannel:99]; 

        }

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end WS

- (void) mTokens
{
    // SimpleC.g:1:8: ( T21 | T22 | T23 | T24 | T25 | T26 | FOR | INT_TYPE | CHAR | VOID | ID | INT | EQ | EQEQ | LT | PLUS | WS ) //ruleblock
    int alt4=17;
    switch ([input LA:1]) {
    	case ';':
    		alt4 = 1;
    		break;
    	case '(':
    		alt4 = 2;
    		break;
    	case ',':
    		alt4 = 3;
    		break;
    	case ')':
    		alt4 = 4;
    		break;
    	case '{':
    		alt4 = 5;
    		break;
    	case '}':
    		alt4 = 6;
    		break;
    	case 'f':
    		{
    			int LA4_7 = [input LA:2];
    			if ( LA4_7=='o' ) {
    				{
    					int LA4_17 = [input LA:3];
    					if ( LA4_17=='r' ) {
    						{
    							int LA4_23 = [input LA:4];
    							if ( (LA4_23>='0' && LA4_23<='9')||(LA4_23>='A' && LA4_23<='Z')||LA4_23=='_'||(LA4_23>='a' && LA4_23<='z') ) {
    								alt4 = 11;
    							}
    						else {
    							alt4 = 7;	}
    						}
    					}
    				else {
    					alt4 = 11;	}
    				}
    			}
    		else {
    			alt4 = 11;	}
    		}
    		break;
    	case 'i':
    		{
    			int LA4_8 = [input LA:2];
    			if ( LA4_8=='n' ) {
    				{
    					int LA4_18 = [input LA:3];
    					if ( LA4_18=='t' ) {
    						{
    							int LA4_24 = [input LA:4];
    							if ( (LA4_24>='0' && LA4_24<='9')||(LA4_24>='A' && LA4_24<='Z')||LA4_24=='_'||(LA4_24>='a' && LA4_24<='z') ) {
    								alt4 = 11;
    							}
    						else {
    							alt4 = 8;	}
    						}
    					}
    				else {
    					alt4 = 11;	}
    				}
    			}
    		else {
    			alt4 = 11;	}
    		}
    		break;
    	case 'c':
    		{
    			int LA4_9 = [input LA:2];
    			if ( LA4_9=='h' ) {
    				{
    					int LA4_19 = [input LA:3];
    					if ( LA4_19=='a' ) {
    						{
    							int LA4_25 = [input LA:4];
    							if ( LA4_25=='r' ) {
    								{
    									int LA4_29 = [input LA:5];
    									if ( (LA4_29>='0' && LA4_29<='9')||(LA4_29>='A' && LA4_29<='Z')||LA4_29=='_'||(LA4_29>='a' && LA4_29<='z') ) {
    										alt4 = 11;
    									}
    								else {
    									alt4 = 9;	}
    								}
    							}
    						else {
    							alt4 = 11;	}
    						}
    					}
    				else {
    					alt4 = 11;	}
    				}
    			}
    		else {
    			alt4 = 11;	}
    		}
    		break;
    	case 'v':
    		{
    			int LA4_10 = [input LA:2];
    			if ( LA4_10=='o' ) {
    				{
    					int LA4_20 = [input LA:3];
    					if ( LA4_20=='i' ) {
    						{
    							int LA4_26 = [input LA:4];
    							if ( LA4_26=='d' ) {
    								{
    									int LA4_30 = [input LA:5];
    									if ( (LA4_30>='0' && LA4_30<='9')||(LA4_30>='A' && LA4_30<='Z')||LA4_30=='_'||(LA4_30>='a' && LA4_30<='z') ) {
    										alt4 = 11;
    									}
    								else {
    									alt4 = 10;	}
    								}
    							}
    						else {
    							alt4 = 11;	}
    						}
    					}
    				else {
    					alt4 = 11;	}
    				}
    			}
    		else {
    			alt4 = 11;	}
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
    	case '_':
    	case 'a':
    	case 'b':
    	case 'd':
    	case 'e':
    	case 'g':
    	case 'h':
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
    	case 'w':
    	case 'x':
    	case 'y':
    	case 'z':
    		alt4 = 11;
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
    		alt4 = 12;
    		break;
    	case '=':
    		{
    			int LA4_13 = [input LA:2];
    			if ( LA4_13=='=' ) {
    				alt4 = 14;
    			}
    		else {
    			alt4 = 13;	}
    		}
    		break;
    	case '<':
    		alt4 = 15;
    		break;
    	case '+':
    		alt4 = 16;
    		break;
    	case '\t':
    	case '\n':
    	case '\r':
    	case ' ':
    		alt4 = 17;
    		break;
    default:
     {
        ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:4 state:0 stream:input];
    	@throw nvae;

    	}}
    switch (alt4) {
    	case 1 :
    	    // SimpleC.g:1:10: T21 // alt
    	    {
    	    [self mT21];



    	    }
    	    break;
    	case 2 :
    	    // SimpleC.g:1:14: T22 // alt
    	    {
    	    [self mT22];



    	    }
    	    break;
    	case 3 :
    	    // SimpleC.g:1:18: T23 // alt
    	    {
    	    [self mT23];



    	    }
    	    break;
    	case 4 :
    	    // SimpleC.g:1:22: T24 // alt
    	    {
    	    [self mT24];



    	    }
    	    break;
    	case 5 :
    	    // SimpleC.g:1:26: T25 // alt
    	    {
    	    [self mT25];



    	    }
    	    break;
    	case 6 :
    	    // SimpleC.g:1:30: T26 // alt
    	    {
    	    [self mT26];



    	    }
    	    break;
    	case 7 :
    	    // SimpleC.g:1:34: FOR // alt
    	    {
    	    [self mFOR];



    	    }
    	    break;
    	case 8 :
    	    // SimpleC.g:1:38: INT_TYPE // alt
    	    {
    	    [self mINT_TYPE];



    	    }
    	    break;
    	case 9 :
    	    // SimpleC.g:1:47: CHAR // alt
    	    {
    	    [self mCHAR];



    	    }
    	    break;
    	case 10 :
    	    // SimpleC.g:1:52: VOID // alt
    	    {
    	    [self mVOID];



    	    }
    	    break;
    	case 11 :
    	    // SimpleC.g:1:57: ID // alt
    	    {
    	    [self mID];



    	    }
    	    break;
    	case 12 :
    	    // SimpleC.g:1:60: INT // alt
    	    {
    	    [self mINT];



    	    }
    	    break;
    	case 13 :
    	    // SimpleC.g:1:64: EQ // alt
    	    {
    	    [self mEQ];



    	    }
    	    break;
    	case 14 :
    	    // SimpleC.g:1:67: EQEQ // alt
    	    {
    	    [self mEQEQ];



    	    }
    	    break;
    	case 15 :
    	    // SimpleC.g:1:72: LT // alt
    	    {
    	    [self mLT];



    	    }
    	    break;
    	case 16 :
    	    // SimpleC.g:1:75: PLUS // alt
    	    {
    	    [self mPLUS];



    	    }
    	    break;
    	case 17 :
    	    // SimpleC.g:1:80: WS // alt
    	    {
    	    [self mWS];



    	    }
    	    break;

    }

}

@end