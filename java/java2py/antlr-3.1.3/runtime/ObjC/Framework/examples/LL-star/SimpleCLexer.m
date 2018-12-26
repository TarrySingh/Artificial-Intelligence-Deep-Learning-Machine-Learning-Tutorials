// $ANTLR 3.0 SimpleC.g 2007-07-25 20:12:42

#import "SimpleCLexer.h"

/** As per Terence: No returns for lexer rules!
#pragma mark Rule return scopes start
#pragma mark Rule return scopes end
*/
@implementation SimpleCLexer


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

- (NSString *) grammarFileName
{
	return @"SimpleC.g";
}


- (void) mT7
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T7;
        // SimpleC.g:7:6: ( ';' ) // ruleBlockSingleAlt
        // SimpleC.g:7:6: ';' // alt
        {
        [self matchChar:';'];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T7


- (void) mT8
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T8;
        // SimpleC.g:8:6: ( '(' ) // ruleBlockSingleAlt
        // SimpleC.g:8:6: '(' // alt
        {
        [self matchChar:'('];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T8


- (void) mT9
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T9;
        // SimpleC.g:9:6: ( ',' ) // ruleBlockSingleAlt
        // SimpleC.g:9:6: ',' // alt
        {
        [self matchChar:','];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T9


- (void) mT10
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T10;
        // SimpleC.g:10:7: ( ')' ) // ruleBlockSingleAlt
        // SimpleC.g:10:7: ')' // alt
        {
        [self matchChar:')'];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T10


- (void) mT11
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T11;
        // SimpleC.g:11:7: ( 'int' ) // ruleBlockSingleAlt
        // SimpleC.g:11:7: 'int' // alt
        {
        [self matchString:@"int"];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T11


- (void) mT12
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T12;
        // SimpleC.g:12:7: ( 'char' ) // ruleBlockSingleAlt
        // SimpleC.g:12:7: 'char' // alt
        {
        [self matchString:@"char"];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T12


- (void) mT13
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T13;
        // SimpleC.g:13:7: ( 'void' ) // ruleBlockSingleAlt
        // SimpleC.g:13:7: 'void' // alt
        {
        [self matchString:@"void"];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T13


- (void) mT14
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T14;
        // SimpleC.g:14:7: ( '{' ) // ruleBlockSingleAlt
        // SimpleC.g:14:7: '{' // alt
        {
        [self matchChar:'{'];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T14


- (void) mT15
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T15;
        // SimpleC.g:15:7: ( '}' ) // ruleBlockSingleAlt
        // SimpleC.g:15:7: '}' // alt
        {
        [self matchChar:'}'];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T15


- (void) mT16
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T16;
        // SimpleC.g:16:7: ( 'for' ) // ruleBlockSingleAlt
        // SimpleC.g:16:7: 'for' // alt
        {
        [self matchString:@"for"];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T16


- (void) mT17
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T17;
        // SimpleC.g:17:7: ( '=' ) // ruleBlockSingleAlt
        // SimpleC.g:17:7: '=' // alt
        {
        [self matchChar:'='];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T17


- (void) mT18
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T18;
        // SimpleC.g:18:7: ( '==' ) // ruleBlockSingleAlt
        // SimpleC.g:18:7: '==' // alt
        {
        [self matchString:@"=="];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T18


- (void) mT19
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T19;
        // SimpleC.g:19:7: ( '<' ) // ruleBlockSingleAlt
        // SimpleC.g:19:7: '<' // alt
        {
        [self matchChar:'<'];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T19


- (void) mT20
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_T20;
        // SimpleC.g:20:7: ( '+' ) // ruleBlockSingleAlt
        // SimpleC.g:20:7: '+' // alt
        {
        [self matchChar:'+'];



        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end T20


- (void) mID
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_ID;
        // SimpleC.g:94:9: ( ( 'a' .. 'z' | 'A' .. 'Z' | '_' ) ( 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' )* ) // ruleBlockSingleAlt
        // SimpleC.g:94:9: ( 'a' .. 'z' | 'A' .. 'Z' | '_' ) ( 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' )* // alt
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

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end ID


- (void) mINT
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_INT;
        // SimpleC.g:97:7: ( ( '0' .. '9' )+ ) // ruleBlockSingleAlt
        // SimpleC.g:97:7: ( '0' .. '9' )+ // alt
        {
        // SimpleC.g:97:7: ( '0' .. '9' )+	// positiveClosureBlock
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
        	    // SimpleC.g:97:8: '0' .. '9' // alt
        	    {
        	    [self matchRangeFromChar:'0' to:'9'];

        	    }
        	    break;

        	default :
        	    if ( cnt2 >= 1 )  goto loop2;
        			ANTLREarlyExitException *eee = [ANTLREarlyExitException exceptionWithStream:input decisionNumber:2];
        			@throw eee;
            }
            cnt2++;
        } while (YES); loop2: ;


        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end INT


- (void) mWS
{
    @try {
        ruleNestingLevel++;
        int _type = SimpleCLexer_WS;
        // SimpleC.g:100:9: ( ( ' ' | '\\t' | '\\r' | '\\n' )+ ) // ruleBlockSingleAlt
        // SimpleC.g:100:9: ( ' ' | '\\t' | '\\r' | '\\n' )+ // alt
        {
        // SimpleC.g:100:9: ( ' ' | '\\t' | '\\r' | '\\n' )+	// positiveClosureBlock
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

         _channel=99; 

        }

        self->_tokenType = _type;
    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token labels
        // token+rule list labels
        // rule labels

    }
    return;
}
// $ANTLR end WS

- (void) mTokens
{
    // SimpleC.g:1:10: ( T7 | T8 | T9 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 | T20 | ID | INT | WS ) //ruleblock
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
    	case 'i':
    		{
    			int LA4_5 = [input LA:2];
    			if ( LA4_5=='n' ) {
    				{
    					int LA4_17 = [input LA:3];
    					if ( LA4_17=='t' ) {
    						{
    							int LA4_23 = [input LA:4];
    							if ( (LA4_23>='0' && LA4_23<='9')||(LA4_23>='A' && LA4_23<='Z')||LA4_23=='_'||(LA4_23>='a' && LA4_23<='z') ) {
    								alt4 = 15;
    							}
    						else {
    							alt4 = 5;	}
    						}
    					}
    				else {
    					alt4 = 15;	}
    				}
    			}
    		else {
    			alt4 = 15;	}
    		}
    		break;
    	case 'c':
    		{
    			int LA4_6 = [input LA:2];
    			if ( LA4_6=='h' ) {
    				{
    					int LA4_18 = [input LA:3];
    					if ( LA4_18=='a' ) {
    						{
    							int LA4_24 = [input LA:4];
    							if ( LA4_24=='r' ) {
    								{
    									int LA4_28 = [input LA:5];
    									if ( (LA4_28>='0' && LA4_28<='9')||(LA4_28>='A' && LA4_28<='Z')||LA4_28=='_'||(LA4_28>='a' && LA4_28<='z') ) {
    										alt4 = 15;
    									}
    								else {
    									alt4 = 6;	}
    								}
    							}
    						else {
    							alt4 = 15;	}
    						}
    					}
    				else {
    					alt4 = 15;	}
    				}
    			}
    		else {
    			alt4 = 15;	}
    		}
    		break;
    	case 'v':
    		{
    			int LA4_7 = [input LA:2];
    			if ( LA4_7=='o' ) {
    				{
    					int LA4_19 = [input LA:3];
    					if ( LA4_19=='i' ) {
    						{
    							int LA4_25 = [input LA:4];
    							if ( LA4_25=='d' ) {
    								{
    									int LA4_29 = [input LA:5];
    									if ( (LA4_29>='0' && LA4_29<='9')||(LA4_29>='A' && LA4_29<='Z')||LA4_29=='_'||(LA4_29>='a' && LA4_29<='z') ) {
    										alt4 = 15;
    									}
    								else {
    									alt4 = 7;	}
    								}
    							}
    						else {
    							alt4 = 15;	}
    						}
    					}
    				else {
    					alt4 = 15;	}
    				}
    			}
    		else {
    			alt4 = 15;	}
    		}
    		break;
    	case '{':
    		alt4 = 8;
    		break;
    	case '}':
    		alt4 = 9;
    		break;
    	case 'f':
    		{
    			int LA4_10 = [input LA:2];
    			if ( LA4_10=='o' ) {
    				{
    					int LA4_20 = [input LA:3];
    					if ( LA4_20=='r' ) {
    						{
    							int LA4_26 = [input LA:4];
    							if ( (LA4_26>='0' && LA4_26<='9')||(LA4_26>='A' && LA4_26<='Z')||LA4_26=='_'||(LA4_26>='a' && LA4_26<='z') ) {
    								alt4 = 15;
    							}
    						else {
    							alt4 = 10;	}
    						}
    					}
    				else {
    					alt4 = 15;	}
    				}
    			}
    		else {
    			alt4 = 15;	}
    		}
    		break;
    	case '=':
    		{
    			int LA4_11 = [input LA:2];
    			if ( LA4_11=='=' ) {
    				alt4 = 12;
    			}
    		else {
    			alt4 = 11;	}
    		}
    		break;
    	case '<':
    		alt4 = 13;
    		break;
    	case '+':
    		alt4 = 14;
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
    		alt4 = 15;
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
    	    // SimpleC.g:1:10: T7 // alt
    	    {
    	    [self mT7];



    	    }
    	    break;
    	case 2 :
    	    // SimpleC.g:1:13: T8 // alt
    	    {
    	    [self mT8];



    	    }
    	    break;
    	case 3 :
    	    // SimpleC.g:1:16: T9 // alt
    	    {
    	    [self mT9];



    	    }
    	    break;
    	case 4 :
    	    // SimpleC.g:1:19: T10 // alt
    	    {
    	    [self mT10];



    	    }
    	    break;
    	case 5 :
    	    // SimpleC.g:1:23: T11 // alt
    	    {
    	    [self mT11];



    	    }
    	    break;
    	case 6 :
    	    // SimpleC.g:1:27: T12 // alt
    	    {
    	    [self mT12];



    	    }
    	    break;
    	case 7 :
    	    // SimpleC.g:1:31: T13 // alt
    	    {
    	    [self mT13];



    	    }
    	    break;
    	case 8 :
    	    // SimpleC.g:1:35: T14 // alt
    	    {
    	    [self mT14];



    	    }
    	    break;
    	case 9 :
    	    // SimpleC.g:1:39: T15 // alt
    	    {
    	    [self mT15];



    	    }
    	    break;
    	case 10 :
    	    // SimpleC.g:1:43: T16 // alt
    	    {
    	    [self mT16];



    	    }
    	    break;
    	case 11 :
    	    // SimpleC.g:1:47: T17 // alt
    	    {
    	    [self mT17];



    	    }
    	    break;
    	case 12 :
    	    // SimpleC.g:1:51: T18 // alt
    	    {
    	    [self mT18];



    	    }
    	    break;
    	case 13 :
    	    // SimpleC.g:1:55: T19 // alt
    	    {
    	    [self mT19];



    	    }
    	    break;
    	case 14 :
    	    // SimpleC.g:1:59: T20 // alt
    	    {
    	    [self mT20];



    	    }
    	    break;
    	case 15 :
    	    // SimpleC.g:1:63: ID // alt
    	    {
    	    [self mID];



    	    }
    	    break;
    	case 16 :
    	    // SimpleC.g:1:66: INT // alt
    	    {
    	    [self mINT];



    	    }
    	    break;
    	case 17 :
    	    // SimpleC.g:1:70: WS // alt
    	    {
    	    [self mWS];



    	    }
    	    break;

    }

}

@end