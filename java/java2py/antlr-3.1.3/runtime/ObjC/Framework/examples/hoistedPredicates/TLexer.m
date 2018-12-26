// $ANTLR 3.0 T.g 2007-07-25 20:12:43

#import "TLexer.h"

/** As per Terence: No returns for lexer rules!
#pragma mark Rule return scopes start
#pragma mark Rule return scopes end
*/
@implementation TLexer


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
	return @"T.g";
}


- (void) mT7
{
    @try {
        ruleNestingLevel++;
        int _type = TLexer_T7;
        // T.g:7:6: ( 'enum' ) // ruleBlockSingleAlt
        // T.g:7:6: 'enum' // alt
        {
        [self matchString:@"enum"];



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


- (void) mID
{
    @try {
        ruleNestingLevel++;
        int _type = TLexer_ID;
        // T.g:37:9: ( ( 'a' .. 'z' | 'A' .. 'Z' | '_' ) ( 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' )* ) // ruleBlockSingleAlt
        // T.g:37:9: ( 'a' .. 'z' | 'A' .. 'Z' | '_' ) ( 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' )* // alt
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
        	    // T.g: // alt
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
        int _type = TLexer_INT;
        // T.g:40:7: ( ( '0' .. '9' )+ ) // ruleBlockSingleAlt
        // T.g:40:7: ( '0' .. '9' )+ // alt
        {
        // T.g:40:7: ( '0' .. '9' )+	// positiveClosureBlock
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
        	    // T.g:40:8: '0' .. '9' // alt
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
        int _type = TLexer_WS;
        // T.g:43:9: ( ( ' ' | '\\t' | '\\r' | '\\n' )+ ) // ruleBlockSingleAlt
        // T.g:43:9: ( ' ' | '\\t' | '\\r' | '\\n' )+ // alt
        {
        // T.g:43:9: ( ' ' | '\\t' | '\\r' | '\\n' )+	// positiveClosureBlock
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
        	    // T.g: // alt
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
    // T.g:1:10: ( T7 | ID | INT | WS ) //ruleblock
    int alt4=4;
    switch ([input LA:1]) {
    	case 'e':
    		{
    			int LA4_1 = [input LA:2];
    			if ( LA4_1=='n' ) {
    				{
    					int LA4_5 = [input LA:3];
    					if ( LA4_5=='u' ) {
    						{
    							int LA4_6 = [input LA:4];
    							if ( LA4_6=='m' ) {
    								{
    									int LA4_7 = [input LA:5];
    									if ( (LA4_7>='0' && LA4_7<='9')||(LA4_7>='A' && LA4_7<='Z')||LA4_7=='_'||(LA4_7>='a' && LA4_7<='z') ) {
    										alt4 = 2;
    									}
    								else {
    									alt4 = 1;	}
    								}
    							}
    						else {
    							alt4 = 2;	}
    						}
    					}
    				else {
    					alt4 = 2;	}
    				}
    			}
    		else {
    			alt4 = 2;	}
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
    	case 'c':
    	case 'd':
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
    		alt4 = 2;
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
    		alt4 = 3;
    		break;
    	case '\t':
    	case '\n':
    	case '\r':
    	case ' ':
    		alt4 = 4;
    		break;
    default:
     {
        ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:4 state:0 stream:input];
    	@throw nvae;

    	}}
    switch (alt4) {
    	case 1 :
    	    // T.g:1:10: T7 // alt
    	    {
    	    [self mT7];



    	    }
    	    break;
    	case 2 :
    	    // T.g:1:13: ID // alt
    	    {
    	    [self mID];



    	    }
    	    break;
    	case 3 :
    	    // T.g:1:16: INT // alt
    	    {
    	    [self mINT];



    	    }
    	    break;
    	case 4 :
    	    // T.g:1:20: WS // alt
    	    {
    	    [self mWS];



    	    }
    	    break;

    }

}

@end