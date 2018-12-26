// $ANTLR 3.0 SymbolTable.g 2007-07-25 20:12:44

#import "SymbolTableLexer.h"

/** As per Terence: No returns for lexer rules!
#pragma mark Rule return scopes start
#pragma mark Rule return scopes end
*/
@implementation SymbolTableLexer


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
	return @"SymbolTable.g";
}


- (void) mT7
{
    @try {
        ruleNestingLevel++;
        int _type = SymbolTableLexer_T7;
        // SymbolTable.g:7:6: ( 'method' ) // ruleBlockSingleAlt
        // SymbolTable.g:7:6: 'method' // alt
        {
        [self matchString:@"method"];



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
        int _type = SymbolTableLexer_T8;
        // SymbolTable.g:8:6: ( '(' ) // ruleBlockSingleAlt
        // SymbolTable.g:8:6: '(' // alt
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
        int _type = SymbolTableLexer_T9;
        // SymbolTable.g:9:6: ( ')' ) // ruleBlockSingleAlt
        // SymbolTable.g:9:6: ')' // alt
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
// $ANTLR end T9


- (void) mT10
{
    @try {
        ruleNestingLevel++;
        int _type = SymbolTableLexer_T10;
        // SymbolTable.g:10:7: ( '{' ) // ruleBlockSingleAlt
        // SymbolTable.g:10:7: '{' // alt
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
// $ANTLR end T10


- (void) mT11
{
    @try {
        ruleNestingLevel++;
        int _type = SymbolTableLexer_T11;
        // SymbolTable.g:11:7: ( '}' ) // ruleBlockSingleAlt
        // SymbolTable.g:11:7: '}' // alt
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
// $ANTLR end T11


- (void) mT12
{
    @try {
        ruleNestingLevel++;
        int _type = SymbolTableLexer_T12;
        // SymbolTable.g:12:7: ( '=' ) // ruleBlockSingleAlt
        // SymbolTable.g:12:7: '=' // alt
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
// $ANTLR end T12


- (void) mT13
{
    @try {
        ruleNestingLevel++;
        int _type = SymbolTableLexer_T13;
        // SymbolTable.g:13:7: ( ';' ) // ruleBlockSingleAlt
        // SymbolTable.g:13:7: ';' // alt
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
// $ANTLR end T13


- (void) mT14
{
    @try {
        ruleNestingLevel++;
        int _type = SymbolTableLexer_T14;
        // SymbolTable.g:14:7: ( 'int' ) // ruleBlockSingleAlt
        // SymbolTable.g:14:7: 'int' // alt
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
// $ANTLR end T14


- (void) mID
{
    @try {
        ruleNestingLevel++;
        int _type = SymbolTableLexer_ID;
        // SymbolTable.g:66:9: ( ( 'a' .. 'z' )+ ) // ruleBlockSingleAlt
        // SymbolTable.g:66:9: ( 'a' .. 'z' )+ // alt
        {
        // SymbolTable.g:66:9: ( 'a' .. 'z' )+	// positiveClosureBlock
        int cnt1=0;

        do {
            int alt1=2;
            {
            	int LA1_0 = [input LA:1];
            	if ( (LA1_0>='a' && LA1_0<='z') ) {
            		alt1 = 1;
            	}

            }
            switch (alt1) {
        	case 1 :
        	    // SymbolTable.g:66:10: 'a' .. 'z' // alt
        	    {
        	    [self matchRangeFromChar:'a' to:'z'];

        	    }
        	    break;

        	default :
        	    if ( cnt1 >= 1 )  goto loop1;
        			ANTLREarlyExitException *eee = [ANTLREarlyExitException exceptionWithStream:input decisionNumber:1];
        			@throw eee;
            }
            cnt1++;
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
        int _type = SymbolTableLexer_INT;
        // SymbolTable.g:69:9: ( ( '0' .. '9' )+ ) // ruleBlockSingleAlt
        // SymbolTable.g:69:9: ( '0' .. '9' )+ // alt
        {
        // SymbolTable.g:69:9: ( '0' .. '9' )+	// positiveClosureBlock
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
        	    // SymbolTable.g:69:10: '0' .. '9' // alt
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
        int _type = SymbolTableLexer_WS;
        // SymbolTable.g:72:9: ( ( ' ' | '\\n' | '\\r' )+ ) // ruleBlockSingleAlt
        // SymbolTable.g:72:9: ( ' ' | '\\n' | '\\r' )+ // alt
        {
        // SymbolTable.g:72:9: ( ' ' | '\\n' | '\\r' )+	// positiveClosureBlock
        int cnt3=0;

        do {
            int alt3=2;
            {
            	int LA3_0 = [input LA:1];
            	if ( LA3_0=='\n'||LA3_0=='\r'||LA3_0==' ' ) {
            		alt3 = 1;
            	}

            }
            switch (alt3) {
        	case 1 :
        	    // SymbolTable.g: // alt
        	    {
        	    if ([input LA:1]=='\n'||[input LA:1]=='\r'||[input LA:1]==' ') {
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
    // SymbolTable.g:1:10: ( T7 | T8 | T9 | T10 | T11 | T12 | T13 | T14 | ID | INT | WS ) //ruleblock
    int alt4=11;
    switch ([input LA:1]) {
    	case 'm':
    		{
    			int LA4_1 = [input LA:2];
    			if ( LA4_1=='e' ) {
    				{
    					int LA4_12 = [input LA:3];
    					if ( LA4_12=='t' ) {
    						{
    							int LA4_14 = [input LA:4];
    							if ( LA4_14=='h' ) {
    								{
    									int LA4_16 = [input LA:5];
    									if ( LA4_16=='o' ) {
    										{
    											int LA4_18 = [input LA:6];
    											if ( LA4_18=='d' ) {
    												{
    													int LA4_19 = [input LA:7];
    													if ( (LA4_19>='a' && LA4_19<='z') ) {
    														alt4 = 9;
    													}
    												else {
    													alt4 = 1;	}
    												}
    											}
    										else {
    											alt4 = 9;	}
    										}
    									}
    								else {
    									alt4 = 9;	}
    								}
    							}
    						else {
    							alt4 = 9;	}
    						}
    					}
    				else {
    					alt4 = 9;	}
    				}
    			}
    		else {
    			alt4 = 9;	}
    		}
    		break;
    	case '(':
    		alt4 = 2;
    		break;
    	case ')':
    		alt4 = 3;
    		break;
    	case '{':
    		alt4 = 4;
    		break;
    	case '}':
    		alt4 = 5;
    		break;
    	case '=':
    		alt4 = 6;
    		break;
    	case ';':
    		alt4 = 7;
    		break;
    	case 'i':
    		{
    			int LA4_8 = [input LA:2];
    			if ( LA4_8=='n' ) {
    				{
    					int LA4_13 = [input LA:3];
    					if ( LA4_13=='t' ) {
    						{
    							int LA4_15 = [input LA:4];
    							if ( (LA4_15>='a' && LA4_15<='z') ) {
    								alt4 = 9;
    							}
    						else {
    							alt4 = 8;	}
    						}
    					}
    				else {
    					alt4 = 9;	}
    				}
    			}
    		else {
    			alt4 = 9;	}
    		}
    		break;
    	case 'a':
    	case 'b':
    	case 'c':
    	case 'd':
    	case 'e':
    	case 'f':
    	case 'g':
    	case 'h':
    	case 'j':
    	case 'k':
    	case 'l':
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
    		alt4 = 9;
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
    		alt4 = 10;
    		break;
    	case '\n':
    	case '\r':
    	case ' ':
    		alt4 = 11;
    		break;
    default:
     {
        ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:4 state:0 stream:input];
    	@throw nvae;

    	}}
    switch (alt4) {
    	case 1 :
    	    // SymbolTable.g:1:10: T7 // alt
    	    {
    	    [self mT7];



    	    }
    	    break;
    	case 2 :
    	    // SymbolTable.g:1:13: T8 // alt
    	    {
    	    [self mT8];



    	    }
    	    break;
    	case 3 :
    	    // SymbolTable.g:1:16: T9 // alt
    	    {
    	    [self mT9];



    	    }
    	    break;
    	case 4 :
    	    // SymbolTable.g:1:19: T10 // alt
    	    {
    	    [self mT10];



    	    }
    	    break;
    	case 5 :
    	    // SymbolTable.g:1:23: T11 // alt
    	    {
    	    [self mT11];



    	    }
    	    break;
    	case 6 :
    	    // SymbolTable.g:1:27: T12 // alt
    	    {
    	    [self mT12];



    	    }
    	    break;
    	case 7 :
    	    // SymbolTable.g:1:31: T13 // alt
    	    {
    	    [self mT13];



    	    }
    	    break;
    	case 8 :
    	    // SymbolTable.g:1:35: T14 // alt
    	    {
    	    [self mT14];



    	    }
    	    break;
    	case 9 :
    	    // SymbolTable.g:1:39: ID // alt
    	    {
    	    [self mID];



    	    }
    	    break;
    	case 10 :
    	    // SymbolTable.g:1:42: INT // alt
    	    {
    	    [self mINT];



    	    }
    	    break;
    	case 11 :
    	    // SymbolTable.g:1:46: WS // alt
    	    {
    	    [self mWS];



    	    }
    	    break;

    }

}

@end