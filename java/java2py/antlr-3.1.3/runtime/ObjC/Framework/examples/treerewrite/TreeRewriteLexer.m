// $ANTLR 3.1b1 TreeRewrite.g 2007-11-04 03:34:43

#import "TreeRewriteLexer.h"

/** As per Terence: No returns for lexer rules!
#pragma mark Rule return scopes start
#pragma mark Rule return scopes end
*/
@implementation TreeRewriteLexer

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
	return @"TreeRewrite.g";
}


- (void) mINT
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = TreeRewriteLexer_INT;
        // TreeRewrite.g:15:5: ( ( '0' .. '9' )+ ) // ruleBlockSingleAlt
        // TreeRewrite.g:15:7: ( '0' .. '9' )+ // alt
        {
        // TreeRewrite.g:15:7: ( '0' .. '9' )+	// positiveClosureBlock
        int cnt1=0;

        do {
            int alt1=2;
            {
            	int LA1_0 = [input LA:1];
            	if ( (LA1_0>='0' && LA1_0<='9') ) {
            		alt1 = 1;
            	}

            }
            switch (alt1) {
        	case 1 :
        	    // TreeRewrite.g:15:8: '0' .. '9' // alt
        	    {
        	    [self matchRangeFromChar:'0' to:'9'];

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

        [_state setTokenType:_type];
    }
    @finally {
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end INT


- (void) mWS
{
    ANTLRLexerState *_state = [self state];
    @try {
        int _type = TreeRewriteLexer_WS;
        // TreeRewrite.g:18:5: ( ' ' ) // ruleBlockSingleAlt
        // TreeRewrite.g:18:9: ' ' // alt
        {
        [self matchChar:' '];


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
    // TreeRewrite.g:1:8: ( INT | WS ) //ruleblock
    int alt2=2;
    {
    	int LA2_0 = [input LA:1];
    	if ( (LA2_0>='0' && LA2_0<='9') ) {
    		alt2 = 1;
    	}
    	else if ( LA2_0==' ' ) {
    		alt2 = 2;
    	}
    else {
        ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:2 state:0 stream:input];
    	@throw nvae;
    	}
    }
    switch (alt2) {
    	case 1 :
    	    // TreeRewrite.g:1:10: INT // alt
    	    {
    	    [self mINT];



    	    }
    	    break;
    	case 2 :
    	    // TreeRewrite.g:1:14: WS // alt
    	    {
    	    [self mWS];



    	    }
    	    break;

    }

}

@end