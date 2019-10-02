// $ANTLR 3.0 Test.gl 2007-08-04 15:59:43

#import "TestLexer.h"

/** As per Terence: No returns for lexer rules!
#pragma mark Rule return scopes start
#pragma mark Rule return scopes end
*/
@implementation TestLexer

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
	return @"Test.gl";
}


- (void) mID
{
    @try {
        ruleNestingLevel++;
        int _type = TestLexer_ID;
        // Test.gl:8:6: ( LETTER ( LETTER | DIGIT )* ) // ruleBlockSingleAlt
        // Test.gl:8:6: LETTER ( LETTER | DIGIT )* // alt
        {
        [self mLETTER];


        do {
            int alt1=2;
            {
            	int LA1_0 = [input LA:1];
            	if ( (LA1_0>='0' && LA1_0<='9')||(LA1_0>='A' && LA1_0<='Z')||(LA1_0>='a' && LA1_0<='z') ) {
            		alt1 = 1;
            	}

            }
            switch (alt1) {
        	case 1 :
        	    // Test.gl: // alt
        	    {
        	    if (([input LA:1]>='0' && [input LA:1]<='9')||([input LA:1]>='A' && [input LA:1]<='Z')||([input LA:1]>='a' && [input LA:1]<='z')) {
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
        // token+rule list labels

    }
    return;
}
// $ANTLR end ID


- (void) mDIGIT
{
    @try {
        ruleNestingLevel++;
        // Test.gl:11:18: ( '0' .. '9' ) // ruleBlockSingleAlt
        // Test.gl:11:18: '0' .. '9' // alt
        {
        [self matchRangeFromChar:'0' to:'9'];

        }

    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end DIGIT


- (void) mLETTER
{
    @try {
        ruleNestingLevel++;
        // Test.gl:15:4: ( 'a' .. 'z' | 'A' .. 'Z' ) // ruleBlockSingleAlt
        // Test.gl: // alt
        {
        if (([input LA:1]>='A' && [input LA:1]<='Z')||([input LA:1]>='a' && [input LA:1]<='z')) {
        	[input consume];

        } else {
        	ANTLRMismatchedSetException *mse = [ANTLRMismatchedSetException exceptionWithSet:nil stream:input];
        	[self recover:mse];	@throw mse;
        }


        }

    }
    @finally {
        ruleNestingLevel--;
        // rule cleanup
        // token+rule list labels

    }
    return;
}
// $ANTLR end LETTER

- (void) mTokens
{
    // Test.gl:1:10: ( ID ) // ruleBlockSingleAlt
    // Test.gl:1:10: ID // alt
    {
    [self mID];



    }


}

@end