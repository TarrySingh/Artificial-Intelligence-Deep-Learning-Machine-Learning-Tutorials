// $ANTLR 3.0 T.g 2007-07-25 20:12:43

#import "TParser.h"
/** Demonstrates how semantic predicates get hoisted out of the rule in 
 *  which they are found and used in other decisions.  This grammar illustrates
 *  how predicates can be used to distinguish between enum as a keyword and
 *  an ID *dynamically*. :)

 * Run "java org.antlr.Tool -dfa t.g" to generate DOT (graphviz) files.  See
 * the T_dec-1.dot file to see the predicates in action.
 */


#pragma mark Bitsets
const static unsigned long long FOLLOW_identifier_in_stat34_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_identifier_in_stat34;
const static unsigned long long FOLLOW_enumAsKeyword_in_stat47_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_enumAsKeyword_in_stat47;
const static unsigned long long FOLLOW_TParser_ID_in_identifier66_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_TParser_ID_in_identifier66;
const static unsigned long long FOLLOW_enumAsID_in_identifier74_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_enumAsID_in_identifier74;
const static unsigned long long FOLLOW_7_in_enumAsKeyword89_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_7_in_enumAsKeyword89;
const static unsigned long long FOLLOW_7_in_enumAsID100_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_7_in_enumAsID100;


#pragma mark Dynamic Global Scopes

#pragma mark Dynamic Rule Scopes

#pragma mark Rule return scopes start

@implementation TParser

+ (void) initialize
{
	FOLLOW_identifier_in_stat34 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_identifier_in_stat34_data count:1];
	FOLLOW_enumAsKeyword_in_stat47 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_enumAsKeyword_in_stat47_data count:1];
	FOLLOW_TParser_ID_in_identifier66 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_TParser_ID_in_identifier66_data count:1];
	FOLLOW_enumAsID_in_identifier74 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_enumAsID_in_identifier74_data count:1];
	FOLLOW_7_in_enumAsKeyword89 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_7_in_enumAsKeyword89_data count:1];
	FOLLOW_7_in_enumAsID100 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_7_in_enumAsID100_data count:1];

}

- (id) initWithTokenStream:(id<ANTLRTokenStream>)aStream
{
	if ((self = [super initWithTokenStream:aStream])) {

		tokenNames = [[NSArray alloc] initWithObjects:@"<invalid>", @"<EOR>", @"<DOWN>", @"<UP>", 
	@"ID", @"INT", @"WS", @"'enum'", nil];

										

		enableEnum = NO;

	}
	return self;
}

- (void) dealloc
{
	[tokenNames release];

	[super dealloc];
}

- (NSString *) grammarFileName
{
	return @"T.g";
}


// $ANTLR start stat
// T.g:24:1: stat : ( identifier | enumAsKeyword );
- (void) stat
{
    @try {
        // T.g:24:7: ( identifier | enumAsKeyword ) //ruleblock
        int alt1=2;
        {
        	int LA1_0 = [input LA:1];
        	if ( LA1_0==TParser_ID ) {
        		alt1 = 1;
        	}
        	else if ( LA1_0==7 ) {
        		{
        			int LA1_2 = [input LA:2];
        			if ( !enableEnum ) {
        				alt1 = 1;
        			}
        			else if ( enableEnum ) {
        				alt1 = 2;
        			}
        		else {
        		    ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:1 state:2 stream:input];
        			@throw nvae;
        			}
        		}
        	}
        else {
            ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:1 state:0 stream:input];
        	@throw nvae;
        	}
        }
        switch (alt1) {
        	case 1 :
        	    // T.g:24:7: identifier // alt
        	    {
        	    [following addObject:FOLLOW_identifier_in_stat34];
        	    [self identifier];
        	    [following removeLastObject];


        	    NSLog(@"enum is an ID");

        	    }
        	    break;
        	case 2 :
        	    // T.g:25:7: enumAsKeyword // alt
        	    {
        	    [following addObject:FOLLOW_enumAsKeyword_in_stat47];
        	    [self enumAsKeyword];
        	    [following removeLastObject];


        	    NSLog(@"enum is a keyword");

        	    }
        	    break;

        }
    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

	}
	return ;
}
// $ANTLR end stat

// $ANTLR start identifier
// T.g:28:1: identifier : ( ID | enumAsID );
- (void) identifier
{
    @try {
        // T.g:29:7: ( ID | enumAsID ) //ruleblock
        int alt2=2;
        {
        	int LA2_0 = [input LA:1];
        	if ( LA2_0==TParser_ID ) {
        		alt2 = 1;
        	}
        	else if ( LA2_0==7 ) {
        		alt2 = 2;
        	}
        else {
            ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:2 state:0 stream:input];
        	@throw nvae;
        	}
        }
        switch (alt2) {
        	case 1 :
        	    // T.g:29:7: ID // alt
        	    {
        	    [self match:input tokenType:TParser_ID follow:FOLLOW_TParser_ID_in_identifier66]; 

        	    }
        	    break;
        	case 2 :
        	    // T.g:30:7: enumAsID // alt
        	    {
        	    [following addObject:FOLLOW_enumAsID_in_identifier74];
        	    [self enumAsID];
        	    [following removeLastObject];



        	    }
        	    break;

        }
    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

	}
	return ;
}
// $ANTLR end identifier

// $ANTLR start enumAsKeyword
// T.g:33:1: enumAsKeyword : {...}? 'enum' ;
- (void) enumAsKeyword
{
    @try {
        // T.g:33:17: ({...}? 'enum' ) // ruleBlockSingleAlt
        // T.g:33:17: {...}? 'enum' // alt
        {
        if ( !(enableEnum) ) {
            @throw [ANTLRFailedPredicateException exceptionWithRuleName:@"enumAsKeyword" predicate:@"enableEnum" stream:input];
        }
        [self match:input tokenType:7 follow:FOLLOW_7_in_enumAsKeyword89]; 

        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

	}
	return ;
}
// $ANTLR end enumAsKeyword

// $ANTLR start enumAsID
// T.g:35:1: enumAsID : {...}? 'enum' ;
- (void) enumAsID
{
    @try {
        // T.g:35:12: ({...}? 'enum' ) // ruleBlockSingleAlt
        // T.g:35:12: {...}? 'enum' // alt
        {
        if ( !(!enableEnum) ) {
            @throw [ANTLRFailedPredicateException exceptionWithRuleName:@"enumAsID" predicate:@"!enableEnum" stream:input];
        }
        [self match:input tokenType:7 follow:FOLLOW_7_in_enumAsID100]; 

        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

	}
	return ;
}
// $ANTLR end enumAsID



@end