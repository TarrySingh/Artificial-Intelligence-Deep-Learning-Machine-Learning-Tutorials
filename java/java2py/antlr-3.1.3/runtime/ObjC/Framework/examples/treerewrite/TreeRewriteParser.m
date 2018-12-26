// $ANTLR 3.1b1 TreeRewrite.g 2007-11-04 03:34:43

#import "TreeRewriteParser.h"

#import <ANTLR/ANTLR.h>



#pragma mark Bitsets
const static unsigned long long FOLLOW_TreeRewriteParser_INT_in_rule26_data[] = {0x0000000000000010LL};
static ANTLRBitSet *FOLLOW_TreeRewriteParser_INT_in_rule26;
const static unsigned long long FOLLOW_subrule_in_rule28_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_subrule_in_rule28;
const static unsigned long long FOLLOW_TreeRewriteParser_INT_in_subrule53_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_TreeRewriteParser_INT_in_subrule53;


#pragma mark Dynamic Global Scopes

#pragma mark Dynamic Rule Scopes

#pragma mark Rule return scopes start
@implementation TreeRewriteParser_rule_return
- (id) tree
{
    return tree;
}
- (void) setTree:(id)aTree
{
    if (tree != aTree) {
        [aTree retain];
        [tree release];
        tree = aTree;
    }
}

- (void) dealloc
{
    [self setTree:nil];
    [super dealloc];
}
@end
@implementation TreeRewriteParser_subrule_return
- (id) tree
{
    return tree;
}
- (void) setTree:(id)aTree
{
    if (tree != aTree) {
        [aTree retain];
        [tree release];
        tree = aTree;
    }
}

- (void) dealloc
{
    [self setTree:nil];
    [super dealloc];
}
@end


@implementation TreeRewriteParser

static NSArray *tokenNames;

+ (void) initialize
{
	FOLLOW_TreeRewriteParser_INT_in_rule26 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_TreeRewriteParser_INT_in_rule26_data count:1];
	FOLLOW_subrule_in_rule28 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_subrule_in_rule28_data count:1];
	FOLLOW_TreeRewriteParser_INT_in_subrule53 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_TreeRewriteParser_INT_in_subrule53_data count:1];

	tokenNames = [[NSArray alloc] initWithObjects:@"<invalid>", @"<EOR>", @"<DOWN>", @"<UP>", 
	@"INT", @"WS", nil];
}

- (id) initWithTokenStream:(id<ANTLRTokenStream>)aStream
{
	if ((self = [super initWithTokenStream:aStream])) {


						
		[self setTreeAdaptor:[[[ANTLRCommonTreeAdaptor alloc] init] autorelease]];
	}
	return self;
}

- (void) dealloc
{

	[self setTreeAdaptor:nil];

	[super dealloc];
}

- (NSString *) grammarFileName
{
	return @"TreeRewrite.g";
}


// $ANTLR start rule
// TreeRewrite.g:8:1: rule : INT subrule -> ^( subrule INT ) ;
- (TreeRewriteParser_rule_return *) rule
{
    ANTLRBaseRecognizerState *_state = [self state];
    TreeRewriteParser_rule_return * _retval = [[[TreeRewriteParser_rule_return alloc] init] autorelease];
    [_retval setStart:[input LT:1]];

    id root_0 = nil;

    id<ANTLRToken>  _INT1 = nil;
    TreeRewriteParser_subrule_return * _subrule2 = nil;


    id _INT1_tree = nil;
    ANTLRRewriteRuleTokenStream *_stream_TreeRewriteParser_INT=[[ANTLRRewriteRuleTokenStream alloc] initWithTreeAdaptor:treeAdaptor description:@"token TreeRewriteParser_INT"];
    ANTLRRewriteRuleSubtreeStream *_stream_subrule=[[ANTLRRewriteRuleSubtreeStream alloc] initWithTreeAdaptor:treeAdaptor description:@"rule subrule"];

    @try {
        // TreeRewrite.g:8:5: ( INT subrule -> ^( subrule INT ) ) // ruleBlockSingleAlt
        // TreeRewrite.g:8:7: INT subrule // alt
        {
        _INT1=(id<ANTLRToken> )[input LT:1];
        [self match:input tokenType:TreeRewriteParser_INT follow:FOLLOW_TreeRewriteParser_INT_in_rule26]; 
        [_stream_TreeRewriteParser_INT addElement:_INT1];

        [[_state following] addObject:FOLLOW_subrule_in_rule28];
        _subrule2 = [self subrule];
        [[_state following] removeLastObject];


        [_stream_subrule addElement:[_subrule2 tree]];

        // AST REWRITE
        // elements: TreeRewriteParser_INT, subrule
        // token labels: 
        // rule labels: retval
        // token list labels: 
        // rule list labels: 
        int i_0 = 0;
        root_0 = (id)[treeAdaptor newEmptyTree];
        [_retval setTree:root_0];
        ANTLRRewriteRuleSubtreeStream *_stream_retval=[[ANTLRRewriteRuleSubtreeStream alloc] initWithTreeAdaptor:treeAdaptor description:@"token retval" element:_retval!=nil?[_retval tree]:nil];

        // 8:19: -> ^( subrule INT )
        {
            // TreeRewrite.g:8:22: ^( subrule INT )
            {
            id root_1 = (id)[treeAdaptor newEmptyTree];
            root_1 = (id)[treeAdaptor makeNode:(id<ANTLRTree>)[_stream_subrule next] parentOf:root_1];

            [treeAdaptor addChild:[_stream_TreeRewriteParser_INT next] toTree:root_1];


            [treeAdaptor addChild:root_1 toTree:root_0];
            [root_1 release];
            }

        }

        [_stream_retval release];


        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token+rule list labels
		[_retval setStop:[input LT:-1]];

		[_stream_TreeRewriteParser_INT release];
		[_stream_subrule release];
		    [_retval setTree:(id)[treeAdaptor postProcessTree:root_0]];
		    [treeAdaptor setBoundariesForTree:[_retval tree] fromToken:[_retval start] toToken:[_retval stop]];
		[root_0 release];
	}
	return _retval;
}
// $ANTLR end rule

// $ANTLR start subrule
// TreeRewrite.g:11:1: subrule : INT ;
- (TreeRewriteParser_subrule_return *) subrule
{
    ANTLRBaseRecognizerState *_state = [self state];
    TreeRewriteParser_subrule_return * _retval = [[[TreeRewriteParser_subrule_return alloc] init] autorelease];
    [_retval setStart:[input LT:1]];

    id root_0 = nil;

    id<ANTLRToken>  _INT3 = nil;

    id _INT3_tree = nil;

    @try {
        // TreeRewrite.g:12:5: ( INT ) // ruleBlockSingleAlt
        // TreeRewrite.g:12:9: INT // alt
        {
        root_0 = (id)[treeAdaptor newEmptyTree];

        _INT3=(id<ANTLRToken> )[input LT:1];
        [self match:input tokenType:TreeRewriteParser_INT follow:FOLLOW_TreeRewriteParser_INT_in_subrule53]; 
        _INT3_tree = (id)[treeAdaptor newTreeWithToken:_INT3];
        [treeAdaptor addChild:_INT3_tree toTree:root_0];
        [_INT3_tree release];


        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token+rule list labels
		[_retval setStop:[input LT:-1]];

		    [_retval setTree:(id)[treeAdaptor postProcessTree:root_0]];
		    [treeAdaptor setBoundariesForTree:[_retval tree] fromToken:[_retval start] toToken:[_retval stop]];
		[root_0 release];
	}
	return _retval;
}
// $ANTLR end subrule


- (id<ANTLRTreeAdaptor>) treeAdaptor
{
	return treeAdaptor;
}

- (void) setTreeAdaptor:(id<ANTLRTreeAdaptor>)aTreeAdaptor
{
	if (aTreeAdaptor != treeAdaptor) {
		[aTreeAdaptor retain];
		[treeAdaptor release];
		treeAdaptor = aTreeAdaptor;
	}
}

@end