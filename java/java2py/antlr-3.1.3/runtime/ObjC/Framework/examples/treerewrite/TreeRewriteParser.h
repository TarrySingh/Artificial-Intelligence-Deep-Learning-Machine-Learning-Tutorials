// $ANTLR 3.1b1 TreeRewrite.g 2007-11-04 03:34:43

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>



#pragma mark Tokens
#define TreeRewriteParser_INT	4
#define TreeRewriteParser_WS	5
#define TreeRewriteParser_EOF	-1

#pragma mark Dynamic Global Scopes

#pragma mark Dynamic Rule Scopes

#pragma mark Rule Return Scopes
@interface TreeRewriteParser_rule_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface TreeRewriteParser_subrule_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end



@interface TreeRewriteParser : ANTLRParser {

			
	id<ANTLRTreeAdaptor> treeAdaptor;

 }


- (TreeRewriteParser_rule_return *) rule;
- (TreeRewriteParser_subrule_return *) subrule;


- (id<ANTLRTreeAdaptor>) treeAdaptor;
- (void) setTreeAdaptor:(id<ANTLRTreeAdaptor>)theTreeAdaptor;

@end