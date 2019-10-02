// $ANTLR 3.1b1 SimpleCTP.gtp 2007-12-16 20:47:43

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>



#pragma mark Tokens
#define SimpleCTP_FUNC_DEF	8
#define SimpleCTP_WS	20
#define SimpleCTP_CHAR	15
#define SimpleCTP_EQ	11
#define SimpleCTP_FUNC_HDR	6
#define SimpleCTP_LT	18
#define SimpleCTP_ARG_DEF	5
#define SimpleCTP_EQEQ	17
#define SimpleCTP_BLOCK	9
#define SimpleCTP_INT	12
#define SimpleCTP_EOF	-1
#define SimpleCTP_VOID	16
#define SimpleCTP_FOR	13
#define SimpleCTP_PLUS	19
#define SimpleCTP_FUNC_DECL	7
#define SimpleCTP_INT_TYPE	14
#define SimpleCTP_ID	10
#define SimpleCTP_VAR_DEF	4

#pragma mark Dynamic Global Scopes

#pragma mark Dynamic Rule Scopes

#pragma mark Rule Return Scopes
@interface SimpleCTP_expr_return : ANTLRTreeParserRuleReturnScope {
}
@end



@interface SimpleCTP : ANTLRTreeParser {

													

 }


- (void) program;
- (void) declaration;
- (void) variable;
- (void) declarator;
- (void) functionHeader;
- (void) formalParameter;
- (void) type;
- (void) block;
- (void) stat;
- (void) forStat;
- (SimpleCTP_expr_return *) expr;
- (void) atom;



@end