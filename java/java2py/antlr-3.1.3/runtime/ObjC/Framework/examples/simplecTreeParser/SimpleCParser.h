// $ANTLR 3.1b1 SimpleC.g 2007-12-16 20:47:42

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>


#pragma mark Cyclic DFA interface start SimpleCParserDFA2
@interface SimpleCParserDFA2 : ANTLRDFA {} @end

#pragma mark Cyclic DFA interface end SimpleCParserDFA2

#pragma mark Tokens
#define SimpleCParser_FUNC_DEF	8
#define SimpleCParser_WS	20
#define SimpleCParser_CHAR	15
#define SimpleCParser_EQ	11
#define SimpleCParser_FUNC_HDR	6
#define SimpleCParser_LT	18
#define SimpleCParser_ARG_DEF	5
#define SimpleCParser_EQEQ	17
#define SimpleCParser_BLOCK	9
#define SimpleCParser_INT	12
#define SimpleCParser_EOF	-1
#define SimpleCParser_VOID	16
#define SimpleCParser_FOR	13
#define SimpleCParser_PLUS	19
#define SimpleCParser_FUNC_DECL	7
#define SimpleCParser_INT_TYPE	14
#define SimpleCParser_VAR_DEF	4
#define SimpleCParser_ID	10

#pragma mark Dynamic Global Scopes

#pragma mark Dynamic Rule Scopes

#pragma mark Rule Return Scopes
@interface SimpleCParser_program_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_declaration_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_variable_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_declarator_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_functionHeader_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_formalParameter_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_type_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_block_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_stat_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_forStat_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_assignStat_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_expr_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_condExpr_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_aexpr_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end
@interface SimpleCParser_atom_return : ANTLRParserRuleReturnScope {
    id tree;
}
- (id) tree;
- (void) setTree:(id)aTree;
@end



@interface SimpleCParser : ANTLRParser {

	SimpleCParserDFA2 *dfa2;
																
	id<ANTLRTreeAdaptor> treeAdaptor;

 }


- (SimpleCParser_program_return *) program;
- (SimpleCParser_declaration_return *) declaration;
- (SimpleCParser_variable_return *) variable;
- (SimpleCParser_declarator_return *) declarator;
- (SimpleCParser_functionHeader_return *) functionHeader;
- (SimpleCParser_formalParameter_return *) formalParameter;
- (SimpleCParser_type_return *) type;
- (SimpleCParser_block_return *) block;
- (SimpleCParser_stat_return *) stat;
- (SimpleCParser_forStat_return *) forStat;
- (SimpleCParser_assignStat_return *) assignStat;
- (SimpleCParser_expr_return *) expr;
- (SimpleCParser_condExpr_return *) condExpr;
- (SimpleCParser_aexpr_return *) aexpr;
- (SimpleCParser_atom_return *) atom;


- (id<ANTLRTreeAdaptor>) treeAdaptor;
- (void) setTreeAdaptor:(id<ANTLRTreeAdaptor>)theTreeAdaptor;

@end