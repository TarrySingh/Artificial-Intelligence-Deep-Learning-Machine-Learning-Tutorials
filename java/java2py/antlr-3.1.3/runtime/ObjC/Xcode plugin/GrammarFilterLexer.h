// $ANTLR 3.0b5 /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g 2006-11-12 20:15:18

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>


#pragma mark Cyclic DFA start
@interface GrammarFilterLexerDFA13 : ANTLRDFA {} @end

#pragma mark Cyclic DFA end

#pragma mark Rule return scopes start
#pragma mark Rule return scopes end

#pragma mark Tokens
#define GrammarFilterLexer_GRAMMAR_TYPE	4
#define GrammarFilterLexer_EOF	-1
#define GrammarFilterLexer_WS	5
#define GrammarFilterLexer_STRING	13
#define GrammarFilterLexer_Tokens	14
#define GrammarFilterLexer_LEXER_RULE	11
#define GrammarFilterLexer_OPTIONS	10
#define GrammarFilterLexer_ACTION	12
#define GrammarFilterLexer_COMMENT	9
#define GrammarFilterLexer_GRAMMAR	7
#define GrammarFilterLexer_SL_COMMENT	8
#define GrammarFilterLexer_ID	6

@interface GrammarFilterLexer : ANTLRLexer {
	GrammarFilterLexerDFA13 *dfa13;
	SEL synpred7SyntacticPredicate;
	SEL synpred3SyntacticPredicate;
	SEL synpred8SyntacticPredicate;

		id delegate;

}


- (void) setDelegate:(id)theDelegate;


- (void) mGRAMMAR;
- (void) mGRAMMAR_TYPE;
- (void) mOPTIONS;
- (void) mLEXER_RULE;
- (void) mCOMMENT;
- (void) mSL_COMMENT;
- (void) mACTION;
- (void) mSTRING;
- (void) mID;
- (void) mWS;
- (void) mTokens;
- (void) synpred3;
- (void) synpred7;
- (void) synpred8;



@end