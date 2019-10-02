// $ANTLR 3.0 FuzzyJava.gl 2007-07-25 20:12:38

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>


#pragma mark Rule return scopes start
#pragma mark Rule return scopes end

#pragma mark Tokens
#define FuzzyJavaLexer_QIDStar	5
#define FuzzyJavaLexer_TYPE	11
#define FuzzyJavaLexer_STAT	15
#define FuzzyJavaLexer_WS	4
#define FuzzyJavaLexer_CHAR	21
#define FuzzyJavaLexer_QID	9
#define FuzzyJavaLexer_STRING	20
#define FuzzyJavaLexer_METHOD	13
#define FuzzyJavaLexer_COMMENT	17
#define FuzzyJavaLexer_ESC	19
#define FuzzyJavaLexer_IMPORT	6
#define FuzzyJavaLexer_FIELD	14
#define FuzzyJavaLexer_CLASS	10
#define FuzzyJavaLexer_RETURN	7
#define FuzzyJavaLexer_ARG	12
#define FuzzyJavaLexer_EOF	-1
#define FuzzyJavaLexer_CALL	16
#define FuzzyJavaLexer_Tokens	22
#define FuzzyJavaLexer_SL_COMMENT	18
#define FuzzyJavaLexer_ID	8

@interface FuzzyJavaLexer : ANTLRLexer {
	SEL synpred4SyntacticPredicate;
	SEL synpred9SyntacticPredicate;
	SEL synpred2SyntacticPredicate;
	SEL synpred7SyntacticPredicate;
	SEL synpred3SyntacticPredicate;
	SEL synpred1SyntacticPredicate;
	SEL synpred5SyntacticPredicate;
	SEL synpred6SyntacticPredicate;
	SEL synpred8SyntacticPredicate;
}


- (void) mIMPORT;
- (void) mRETURN;
- (void) mCLASS;
- (void) mMETHOD;
- (void) mFIELD;
- (void) mSTAT;
- (void) mCALL;
- (void) mCOMMENT;
- (void) mSL_COMMENT;
- (void) mSTRING;
- (void) mCHAR;
- (void) mWS;
- (void) mQID;
- (void) mQIDStar;
- (void) mTYPE;
- (void) mARG;
- (void) mID;
- (void) mESC;
- (void) mTokens;
- (void) synpred1;
- (void) synpred2;
- (void) synpred3;
- (void) synpred4;
- (void) synpred5;
- (void) synpred6;
- (void) synpred7;
- (void) synpred8;
- (void) synpred9;



@end