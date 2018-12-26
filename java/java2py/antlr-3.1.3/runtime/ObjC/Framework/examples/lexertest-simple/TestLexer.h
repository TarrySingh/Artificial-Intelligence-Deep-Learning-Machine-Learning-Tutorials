// $ANTLR 3.0 Test.gl 2007-08-04 15:59:43

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>


#pragma mark Rule return scopes start
#pragma mark Rule return scopes end

#pragma mark Tokens
#define TestLexer_LETTER	4
#define TestLexer_EOF	-1
#define TestLexer_Tokens	7
#define TestLexer_DIGIT	5
#define TestLexer_ID	6

@interface TestLexer : ANTLRLexer {
}


- (void) mID;
- (void) mDIGIT;
- (void) mLETTER;
- (void) mTokens;



@end