#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>
#import "TLexer.h"
#import "TParser.h"

int main() {
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
	
	NSString *string = [NSString stringWithContentsOfFile:@"examples/hoistedPredicates/input"];
	NSLog(@"input is : %@", string);
	ANTLRStringStream *stream = [[ANTLRStringStream alloc] initWithStringNoCopy:string];
	TLexer *lexer = [[TLexer alloc] initWithCharStream:stream];
	
	//	ANTLRToken *currentToken;
	//	while ((currentToken = [lexer nextToken]) && [currentToken type] != ANTLRTokenTypeEOF) {
	//		NSLog(@"%@", currentToken);
	//	}
	
	ANTLRCommonTokenStream *tokenStream = [[ANTLRCommonTokenStream alloc] initWithTokenSource:lexer];
	TParser *parser = [[TParser alloc] initWithTokenStream:tokenStream];
	[parser stat];
	[lexer release];
	[stream release];
	[tokenStream release];
	[parser release];
	
	[pool release];
	return 0;
}