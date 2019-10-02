#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>
#import "SimpleCLexer.h"
#import "SimpleCParser.h"

int main() {
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

	NSString *string = [NSString stringWithContentsOfFile:@"examples/LL-star/input"];
	NSLog(@"input is: %@", string);
	ANTLRStringStream *stream = [[ANTLRStringStream alloc] initWithStringNoCopy:string];
	SimpleCLexer *lexer = [[SimpleCLexer alloc] initWithCharStream:stream];

//	ANTLRToken *currentToken;
//	while ((currentToken = [lexer nextToken]) && [currentToken type] != ANTLRTokenTypeEOF) {
//		NSLog(@"%@", currentToken);
//	}
	
	ANTLRCommonTokenStream *tokenStream = [[ANTLRCommonTokenStream alloc] initWithTokenSource:lexer];
	SimpleCParser *parser = [[SimpleCParser alloc] initWithTokenStream:tokenStream];
	[parser program];
	[lexer release];
	[stream release];
	[tokenStream release];
	[parser release];

	[pool release];
	return 0;
}