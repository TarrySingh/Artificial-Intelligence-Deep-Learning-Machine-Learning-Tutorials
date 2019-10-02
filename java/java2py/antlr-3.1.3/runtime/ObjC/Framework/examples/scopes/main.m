#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>
#import "SymbolTableLexer.h"
#import "SymbolTableParser.h"

int main() {
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
	
	NSString *string = [NSString stringWithContentsOfFile:@"examples/scopes/input"];
	NSLog(@"input is : %@", string);
	ANTLRStringStream *stream = [[ANTLRStringStream alloc] initWithStringNoCopy:string];
	SymbolTableLexer *lexer = [[SymbolTableLexer alloc] initWithCharStream:stream];
	
	//	ANTLRToken *currentToken;
	//	while ((currentToken = [lexer nextToken]) && [currentToken type] != ANTLRTokenTypeEOF) {
	//		NSLog(@"%@", currentToken);
	//	}
	
	ANTLRCommonTokenStream *tokenStream = [[ANTLRCommonTokenStream alloc] initWithTokenSource:lexer];
	SymbolTableParser *parser = [[SymbolTableParser alloc] initWithTokenStream:tokenStream];
	[parser prog];
	[lexer release];
	[stream release];
	[tokenStream release];
	[parser release];
	
	[pool release];
	return 0;
}