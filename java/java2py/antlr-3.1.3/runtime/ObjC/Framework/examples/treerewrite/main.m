#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>
#import "TreeRewriteLexer.h"
#import "TreeRewriteParser.h"
#import "stdio.h"
#include <unistd.h>

int main(int argc, const char * argv[]) {
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

	ANTLRStringStream *stream = [[ANTLRStringStream alloc] initWithStringNoCopy:@"23 42"];
	TreeRewriteLexer *lexer = [[TreeRewriteLexer alloc] initWithCharStream:stream];
	
//    id<ANTLRToken> currentToken;
//    while ((currentToken = [lexer nextToken]) && [currentToken type] != ANTLRTokenTypeEOF) {
//        NSLog(@"%@", currentToken);
//    }
	
	ANTLRCommonTokenStream *tokenStream = [[ANTLRCommonTokenStream alloc] initWithTokenSource:lexer];
	TreeRewriteParser *parser = [[TreeRewriteParser alloc] initWithTokenStream:tokenStream];
	ANTLRCommonTree *rule_tree = [[parser rule] tree];
	NSLog(@"tree: %@", [rule_tree treeDescription]);
//	ANTLRCommonTreeNodeStream *treeStream = [[ANTLRCommonTreeNodeStream alloc] initWithTree:program_tree];
//	SimpleCTP *walker = [[SimpleCTP alloc] initWithTreeNodeStream:treeStream];
//	[walker program];

	[lexer release];
	[stream release];
	[tokenStream release];
	[parser release];
//	[treeStream release];
//	[walker release];

	[pool release];
    // sleep for objectalloc
    while(1) sleep(60);
	return 0;
}