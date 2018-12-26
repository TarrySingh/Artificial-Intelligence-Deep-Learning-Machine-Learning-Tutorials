// $ANTLR 3.0 T.g 2007-07-25 20:12:43

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLR.h>



#pragma mark Tokens
#define TParser_INT	5
#define TParser_WS	6
#define TParser_EOF	-1
#define TParser_ID	4

#pragma mark Dynamic Global Scopes

#pragma mark Dynamic Rule Scopes

#pragma mark Rule Return Scopes


@interface TParser : ANTLRParser {

					


	/** With this true, enum is seen as a keyword.  False, it's an identifier */
	BOOL enableEnum;

 }


- (void) stat;
- (void) identifier;
- (void) enumAsKeyword;
- (void) enumAsID;



@end