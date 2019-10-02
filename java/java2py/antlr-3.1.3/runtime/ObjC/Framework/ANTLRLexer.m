// [The "BSD licence"]
// Copyright (c) 2006-2007 Kay Roepke
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#import "ANTLRLexer.h"


@implementation ANTLRLexer

// init
#pragma mark Initializer
- (id) initWithCharStream:(id<ANTLRCharStream>)anInput
{
	if (nil != (self = [super init])) {
		[self setInput:anInput];
		tokenStartCharIndex = -1;
		ruleNestingLevel = 0;
	}
	return self;
}

- (void) reset
{
	tokenStartCharIndex = -1;
	ruleNestingLevel = 0;
	[super reset];
}

- (void) dealloc
{
    [self setToken:nil];
    [self setInput:nil];
    [self setText:nil];
    [super dealloc];
}

	// token stuff
#pragma mark Tokens

- (id<ANTLRToken>) token
{
    return token; 
}

- (void) setToken: (id<ANTLRToken>) aToken
{
    if (token != aToken) {
        [aToken retain];
        [token release];
        token = aToken;
    }
}


// this method may be overridden in the generated lexer if we generate a filtering lexer.
- (id<ANTLRToken>) nextToken
{
	[self setToken:nil];
    _channel = ANTLRTokenChannelDefault;
    _tokenStartLine = [input line];
    _tokenCharPositionInLine = [input charPositionInLine];
	tokenStartCharIndex = [self charIndex];
    
	while (YES) {
        [self setText:nil];
		if ([input LA:1] == ANTLRCharStreamEOF) {
			return [ANTLRCommonToken eofToken];
			return nil;
		}
		@try {
			[self mTokens];
            if (token == nil)
                [self emit];
			return token;
		}
		@catch (ANTLRRecognitionException *e) {
			[self reportError:e];
			[self recover:e];
		}
	}
}

- (void) mTokens { [self doesNotRecognizeSelector:_cmd]; }		// abstract, defined in generated source as a starting point for matching
- (id<ANTLRCharStream>) input
{
    return input; 
}

- (void) setInput: (id<ANTLRCharStream>) anInput
{
    if (input != anInput) {
        [anInput retain];
        [input release];
        input = anInput;
    }
	[self setToken:nil];
	tokenStartCharIndex = -1;
}

// this method is used by the code generator to automatically emit tokens from the lexer.
// for now it will always return ANTLRCommonTokens
// use a manual emit: in the grammar to return custom tokens or override this method in your
// lexer
- (void) emit
{
	id<ANTLRToken> aToken = [[ANTLRCommonToken alloc] initWithInput:input 
                                                          tokenType:_tokenType 
                                                            channel:_channel
                                                              start:tokenStartCharIndex
                                                               stop:[self charIndex]-1];
	[aToken setLine:_tokenStartLine];
	[aToken setCharPositionInLine:_tokenCharPositionInLine];
    if (text != nil)
        [aToken setText:text];
	[self emit:aToken];
	[aToken release];
}

// use this to emit custom tokens from a lexer rule
- (void) emit:(id<ANTLRToken>)aToken
{
	[self setToken:aToken];
}

// matching
#pragma mark Matching
- (void) matchString:(NSString *)aString
{
	unsigned int i = 0;
	unsigned int stringLength = [aString length];
	while ( i < stringLength ) {
		if ((unichar)[input LA:1] != [aString characterAtIndex:i]) {
			if (backtracking > 0) {
				failed = YES;
				return;
			}
			ANTLRMismatchedTokenException  *mte = [ANTLRMismatchedTokenException exceptionWithCharacter:[aString characterAtIndex:i] stream:input];
			[self recover:mte];
			@throw mte;
		}
		i++;
		[input consume];
		failed = NO;
	}
}

- (void) matchAny
{
	[input consume];
}

- (void) matchChar:(unichar) aChar
{
	// TODO: -LA: is returning an int because it sometimes is used in the generated parser to compare lookahead with a tokentype.
	//		 try to change all those occurrences to -LT: if possible (i.e. if ANTLR can be made to generate LA only for lexer code)
	if ((unichar)[input LA:1] != aChar) {
		if (backtracking > 0) {
			failed = YES;
			return;
		}
		ANTLRMismatchedTokenException  *mte = [ANTLRMismatchedTokenException exceptionWithCharacter:aChar stream:input];
		[self recover:mte];
		@throw mte;
	}
	[input consume];
	failed = NO;
}

- (void) matchRangeFromChar:(unichar)fromChar to:(unichar)toChar
{
	unichar charLA = (unichar)[input LA:1];
	if ( charLA < fromChar || charLA > toChar ) {
		if (backtracking > 0) {
			failed = YES;
			return;
		}
		ANTLRMismatchedRangeException  *mre = [ANTLRMismatchedRangeException
					exceptionWithRange:NSMakeRange((unsigned int)fromChar,(unsigned int)toChar)
							   stream:input];
		[self recover:mre];
		@throw mre;
	}		
	[input consume];
	failed = NO;
}

	// info
#pragma mark Informational

- (unsigned int) line
{
	return [input line];
}
- (unsigned int) charPositionInLine
{
	return [input charPositionInLine];
}
- (unsigned int) charIndex
{
	return [input index];
}
- (NSString *) text
{
	return [input substringWithRange:NSMakeRange(tokenStartCharIndex, [self charIndex]-tokenStartCharIndex)];
}
- (void) setText:(NSString *) theText
{
    if (theText != text) {
        [text release];
        text = [theText retain];
    }
}

	// error handling
- (void) reportError:(ANTLRRecognitionException *)e
{
	NSLog(@"%@", e);
}

- (void) recover:(ANTLRRecognitionException *)e
{
	[input consume];
}

@end
