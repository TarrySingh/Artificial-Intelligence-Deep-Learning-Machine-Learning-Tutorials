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



#import "ANTLRBaseRecognizer.h"
#import "ANTLRBitSet.h"
#import "ANTLRCommonToken.h"

@class ANTLRRuleReturnScope;
@implementation ANTLRBaseRecognizer

- (id) init
{
	if (nil != (self = [super init])) {
		state = [[[self stateClass] alloc] init];
	}
	return self;
}

- (void) dealloc
{
	[state release];
	[super dealloc];
}

- (BOOL) isFailed
{
	return [state isFailed];
}

- (void) setIsFailed: (BOOL) flag
{
	[state setIsFailed:flag];
}

- (BOOL) isBacktracking
{
	return [state isBacktracking];
}

- (int) backtrackingLevel
{
	return [state backtracking];
}

- (void) setBacktrackingLevel:(int) level
{
	[state setBacktracking:level];
}

- (ANTLRBaseRecognizerState *) state
{
	return state;
}

- (void) setState:(ANTLRBaseRecognizerState *) theState
{
	if (state != theState) {
		[state release];
		state = [theState retain];
	}
}

- (Class) stateClass
{
	return [ANTLRBaseRecognizerState class];
}


// reset the recognizer to the initial state. does not touch the token source!
// this can be extended by the grammar writer to reset custom ivars
- (void) reset
{
	[state reset];
}

// where do we get our input from - subclass responsibility
- (id) input {
    // subclass responsibility
    return nil;
}

- (void) setInput:(id)theInput
{
    // subclass responsibility
}

// match the next token on the input stream. try to recover with the FOLLOW set is there is a mismatch
- (void) match:(id<ANTLRIntStream>)input 
	 tokenType:(ANTLRTokenType) ttype
		follow:(ANTLRBitSet *)follow
{
	ANTLRTokenType _ttype = [input LA:1];
	if (_ttype == ttype) {
		[input consume];
		[state setIsErrorRecovery:NO];
		[state setIsFailed:NO];
		return;
	}
	if ([state isBacktracking]) {
		[state setIsFailed:YES];
		return;
	}
	[self mismatch:input tokenType:ttype follow:follow];
}

// prepare and exception and try to recover from a mismatch
- (void) mismatch:(id<ANTLRIntStream>)aStream tokenType:(int)aTType follow:(ANTLRBitSet *)aBitset
{
	ANTLRMismatchedTokenException *mte = [ANTLRMismatchedTokenException exceptionWithTokenType:aTType stream:aStream];
	[self recoverFromMismatchedToken:aStream exception:mte tokenType:aTType follow:aBitset];
}

// just consume the next symbol and reset the error ivars
- (void) matchAny:(id<ANTLRIntStream>)input
{
	[state setIsErrorRecovery:NO];
	[state setIsFailed:NO];
	[input consume];
}

// everything failed. report the error
- (void) reportError:(NSException *)e
{
	if ([state isErrorRecovery]) {
		return;
	}
	[state setIsErrorRecovery:YES];
	[self displayRecognitionError:NSStringFromClass([self class]) tokenNames:[self tokenNames] exception:e];
}

// override to implement a different display strategy.
- (void) displayRecognitionError:(NSString *)name tokenNames:(NSArray *)tokenNames exception:(NSException *)e
{
	NSLog(@"%@", [e description]);
}

// try to recover from a mismatch by resyncing
- (void) recover:(id<ANTLRIntStream>)input exception:(NSException *)e
{
	if ([state lastErrorIndex] == [input index]) {
		[input consume];
	}
	[state setLastErrorIndex:[input index]];
	ANTLRBitSet *followSet = [self computeErrorRecoverySet];
	[self beginResync];
	[self consumeUntil:input bitSet:followSet];
	[self endResync];
}

// code smell...:(
// hooks for debugger
- (void) beginResync
{
}

- (void) endResync
{
}

- (void)beginBacktracking:(int)level
{
}

- (void)endBacktracking:(int)level wasSuccessful:(BOOL)successful
{
}

// end hooks for debugger

- (ANTLRBitSet *)computeErrorRecoverySet
{
	return [self combineFollowsExact:NO];
}

- (ANTLRBitSet *)computeContextSensitiveRuleFOLLOW
{
	return [self combineFollowsExact:YES];
}

// compute a new FOLLOW set for recovery using the rules we have descended through
- (ANTLRBitSet *)combineFollowsExact:(BOOL)exact
{
	ANTLRBitSet *followSet = [[[ANTLRBitSet alloc] init] autorelease];
	int i;
	NSMutableArray *following = [state following];
	for (i = [following count]-1; i >= 0; i--) {
		ANTLRBitSet *localFollowSet = [following objectAtIndex:i];
		[followSet orInPlace:localFollowSet];
		if (exact && ![localFollowSet isMember:ANTLRTokenTypeEOR]) {
			break;
		}
	}
	[followSet remove:ANTLRTokenTypeEOR];
	return followSet;
}


// delete one token and try to carry on.
- (void) recoverFromMismatchedToken:(id<ANTLRIntStream>)input 
						  exception:(NSException *)e 
						  tokenType:(ANTLRTokenType)ttype 
							 follow:(ANTLRBitSet *)follow
{
	if ([input LA:2] == ttype) {
		[self reportError:e];
		[self beginResync];
		[input consume];
		[self endResync];
		[input consume];
		return;
	}
	if (![self recoverFromMismatchedElement:input exception:e follow:follow]) {
		@throw e;
	}
}

- (void) recoverFromMismatchedSet:(id<ANTLRIntStream>)input
						exception:(NSException *)e
						   follow:(ANTLRBitSet *)follow
{
	// TODO - recovery is currently incomplete in ANTLR
	if (![self recoverFromMismatchedElement:input exception:e follow:follow]) {
		@throw e;
	}
}

// this code handles single token insertion recovery
- (BOOL) recoverFromMismatchedElement:(id<ANTLRIntStream>)input
							exception:(NSException *)e
							   follow:(ANTLRBitSet *)follow
{
	if (follow == nil) {
		return NO;
	}
	
	// compute the viable symbols that can follow the current rule
	ANTLRBitSet *localFollow = follow;
	if ([follow isMember:ANTLRTokenTypeEOR]) {
		ANTLRBitSet *viableTokensFollowingThisRule = [self computeContextSensitiveRuleFOLLOW];
		localFollow = [follow or:viableTokensFollowingThisRule];
		[localFollow remove:ANTLRTokenTypeEOR];
	}
	// if the current token could follow the missing token we tell the user and proceed with matching
	if ([localFollow isMember:[input LA:1]]) {
		[self reportError:e];
		// clean up the temporary follow set, if we created one
		if (localFollow != follow)
			[localFollow release];
		return YES;
	}
	// clean up the temporary follow set, if we created one
	if (localFollow != follow)
		[localFollow release];
	// otherwise the match fails
	return NO;
}

// used in resyncing to skip to next token of a known type
- (void) consumeUntil:(id<ANTLRIntStream>)input
			tokenType:(ANTLRTokenType)theTtype
{
	ANTLRTokenType ttype = [input LA:1];
	while (ttype != ANTLRTokenTypeEOF && ttype != theTtype) {
		[input consume];
		ttype = [input LA:1];
	}
}

// used in resyncing to skip to the next token whose type in the bitset
- (void) consumeUntil:(id<ANTLRIntStream>)input
			   bitSet:(ANTLRBitSet *)bitSet
{
	ANTLRTokenType ttype = [input LA:1];
	while (ttype != ANTLRTokenTypeEOF && ![bitSet isMember:ttype]) {
		[input consume];
		ttype = [input LA:1];
	}
}

- (void) pushFollow:(ANTLRBitSet *)follow
{
	[[state following] addObject:follow];
}


- (NSArray *) ruleInvocationStack
{
	return [self ruleInvocationStack:nil recognizer:[self class]];
}


- (NSArray *) ruleInvocationStack:(id) exception
					   recognizer:(Class) recognizerClass
{
#warning TODO: ruleInvocationStack:recognizer:
	return [NSArray arrayWithObject:[@"not implemented yet: " stringByAppendingString:NSStringFromClass(recognizerClass)]];
}

+ (NSString *) tokenNameForType:(int)aTokenType
{
    // subclass responsibility
    return nil;
}

- (NSString *) tokenNameForType:(int)aTokenType
{
    return [[self class] tokenNameForType:aTokenType];
}

+ (NSArray *) tokenNames
{
    // subclass responsibility
    return nil;
}

- (NSArray *) tokenNames
{
    return [[self class] tokenNames];
}

- (NSString *) grammarFileName
{
	return grammarFileName;
}

// pure convenience
- (NSArray *) toStrings:(NSArray *)tokens
{
	if (tokens == nil ) {
		return nil;
	}
	NSMutableArray *strings = [[[NSArray alloc] init] autorelease];
	NSEnumerator *tokensEnumerator = [tokens objectEnumerator];
	id value;
	while (nil != (value = [tokensEnumerator nextObject])) {
		[strings addObject:[(id<ANTLRToken>)value text]];
	}
	return strings;
}


// TODO need an Objective-C StringTemplate implementation for this
- (NSArray *) toTemplates:(NSArray *)retvals
{
	return nil;
#warning TODO: Templates are not yet supported in ObjC!
}

// the following methods handle the "memoization" caching functionality
// they work by associating token indices with rule numbers.
// that way, when we are about to parse a rule and have parsed the rule previously, e.g. in prediction,
// we don't have to do it again but can simply return the token index to continue up parsing at.

- (int) ruleMemoization:(unsigned int)ruleIndex startIndex:(int)ruleStartIndex
{
	NSMutableArray *ruleMemo = [state ruleMemo];
	NSAssert([ruleMemo count] >= ruleIndex, @"memoization ruleIndex is out of bounds!");

	NSNumber *stopIndexNumber = [[ruleMemo objectAtIndex:ruleIndex] objectForKey:[NSNumber numberWithInt:ruleStartIndex]];
	if (stopIndexNumber == nil) {
		return ANTLR_MEMO_RULE_UNKNOWN;
	} else {
		return [stopIndexNumber intValue];
	}
}

- (BOOL) alreadyParsedRule:(id<ANTLRIntStream>)input ruleIndex:(unsigned int)ruleIndex
{
	int stopIndex = [self ruleMemoization:ruleIndex startIndex:[input index]];
	if (stopIndex == ANTLR_MEMO_RULE_UNKNOWN) {
		return NO;
	}
	if (stopIndex == ANTLR_MEMO_RULE_FAILED) {
		[state setIsFailed:YES];
	} else {
		[input seek:stopIndex+1];
	}
	return YES;
}

- (void) memoize:(id<ANTLRIntStream>)input
	   ruleIndex:(int)ruleIndex
	  startIndex:(int)ruleStartIndex
{
	NSMutableArray *ruleMemo = [state ruleMemo];
	NSAssert([ruleMemo count] >= ruleIndex, @"memoization ruleIndex is out of bounds!");

	int stopTokenIndex = [state isFailed] ? ANTLR_MEMO_RULE_FAILED : [input index]-1;
	NSMutableDictionary *ruleMemoDict = [ruleMemo objectAtIndex:ruleIndex];
	if ([ruleMemoDict objectForKey:[NSNumber numberWithInt:ruleStartIndex]] == nil) {
		[ruleMemoDict setObject:[NSNumber numberWithInt:stopTokenIndex] forKey:[NSNumber numberWithInt:ruleStartIndex]];
	}
}

- (int) ruleMemoizationCacheSize
{
	int n = 0;
	NSMutableArray *ruleMemo = [state ruleMemo];
	NSEnumerator *ruleEnumerator = [ruleMemo objectEnumerator];
	id value;
	while ((value = [ruleEnumerator nextObject])) {
		n += [value count];
	}
	return n;
}

// call a syntactic predicate methods using its selector. this way we can support arbitrary synpreds.
- (BOOL) evaluateSyntacticPredicate:(SEL)synpredFragment // stream:(id<ANTLRIntStream>)input
{
    [state increaseBacktracking];
	[self beginBacktracking:[state backtracking]];
	int start = [[self input] mark];
    @try {
        [self performSelector:synpredFragment];
    }
    @catch (ANTLRRecognitionException *re) {
        NSLog(@"impossible synpred: %@", re);
    }
    BOOL success = ![state isFailed];
    [[self input] rewind:start];
	[self endBacktracking:[state backtracking] wasSuccessful:success];
	[state decreaseBacktracking];
	[state setIsFailed:NO];
    return success;
}	

@end
