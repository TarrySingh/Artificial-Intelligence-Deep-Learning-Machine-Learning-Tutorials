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


#import "ANTLRUnbufferedCommonTreeNodeStream.h"
#import "ANTLRUnbufferedCommonTreeNodeStreamState.h"
#import "ANTLRTree.h"
#import "ANTLRToken.h"

@interface ANTLRTreeNavigationNode : ANTLRTree {
}
- (id) copyWithZone:(NSZone *)aZone;
@end
@interface ANTLRTreeNavigationNodeDown : ANTLRTreeNavigationNode {
}
- (int) tokenType;
- (NSString *) description;
@end
@interface ANTLRTreeNavigationNodeUp : ANTLRTreeNavigationNode {
}
- (int) tokenType;
- (NSString *) description;
@end
@interface ANTLRTreeNavigationNodeEOF : ANTLRTreeNavigationNode {
}
- (int) tokenType;
- (NSString *) description;
@end

#pragma mark -

static ANTLRTreeNavigationNodeDown *downNavigationNode;
static ANTLRTreeNavigationNodeUp *upNavigationNode;
static ANTLRTreeNavigationNodeEOF *eofNavigationNode;

#define INITIAL_LOOKAHEAD_BUFFER_SIZE 5
@implementation ANTLRUnbufferedCommonTreeNodeStream

+ (void) initialize
{
	downNavigationNode = [[ANTLRTreeNavigationNodeDown alloc] init];
	upNavigationNode = [[ANTLRTreeNavigationNodeUp alloc] init];
	eofNavigationNode = [[ANTLRTreeNavigationNodeEOF alloc] init];
}

- (id) initWithTree:(ANTLRCommonTree *)theTree;
{
	return [self initWithTree:theTree treeAdaptor:nil];
}

- (id) initWithTree:(ANTLRCommonTree *)theTree treeAdaptor:(ANTLRCommonTreeAdaptor *)theAdaptor;
{
	if ((self = [super init]) != nil) {
		[self setRoot:theTree];
		if (!theAdaptor) 
			[self setTreeAdaptor:[[[ANTLRCommonTreeAdaptor alloc] init] autorelease]];
		else
			[self setTreeAdaptor:theAdaptor];
		nodeStack = [[NSMutableArray alloc] init];
		indexStack = [[NSMutableArray alloc] init];
		markers = [[NSMutableArray alloc] initWithObjects:[NSNull null], nil];	// markers is one based - maybe fix this later
		lookahead = [[NSMutableArray alloc] initWithCapacity:INITIAL_LOOKAHEAD_BUFFER_SIZE];	// lookahead is filled with [NSNull null] in -reset
		[self reset];
	}
	return self;
}

- (void) dealloc
{
	[self setRoot:nil];
	[self setTreeAdaptor:nil];
	
	[nodeStack release];	nodeStack = nil;
	[indexStack release];	indexStack = nil;
	[markers release];		markers = nil;
	[lookahead release];	lookahead = nil;
	
	[super dealloc];
}

- (void) reset;
{
	currentNode = root;
	previousNode = nil;
	currentChildIndex = -1;
	absoluteNodeIndex = -1;
	head = tail = 0;
	[nodeStack removeAllObjects];
	[indexStack removeAllObjects];
	[markers removeAllObjects];
	[markers addObject:[NSNull null]];	// markers are 1-based.
	[lookahead removeAllObjects];
	// TODO: this is not ideal, but works for now. optimize later
	int i;
	for (i = 0; i < INITIAL_LOOKAHEAD_BUFFER_SIZE; i++)
		[lookahead addObject:[NSNull null]];
}


#pragma mark ANTLRTreeNodeStream conformance

- (id) LT:(int)k;
{
	if (k == -1)
		return previousNode;
	if (k < 0)
		@throw [NSException exceptionWithName:@"ANTLRTreeException" reason:@"-LT: looking back more than one node unsupported for unbuffered streams" userInfo:nil];
	if (k == 0)
		return [ANTLRTree invalidNode];
	[self fillBufferWithLookahead:k];
	return [lookahead objectAtIndex:(head+k-1) % [lookahead count]];
}

- (id) treeSource;
{
	return [self root];
}

- (id<ANTLRTreeAdaptor>) treeAdaptor;
{
	return treeAdaptor;
}

- (id<ANTLRTokenStream>) tokenStream; 
{
	return tokenStream;
}

- (void) setUsesUniqueNavigationNodes:(BOOL)flag;
{
	shouldUseUniqueNavigationNodes = flag;
}

- (id) nodeAtIndex:(unsigned int) idx;
{
	@throw [NSException exceptionWithName:@"ANTLRTreeException" reason:@"-nodeAtIndex: unsupported for unbuffered streams" userInfo:nil];
}

- (NSString *) stringValue;
{
	@throw [NSException exceptionWithName:@"ANTLRTreeException" reason:@"-stringValue unsupported for unbuffered streams" userInfo:nil];
}

- (NSString *) stringValueWithRange:(NSRange) aRange;
{
	@throw [NSException exceptionWithName:@"ANTLRTreeException" reason:@"-stringValue: unsupported for unbuffered streams" userInfo:nil];
}

- (NSString *) stringValueFromNode:(id)startNode toNode:(id)stopNode;
{
	@throw [NSException exceptionWithName:@"ANTLRTreeException" reason:@"-stringValueFromNode:toNode: unsupported for unbuffered streams" userInfo:nil];
}

#pragma mark ANTLRIntStream conformance

- (void) consume;
{
	[self fillBufferWithLookahead:1];
	absoluteNodeIndex++;
	previousNode = [lookahead objectAtIndex:head];
	head = (head+1) % [lookahead count];
}

- (int) LA:(unsigned int) i;
{
	ANTLRCommonTree *node = [self LT:i];
	if (!node) 
		return ANTLRTokenTypeInvalid;
	int ttype = [node tokenType];
	return ttype;
}

- (unsigned int) mark;
{
	ANTLRUnbufferedCommonTreeNodeStreamState *state = [[ANTLRUnbufferedCommonTreeNodeStreamState alloc] init];
	[state setCurrentNode:currentNode];
	[state setPreviousNode:previousNode];
	[state setIndexStackSize:[indexStack count]];
	[state setNodeStackSize:[nodeStack count]];
	[state setCurrentChildIndex:currentChildIndex];
	[state setAbsoluteNodeIndex:absoluteNodeIndex];
	unsigned int lookaheadSize = [self lookaheadSize];
	unsigned int k;
	for ( k = 1; k <= lookaheadSize; k++) {
		[state addToLookahead:[self LT:k]];
	}
	[markers addObject:state];
	[state release];
	return [markers count];
}

- (unsigned int) index;
{
	return absoluteNodeIndex + 1;
}

- (void) rewind:(unsigned int) marker;
{
	if ( [markers count] < marker ) {
		return;
	}
	ANTLRUnbufferedCommonTreeNodeStreamState *state = [[markers objectAtIndex:marker-1] retain];
	[markers removeObjectAtIndex:marker-1];

	absoluteNodeIndex = [state absoluteNodeIndex];
	currentChildIndex = [state currentChildIndex];
	currentNode = [state currentNode];
	previousNode = [state previousNode];
	// drop node and index stacks back to old size
	[nodeStack removeObjectsInRange:NSMakeRange([state nodeStackSize], [nodeStack count]-[state nodeStackSize])];
	[indexStack removeObjectsInRange:NSMakeRange([state indexStackSize], [indexStack count]-[state indexStackSize])];
	
	head = tail = 0; // wack lookahead buffer and then refill
	[lookahead release];
	lookahead = [[NSMutableArray alloc] initWithArray:[state lookahead]];
	tail = [lookahead count];
	// make some room after the restored lookahead, so that the above line is not a bug ;)
	// this also ensures that a subsequent -addLookahead: will not immediately need to resize the buffer
	[lookahead addObjectsFromArray:[NSArray arrayWithObjects:[NSNull null], [NSNull null], [NSNull null], [NSNull null], [NSNull null], nil]];
}

- (void) rewind
{
	[self rewind:[markers count]];
}

- (void) release:(unsigned int) marker;
{
	@throw [NSException exceptionWithName:@"ANTLRTreeException" reason:@"-release: unsupported for unbuffered streams" userInfo:nil];
}

- (void) seek:(unsigned int) index;
{
	if ( index < [self index] )
		@throw [NSException exceptionWithName:@"ANTLRTreeException" reason:@"-seek: backwards unsupported for unbuffered streams" userInfo:nil];
	while ( [self index] < index ) {
		[self consume];
	}
}

- (unsigned int) count;
{
	return absoluteNodeIndex + 1;	// not entirely correct, but cheap.
}


#pragma mark Lookahead Handling
- (void) addLookahead:(id<ANTLRTree>)aNode;
{
	[lookahead replaceObjectAtIndex:tail withObject:aNode];
	tail = (tail+1) % [lookahead count];
	
	if ( tail == head ) {
		NSMutableArray *newLookahead = [[NSMutableArray alloc] initWithCapacity:[lookahead count]*2];
		
		NSRange headRange = NSMakeRange(head, [lookahead count]-head);
		NSRange tailRange = NSMakeRange(0, tail);
		
		[newLookahead addObjectsFromArray:[lookahead objectsAtIndexes:[NSIndexSet indexSetWithIndexesInRange:headRange]]];
		[newLookahead addObjectsFromArray:[lookahead objectsAtIndexes:[NSIndexSet indexSetWithIndexesInRange:tailRange]]];
		
		unsigned int i;
		unsigned int lookaheadCount = [newLookahead count];
		for (i = 0; i < lookaheadCount; i++)
			[newLookahead addObject:[NSNull null]];
		[lookahead release];
		lookahead = newLookahead;
		
		head = 0;
		tail = lookaheadCount;	// tail is the location the _next_ lookahead node will end up in, not the last element's idx itself!
	}
	
}

- (unsigned int) lookaheadSize;
{
	return tail < head
		? ([lookahead count] - head + tail) 
		: (tail - head);
}

- (void) fillBufferWithLookahead:(int)k;
{
	unsigned int n = [self lookaheadSize];
	unsigned int i;
	id lookaheadObject = self; // any valid object would do.
	for (i=1; i <= k-n && lookaheadObject != nil; i++) {
		lookaheadObject = [self nextObject];
	}
}

- (id) nextObject
{
	// NOTE: this could/should go into an NSEnumerator subclass for treenode streams.
	if (currentNode == nil) {
		[self addLookahead:eofNavigationNode];
		return nil;
	}
	if (currentChildIndex == -1) {
		return [self handleRootNode];
	}
	if (currentChildIndex < (int)[currentNode childCount]) {
		return [self visitChild:currentChildIndex];
	}
	[self walkBackToMostRecentNodeWithUnvisitedChildren];
	if (currentNode != nil) {
		return [self visitChild:currentChildIndex];
	}
	
	return nil;
}	

#pragma mark Node visiting
- (ANTLRCommonTree *) handleRootNode;
{
	ANTLRCommonTree *node = currentNode;
	currentChildIndex = 0;
	if ([node isEmpty]) {
		node = [self visitChild:currentChildIndex];
	} else {
		[self addLookahead:node];
		if ([currentNode childCount] == 0) {
			currentNode = nil;
		}
	}
	return node;
}

- (ANTLRCommonTree *) visitChild:(int)childNumber;
{
	ANTLRCommonTree *node = nil;
	
	[nodeStack addObject:currentNode];
	[indexStack addObject:[NSNumber numberWithInt:childNumber]];
	if (childNumber == 0 && ![currentNode isEmpty])
		[self addNavigationNodeWithType:ANTLRTokenTypeDOWN];

	currentNode = [currentNode childAtIndex:childNumber];
	currentChildIndex = 0;
	node = currentNode;  // record node to return
	[self addLookahead:node];
	[self walkBackToMostRecentNodeWithUnvisitedChildren];
	return node;
}

- (void) walkBackToMostRecentNodeWithUnvisitedChildren;
{
	while (currentNode != nil && currentChildIndex >= (int)[currentNode childCount])
	{
		currentNode = (ANTLRCommonTree *)[nodeStack lastObject];
		[nodeStack removeLastObject];
		currentChildIndex = [(NSNumber *)[indexStack lastObject] intValue];
		[indexStack removeLastObject];
		currentChildIndex++; // move to next child
		if (currentChildIndex >= (int)[currentNode childCount]) {
			if (![currentNode isEmpty]) {
				[self addNavigationNodeWithType:ANTLRTokenTypeUP];
			}
			if (currentNode == root) { // we done yet?
				currentNode = nil;
			}
		}
	}
	
}

- (void) addNavigationNodeWithType:(int)tokenType;
{
	// TODO: this currently ignores shouldUseUniqueNavigationNodes.
	switch (tokenType) {
		case ANTLRTokenTypeDOWN: {
			[self addLookahead:downNavigationNode];
			break;
		}
		case ANTLRTokenTypeUP: {
			[self addLookahead:upNavigationNode];
			break;
		}
	}
}

#pragma mark Accessors
- (ANTLRCommonTree *) root
{
    return root; 
}

- (void) setRoot: (ANTLRCommonTree *) aRoot
{
    if (root != aRoot) {
        [aRoot retain];
        [root release];
        root = aRoot;
    }
}

- (void) setTreeAdaptor: (ANTLRCommonTreeAdaptor *) aTreeAdaptor
{
    if (treeAdaptor != aTreeAdaptor) {
        [aTreeAdaptor retain];
        [treeAdaptor release];
        treeAdaptor = aTreeAdaptor;
    }
}

- (void) setTokenStream:(id<ANTLRTokenStream>) aTokenStream;
{
	if (tokenStream != aTokenStream) {
		[tokenStream release];
		tokenStream = [aTokenStream retain];
	}
}

@end

#pragma mark -

@implementation ANTLRTreeNavigationNode
- (id) copyWithZone:(NSZone *)aZone
{
	return nil;
}
@end

@implementation ANTLRTreeNavigationNodeDown
- (int) tokenType { return ANTLRTokenTypeDOWN; }
- (NSString *) description { return @"DOWN"; }
@end

@implementation ANTLRTreeNavigationNodeUp
- (int) tokenType { return ANTLRTokenTypeUP; }
- (NSString *) description { return @"UP"; }
@end

@implementation ANTLRTreeNavigationNodeEOF
- (int) tokenType { return ANTLRTokenTypeEOF; }
- (NSString *) description { return @"EOF"; }
@end

