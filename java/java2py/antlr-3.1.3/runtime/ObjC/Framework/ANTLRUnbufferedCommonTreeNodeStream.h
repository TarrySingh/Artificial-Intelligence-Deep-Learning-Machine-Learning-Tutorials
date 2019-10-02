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


#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLRTreeNodeStream.h>
#import <ANTLR/ANTLRCommonTokenStream.h>
#import <ANTLR/ANTLRCommonTree.h>
#import <ANTLR/ANTLRCommonTreeAdaptor.h>

@interface ANTLRUnbufferedCommonTreeNodeStream : NSObject < ANTLRTreeNodeStream > {

	BOOL shouldUseUniqueNavigationNodes;

	ANTLRCommonTree *root;
	ANTLRCommonTree *currentNode;
	ANTLRCommonTree *previousNode;

	ANTLRCommonTreeAdaptor *treeAdaptor;
	
	ANTLRCommonTokenStream *tokenStream;
	
	NSMutableArray *nodeStack;
	NSMutableArray *indexStack;
	NSMutableArray *markers;
	int lastMarker;
	
	int currentChildIndex;
	int absoluteNodeIndex;
	
	NSMutableArray *lookahead;
	unsigned int head;
	unsigned int tail;
}

- (id) initWithTree:(ANTLRCommonTree *)theTree;
- (id) initWithTree:(ANTLRCommonTree *)theTree treeAdaptor:(ANTLRCommonTreeAdaptor *)theAdaptor;

- (void) reset;

#pragma mark ANTLRTreeNodeStream conformance

- (id) LT:(int)k;
- (id) treeSource;
- (id<ANTLRTreeAdaptor>) treeAdaptor;
- (id<ANTLRTokenStream>) tokenStream;
- (void) setTokenStream:(id<ANTLRTokenStream>) aTokenStream;	///< Added by subclass, not in protocol
- (void) setUsesUniqueNavigationNodes:(BOOL)flag;

- (id) nodeAtIndex:(unsigned int) idx;

- (NSString *) stringValue;
- (NSString *) stringValueWithRange:(NSRange) aRange;
- (NSString *) stringValueFromNode:(id)startNode toNode:(id)stopNode;

#pragma mark ANTLRIntStream conformance
- (void) consume;
- (int) LA:(unsigned int) i;
- (unsigned int) mark;
- (unsigned int) index;
- (void) rewind:(unsigned int) marker;
- (void) rewind;
- (void) release:(unsigned int) marker;
- (void) seek:(unsigned int) index;
- (unsigned int) count;

#pragma mark Lookahead Handling
- (void) addLookahead:(id<ANTLRTree>)aNode;
- (unsigned int) lookaheadSize;
- (void) fillBufferWithLookahead:(int)k;
- (id) nextObject;

#pragma mark Node visiting
- (ANTLRCommonTree *) handleRootNode;
- (ANTLRCommonTree *) visitChild:(int)childNumber;
- (void) walkBackToMostRecentNodeWithUnvisitedChildren;
- (void) addNavigationNodeWithType:(int)tokenType;

#pragma mark Accessors
- (ANTLRCommonTree *) root;
- (void) setRoot: (ANTLRCommonTree *) aRoot;

- (void) setTreeAdaptor: (ANTLRCommonTreeAdaptor *) aTreeAdaptor;

@end
