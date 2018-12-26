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

#import "ANTLRTreeAdaptor.h"
#import <ANTLR/ANTLRTreeException.h>

@implementation ANTLRTreeAdaptor


- (id<ANTLRTree>) newEmptyTree
{
	return [self newTreeWithToken:nil];
}


- (id) copyNode:(id<ANTLRTree>)aNode
{
	return [aNode copyWithZone:nil];	// not -copy: to silence warnings
}

- (id) copyTree:(id<ANTLRTree>)aTree
{
	return [aTree deepCopy];
}


- (void) addChild:(id<ANTLRTree>)child toTree:(id<ANTLRTree>)aTree
{
	[aTree addChild:child];
}

- (id) makeNode:(id<ANTLRTree>)newRoot parentOf:(id<ANTLRTree>)oldRoot
{
	id<ANTLRTree> newRootNode = newRoot;

	if (oldRoot == nil)
		return newRootNode;
    // handles ^(nil real-node) case
	if ([newRootNode isEmpty]) {
		if ([newRootNode childCount] > 1) {
#warning TODO: Find a way to the current input stream here!
			@throw [ANTLRTreeException exceptionWithOldRoot:oldRoot newRoot:newRootNode stream:nil];
		}
#warning TODO: double check memory management with respect to code generation
		// remove the empty node, placing its sole child in its role.
		id<ANTLRTree> tmpRootNode = [[newRootNode childAtIndex:0] retain];
		[newRootNode release];
		newRootNode = tmpRootNode;		
	}
	// the handling of an empty node at the root of oldRoot happens in addChild:
	[newRootNode addChild:oldRoot];
    // this release relies on the fact that the ANTLR code generator always assigns the return value of this method
    // to the variable originally holding oldRoot. If we don't release we leak the reference.
    // FIXME: this is totally non-obvious. maybe do it in calling code by comparing pointers and conditionally releasing
    // the old object
    [oldRoot release];
    
    // what happens to newRootNode's retain count? Should we be autoreleasing this one? Probably.
	return [newRootNode retain];
}


- (id<ANTLRTree>) postProcessTree:(id<ANTLRTree>)aTree
{
	id<ANTLRTree> processedNode = aTree;
	if (aTree != nil && [aTree isEmpty] != NO && [aTree childCount] == 1) {
		processedNode = [aTree childAtIndex:0];
	}
	return processedNode;
}


- (unsigned int) uniqueIdForTree:(id<ANTLRTree>)aNode
{
	// TODO: is hash appropriate here?
	return [aNode hash];
}


#pragma mark Rewrite Rules

- (id<ANTLRTree>) newTreeWithTokenType:(int)tokenType
{
	id<ANTLRToken> newToken = [self newTokenWithTokenType:tokenType text:nil];
	
	id<ANTLRTree> newTree = [self newTreeWithToken:newToken];
	[newToken release];
	return newTree;
}

- (id<ANTLRTree>) newTreeWithTokenType:(int)tokenType text:(NSString *)tokenText
{
	id<ANTLRToken> newToken = [self newTokenWithTokenType:tokenType text:tokenText];
	
	id<ANTLRTree> newTree = [self newTreeWithToken:newToken];
	[newToken release];
	return newTree;
}

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken tokenType:(int)tokenType
{
	id<ANTLRToken> newToken = [self newTokenWithToken:fromToken];
	[newToken setType:tokenType];

	id<ANTLRTree> newTree = [self newTreeWithToken:newToken];
	[newToken release];
	return newTree;
}

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken tokenType:(int)tokenType text:(NSString *)tokenText
{
	id<ANTLRToken> newToken = [self newTokenWithToken:fromToken];
	[newToken setType:tokenType];
	[newToken setText:tokenText];

	id<ANTLRTree> newTree = [self newTreeWithToken:newToken];
	[newToken release];
	return newTree;
}

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken text:(NSString *)tokenText
{
	id<ANTLRToken> newToken = [self newTokenWithToken:fromToken];
	[newToken setText:tokenText];
	
	id<ANTLRTree> newTree = [self newTreeWithToken:newToken];
	[newToken release];
	return newTree;
}


#pragma mark Content

- (int) tokenTypeForNode:(id<ANTLRTree>)aNode
{
	return [aNode tokenType];
}

- (void) setTokenType:(int)tokenType forNode:(id)aNode
{
	// currently unimplemented
}


- (NSString *) textForNode:(id<ANTLRTree>)aNode
{
	return [aNode text];
}

- (void) setText:(NSString *)tokenText forNode:(id<ANTLRTree>)aNode
{
	// currently unimplemented
}


#pragma mark Navigation / Tree Parsing

- (id<ANTLRTree>) childForNode:(id<ANTLRTree>) aNode atIndex:(int) i
{
	// currently unimplemented
	return nil;
}

- (int) childCountForTree:(id<ANTLRTree>) aTree
{
	// currently unimplemented
	return 0;
}

#pragma mark Subclass Responsibilties

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>) payload
{
	// subclass responsibility
	return nil;
}

- (id<ANTLRToken>) newTokenWithToken:(id<ANTLRToken>)fromToken
{
	// subclass responsibility
	return nil;
}

- (id<ANTLRToken>) newTokenWithTokenType:(int)tokenType text:(NSString *)tokenText
{
	// subclass responsibility
	return nil;
}

- (void) setBoundariesForTree:(id<ANTLRTree>)aTree fromToken:(id<ANTLRToken>)startToken toToken:(id<ANTLRToken>)stopToken
{
	// subclass responsibility
}

- (int) tokenStartIndexForTree:(id<ANTLRTree>)aTree
{
	// subclass responsibility
	return 0;
}

- (int) tokenStopIndexForTree:(id<ANTLRTree>)aTree
{
	// subclass responsibility
	return 0;
}


@end
