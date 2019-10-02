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

@protocol ANTLRTree < NSObject, NSCopying >

+ (id<ANTLRTree>) invalidNode;

- (id<ANTLRTree>) childAtIndex:(unsigned int) index;
- (unsigned int) childCount;

- (NSArray *) allChildren;
// Add t as a child to this node.  If t is null, do nothing.  If t
//  is nil, add all children of t to this' children.

- (void) addChild:(id<ANTLRTree>) tree;
- (void) addChildren:(NSArray *) theChildren;
- (void) removeAllChildren;

// Indicates the node is a nil node but may still have children, meaning
// the tree is a flat list.

- (BOOL) isEmpty;
- (void) setIsEmpty:(BOOL)emptyFlag;

#pragma mark Copying
- (id) copyWithZone:(NSZone *)aZone;	// the children themselves are not copied here!
- (id) deepCopy;					// performs a deepCopyWithZone: with the default zone
- (id) deepCopyWithZone:(NSZone *)aZone;

#pragma mark Tree Parser support
- (int) tokenType;
- (NSString *) text;
// In case we don't have a token payload, what is the line for errors?
- (int) line;
- (int) charPositionInLine;

#pragma mark Informational
- (NSString *) treeDescription;
- (NSString *) description;

@end


@interface ANTLRTree : NSObject < ANTLRTree >
{
	NSMutableArray *children;
	BOOL isEmptyNode;
}

+ (id<ANTLRTree>) invalidNode;
- (id<ANTLRTree>) init;

- (id<ANTLRTree>) childAtIndex:(unsigned int) index;
- (unsigned int) childCount;
- (NSArray *) allChildren;
- (void) removeAllChildren;

	// Add t as a child to this node.  If t is null, do nothing.  If t
	//  is nil, add all children of t to this' children.

- (void) addChild:(id<ANTLRTree>) tree;
- (void) addChildren:(NSArray *) theChildren;
	// Indicates the node is a nil node but may still have children, meaning
	// the tree is a flat list.

- (BOOL) isEmpty;
- (void) setIsEmpty:(BOOL)emptyFlag;

- (id) copyWithZone:(NSZone *)aZone;
- (id) deepCopy;					// performs a deepCopyWithZone: with the default zone
- (id) deepCopyWithZone:(NSZone *)aZone;

	// Return a token type; needed for tree parsing
- (int) tokenType;

- (NSString *) text;

	// In case we don't have a token payload, what is the line for errors?
- (int) line;
- (int) charPositionInLine;

- (NSString *) treeDescription;
- (NSString *) description;

@end
