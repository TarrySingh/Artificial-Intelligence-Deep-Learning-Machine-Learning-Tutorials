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

#import "ANTLRTree.h"
#import "ANTLRToken.h"
// TODO: this shouldn't be here...but needed for invalidNode
#import "ANTLRCommonTree.h"

@interface ANTLRTree (TreeMaintenance)
- (void) _createChildrenList;
@end

@implementation ANTLRTree

#pragma mark ANTLRTree protocol conformance

+ (id<ANTLRTree>) invalidNode
{
	static id<ANTLRTree> invalidNode = nil;
	if (!invalidNode) {
		invalidNode = [[ANTLRCommonTree alloc] initWithTokenType:ANTLRTokenTypeInvalid];
	}
	return invalidNode;
}

- (id<ANTLRTree>) init
{
	if ((self = [super init]) != nil) {
		isEmptyNode = NO;
		children = nil;
	}
	return self;
}

- (void) dealloc
{
	[children release];
	children = nil;
	[super dealloc];
}

- (id<ANTLRTree>) childAtIndex:(unsigned int) index
{
	if (children && index < [children count]) {
		return [children objectAtIndex:index];
	}
	return nil;
}

- (unsigned int) childCount
{
	if (children) 
		return [children count];
	return 0;
}

- (NSArray *) allChildren
{
	return [[children retain] autorelease];
}

	// Add tree as a child to this node.  If tree is nil, do nothing.  If tree
	// is an empty node, add all children of tree to our children.

- (void) addChild:(id<ANTLRTree>) tree
{
	if (!tree) return;
	if ([tree isEmpty]) {
		// TODO: handle self add
		if ([tree childCount]) {
			if (!children)
				[self _createChildrenList];
			[children addObjectsFromArray:[tree allChildren]];
		}
	} else {
		if (!children)
			[self _createChildrenList];
		[children addObject:tree];
	}
}

- (void) addChildren:(NSArray *) theChildren
{
	if (!theChildren) return;
	if (!children)
		[self _createChildrenList];
	[children addObjectsFromArray:theChildren];
}

- (void) removeAllChildren
{
	if (children)
		[children removeAllObjects];
}

	// Indicates the node is an empty node but may still have children, meaning
	// the tree is a flat list.

- (BOOL) isEmpty
{
	return isEmptyNode;
}

- (void) setIsEmpty:(BOOL)emptyFlag
{
	isEmptyNode = emptyFlag;
}

#pragma mark Copying

// the children themselves are not copied here!
- (id) copyWithZone:(NSZone *)aZone
{
	id<ANTLRTree> theCopy = [[[self class] alloc] init];
	[theCopy addChildren:[self allChildren]];
	[theCopy setIsEmpty:[self isEmpty]];
	return theCopy;
}

- (id) deepCopy 					// performs a deepCopyWithZone: with the default zone
{
	return [self deepCopyWithZone:NULL];
}

- (id) deepCopyWithZone:(NSZone *)aZone
{
	id<ANTLRTree> theCopy = [self copyWithZone:aZone];
	
	NSArray *childrenCopy = [[theCopy allChildren] copy];
	[theCopy removeAllChildren];
	unsigned int childIdx = 0;
	for (childIdx = 0; childIdx < [childrenCopy count]; childIdx++) {
		id<ANTLRTree> childCopy = [[childrenCopy objectAtIndex:childIdx] deepCopyWithZone:aZone];
		[theCopy addChild:childCopy];
	}
	[childrenCopy release];
	
	return theCopy;
}

#pragma mark ANTLRTree abstract base class

	// Return a token type; needed for tree parsing
- (int) tokenType
{
	return 0;
}

- (NSString *) text
{
	return [self description];
}

	// In case we don't have a token payload, what is the line for errors?
- (int) line
{
	return 0;
}

- (int) charPositionInLine
{
	return 0;
}

- (NSString *) treeDescription
{
	return @"";
}

- (NSString *) description
{
	return @"";
}


@end

@implementation ANTLRTree (TreeMaintenance)

- (void) _createChildrenList
{
	if (!children)
		children = [[NSMutableArray alloc] init];
}

@end