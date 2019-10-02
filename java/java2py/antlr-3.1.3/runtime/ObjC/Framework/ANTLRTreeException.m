//
//  ANTLRTreeException.m
//  ANTLR
//
//  Created by Kay Röpke on 24.10.2006.
//  Copyright 2006 classDump. All rights reserved.
//

#import "ANTLRTreeException.h"


@implementation ANTLRTreeException

+ (id) exceptionWithOldRoot:(id<ANTLRTree>)theOldRoot newRoot:(id<ANTLRTree>)theNewRoot stream:(id<ANTLRIntStream>)aStream;
{
	return [[ANTLRTreeException alloc] initWithOldRoot:theOldRoot newRoot:theNewRoot stream:aStream];
}

- (id) initWithOldRoot:(id<ANTLRTree>)theOldRoot newRoot:(id<ANTLRTree>)theNewRoot stream:(id<ANTLRIntStream>)aStream;
{
	if ((self = [super initWithStream:aStream reason:@"The new root has more than one child. Cannot make it the root node."])) {
		[self setOldRoot:theOldRoot];
		[self setNewRoot:theNewRoot];
	}
	return self;
}

- (void) dealloc
{
	[self setOldRoot:nil];
	[self setNewRoot:nil];
	[super dealloc];
}

- (void) setNewRoot:(id<ANTLRTree>)aTree
{
	if (newRoot != aTree) {
		[aTree retain];
		[newRoot release];
		newRoot = aTree;
	}
}

- (void) setOldRoot:(id<ANTLRTree>)aTree
{
	if (oldRoot != aTree) {
		[aTree retain];
		[oldRoot release];
		oldRoot = aTree;
	}
}

- (NSString *) description
{
	 return [NSMutableString stringWithFormat:@"%@ old root: <%@> new root: <%@>", [super description], [oldRoot treeDescription], [newRoot treeDescription]];
}

@end
