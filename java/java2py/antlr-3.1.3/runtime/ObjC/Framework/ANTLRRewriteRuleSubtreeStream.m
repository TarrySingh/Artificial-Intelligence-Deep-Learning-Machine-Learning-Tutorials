//
//  ANTLRRewriteRuleSubtreeStream.m
//  ANTLR
//
//  Created by Kay Röpke on 7/16/07.
//  Copyright 2007 classDump. All rights reserved.
//

#import "ANTLRRewriteRuleSubtreeStream.h"


@implementation ANTLRRewriteRuleSubtreeStream

- (id) nextNode
{
    if (shouldCopyElements || (cursor >= [self count] && [self count] == 1))
        return [[treeAdaptor copyNode:[self _next]] autorelease];
    else 
        return [self _next];
}

- (id) copyElement:(id)element
{
    return [treeAdaptor copyTree:element];
}

@end
