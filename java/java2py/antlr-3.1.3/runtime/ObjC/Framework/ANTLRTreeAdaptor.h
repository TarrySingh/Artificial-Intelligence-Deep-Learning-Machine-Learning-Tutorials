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

#import <ANTLR/ANTLRTree.h>

#import <ANTLR/ANTLRToken.h>
#pragma warning tree/node diction is broken.
@protocol ANTLRTreeAdaptor <NSObject>

#pragma mark Construction

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>) payload;
- (id<ANTLRTree>) newEmptyTree;

- (id) copyNode:(id<ANTLRTree>)aNode;	// copies just the node
- (id) copyTree:(id<ANTLRTree>)aTree;	// copies the entire subtree, recursively

- (void) addChild:(id<ANTLRTree>)child toTree:(id<ANTLRTree>)aTree;
- (id) makeNode:(id<ANTLRTree>)newRoot parentOf:(id<ANTLRTree>)oldRoot;

- (id<ANTLRTree>) postProcessTree:(id<ANTLRTree>)aTree;

- (unsigned int) uniqueIdForTree:(id<ANTLRTree>)aNode;

#pragma mark Rewrite Rules

- (id<ANTLRTree>) newTreeWithTokenType:(int)tokenType;
- (id<ANTLRTree>) newTreeWithTokenType:(int)tokenType text:(NSString *)tokenText;
- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken tokenType:(int)tokenType;
- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken tokenType:(int)tokenType text:(NSString *)tokenText;
- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken text:(NSString *)tokenText;

#pragma mark Content

- (int) tokenTypeForNode:(id<ANTLRTree>)aNode;
- (void) setTokenType:(int)tokenType forNode:(id)aNode;

- (NSString *) textForNode:(id<ANTLRTree>)aNode;
- (void) setText:(NSString *)tokenText forNode:(id<ANTLRTree>)aNode;

- (void) setBoundariesForTree:(id<ANTLRTree>)aTree fromToken:(id<ANTLRToken>)startToken toToken:(id<ANTLRToken>)stopToken;
- (int) tokenStartIndexForTree:(id<ANTLRTree>)aTree;
- (int) tokenStopIndexForTree:(id<ANTLRTree>)aTree;


#pragma mark Navigation / Tree Parsing

- (id<ANTLRTree>) childForNode:(id<ANTLRTree>) aNode atIndex:(int) i;
- (int) childCountForTree:(id<ANTLRTree>) aTree;

@end

#pragma mark Abstract Base Class
@interface ANTLRTreeAdaptor : NSObject <ANTLRTreeAdaptor> {
}

#pragma mark Construction

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>) payload;
- (id<ANTLRTree>) newEmptyTree;

- (id) copyNode:(id<ANTLRTree>)aNode;	// copies just the node
- (id) copyTree:(id<ANTLRTree>)aTree;	// copies the entire subtree, recursively

- (void) addChild:(id<ANTLRTree>)child toTree:(id<ANTLRTree>)aTree;
- (id) makeNode:(id<ANTLRTree>)newRoot parentOf:(id<ANTLRTree>)oldRoot;

- (id<ANTLRTree>) postProcessTree:(id<ANTLRTree>)aTree;

- (unsigned int) uniqueIdForTree:(id<ANTLRTree>)aNode;

#pragma mark Rewrite Rules

- (id<ANTLRTree>) newTreeWithTokenType:(int)tokenType;
- (id<ANTLRTree>) newTreeWithTokenType:(int)tokenType text:(NSString *)tokenText;
- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken tokenType:(int)tokenType;
- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken tokenType:(int)tokenType text:(NSString *)tokenText;
- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken text:(NSString *)tokenText;

// these are not part of the protocol, but are here for the benefit of ANTLRCommonTreeAdaptor
// only, they are not required for ANTLR trees. simply an implementation detail, leave'em out in your
// custom tree adaptors if you wish.
- (id<ANTLRToken>) newTokenWithToken:(id<ANTLRToken>)fromToken;
- (id<ANTLRToken>) newTokenWithTokenType:(int)tokenType text:(NSString *)tokenText;

#pragma mark Content

- (int) tokenTypeForNode:(id<ANTLRTree>)aNode;
- (void) setTokenType:(int)tokenType forNode:(id)aNode;

- (NSString *) textForNode:(id<ANTLRTree>)aNode;
- (void) setText:(NSString *)tokenText forNode:(id<ANTLRTree>)aNode;

- (void) setBoundariesForTree:(id<ANTLRTree>)aTree fromToken:(id<ANTLRToken>)startToken toToken:(id<ANTLRToken>)stopToken;
- (int) tokenStartIndexForTree:(id<ANTLRTree>)aTree;
- (int) tokenStopIndexForTree:(id<ANTLRTree>)aTree;


#pragma mark Navigation / Tree Parsing

- (id<ANTLRTree>) childForNode:(id<ANTLRTree>) aNode atIndex:(int) i;
- (int) childCountForTree:(id<ANTLRTree>) aTree;


@end
