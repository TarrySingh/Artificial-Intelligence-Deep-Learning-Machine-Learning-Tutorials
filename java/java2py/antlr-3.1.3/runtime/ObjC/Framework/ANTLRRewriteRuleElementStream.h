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
#import <ANTLR/ANTLRTreeAdaptor.h>

// TODO: this should be separated into stream and enumerator classes
@interface ANTLRRewriteRuleElementStream : NSObject {
    @public
    int cursor;
    BOOL shouldCopyElements;        ///< indicates whether the stream should return copies of its elements, set to true after a call to -reset
    
    BOOL isSingleElement;
    union {
        id single;
        NSMutableArray *multiple;
    } elements;
    
    NSString *elementDescription;
    id<ANTLRTreeAdaptor> treeAdaptor;
}

- (id) initWithTreeAdaptor:(id<ANTLRTreeAdaptor>)aTreeAdaptor description:(NSString *)anElementDescription;
- (id) initWithTreeAdaptor:(id<ANTLRTreeAdaptor>)aTreeAdaptor description:(NSString *)anElementDescription element:(id)anElement;
- (id) initWithTreeAdaptor:(id<ANTLRTreeAdaptor>)aTreeAdaptor description:(NSString *)anElementDescription elements:(NSArray *)theElements;

- (void) reset;

- (id<ANTLRTreeAdaptor>) treeAdaptor;
- (void) setTreeAdaptor:(id<ANTLRTreeAdaptor>)aTreeAdaptor;

- (void) addElement:(id)anElement;
- (unsigned int) count;
 
- (BOOL) hasNext;
- (id) next;
- (id) _next;       // internal: TODO: redesign if necessary. maybe delegate

- (id) copyElement:(id)element;
- (id) toTree:(id)element;

- (NSString *) description;
- (void) setDescription:(NSString *) description;

@end

