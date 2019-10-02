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
#import <CoreFoundation/CoreFoundation.h>

// A simple wrapper around CoreFoundation bit vectors to shield the rest of the implementation
// from the specifics of the BitVector initialization and query functions.
// This is fast, so there is no need to reinvent the wheel just yet.

@interface ANTLRBitSet : NSObject < NSCopying > {
	CFMutableBitVectorRef bitVector;
}

#pragma mark Initializer

- (ANTLRBitSet *) init;
- (ANTLRBitSet *) initWithBitVector:(CFMutableBitVectorRef)theBitVector;
- (ANTLRBitSet *) initWithBits:(const unsigned long long const*)theBits count:(unsigned int)theCount;
- (ANTLRBitSet *) initWithArrayOfBits:(NSArray *)theArray;

#pragma mark Operations
- (ANTLRBitSet *) or:(ANTLRBitSet *) aBitSet;
- (void) orInPlace:(ANTLRBitSet *) aBitSet;
- (void) add:(unsigned int) bit;
- (void) remove:(unsigned int) bit;

- (unsigned int) size;
- (void) setSize:(unsigned int) noOfWords;

#pragma mark Informational
- (unsigned long long) bitMask:(unsigned int) bitNumber;
- (BOOL) isMember:(unsigned int) bitNumber;
- (BOOL) isNil;
- (NSString *) toString;
- (NSString *) description;

#pragma mark NSCopying support

- (id) copyWithZone:(NSZone *) theZone;


//private
- (CFMutableBitVectorRef) _bitVector;
@end
