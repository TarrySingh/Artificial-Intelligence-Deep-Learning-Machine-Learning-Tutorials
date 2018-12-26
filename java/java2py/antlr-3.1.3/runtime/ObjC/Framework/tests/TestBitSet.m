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


#import "TestBitSet.h"
#import "ANTLRBitSet.h"
#import <CoreFoundation/CoreFoundation.h>
#import <CoreFoundation/CFBitVector.h>

@implementation TestBitSet

- (void) testBitSetCreationFromLongs
{
	static const unsigned long long bitData[] = {3LL, 1LL};
	ANTLRBitSet *bitSet = [[ANTLRBitSet alloc] initWithBits:bitData count:2];
	CFMutableBitVectorRef bitVector = [bitSet _bitVector];
	NSLog(@"%@", [bitSet toString]);
    NSLog(@"getcount %d", CFBitVectorGetCount(bitVector));
    
    CFIndex actual = CFBitVectorGetCountOfBit(bitVector,CFRangeMake(0,CFBitVectorGetCount(bitVector)),1);
    CFIndex expected = 3;
	
    STAssertEquals(actual,
                   expected,
                   @"There should be three bits set in bitvector. But I have %d",
                   CFBitVectorGetCountOfBit(bitVector,CFRangeMake(0,CFBitVectorGetCount(bitVector)),1)
                   );
}

@end
