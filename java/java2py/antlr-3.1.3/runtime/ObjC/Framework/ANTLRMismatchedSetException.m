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

#import "ANTLRMismatchedSetException.h"


@implementation ANTLRMismatchedSetException

+ (id) exceptionWithSet:(NSSet *) theExpectedSet stream:(id<ANTLRIntStream>) theStream
{
	return [[[self alloc] initWithSet:theExpectedSet stream:theStream] autorelease];
}

- (id) initWithSet:(NSSet *) theExpectedSet stream:(id<ANTLRIntStream>) theStream
{
	if (nil != (self = [super initWithStream:theStream])) {
		[self setExpectedSet:theExpectedSet];
	}
	return self;
}

- (void) dealloc
{
	[self setExpectedSet:nil];
	[super dealloc];
}

- (NSString *) description
{
	NSMutableString *desc =(NSMutableString *)[super description];
	[desc appendFormat:@" set:%@", expectedSet];
	return desc;
}


//---------------------------------------------------------- 
//  expectedSet 
//---------------------------------------------------------- 
- (NSSet *) expectedSet
{
    return expectedSet; 
}

- (void) setExpectedSet: (NSSet *) anExpectedSet
{
    if (expectedSet != anExpectedSet) {
        [anExpectedSet retain];
        [expectedSet release];
        expectedSet = anExpectedSet;
    }
}


@end
