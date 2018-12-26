// [The "BSD licence"]
// Copyright (c) 2007 Kay Roepke
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

#import "ANTLRBaseRecognizerState.h"


@implementation ANTLRBaseRecognizerState

- (id) init
{
	if (nil != (self = [super init])) {
		following = [[NSMutableArray alloc] init];
		ruleMemo = [[NSMutableArray alloc] init];
		errorRecovery = NO;
		lastErrorIndex = -1;
		failed = NO;
		backtracking = 0;
	}
	return self;
}

- (void) reset
{
	errorRecovery = NO;
	lastErrorIndex = -1;
	failed = NO;
	backtracking = 0;
	[following removeAllObjects]; 
	[ruleMemo removeAllObjects];
}

- (void) dealloc
{
	[following release];
	[ruleMemo release];
	[super dealloc];
}


- (NSMutableArray *) following
{
	return following;
}

- (NSMutableArray *) ruleMemo
{
	return ruleMemo;
}


- (BOOL) isErrorRecovery
{
	return errorRecovery;
}

- (void) setIsErrorRecovery: (BOOL) flag
{
	errorRecovery = flag;
}


- (BOOL) isFailed
{
	return failed;
}

- (void) setIsFailed: (BOOL) flag
{
	failed = flag;
}


- (int) backtracking
{
	return backtracking;
}

- (void) setBacktracking:(int) value
{
	backtracking = value;
}

- (void) increaseBacktracking
{
	backtracking++;
}

- (void) decreaseBacktracking
{
	backtracking--;
}

- (BOOL) isBacktracking
{
	return backtracking > 0;
}


- (int) lastErrorIndex
{
    return lastErrorIndex;
}

- (void) setLastErrorIndex:(int) value
{
	lastErrorIndex = value;
}


@end
