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
#import <ANTLR/ANTLRCharStream.h>

@interface ANTLRStringStream : NSObject < ANTLRCharStream > {
	NSMutableArray *markers;
	NSString *data;
	
	unsigned  p;
	unsigned  line;
	unsigned  charPositionInLine;
	unsigned  markDepth;
	unsigned lastMarker;
}

- (id) init;

// this initializer copies the string
- (id) initWithString:(NSString *) theString;

// This is the preferred constructor as no data is copied
- (id) initWithStringNoCopy:(NSString *) theString;

- (void) dealloc;

// reset the stream's state, but keep the data to feed off
- (void) reset;
// consume one character from the stream
- (void) consume;

// look ahead i characters
- (int) LA:(int) i;

// returns the position of the current input symbol
- (unsigned int) index;
// total length of the input data
- (unsigned int) count;

// seek and rewind in the stream
- (unsigned int) mark;
- (void) rewind;
- (void) rewind:(unsigned int) marker;
- (void) release:(unsigned int) marker;
- (void) seek:(unsigned int) index;

// provide the streams data (e.g. for tokens using indices)
- (NSString *) substringWithRange:(NSRange) theRange;

// used for tracking the current position in the input stream
- (unsigned int) line;
- (void) setLine:(unsigned int) theLine;
- (unsigned int) charPositionInLine;
- (void) setCharPositionInLine:(unsigned int) thePos;

// accessors to the raw data of this stream
- (NSString *) data;
- (void) setData: (NSString *) aData;


@end
