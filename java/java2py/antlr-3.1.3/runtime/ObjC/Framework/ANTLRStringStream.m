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


#import "ANTLRStringStream.h"
#import "ANTLRStringStreamState.h"

@implementation ANTLRStringStream

- (id) init
{
	if (nil != (self = [super init])) {
		markers = [[NSMutableArray alloc] init];
		[self reset];			// rely on internal implementation to reset the state, instead of duplicating here.
	}
	return self;
}

- (id) initWithString:(NSString *) theString
{
	if (nil != (self = [self init])) {
		[self setData:[theString copy]];
	}
	return self;
}

- (id) initWithStringNoCopy:(NSString *) theString
{
	if (nil != (self = [self init])) {
		[self setData:theString];
	}
	return self;
}

- (void) dealloc
{
	[markers release];
	markers = nil;
    [self setData:nil];
	[super dealloc];
}


// reset the streams state
// the streams content is not reset!
- (void) reset
{
	p = 0;
	line = 1;
	charPositionInLine = 0;
	markDepth = 0;
	[markers removeAllObjects];
	[markers addObject:[NSNull null]];		// ANTLR generates code that assumes markers to be 1-based,
											// thus the initial null in the array!
}

// read one character off the stream, tracking line numbers and character positions
// automatically.
// Override this in subclasses if you want to avoid the overhead of automatic line/pos
// handling. Do not call super in that case.
- (void) consume 
{
	if ( p < [data length] ) {
		charPositionInLine++;
		if ( [data characterAtIndex:p] == '\n' ) {
			line++;
			charPositionInLine=0;
		}
		p++;
	}
}

// implement the lookahead method used in lexers
- (int) LA:(int) i 
{
	if ( (p+i-1) >= [data length] ) {
		return ANTLRCharStreamEOF;
	}
	return (int)[data characterAtIndex:p+i-1];
}

// current input position
- (unsigned int) index 
{
	return p;
}

- (unsigned int) count 
{
	return [data length];
}

// push the current state of the stream onto a stack
// returns the depth of the stack, to be used as a marker to rewind the stream.
// Note: markers are 1-based!
- (unsigned int) mark 
{
	markDepth++;
	ANTLRStringStreamState *state = nil;
	if ( markDepth >= [markers count] ) {
		state = [[ANTLRStringStreamState alloc] init];
		[markers addObject:state];
		[state release];
	}
	else {
		state = (ANTLRStringStreamState *)[markers objectAtIndex:markDepth];
	}
	[state setIndex:p];
	[state setLine:line];
	[state setCharPositionInLine:charPositionInLine];
	lastMarker = markDepth;
	return markDepth;
}

- (void) rewind
{
	[self rewind:lastMarker];
}

- (void) rewind:(unsigned int) marker 
{
	[self release:marker];
	ANTLRStringStreamState *state = (ANTLRStringStreamState *)[markers objectAtIndex:marker];
	// restore stream state
	[self seek:[state index]];
	line = [state line];
	charPositionInLine = [state charPositionInLine];
}

// remove stream states on top of 'marker' from the marker stack
// returns the new markDepth of the stack.
// Note: unfortunate naming for Objective-C, but to keep close to the Java target this is named release:
- (void) release:(unsigned int) marker 
{
	// unwind any other markers made after marker and release marker
	markDepth = marker;
	markDepth--;
}

// when seeking forward we must handle character position and line numbers.
// seeking backward already has the correct line information on the markers stack, 
// so we just take it from there.
- (void) seek:(unsigned int) index 
{
	if ( index<=p ) {
		p = index; // just jump; don't update stream state (line, ...)
		return;
	}
	// seek forward, consume until p hits index
	while ( p<index ) {
		[self consume];
	}
}

// get a substring from our raw data.
- (NSString *) substringWithRange:(NSRange) theRange 
{
	return [data substringWithRange:theRange];
}

- (unsigned int) line 
{
	return line;
}

- (unsigned int) charPositionInLine 
{
	return charPositionInLine;
}

- (void) setLine:(unsigned int) theLine 
{
	line = theLine;
}

- (void) setCharPositionInLine:(unsigned int) thePos 
{
	charPositionInLine = thePos;
}

//---------------------------------------------------------- 
//  data 
//---------------------------------------------------------- 
- (NSString *) data
{
    return data; 
}

- (void) setData: (NSString *) aData
{
    if (data != aData) {
        [aData retain];
        [data release];
        data = aData;
    }
}

@end
