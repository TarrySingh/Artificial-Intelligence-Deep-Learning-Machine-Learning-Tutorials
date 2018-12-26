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

#import "ANTLRDFA.h"
#import <ANTLR/ANTLRToken.h>
#import <ANTLR/ANTLRNoViableAltException.h>

static BOOL debug = NO;

@implementation ANTLRDFA

- (id) initWithRecognizer:(ANTLRBaseRecognizer *) theRecognizer
{
	if ((self = [super init])) {
		recognizer = theRecognizer;
	}
	return self;
}

// using the tables ANTLR generates for the DFA based prediction this method simulates the DFA
// and returns the prediction of the alternative to be used.
- (int) predict
{
	id<ANTLRIntStream> stream = [recognizer input];
	
	unsigned int mark = [stream mark];
	int s = 0;
	@try {
		while (YES) {
			if ( debug ) NSLog(@"DFA %d state %d LA(1)=%c(%d)", decisionNumber, s, (unichar)[stream LA:1], [stream LA:1]);
			int specialState = special[s];
			if (specialState >= 0) {
				// this state is special in that it has some code associated with it. we cannot do this in a pure DFA so
				// we signal the caller accordingly.
				if ( debug ) NSLog(@"DFA %d state %d is special state %d", decisionNumber, s, specialState);
				s = [self specialStateTransition:specialState];
				[stream consume];
				continue;
			}
			// if this is an accepting state return the prediction
			if (accept[s] >= 1) {
				if ( debug ) NSLog(@"accept; predict %d from state %d", accept[s], s);
				return accept[s];
			}
			// based on the lookahead lookup the next transition, consume and do transition
			// or signal that we have no viable alternative
			int c = [stream LA:1];
			if ( (unichar)c >= min[s] && (unichar)c <= max[s]) {
				int snext = transition[s][c-min[s]];
				if (snext < 0) {
					if (eot[s] >= 0) {
						if ( debug ) NSLog(@"EOT transition");
						s = eot[s];
						[stream consume];
						continue;
					}
					[self noViableAlt:s];
					return 0;
				}
				s = snext;
				[stream consume];
				continue;
			}
			// we are the end of the token, consume and do transition. we may still accept the input in the next state
			if (eot[s] >= 0) {
				if ( debug ) NSLog(@"EOT transition");
				s = eot[s];
				[stream consume];
				continue;
			}
			// we are at EOF and may even accept the input.
			if ( c == ANTLRTokenTypeEOF && eof[s] >= 0) {
				if ( debug ) NSLog(@"accept via EOF; predict %d from %d", accept[eof[s]], eof[s]);
				return accept[eof[s]];
			}
			if (debug) NSLog(@"no viable alt!");
			[self noViableAlt:s];
		}
	}
	@finally {
		[stream rewind:mark];
	}
	return 0; // silence warning
}

- (void) noViableAlt:(int) state
{
	id<ANTLRIntStream> stream = [recognizer input];
	if ([recognizer isBacktracking]) {
		[recognizer setIsFailed:YES];
		return;
	}
	ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:decisionNumber state:state stream:stream];
	[self error:nvae];
	@throw nvae;
}

- (int) specialStateTransition:(int) state
{
	return 0;
}

- (void) error:(ANTLRNoViableAltException *)nvae
{
	// empty, hook for debugger support
}

- (NSString *) description
{
	return @"subclass responsibility";
}

- (BOOL) evaluateSyntacticPredicate:(SEL)synpredFragment
{
	return [recognizer evaluateSyntacticPredicate:synpredFragment];
}

+ (void) setIsEmittingDebugInfo:(BOOL) shouldEmitDebugInfo
{
	debug = shouldEmitDebugInfo;
}

@end
