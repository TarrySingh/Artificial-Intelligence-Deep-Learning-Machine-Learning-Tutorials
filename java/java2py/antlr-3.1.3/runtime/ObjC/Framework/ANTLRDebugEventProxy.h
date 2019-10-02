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
#import <ANTLR/ANTLRParser.h>
#import <ANTLR/ANTLRDebugEventListener.h>
#import <sys/socket.h>
#import <netinet/in.h>
#import <netinet/tcp.h>
#include <arpa/inet.h>

// default port for ANTLRWorks
#define DEFAULT_DEBUGGER_PORT 0xC001

@interface ANTLRDebugEventProxy : NSObject <ANTLRDebugEventListener> {
	int serverSocket;
	
	struct sockaddr debugger_sockaddr;
	socklen_t debugger_socklen;
	int debuggerSocket;
	NSFileHandle *debuggerFH;
	
	NSString *grammarName;
	int debuggerPort;
}

- (id) init;
- (id) initWithGrammarName:(NSString *)aGrammarName debuggerPort:(int)aPort;
- (void) waitForDebuggerConnection;
- (void) waitForAck;
- (void) sendToDebugger:(NSString *)message;
- (void) sendToDebugger:(NSString *)message waitForResponse:(BOOL)wait;

- (int) serverSocket;
- (void) setServerSocket: (int) aServerSocket;

- (int) debuggerSocket;
- (void) setDebuggerSocket: (int) aDebuggerSocket;

- (NSString *) grammarName;
- (void) setGrammarName: (NSString *) aGrammarName;

- (int) debuggerPort;
- (void) setDebuggerPort: (int) aDebuggerPort;

- (NSString *) escapeNewlines:(NSString *)aString;

#pragma mark -

#pragma mark DebugEventListener Protocol
- (void) enterRule:(NSString *)ruleName;
- (void) enterAlt:(int)alt;
- (void) exitRule:(NSString *)ruleName;
- (void) enterSubRule:(int)decisionNumber;
- (void) exitSubRule:(int)decisionNumber;
- (void) enterDecision:(int)decisionNumber;
- (void) exitDecision:(int)decisionNumber;
- (void) consumeToken:(id<ANTLRToken>)t;
- (void) consumeHiddenToken:(id<ANTLRToken>)t;
- (void) LT:(int)i foundToken:(id<ANTLRToken>)t;
- (void) mark:(int)marker;
- (void) rewind:(int)marker;
- (void) rewind;
- (void) beginBacktrack:(int)level;
- (void) endBacktrack:(int)level wasSuccessful:(BOOL)successful;
- (void) locationLine:(int)line column:(int)pos;
- (void) recognitionException:(ANTLRRecognitionException *)e;
- (void) beginResync;
- (void) endResync;
- (void) semanticPredicate:(NSString *)predicate matched:(BOOL)result;
- (void) commence;
- (void) terminate;


#pragma mark Tree Parsing
- (void) consumeNode:(unsigned)nodeHash ofType:(int)type text:(NSString *)text;
- (void) LT:(int)i foundNode:(unsigned)nodeHash ofType:(int)type text:(NSString *)text;


#pragma mark AST Events

- (void) createNilNode:(unsigned)hash;
- (void) createNode:(unsigned)hash text:(NSString *)text type:(int)type;
- (void) createNode:(unsigned)hash fromTokenAtIndex:(int)tokenIndex;
- (void) makeNode:(unsigned)newRootHash parentOf:(unsigned)oldRootHash;
- (void) addChild:(unsigned)childHash toTree:(unsigned)treeHash;
- (void) setTokenBoundariesForTree:(unsigned)nodeHash start:(int)tokenStartIndex stop:(int)tokenStopIndex;

@end
