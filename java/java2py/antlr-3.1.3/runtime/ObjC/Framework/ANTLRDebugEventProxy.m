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

#import "ANTLRDebugEventProxy.h"
#import "ANTLRToken+DebuggerSupport.h"
#include <string.h>

static NSData *newlineData = nil;
static unsigned lengthOfUTF8Ack = 0;

@implementation ANTLRDebugEventProxy

+ (void) initialize
{
	if (!newlineData) newlineData = [@"\n" dataUsingEncoding:NSUTF8StringEncoding];
	if (!lengthOfUTF8Ack) lengthOfUTF8Ack = [[@"ack\n" dataUsingEncoding:NSUTF8StringEncoding] length];
}

- (id) init
{
	return [self initWithGrammarName:nil debuggerPort:DEFAULT_DEBUGGER_PORT];
}

- (id) initWithGrammarName:(NSString *)aGrammarName debuggerPort:(int)aPort
{
	self = [super init];
	if (self) {
		serverSocket = -1;
		[self setGrammarName:aGrammarName];
		if (aPort == -1) aPort = DEFAULT_DEBUGGER_PORT;
		[self setDebuggerPort:aPort];
	}
	return self;
}

- (void) dealloc
{
	if (serverSocket != -1) 
		shutdown(serverSocket,SHUT_RDWR);
	serverSocket = -1;
	[debuggerFH release];
    [self setGrammarName:nil];
    [super dealloc];
}

- (void) waitForDebuggerConnection
{
	if (serverSocket == -1) {
		serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		
		NSAssert1(serverSocket != -1, @"Failed to create debugger socket. %s", strerror(errno));
		
		int yes = 1;
		setsockopt(serverSocket, SOL_SOCKET, SO_KEEPALIVE|SO_REUSEPORT|SO_REUSEADDR|TCP_NODELAY, (void *)&yes, sizeof(int));

		struct sockaddr_in server_addr;
		bzero(&server_addr, sizeof(struct sockaddr_in));
		server_addr.sin_family = AF_INET;
		server_addr.sin_port = htons([self debuggerPort]);
		server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
		NSAssert1( bind(serverSocket, (struct sockaddr *)&server_addr, sizeof(struct sockaddr)) != -1, @"bind(2) failed. %s", strerror(errno));

		NSAssert1(listen(serverSocket,50) == 0, @"listen(2) failed. %s", strerror(errno));
		
		NSLog(@"ANTLR waiting for debugger attach (grammar %@)", [self grammarName]);
		
		debuggerSocket = accept(serverSocket, &debugger_sockaddr, &debugger_socklen);
		NSAssert1( debuggerSocket != -1, @"accept(2) failed. %s", strerror(errno));
		
		debuggerFH = [[NSFileHandle alloc] initWithFileDescriptor:debuggerSocket];
		[self sendToDebugger:[NSString stringWithFormat:@"ANTLR %d", ANTLRDebugProtocolVersion] waitForResponse:NO];
		[self sendToDebugger:[NSString stringWithFormat:@"grammar \"%@", [self grammarName]] waitForResponse:NO];
	}
}

- (void) waitForAck
{
	NSString *response;
	@try {
		NSData *newLine = [debuggerFH readDataOfLength:lengthOfUTF8Ack];
		response = [[NSString alloc] initWithData:newLine encoding:NSUTF8StringEncoding];
		if (![response isEqualToString:@"ack\n"]) @throw [NSException exceptionWithName:@"ANTLRDebugEventProxy" reason:@"illegal response from debugger" userInfo:nil];
	}
	@catch (NSException *e) {
		NSLog(@"socket died or debugger misbehaved: %@ read <%@>", e, response);
	}
	@finally {
		[response release];
	}
}

- (void) sendToDebugger:(NSString *)message
{
	[self sendToDebugger:message waitForResponse:YES];
}

- (void) sendToDebugger:(NSString *)message waitForResponse:(BOOL)wait
{
	if (! debuggerFH ) return;
	[debuggerFH writeData:[message dataUsingEncoding:NSUTF8StringEncoding]];
	[debuggerFH writeData:newlineData];
	if (wait) [self waitForAck];
}

- (int) serverSocket
{
    return serverSocket;
}

- (void) setServerSocket: (int) aServerSocket
{
    serverSocket = aServerSocket;
}

- (int) debuggerSocket
{
    return debuggerSocket;
}

- (void) setDebuggerSocket: (int) aDebuggerSocket
{
    debuggerSocket = aDebuggerSocket;
}

- (NSString *) grammarName
{
    return grammarName; 
}

- (void) setGrammarName: (NSString *) aGrammarName
{
    if (grammarName != aGrammarName) {
        [aGrammarName retain];
        [grammarName release];
        grammarName = aGrammarName;
    }
}

- (int) debuggerPort
{
    return debuggerPort;
}

- (void) setDebuggerPort: (int) aDebuggerPort
{
    debuggerPort = aDebuggerPort;
}

- (NSString *) escapeNewlines:(NSString *)aString
{
	NSMutableString *escapedText;
	if (aString) {
		escapedText = [NSMutableString stringWithString:aString];
		NSRange wholeString = NSMakeRange(0,[escapedText length]);
		[escapedText replaceOccurrencesOfString:@"%" withString:@"%25" options:0 range:wholeString];
		[escapedText replaceOccurrencesOfString:@"\n" withString:@"%0A" options:0 range:wholeString];
		[escapedText replaceOccurrencesOfString:@"\r" withString:@"%0D" options:0 range:wholeString];
	} else {
		escapedText = [NSMutableString stringWithString:@""];
	}
	return escapedText;
}

#pragma mark -

#pragma mark DebugEventListener Protocol
- (void) enterRule:(NSString *)ruleName
{
	[self sendToDebugger:[NSString stringWithFormat:@"enterRule %@", ruleName]];
}

- (void) enterAlt:(int)alt
{
	[self sendToDebugger:[NSString stringWithFormat:@"enterAlt %d", alt]]; 
}

- (void) exitRule:(NSString *)ruleName
{
	[self sendToDebugger:[NSString stringWithFormat:@"exitRule %@", ruleName]];
}

- (void) enterSubRule:(int)decisionNumber
{
	[self sendToDebugger:[NSString stringWithFormat:@"enterSubRule %d", decisionNumber]];
}

- (void) exitSubRule:(int)decisionNumber
{
	[self sendToDebugger:[NSString stringWithFormat:@"exitSubRule %d", decisionNumber]];
}

- (void) enterDecision:(int)decisionNumber
{
	[self sendToDebugger:[NSString stringWithFormat:@"enterDecision %d", decisionNumber]];
}

- (void) exitDecision:(int)decisionNumber
{
	[self sendToDebugger:[NSString stringWithFormat:@"exitDecision %d", decisionNumber]];
}

- (void) consumeToken:(id<ANTLRToken>)t
{
	[self sendToDebugger:[NSString stringWithFormat:@"consumeToken %@", [self escapeNewlines:[t debuggerDescription]]]];
}

- (void) consumeHiddenToken:(id<ANTLRToken>)t
{
	[self sendToDebugger:[NSString stringWithFormat:@"consumeHiddenToken %@", [self escapeNewlines:[t debuggerDescription]]]];
}

- (void) LT:(int)i foundToken:(id<ANTLRToken>)t
{
	[self sendToDebugger:[NSString stringWithFormat:@"LT %d %@", i, [self escapeNewlines:[t debuggerDescription]]]];
}

- (void) mark:(int)marker
{
	[self sendToDebugger:[NSString stringWithFormat:@"mark %d", marker]];
}
- (void) rewind:(int)marker
{
	[self sendToDebugger:[NSString stringWithFormat:@"rewind %d", marker]];
}

- (void) rewind
{
	[self sendToDebugger:@"rewind"];
}

- (void) beginBacktrack:(int)level
{
	[self sendToDebugger:[NSString stringWithFormat:@"beginBacktrack %d", level]];
}

- (void) endBacktrack:(int)level wasSuccessful:(BOOL)successful
{
	[self sendToDebugger:[NSString stringWithFormat:@"endBacktrack %d %d", level, successful ? 1 : 0]];
}

- (void) locationLine:(int)line column:(int)pos
{
	[self sendToDebugger:[NSString stringWithFormat:@"location %d %d", line, pos]];
}

- (void) recognitionException:(ANTLRRecognitionException *)e
{
#warning TODO: recognition exceptions
	// these must use the names of the corresponding Java exception classes, because ANTLRWorks recreates the exception
	// objects on the Java side.
	// Write categories for Objective-C exceptions to provide those names
}

- (void) beginResync
{
	[self sendToDebugger:@"beginResync"];
}
	
- (void) endResync
{
	[self sendToDebugger:@"endResync"];
}

- (void) semanticPredicate:(NSString *)predicate matched:(BOOL)result
{
	[self sendToDebugger:[NSString stringWithFormat:@"semanticPredicate %d %@", result?1:0, [self escapeNewlines:predicate]]];
}

- (void) commence
{
	// no need to send event
}

- (void) terminate
{
	[self sendToDebugger:@"terminate"];
	@try {
		[debuggerFH closeFile];
	}
	@finally {
#warning TODO: make socket handling robust. too lazy now...
		shutdown(serverSocket,SHUT_RDWR);
		serverSocket = -1;
	}
}


#pragma mark Tree Parsing
- (void) consumeNode:(unsigned)nodeHash ofType:(int)type text:(NSString *)text
{
	[self sendToDebugger:[NSString stringWithFormat:@"consumeNode %u %d %@",
		nodeHash,
		type,
		[self escapeNewlines:text]
		]];
}

- (void) LT:(int)i foundNode:(unsigned)nodeHash ofType:(int)type text:(NSString *)text
{
	[self sendToDebugger:[NSString stringWithFormat:@"LN %d %u %d %@",
		i,
		nodeHash,
		type,
		[self escapeNewlines:text]
		]];
}


#pragma mark AST Events

- (void) createNilNode:(unsigned)hash
{
	[self sendToDebugger:[NSString stringWithFormat:@"nilNode %u", hash]];
}

- (void) createNode:(unsigned)hash text:(NSString *)text type:(int)type
{
	[self sendToDebugger:[NSString stringWithFormat:@"createNodeFromToken %u %d %@", 
		hash,
		type,
		[self escapeNewlines:text]
		]];
}

- (void) createNode:(unsigned)hash fromTokenAtIndex:(int)tokenIndex
{
	[self sendToDebugger:[NSString stringWithFormat:@"createNode %u %d", hash, tokenIndex]];
}

- (void) makeNode:(unsigned)newRootHash parentOf:(unsigned)oldRootHash
{
	[self sendToDebugger:[NSString stringWithFormat:@"becomeRoot %u %u", newRootHash, oldRootHash]];
}

- (void) addChild:(unsigned)childHash toTree:(unsigned)treeHash
{
	[self sendToDebugger:[NSString stringWithFormat:@"addChild %u %u", treeHash, childHash]];
}

- (void) setTokenBoundariesForTree:(unsigned)nodeHash start:(int)tokenStartIndex stop:(int)tokenStopIndex
{
	[self sendToDebugger:[NSString stringWithFormat:@"setTokenBoundaries %u %d %d", nodeHash, tokenStartIndex, tokenStopIndex]];
}



@end
