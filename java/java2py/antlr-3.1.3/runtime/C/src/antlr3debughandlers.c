/// \file
/// Provides the debugging functions invoked by a recognizer
/// built using the debug generator mode of the antlr tool.
/// See antlr3debugeventlistener.h for documentation.
///

// [The "BSD licence"]
// Copyright (c) 2005-2009 Jim Idle, Temporal Wave LLC
// http://www.temporal-wave.com
// http://www.linkedin.com/in/jimidle
//
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

#include    <antlr3.h>

// Not everyone wishes to include the debugger stuff in their final deployment because
// it will then rely on being linked with the socket libraries. Hence if the programmer turns
// off the debugging, we do some dummy stuff that satifies compilers etc but means there is
// no debugger and no reliance on the socket librarires. If you set this flag, then using the -debug
// option to generate your code will produce code that just crashes, but then I presme you are smart
// enough to realize that building the libraries without debugger support means you can't call the
// debugger ;-)
// 
#ifdef ANTLR3_NODEBUGGER
ANTLR3_API pANTLR3_DEBUG_EVENT_LISTENER
antlr3DebugListenerNew()
{
		ANTLR3_PRINTF("C runtime was compiled without debugger support. This program will crash!!");
		return NULL;
}
#else

static	ANTLR3_BOOLEAN	handshake		(pANTLR3_DEBUG_EVENT_LISTENER delboy);
static	void	enterRule				(pANTLR3_DEBUG_EVENT_LISTENER delboy, const char * grammarFileName, const char * ruleName);
static	void	enterAlt				(pANTLR3_DEBUG_EVENT_LISTENER delboy, int alt);
static	void	exitRule				(pANTLR3_DEBUG_EVENT_LISTENER delboy, const char * grammarFileName, const char * ruleName);
static	void	enterSubRule			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber);
static	void	exitSubRule				(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber);
static	void	enterDecision			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber);
static	void	exitDecision			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber);
static	void	consumeToken			(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_COMMON_TOKEN t);
static	void	consumeHiddenToken		(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_COMMON_TOKEN t);
static	void	LT						(pANTLR3_DEBUG_EVENT_LISTENER delboy, int i, pANTLR3_COMMON_TOKEN t);
static	void	mark					(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_MARKER marker);
static	void	rewindMark				(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_MARKER marker);
static	void	rewindLast				(pANTLR3_DEBUG_EVENT_LISTENER delboy);
static	void	beginBacktrack			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int level);
static	void	endBacktrack			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int level, ANTLR3_BOOLEAN successful);
static	void	location				(pANTLR3_DEBUG_EVENT_LISTENER delboy, int line, int pos);
static	void	recognitionException	(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_EXCEPTION e);
static	void	beginResync				(pANTLR3_DEBUG_EVENT_LISTENER delboy);
static	void	endResync				(pANTLR3_DEBUG_EVENT_LISTENER delboy);
static	void	semanticPredicate		(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_BOOLEAN result, const char * predicate);
static	void	commence				(pANTLR3_DEBUG_EVENT_LISTENER delboy);
static	void	terminate				(pANTLR3_DEBUG_EVENT_LISTENER delboy);
static	void	consumeNode				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t);
static	void	LTT						(pANTLR3_DEBUG_EVENT_LISTENER delboy, int i, pANTLR3_BASE_TREE t);
static	void	nilNode					(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t);
static	void	errorNode				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t);
static	void	createNode				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t);
static	void	createNodeTok			(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE node, pANTLR3_COMMON_TOKEN token);
static	void	becomeRoot				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE newRoot, pANTLR3_BASE_TREE oldRoot);
static	void	addChild				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE root, pANTLR3_BASE_TREE child);
static	void	setTokenBoundaries		(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t, ANTLR3_MARKER tokenStartIndex, ANTLR3_MARKER tokenStopIndex);
static	void	ack						(pANTLR3_DEBUG_EVENT_LISTENER delboy);

/// Create and initialize a new debug event listener that can be connected to
/// by ANTLRWorks and any other debugger via a socket.
///
ANTLR3_API pANTLR3_DEBUG_EVENT_LISTENER
antlr3DebugListenerNew()
{
	pANTLR3_DEBUG_EVENT_LISTENER	delboy;

	delboy = ANTLR3_CALLOC(1, sizeof(ANTLR3_DEBUG_EVENT_LISTENER));

	if	(delboy == NULL)
	{
		return NULL;
	}

	// Initialize the API
	//
	delboy->addChild				= addChild;
	delboy->becomeRoot				= becomeRoot;
	delboy->beginBacktrack			= beginBacktrack;
	delboy->beginResync				= beginResync;
	delboy->commence				= commence;
	delboy->consumeHiddenToken		= consumeHiddenToken;
	delboy->consumeNode				= consumeNode;
	delboy->consumeToken			= consumeToken;
	delboy->createNode				= createNode;
	delboy->createNodeTok			= createNodeTok;
	delboy->endBacktrack			= endBacktrack;
	delboy->endResync				= endResync;
	delboy->enterAlt				= enterAlt;
	delboy->enterDecision			= enterDecision;
	delboy->enterRule				= enterRule;
	delboy->enterSubRule			= enterSubRule;
	delboy->exitDecision			= exitDecision;
	delboy->exitRule				= exitRule;
	delboy->exitSubRule				= exitSubRule;
	delboy->handshake				= handshake;
	delboy->location				= location;
	delboy->LT						= LT;
	delboy->LTT						= LTT;
	delboy->mark					= mark;
	delboy->nilNode					= nilNode;
	delboy->recognitionException	= recognitionException;
	delboy->rewind					= rewindMark;
	delboy->rewindLast				= rewindLast;
	delboy->semanticPredicate		= semanticPredicate;
	delboy->setTokenBoundaries		= setTokenBoundaries;
	delboy->terminate				= terminate;
	delboy->errorNode				= errorNode;

	delboy->PROTOCOL_VERSION		= 2;	// ANTLR 3.1 is at protocol version 2

	delboy->port					= DEFAULT_DEBUGGER_PORT;

	return delboy;
}

pANTLR3_DEBUG_EVENT_LISTENER
antlr3DebugListenerNewPort(ANTLR3_UINT32 port)
{
	pANTLR3_DEBUG_EVENT_LISTENER	delboy;

	delboy		 = antlr3DebugListenerNew();

	if	(delboy != NULL)
	{
		delboy->port = port;
	}

	return delboy;
}

//--------------------------------------------------------------------------------
// Support functions for sending stuff over the socket interface
//
static int 
sockSend(SOCKET sock, const char * ptr, int len)
{
	int		sent;
	int		thisSend;

	sent	= 0;
		
	while	(sent < len)
	{
		// Send as many bytes as we can
		//
		thisSend =	send(sock, ptr, len - sent, 0);

		// Check for errors and tell the user if we got one
		//
		if	(thisSend	== -1)
		{
			return	ANTLR3_FALSE;
		}

		// Increment our offset by how many we were able to send
		//
		ptr			+= thisSend;
		sent		+= thisSend;
	}
	return	ANTLR3_TRUE;
}

static	ANTLR3_BOOLEAN	
handshake				(pANTLR3_DEBUG_EVENT_LISTENER delboy)
{
	/// Connection structure with which to wait and accept a connection from
	/// a debugger.
	///
	SOCKET				serverSocket;

	// Connection structures to deal with the client after we accept the connection
	// and the server while we accept a connection.
	//
	ANTLR3_SOCKADDRT	client;
	ANTLR3_SOCKADDRT	server;

	// Buffer to construct our message in
	//
	char	message[256];

	// Specifies the length of the connection structure to accept()
	// Windows use int, everyone else uses size_t
	//
	ANTLR3_SALENT				sockaddr_len;

	// Option holder for setsockopt()
	//
	int		optVal;

	if	(delboy->initialized == ANTLR3_FALSE)
	{
		// Windows requires us to initialize WinSock.
		//
#ifdef ANTLR3_WINDOWS
		{
			WORD		wVersionRequested;
			WSADATA		wsaData;
			int			err;			// Return code from WSAStartup

			// We must initialise the Windows socket system when the DLL is loaded.
			// We are asking for Winsock 1.1 or better as we don't need anything
			// too complicated for this.
			//
			wVersionRequested = MAKEWORD( 1, 1);

			err = WSAStartup( wVersionRequested, &wsaData );

			if ( err != 0 ) 
			{
				// Tell the user that we could not find a usable
				// WinSock DLL
				//
				return FALSE;
			}
		}
#endif

		// Create the server socket, we are the server because we just wait until
		// a debugger connects to the port we are listening on.
		//
		serverSocket	= socket(AF_INET, SOCK_STREAM, 0);

		if	(serverSocket == INVALID_SOCKET)
		{
			return ANTLR3_FALSE;
		}

		// Set the listening port
		//
		server.sin_port			= htons((unsigned short)delboy->port);
		server.sin_family		= AF_INET;
		server.sin_addr.s_addr	= htonl (INADDR_ANY);

		// We could allow a rebind on the same addr/port pair I suppose, but
		// I imagine that most people will just want to start debugging one parser at once.
		// Maybe change this at some point, but rejecting the bind at this point will ensure
		// that people realize they have left something running in the background.
		//
		if	(bind(serverSocket, (pANTLR3_SOCKADDRC)&server, sizeof(server)) == -1)
		{
			return ANTLR3_FALSE;
		}

		// We have bound the socket to the port and address so we now ask the TCP subsystem
		// to start listening on that address/port
		//
		if	(listen(serverSocket, 1) == -1)
		{
			// Some error, just fail
			//
			return	ANTLR3_FALSE;
		}

		// Now we can try to accept a connection on the port
		//
		sockaddr_len	= sizeof(client);
		delboy->socket	= accept(serverSocket, (pANTLR3_SOCKADDRC)&client, &sockaddr_len);

		// Having accepted a connection, we can stop listening and close down the socket
		//
		shutdown		(serverSocket, 0x02);
		ANTLR3_CLOSESOCKET		(serverSocket);

		if	(delboy->socket == -1)
		{
			return ANTLR3_FALSE;
		}

		// Disable Nagle as this is essentially a chat exchange
		//
		optVal	= 1;
		setsockopt(delboy->socket, SOL_SOCKET, TCP_NODELAY, (const void *)&optVal, sizeof(optVal));
		
	}

	// We now have a good socket connection with the debugging client, so we
	// send it the protocol version we are using and what the name of the grammar
	// is that we represent.
	//
	sprintf		(message, "ANTLR %d\n", delboy->PROTOCOL_VERSION);
	sockSend	(delboy->socket, message, (int)strlen(message));
	sprintf		(message, "grammar \"%s\n", delboy->grammarFileName->chars);
	sockSend	(delboy->socket, message, (int)strlen(message));
	ack			(delboy);

	delboy->initialized = ANTLR3_TRUE;

	return	ANTLR3_TRUE;
}

// Send the supplied text and wait for an ack from the client
static void
transmit(pANTLR3_DEBUG_EVENT_LISTENER delboy, const char * ptr)
{
	sockSend(delboy->socket, ptr, (int)strlen(ptr));
	ack(delboy);
}

static	void
ack						(pANTLR3_DEBUG_EVENT_LISTENER delboy)
{
	// Local buffer to read the next character in to
	//
	char	buffer;
	int		rCount;

	// Ack terminates in a line feed, so we just wait for
	// one of those. Speed is not of the essence so we don't need
	// to buffer the input or anything.
	//
	do
	{
		rCount = recv(delboy->socket, &buffer, 1, 0);
	}
	while	(rCount == 1 && buffer != '\n');

	// If the socket ws closed on us, then we will get an error or
	// (with a graceful close), 0. We can assume the the debugger stopped for some reason
	// (such as Java crashing again). Therefore we just exit the program
	// completely if we don't get the terminating '\n' for the ack.
	//
	if	(rCount != 1)
	{
		ANTLR3_PRINTF("Exiting debugger as remote client closed the socket\n");
		ANTLR3_PRINTF("Received char count was %d, and last char received was %02X\n", rCount, buffer);
		exit(0);
	}
}

// Given a buffer string and a source string, serialize the
// text, escaping any newlines and linefeeds. We have no need
// for speed here, this is the debugger.
//
void
serializeText(pANTLR3_STRING buffer, pANTLR3_STRING text)
{
	ANTLR3_UINT32	c;
	ANTLR3_UCHAR	character;

	// strings lead in with a "
	//
	buffer->append(buffer, " \"");

	if	(text == NULL)
	{
		return;
	}

	// Now we replace linefeeds, newlines and the escape
	// leadin character '%' with their hex equivalents
	// prefixed by '%'
	//
	for	(c = 0; c < text->len; c++)
	{
		switch	(character = text->charAt(text, c))
		{
			case	'\n':

				buffer->append(buffer, "%0A");
				break;

			case	'\r':
			
				buffer->append(buffer, "%0D");
				break;

			case	'\\':

				buffer->append(buffer, "%25");
				break;

				// Other characters: The Song Remains the Same.
				//
			default:
					
				buffer->addc(buffer, character);
				break;
		}
	}
}

// Given a token, create a stringified version of it, in the supplied
// buffer. We create a string for this in the debug 'object', if there 
// is not one there already, and then reuse it here if asked to do this
// again.
//
pANTLR3_STRING
serializeToken(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_COMMON_TOKEN t)
{
	// Do we already have a serialization buffer?
	//
	if	(delboy->tokenString == NULL)
	{
		// No, so create one, using the string factory that
		// the grammar name used, which is guaranteed to exist.
		// 64 bytes will do us here for starters. 
		//
		delboy->tokenString = delboy->grammarFileName->factory->newSize(delboy->grammarFileName->factory, 64);
	}

	// Empty string
	//
	delboy->tokenString->set(delboy->tokenString, (const char *)"");

	// Now we serialize the elements of the token.Note that the debugger only
	// uses 32 bits.
	//
	delboy->tokenString->addi(delboy->tokenString, (ANTLR3_INT32)(t->getTokenIndex(t)));
	delboy->tokenString->addc(delboy->tokenString, ' ');
	delboy->tokenString->addi(delboy->tokenString, (ANTLR3_INT32)(t->getType(t)));
	delboy->tokenString->addc(delboy->tokenString, ' ');
	delboy->tokenString->addi(delboy->tokenString, (ANTLR3_INT32)(t->getChannel(t)));
	delboy->tokenString->addc(delboy->tokenString, ' ');
	delboy->tokenString->addi(delboy->tokenString, (ANTLR3_INT32)(t->getLine(t)));
	delboy->tokenString->addc(delboy->tokenString, ' ');
	delboy->tokenString->addi(delboy->tokenString, (ANTLR3_INT32)(t->getCharPositionInLine(t)));

	// Now send the text that the token represents.
	//
	serializeText(delboy->tokenString, t->getText(t));

	// Finally, as the debugger is a Java program it will expect to get UTF-8
	// encoded strings. We don't use UTF-8 internally to the C runtime, so we 
	// must force encode it. We have a method to do this in the string class, but
	// it returns malloc space that we must free afterwards.
	//
	return delboy->tokenString->toUTF8(delboy->tokenString);
}

// Given a tree node, create a stringified version of it in the supplied
// buffer.
//
pANTLR3_STRING
serializeNode(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE node)
{
	pANTLR3_COMMON_TOKEN	token;


	// Do we already have a serialization buffer?
	//
	if	(delboy->tokenString == NULL)
	{
		// No, so create one, using the string factory that
		// the grammar name used, which is guaranteed to exist.
		// 64 bytes will do us here for starters. 
		//
		delboy->tokenString = delboy->grammarFileName->factory->newSize(delboy->grammarFileName->factory, 64);
	}

	// Empty string
	//
	delboy->tokenString->set(delboy->tokenString, (const char *)"");

	// Protect against bugs/errors etc
	//
	if	(node == NULL)
	{
		return delboy->tokenString;
	}

	// Now we serialize the elements of the node.Note that the debugger only
	// uses 32 bits.
	//
	delboy->tokenString->addc(delboy->tokenString, ' ');

	// Adaptor ID
	//
	delboy->tokenString->addi(delboy->tokenString, delboy->adaptor->getUniqueID(delboy->adaptor, node));
	delboy->tokenString->addc(delboy->tokenString, ' ');

	// Type of the current token (which may be imaginary)
	//
	delboy->tokenString->addi(delboy->tokenString, delboy->adaptor->getType(delboy->adaptor, node));

	// See if we have an actual token or just an imaginary
	//
	token	= delboy->adaptor->getToken(delboy->adaptor, node);

	delboy->tokenString->addc(delboy->tokenString, ' ');
	if	(token != NULL)
	{
		// Real token
		//
		delboy->tokenString->addi(delboy->tokenString, (ANTLR3_INT32)(token->getLine(token)));
		delboy->tokenString->addc(delboy->tokenString, ' ');
		delboy->tokenString->addi(delboy->tokenString, (ANTLR3_INT32)(token->getCharPositionInLine(token)));
	}
	else
	{
		// Imaginary tokens have no location
		//
		delboy->tokenString->addi(delboy->tokenString, -1);
		delboy->tokenString->addc(delboy->tokenString, ' ');
		delboy->tokenString->addi(delboy->tokenString, -1);
	}

	// Start Index of the node
	//
	delboy->tokenString->addc(delboy->tokenString, ' ');
	delboy->tokenString->addi(delboy->tokenString, (ANTLR3_UINT32)(delboy->adaptor->getTokenStartIndex(delboy->adaptor, node)));

	// Now send the text that the node represents.
	//
	serializeText(delboy->tokenString, delboy->adaptor->getText(delboy->adaptor, node));

	// Finally, as the debugger is a Java program it will expect to get UTF-8
	// encoded strings. We don't use UTF-8 internally to the C runtime, so we 
	// must force encode it. We have a method to do this in the string class, but
	// there is no utf8 string implementation as of yet
	//
	return delboy->tokenString->toUTF8(delboy->tokenString);
}

//------------------------------------------------------------------------------------------------------------------
// EVENTS
//
static	void
enterRule				(pANTLR3_DEBUG_EVENT_LISTENER delboy, const char * grammarFileName, const char * ruleName)
{
	char	buffer[512];

	// Create the message (speed is not of the essence)
	//
	sprintf(buffer, "enterRule %s %s\n", grammarFileName, ruleName);
	transmit(delboy, buffer);
}

static	void	
enterAlt				(pANTLR3_DEBUG_EVENT_LISTENER delboy, int alt)
{
	char	buffer[512];

	// Create the message (speed is not of the essence)
	//
	sprintf(buffer, "enterAlt %d\n", alt);
	transmit(delboy, buffer);
}

static	void	
exitRule				(pANTLR3_DEBUG_EVENT_LISTENER delboy, const char * grammarFileName, const char * ruleName)
{
	char	buffer[512];

	// Create the message (speed is not of the essence)
	//
	sprintf(buffer, "enterRule %s %s\n", grammarFileName, ruleName);
	transmit(delboy, buffer);
}

static	void	
enterSubRule			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber)
{
	char	buffer[512];

	// Create the message (speed is not of the essence)
	//
	sprintf(buffer, "enterSubRule %d\n", decisionNumber);
	transmit(delboy, buffer);
}

static	void	
exitSubRule				(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber)
{
	char	buffer[512];

	// Create the message (speed is not of the essence)
	//
	sprintf(buffer, "exitSubRule %d\n", decisionNumber);
	transmit(delboy, buffer);
}

static	void	
enterDecision			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber)
{
	char	buffer[512];

	// Create the message (speed is not of the essence)
	//
	sprintf(buffer, "enterDecision %d\n", decisionNumber);
	transmit(delboy, buffer);

}

static	void	
exitDecision			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int decisionNumber)
{
	char	buffer[512];

	// Create the message (speed is not of the essence)
	//
	sprintf(buffer, "exitDecision %d\n", decisionNumber);
	transmit(delboy, buffer);
}

static	void	
consumeToken			(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_COMMON_TOKEN t)
{
	pANTLR3_STRING msg;

	// Create the serialized token
	//
	msg = serializeToken(delboy, t);

	// Insert the debug event indicator
	//
	msg->insert8(msg, 0, "consumeToken ");

	msg->addc(msg, '\n');

	// Transmit the message and wait for ack
	//
	transmit(delboy, (const char *)(msg->chars));
}

static	void	
consumeHiddenToken		(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_COMMON_TOKEN t)
{
	pANTLR3_STRING msg;

	// Create the serialized token
	//
	msg = serializeToken(delboy, t);

	// Insert the debug event indicator
	//
	msg->insert8(msg, 0, "consumeHiddenToken ");

	msg->addc(msg, '\n');

	// Transmit the message and wait for ack
	//
	transmit(delboy, (const char *)(msg->chars));
}

// Looking at the next token event.
//
static	void	
LT						(pANTLR3_DEBUG_EVENT_LISTENER delboy, int i, pANTLR3_COMMON_TOKEN t)
{
	pANTLR3_STRING msg;

	if	(t != NULL)
	{
		// Create the serialized token
		//
		msg = serializeToken(delboy, t);

		// Insert the index parameter
		//
		msg->insert8(msg, 0, " ");
		msg->inserti(msg, 0, i);

		// Insert the debug event indicator
		//
		msg->insert8(msg, 0, "LT ");

		msg->addc(msg, '\n');

		// Transmit the message and wait for ack
		//
		transmit(delboy, (const char *)(msg->chars));
	}
}

static	void	
mark					(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_MARKER marker)
{
	char buffer[128];

	sprintf(buffer, "mark %d\n", (ANTLR3_UINT32)(marker & 0xFFFFFFFF));

	// Transmit the message and wait for ack
	//
	transmit(delboy, buffer);
}

static	void	
rewindMark					(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_MARKER marker)
{
	char buffer[128];

	sprintf(buffer, "rewind %d\n", (ANTLR3_UINT32)(marker & 0xFFFFFFFF));

	// Transmit the message and wait for ack
	//
	transmit(delboy, buffer);

}

static	void	
rewindLast				(pANTLR3_DEBUG_EVENT_LISTENER delboy)
{
	transmit(delboy, (const char *)"rewind\n");
}

static	void	
beginBacktrack			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int level)
{
	char buffer[128];

	sprintf(buffer, "beginBacktrack %d\n", (ANTLR3_UINT32)(level & 0xFFFFFFFF));

	// Transmit the message and wait for ack
	//
	transmit(delboy, buffer);
}

static	void	
endBacktrack			(pANTLR3_DEBUG_EVENT_LISTENER delboy, int level, ANTLR3_BOOLEAN successful)
{
	char buffer[128];

	sprintf(buffer, "endBacktrack %d %d\n", level, successful);

	// Transmit the message and wait for ack
	//
	transmit(delboy, buffer);
}

static	void	
location				(pANTLR3_DEBUG_EVENT_LISTENER delboy, int line, int pos)
{
	char buffer[128];

	sprintf(buffer, "location %d %d\n", line, pos);

	// Transmit the message and wait for ack
	//
	transmit(delboy, buffer);
}

static	void	
recognitionException	(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_EXCEPTION e)
{
	char	buffer[256];

	sprintf(buffer, "exception %s %d %d %d\n", (char *)(e->name), (ANTLR3_INT32)(e->index), e->line, e->charPositionInLine);

	// Transmit the message and wait for ack
	//
	transmit(delboy, buffer);
}

static	void	
beginResync				(pANTLR3_DEBUG_EVENT_LISTENER delboy)
{
	transmit(delboy, (const char *)"beginResync\n");
}

static	void	
endResync				(pANTLR3_DEBUG_EVENT_LISTENER delboy)
{
	transmit(delboy, (const char *)"endResync\n");
}

static	void	
semanticPredicate		(pANTLR3_DEBUG_EVENT_LISTENER delboy, ANTLR3_BOOLEAN result, const char * predicate)
{
	unsigned char * buffer;
	unsigned char * out;

	if	(predicate != NULL)
	{
		buffer	= (unsigned char *)ANTLR3_MALLOC(64 + 2*strlen(predicate));

		if	(buffer != NULL)
		{
			out = buffer + sprintf((char *)buffer, "semanticPredicate %s ", result == ANTLR3_TRUE ? "true" : "false");

			while (*predicate != '\0')
			{
				switch(*predicate)
				{
					case	'\n':
						
						*out++	= '%';
						*out++	= '0';
						*out++	= 'A';
						break;

					case	'\r':

						*out++	= '%';
						*out++	= '0';
						*out++	= 'D';
						break;

					case	'%':

						*out++	= '%';
						*out++	= '0';
						*out++	= 'D';
						break;


					default:

						*out++	= *predicate;
						break;
				}

				predicate++;
			}
			*out++	= '\n';
			*out++	= '\0';
		}

		// Send it and wait for the ack
		//
		transmit(delboy, (const char *)buffer);
	}
}

#ifdef ANTLR3_WINDOWS
#pragma warning	(push)
#pragma warning (disable : 4100)
#endif

static	void	
commence				(pANTLR3_DEBUG_EVENT_LISTENER delboy)
{
	// Nothing to see here
	//
}

#ifdef ANTLR3_WINDOWS
#pragma warning	(pop)
#endif

static	void	
terminate				(pANTLR3_DEBUG_EVENT_LISTENER delboy)
{
	// Terminate sequence
	//
	sockSend(delboy->socket, "terminate\n", 10);		// Send out the command
}

//----------------------------------------------------------------
// Tree parsing events
//
static	void	
consumeNode				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t)
{
	pANTLR3_STRING	buffer;

	buffer = serializeNode	(delboy, t);

	// Now prepend the command
	//
	buffer->insert8	(buffer, 0, "consumeNode ");
	buffer->addc	(buffer, '\n');

	// Send to the debugger and wait for the ack
	//
	transmit		(delboy, (const char *)(delboy->tokenString->toUTF8(delboy->tokenString)->chars));
}

static	void	
LTT						(pANTLR3_DEBUG_EVENT_LISTENER delboy, int i, pANTLR3_BASE_TREE t)
{
	pANTLR3_STRING	buffer;

	buffer = serializeNode	(delboy, t);

	// Now prepend the command
	//
	buffer->insert8	(buffer, 0, " ");
	buffer->inserti	(buffer, 0, i);
	buffer->insert8	(buffer, 0, "LN ");
	buffer->addc	(buffer, '\n');

	// Send to the debugger and wait for the ack
	//
	transmit		(delboy, (const char *)(delboy->tokenString->toUTF8(delboy->tokenString)->chars));
}

static	void	
nilNode					(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t)
{
	char	buffer[128];
	sprintf(buffer, "nilNode %d\n", delboy->adaptor->getUniqueID(delboy->adaptor, t));
	transmit(delboy, buffer);
}

static	void	
createNode				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t)
{
	// Do we already have a serialization buffer?
	//
	if	(delboy->tokenString == NULL)
	{
		// No, so create one, using the string factory that
		// the grammar name used, which is guaranteed to exist.
		// 64 bytes will do us here for starters. 
		//
		delboy->tokenString = delboy->grammarFileName->factory->newSize(delboy->grammarFileName->factory, 64);
	}

	// Empty string
	//
	delboy->tokenString->set8(delboy->tokenString, (const char *)"createNodeFromTokenElements ");

	// Now we serialize the elements of the node.Note that the debugger only
	// uses 32 bits.
	//
	// Adaptor ID
	//
	delboy->tokenString->addi(delboy->tokenString, delboy->adaptor->getUniqueID(delboy->adaptor, t));
	delboy->tokenString->addc(delboy->tokenString, ' ');

	// Type of the current token (which may be imaginary)
	//
	delboy->tokenString->addi(delboy->tokenString, delboy->adaptor->getType(delboy->adaptor, t));

	// The text that this node represents
	//
	serializeText(delboy->tokenString, delboy->adaptor->getText(delboy->adaptor, t));
	delboy->tokenString->addc(delboy->tokenString, '\n');

	// Finally, as the debugger is a Java program it will expect to get UTF-8
	// encoded strings. We don't use UTF-8 internally to the C runtime, so we 
	// must force encode it. We have a method to do this in the string class, but
	// there is no utf8 string implementation as of yet
	//
	transmit(delboy, (const char *)(delboy->tokenString->toUTF8(delboy->tokenString)->chars));

}
static void
errorNode				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t)
{
	// Do we already have a serialization buffer?
	//
	if	(delboy->tokenString == NULL)
	{
		// No, so create one, using the string factory that
		// the grammar name used, which is guaranteed to exist.
		// 64 bytes will do us here for starters. 
		//
		delboy->tokenString = delboy->grammarFileName->factory->newSize(delboy->grammarFileName->factory, 64);
	}

	// Empty string
	//
	delboy->tokenString->set8(delboy->tokenString, (const char *)"errorNode ");

	// Now we serialize the elements of the node.Note that the debugger only
	// uses 32 bits.
	//
	// Adaptor ID
	//
	delboy->tokenString->addi(delboy->tokenString, delboy->adaptor->getUniqueID(delboy->adaptor, t));
	delboy->tokenString->addc(delboy->tokenString, ' ');

	// Type of the current token (which is an error)
	//
	delboy->tokenString->addi(delboy->tokenString, ANTLR3_TOKEN_INVALID);

	// The text that this node represents
	//
	serializeText(delboy->tokenString, delboy->adaptor->getText(delboy->adaptor, t));
	delboy->tokenString->addc(delboy->tokenString, '\n');

	// Finally, as the debugger is a Java program it will expect to get UTF-8
	// encoded strings. We don't use UTF-8 internally to the C runtime, so we 
	// must force encode it. We have a method to do this in the string class, but
	// there is no utf8 string implementation as of yet
	//
	transmit(delboy, (const char *)(delboy->tokenString->toUTF8(delboy->tokenString)->chars));

}

static	void	
createNodeTok			(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE node, pANTLR3_COMMON_TOKEN token)
{
	char	buffer[128];

	sprintf(buffer, "createNode %d %d\n",	delboy->adaptor->getUniqueID(delboy->adaptor, node), (ANTLR3_UINT32)token->getTokenIndex(token));

	transmit(delboy, buffer);
}

static	void	
becomeRoot				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE newRoot, pANTLR3_BASE_TREE oldRoot)
{
	char	buffer[128];

	sprintf(buffer, "becomeRoot %d %d\n",	delboy->adaptor->getUniqueID(delboy->adaptor, newRoot),
											delboy->adaptor->getUniqueID(delboy->adaptor, oldRoot)
											);
	transmit(delboy, buffer);
}


static	void	
addChild				(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE root, pANTLR3_BASE_TREE child)
{
	char	buffer[128];

	sprintf(buffer, "addChild %d %d\n",		delboy->adaptor->getUniqueID(delboy->adaptor, root),
											delboy->adaptor->getUniqueID(delboy->adaptor, child)
											);
	transmit(delboy, buffer);
}

static	void	
setTokenBoundaries		(pANTLR3_DEBUG_EVENT_LISTENER delboy, pANTLR3_BASE_TREE t, ANTLR3_MARKER tokenStartIndex, ANTLR3_MARKER tokenStopIndex)
{
	char	buffer[128];

	sprintf(buffer, "becomeRoot %d %d %d\n",	delboy->adaptor->getUniqueID(delboy->adaptor, t),
												(ANTLR3_UINT32)tokenStartIndex,
												(ANTLR3_UINT32)tokenStopIndex
											);
	transmit(delboy, buffer);
}
#endif

