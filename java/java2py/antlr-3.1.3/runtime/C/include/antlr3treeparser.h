#ifndef	ANTLR3TREEPARSER_H
#define	ANTLR3TREEPARSER_H

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

#include    <antlr3defs.h>
#include    <antlr3baserecognizer.h>
#include    <antlr3commontreenodestream.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Internal structure representing an element in a hash bucket.
 *  Stores the original key so that duplicate keys can be rejected
 *  if necessary, and contains function can be supported If the hash key
 *  could be unique I would have invented the perfect compression algorithm ;-)
 */
typedef	struct	ANTLR3_TREE_PARSER_struct
{
    /** Pointer to any super class
     */
    void    * super;

    /** A pointer to the base recognizer, where most of the parser functions actually
     *  live because they are shared between parser and tree parser and this is the
     *  easier way than copying the interface all over the place. Macros hide this
     *  for the generated code so it is easier on the eye (though not the debugger ;-).
     */
    pANTLR3_BASE_RECOGNIZER		rec;

    /** Pointer to the common tree node stream for the parser
     */
    pANTLR3_COMMON_TREE_NODE_STREAM	ctnstream;

    /** Set the input stream and reset the parser
     */
    void			    (*setTreeNodeStream)    (struct ANTLR3_TREE_PARSER_struct * parser, pANTLR3_COMMON_TREE_NODE_STREAM input);

    /** Return a pointer to the input stream
     */
    pANTLR3_COMMON_TREE_NODE_STREAM (*getTreeNodeStream)    (struct ANTLR3_TREE_PARSER_struct * parser);
    
    /** Pointer to a function that knows how to free resources of an ANTLR3 tree parser.
     */
    void			    (*free)		    (struct ANTLR3_TREE_PARSER_struct * parser);
}
    ANTLR3_TREE_PARSER;

#ifdef __cplusplus
}
#endif

#endif
