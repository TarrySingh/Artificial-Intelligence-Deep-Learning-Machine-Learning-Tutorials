/** \file
 *  Abstraction of Common tree to provide payload and string representation of node.
 *
 * \todo May not need this in the end
 */

#ifndef	ANTLR3_PARSETREE_H
#define	ANTLR3_PARSETREE_H

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

#include    <antlr3basetree.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANTLR3_PARSE_TREE_struct
{
    /** Any interface that implements methods in this interface
     *  may need to point back to itself using this pointer to its
     *  super structure.
     */
    void    * super;

    /** The payload that the parse tree node passes around
     */
    void    * payload;

    /** An encapsulated BASE TREE strcuture (NOT a pointer)
      * that perfoms a lot of the dirty work of node management
      */
    ANTLR3_BASE_TREE	    baseTree;

    /** How to dup this node
     */
    pANTLR3_BASE_TREE	    (*dupNode)	(struct ANTLR3_PARSE_TREE_struct * tree);

    /** Return the type of this node
     */
    ANTLR3_UINT32	    (*getType)	(struct ANTLR3_PARSE_TREE_struct * tree);

    /** Return the string representation of the payload (must be installed
     *  when the payload is added and point to a function that knwos how to 
     *  manifest a pANTLR3_STRING from a node.
     */
    pANTLR3_STRING	    (*toString)	(struct ANTLR3_PARSE_TREE_struct * payload);

    void		    (*free)	(struct ANTLR3_PARSE_TREE_struct * tree);

}
    ANTLR3_PARSE_TREE;

#ifdef __cplusplus
}
#endif

#endif
