/**
 * \file
 * Defines the basic structures of an ANTLR3 bitset. this is a C version of the 
 * cut down Bitset class provided with the java version of antlr 3.
 * 
 * 
 */
#ifndef	_ANTLR3_BITSET_H
#define	_ANTLR3_BITSET_H

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
#include    <antlr3collections.h>

/** How many bits in the elements
 */
#define	ANTLR3_BITSET_BITS	64

/** How many bits in a nible of bits
 */
#define	ANTLR3_BITSET_NIBBLE	4

/** log2 of ANTLR3_BITSET_BITS 2^ANTLR3_BITSET_LOG_BITS = ANTLR3_BITSET_BITS
 */
#define	ANTLR3_BITSET_LOG_BITS	6

/** We will often need to do a mod operator (i mod nbits).
 *  For powers of two, this mod operation is the
 *  same as:
 *   - (i & (nbits-1)).  
 *
 * Since mod is relatively slow, we use an easily
 * precomputed mod mask to do the mod instead.
 */
#define	ANTLR3_BITSET_MOD_MASK	ANTLR3_BITSET_BITS - 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANTLR3_BITSET_LIST_struct
{
	/// Pointer to the allocated array of bits for this bit set, which
    /// is an array of 64 bit elements (of the architecture). If we find a 
    /// machine/C compiler that does not know anything about 64 bit values
    ///	then it should be easy enough to produce a 32 bit (or less) version
    /// of the bitset code. Note that the pointer here may be static if laid down
	/// by the code generation, and it must be copied if it is to be manipulated
	/// to perform followset calculations.
    ///
    pANTLR3_BITWORD   bits;

    /// Length of the current bit set in ANTLR3_UINT64 units.
    ///
    ANTLR3_UINT32    length;
}
	ANTLR3_BITSET_LIST;

typedef	struct ANTLR3_BITSET_struct
{
	/// The actual bits themselves
	///
	ANTLR3_BITSET_LIST				blist;

    pANTLR3_BITSET					(*clone)	    (struct ANTLR3_BITSET_struct  * inSet);
    pANTLR3_BITSET					(*bor)			(struct ANTLR3_BITSET_struct  * bitset1, struct ANTLR3_BITSET_struct * bitset2);
    void							(*borInPlace)   (struct ANTLR3_BITSET_struct  * bitset,  struct ANTLR3_BITSET_struct * bitset2);
    ANTLR3_UINT32					(*size)			(struct ANTLR3_BITSET_struct  * bitset);
    void							(*add)			(struct ANTLR3_BITSET_struct  * bitset, ANTLR3_INT32 bit);
    void							(*grow)			(struct ANTLR3_BITSET_struct  * bitset, ANTLR3_INT32 newSize);
    ANTLR3_BOOLEAN					(*equals)	    (struct ANTLR3_BITSET_struct  * bitset1, struct ANTLR3_BITSET_struct * bitset2);
    ANTLR3_BOOLEAN					(*isMember)	    (struct ANTLR3_BITSET_struct  * bitset, ANTLR3_UINT32 bit);
    ANTLR3_UINT32					(*numBits)	    (struct ANTLR3_BITSET_struct  * bitset);
    void							(*remove)	    (struct ANTLR3_BITSET_struct  * bitset, ANTLR3_UINT32 bit);
    ANTLR3_BOOLEAN					(*isNilNode)	    (struct ANTLR3_BITSET_struct  * bitset);
    pANTLR3_INT32					(*toIntList)    (struct ANTLR3_BITSET_struct  * bitset);

    void							(*free)			(struct ANTLR3_BITSET_struct  * bitset);


}
    ANTLR3_BITSET;

#ifdef __cplusplus
}
#endif



#endif

