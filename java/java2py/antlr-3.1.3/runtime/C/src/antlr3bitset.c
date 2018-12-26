///
/// \file
/// Contains the C implementation of ANTLR3 bitsets as adapted from Terence Parr's
/// Java implementation.
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

#include    <antlr3bitset.h>

// External interface
//

static	pANTLR3_BITSET  antlr3BitsetClone		(pANTLR3_BITSET inSet);
static	pANTLR3_BITSET  antlr3BitsetOR			(pANTLR3_BITSET bitset1, pANTLR3_BITSET bitset2);
static	void			antlr3BitsetORInPlace	(pANTLR3_BITSET bitset, pANTLR3_BITSET bitset2);
static	ANTLR3_UINT32	antlr3BitsetSize		(pANTLR3_BITSET bitset);
static	void			antlr3BitsetAdd			(pANTLR3_BITSET bitset, ANTLR3_INT32 bit);
static	ANTLR3_BOOLEAN	antlr3BitsetEquals		(pANTLR3_BITSET bitset1, pANTLR3_BITSET bitset2);
static	ANTLR3_BOOLEAN	antlr3BitsetMember		(pANTLR3_BITSET bitset, ANTLR3_UINT32 bit);
static	ANTLR3_UINT32	antlr3BitsetNumBits		(pANTLR3_BITSET bitset);
static	void			antlr3BitsetRemove		(pANTLR3_BITSET bitset, ANTLR3_UINT32 bit);
static	ANTLR3_BOOLEAN	antlr3BitsetIsNil		(pANTLR3_BITSET bitset);
static	pANTLR3_INT32	antlr3BitsetToIntList	(pANTLR3_BITSET bitset);

// Local functions
//
static	void			growToInclude		(pANTLR3_BITSET bitset, ANTLR3_INT32 bit);
static	void			grow				(pANTLR3_BITSET bitset, ANTLR3_INT32 newSize);
static	ANTLR3_UINT64	bitMask				(ANTLR3_UINT32 bitNumber);
static	ANTLR3_UINT32	numWordsToHold		(ANTLR3_UINT32 bit);
static	ANTLR3_UINT32	wordNumber			(ANTLR3_UINT32 bit);
static	void			antlr3BitsetFree	(pANTLR3_BITSET bitset);

static void
antlr3BitsetFree(pANTLR3_BITSET bitset)
{
    if	(bitset->blist.bits != NULL)
    {
		ANTLR3_FREE(bitset->blist.bits);
		bitset->blist.bits = NULL;
    }
    ANTLR3_FREE(bitset);

    return;
}

ANTLR3_API pANTLR3_BITSET
antlr3BitsetNew(ANTLR3_UINT32 numBits)
{
	pANTLR3_BITSET  bitset;

	ANTLR3_UINT32   numelements;

	// Allocate memory for the bitset structure itself
	//
	bitset  = (pANTLR3_BITSET) ANTLR3_MALLOC((size_t)sizeof(ANTLR3_BITSET));

	if	(bitset == NULL)
	{
		return	NULL;
	}

	// Avoid memory thrashing at the up front expense of a few bytes
	//
	if	(numBits < (8 * ANTLR3_BITSET_BITS))
	{
		numBits = 8 * ANTLR3_BITSET_BITS;
	}

	// No we need to allocate the memory for the number of bits asked for
	// in multiples of ANTLR3_UINT64. 
	//
	numelements	= ((numBits -1) >> ANTLR3_BITSET_LOG_BITS) + 1;

	bitset->blist.bits    = (pANTLR3_BITWORD) ANTLR3_MALLOC((size_t)(numelements * sizeof(ANTLR3_BITWORD)));
	memset(bitset->blist.bits, 0, (size_t)(numelements * sizeof(ANTLR3_BITWORD)));
	bitset->blist.length  = numelements;

	if	(bitset->blist.bits == NULL)
	{
		ANTLR3_FREE(bitset);
		return	NULL;
	}

	antlr3BitsetSetAPI(bitset);


	// All seems good
	//
	return  bitset;
}

ANTLR3_API void
antlr3BitsetSetAPI(pANTLR3_BITSET bitset)
{
    bitset->clone		=    antlr3BitsetClone;
    bitset->bor			=    antlr3BitsetOR;
    bitset->borInPlace	=    antlr3BitsetORInPlace;
    bitset->size		=    antlr3BitsetSize;
    bitset->add			=    antlr3BitsetAdd;
    bitset->grow		=    grow;
    bitset->equals		=    antlr3BitsetEquals;
    bitset->isMember	=    antlr3BitsetMember;
    bitset->numBits		=    antlr3BitsetNumBits;
    bitset->remove		=    antlr3BitsetRemove;
    bitset->isNilNode		=    antlr3BitsetIsNil;
    bitset->toIntList	=    antlr3BitsetToIntList;

    bitset->free		=    antlr3BitsetFree;
}

ANTLR3_API pANTLR3_BITSET
antlr3BitsetCopy(pANTLR3_BITSET_LIST blist)
{
    pANTLR3_BITSET  bitset;
	int				numElements;

    // Allocate memory for the bitset structure itself
    //
    bitset  = (pANTLR3_BITSET) ANTLR3_MALLOC((size_t)sizeof(ANTLR3_BITSET));

    if	(bitset == NULL)
    {
		return	NULL;
    }

	numElements = blist->length;

    // Avoid memory thrashing at the expense of a few more bytes
    //
    if	(numElements < 8)
    {
		numElements = 8;
    }

    // Install the length in ANTLR3_UINT64 units
    //
    bitset->blist.length  = numElements;

    bitset->blist.bits    = (pANTLR3_BITWORD)ANTLR3_MALLOC((size_t)(numElements * sizeof(ANTLR3_BITWORD)));

    if	(bitset->blist.bits == NULL)
    {
		ANTLR3_FREE(bitset);
		return	NULL;
    }

	ANTLR3_MEMCPY(bitset->blist.bits, blist->bits, (ANTLR3_UINT64)(numElements * sizeof(ANTLR3_BITWORD)));

    // All seems good
    //
    return  bitset;
}

static pANTLR3_BITSET
antlr3BitsetClone(pANTLR3_BITSET inSet)
{
    pANTLR3_BITSET  bitset;

    // Allocate memory for the bitset structure itself
    //
    bitset  = antlr3BitsetNew(ANTLR3_BITSET_BITS * inSet->blist.length);

    if	(bitset == NULL)
    {
		return	NULL;
    }

    // Install the actual bits in the source set
    //
    ANTLR3_MEMCPY(bitset->blist.bits, inSet->blist.bits, (ANTLR3_UINT64)(inSet->blist.length * sizeof(ANTLR3_BITWORD)));

    // All seems good
    //
    return  bitset;
}


ANTLR3_API pANTLR3_BITSET
antlr3BitsetList(pANTLR3_HASH_TABLE list)
{
    pANTLR3_BITSET		bitSet;
    pANTLR3_HASH_ENUM	en;
    pANTLR3_HASH_KEY	key;
    ANTLR3_UINT64		bit;

    // We have no idea what exactly is in the list
    // so create a default bitset and then just add stuff
    // as we enumerate.
    //
    bitSet  = antlr3BitsetNew(0);

    en		= antlr3EnumNew(list);

    while   (en->next(en, &key, (void **)(&bit)) == ANTLR3_SUCCESS)
    {
		bitSet->add(bitSet, (ANTLR3_UINT32)bit);
    }
    en->free(en);

    return NULL;
}

///
/// \brief
/// Creates a new bitset with at least one 64 bit bset of bits, but as
/// many 64 bit sets as are required.
///
/// \param[in] bset
/// A variable number of bits to add to the set, ending in -1 (impossible bit).
/// 
/// \returns
/// A new bit set with all of the specified bitmaps in it and the API
/// initialized.
/// 
/// Call as:
///  - pANTLR3_BITSET = antlrBitsetLoad(bset, bset11, ..., -1);
///  - pANTLR3_BITSET = antlrBitsetOf(-1);  Create empty bitset 
///
/// \remarks
/// Stdargs function - must supply -1 as last paremeter, which is NOT
/// added to the set.
/// 
///
ANTLR3_API pANTLR3_BITSET
antlr3BitsetLoad(pANTLR3_BITSET_LIST inBits)
{
	pANTLR3_BITSET  bitset;
	ANTLR3_UINT32  count;

	// Allocate memory for the bitset structure itself
	// the input parameter is the bit number (0 based)
	// to include in the bitset, so we need at at least
	// bit + 1 bits. If any arguments indicate a 
	// a bit higher than the default number of bits (0 means default size)
	// then Add() will take care
	// of it.
	//
	bitset  = antlr3BitsetNew(0);

	if	(bitset == NULL)
	{
		return	NULL;
	}

	if	(inBits != NULL)
	{
		// Now we can add the element bits into the set
		//
		count=0;
		while (count < inBits->length)
		{
			if  (bitset->blist.length <= count)
			{
				bitset->grow(bitset, count+1);
			}

			bitset->blist.bits[count] = *((inBits->bits)+count);
			count++;
		}
	}

	// return the new bitset
	//
	return  bitset;
}

///
/// \brief
/// Creates a new bitset with at least one element, but as
/// many elements are required.
/// 
/// \param[in] bit
/// A variable number of bits to add to the set, ending in -1 (impossible bit).
/// 
/// \returns
/// A new bit set with all of the specified elements added into it.
/// 
/// Call as:
///  - pANTLR3_BITSET = antlrBitsetOf(n, n1, n2, -1);
///  - pANTLR3_BITSET = antlrBitsetOf(-1);  Create empty bitset 
///
/// \remarks
/// Stdargs function - must supply -1 as last paremeter, which is NOT
/// added to the set.
/// 
///
ANTLR3_API pANTLR3_BITSET
antlr3BitsetOf(ANTLR3_INT32 bit, ...)
{
    pANTLR3_BITSET  bitset;

    va_list ap;

    // Allocate memory for the bitset structure itself
    // the input parameter is the bit number (0 based)
    // to include in the bitset, so we need at at least
    // bit + 1 bits. If any arguments indicate a 
    // a bit higher than the default number of bits (0 menas default size)
    // then Add() will take care
    // of it.
    //
    bitset  = antlr3BitsetNew(0);

    if	(bitset == NULL)
    {
		return	NULL;
    }

    // Now we can add the element bits into the set
    //
    va_start(ap, bit);
    while   (bit != -1)
    {
		antlr3BitsetAdd(bitset, bit);
		bit = va_arg(ap, ANTLR3_UINT32);
    }
    va_end(ap);

    // return the new bitset
    //
    return  bitset;
}

static pANTLR3_BITSET
antlr3BitsetOR(pANTLR3_BITSET bitset1, pANTLR3_BITSET bitset2)
{
    pANTLR3_BITSET  bitset;

    if	(bitset1 == NULL)
    {
		return antlr3BitsetClone(bitset2);
    }

    if	(bitset2 == NULL)
    {
		return	antlr3BitsetClone(bitset1);
    }

    // Allocate memory for the newly ordered bitset structure itself.
    //
    bitset  = antlr3BitsetClone(bitset1);
    
    antlr3BitsetORInPlace(bitset, bitset2);

    return  bitset;

}

static void
antlr3BitsetAdd(pANTLR3_BITSET bitset, ANTLR3_INT32 bit)
{
    ANTLR3_UINT32   word;

    word    = wordNumber(bit);

    if	(word	> bitset->blist.length)
    {
		growToInclude(bitset, bit);
    }

    bitset->blist.bits[word] |= bitMask(bit);

}

static void
grow(pANTLR3_BITSET bitset, ANTLR3_INT32 newSize)
{
    pANTLR3_BITWORD   newBits;

    // Space for newly sized bitset - TODO: come back to this and use realloc?, it may
    // be more efficient...
    //
    newBits = (pANTLR3_BITWORD) ANTLR3_MALLOC((size_t)(newSize * sizeof(ANTLR3_BITWORD)));

    if	(bitset->blist.bits != NULL)
    {
		// Copy existing bits
		//
		ANTLR3_MEMCPY((void *)newBits, (const void *)bitset->blist.bits, (size_t)(bitset->blist.length * sizeof(ANTLR3_BITWORD)));

		// Out with the old bits... de de de derrr
		//
		ANTLR3_FREE(bitset->blist.bits);
    }

    // In with the new bits... keerrrang.
    //
    bitset->blist.bits    = newBits;
}

static void
growToInclude(pANTLR3_BITSET bitset, ANTLR3_INT32 bit)
{
	ANTLR3_UINT32	bl;
	ANTLR3_UINT32	nw;

	bl = (bitset->blist.length << 1);
	nw = numWordsToHold(bit);
	if	(bl > nw)
	{
		bitset->grow(bitset, bl);
	}
	else
	{
		bitset->grow(bitset, nw);
	}
}

static void
antlr3BitsetORInPlace(pANTLR3_BITSET bitset, pANTLR3_BITSET bitset2)
{
    ANTLR3_UINT32   minimum;
    ANTLR3_UINT32   i;

    if	(bitset2 == NULL)
    {
		return;
    }


    // First make sure that the target bitset is big enough
    // for the new bits to be ored in.
    //
    if	(bitset->blist.length < bitset2->blist.length)
    {
		growToInclude(bitset, (bitset2->blist.length * sizeof(ANTLR3_BITWORD)));
    }
    
    // Or the miniimum number of bits after any resizing went on
    //
    if	(bitset->blist.length < bitset2->blist.length)
	{
		minimum = bitset->blist.length;
	}
	else
	{
		minimum = bitset2->blist.length;
	}

    for	(i = minimum; i > 0; i--)
    {
		bitset->blist.bits[i-1] |= bitset2->blist.bits[i-1];
    }
}

static ANTLR3_UINT64
bitMask(ANTLR3_UINT32 bitNumber)
{
    return  ((ANTLR3_UINT64)1) << (bitNumber & (ANTLR3_BITSET_MOD_MASK));
}

static ANTLR3_UINT32
antlr3BitsetSize(pANTLR3_BITSET bitset)
{
    ANTLR3_UINT32   degree;
    ANTLR3_INT32   i;
    ANTLR3_INT8    bit;
    
    // TODO: Come back to this, it may be faster to & with 0x01
    // then shift right a copy of the 4 bits, than shift left a constant of 1.
    // But then again, the optimizer might just work this out
    // anyway.
    //
    degree  = 0;
    for	(i = bitset->blist.length - 1; i>= 0; i--)
    {
		if  (bitset->blist.bits[i] != 0)
		{
			for	(bit = ANTLR3_BITSET_BITS - 1; bit >= 0; bit--)
			{
				if  ((bitset->blist.bits[i] & (((ANTLR3_BITWORD)1) << bit)) != 0)
				{
					degree++;
				}
			}
		}
    }
    return degree;
}

static ANTLR3_BOOLEAN
antlr3BitsetEquals(pANTLR3_BITSET bitset1, pANTLR3_BITSET bitset2)
{
    ANTLR3_INT32   minimum;
    ANTLR3_INT32   i;

    if	(bitset1 == NULL || bitset2 == NULL)
    {
	return	ANTLR3_FALSE;
    }

    // Work out the minimum comparison set
    //
    if	(bitset1->blist.length < bitset2->blist.length)
    {
		minimum = bitset1->blist.length;
    }
    else
    {
		minimum = bitset2->blist.length;
    }

    // Make sure explict in common bits are equal
    //
    for	(i = minimum - 1; i >=0 ; i--)
    {
		if  (bitset1->blist.bits[i] != bitset2->blist.bits[i])
		{
			return  ANTLR3_FALSE;
		}
    }

    // Now make sure the bits of the larger set are all turned
    // off.
    //
    if	(bitset1->blist.length > (ANTLR3_UINT32)minimum)
    {
		for (i = minimum ; (ANTLR3_UINT32)i < bitset1->blist.length; i++)
		{
			if	(bitset1->blist.bits[i] != 0)
			{
				return	ANTLR3_FALSE;
			}
		}
    }
    else if (bitset2->blist.length > (ANTLR3_UINT32)minimum)
    {
		for (i = minimum; (ANTLR3_UINT32)i < bitset2->blist.length; i++)
		{
			if	(bitset2->blist.bits[i] != 0)
			{
				return	ANTLR3_FALSE;
			}
		}
    }

    return  ANTLR3_TRUE;
}

static ANTLR3_BOOLEAN
antlr3BitsetMember(pANTLR3_BITSET bitset, ANTLR3_UINT32 bit)
{
    ANTLR3_UINT32    wordNo;

    wordNo  = wordNumber(bit);

    if	(wordNo >= bitset->blist.length)
    {
		return	ANTLR3_FALSE;
    }
    
    if	((bitset->blist.bits[wordNo] & bitMask(bit)) == 0)
    {
		return	ANTLR3_FALSE;
    }
    else
    {
		return	ANTLR3_TRUE;
    }
}

static void
antlr3BitsetRemove(pANTLR3_BITSET bitset, ANTLR3_UINT32 bit)
{
    ANTLR3_UINT32    wordNo;

    wordNo  = wordNumber(bit);

    if	(wordNo < bitset->blist.length)
    {
		bitset->blist.bits[wordNo] &= ~(bitMask(bit));
    }
}
static ANTLR3_BOOLEAN
antlr3BitsetIsNil(pANTLR3_BITSET bitset)
{
   ANTLR3_INT32    i;

   for	(i = bitset->blist.length -1; i>= 0; i--)
   {
       if   (bitset->blist.bits[i] != 0)
       {
			return ANTLR3_FALSE;
       }
   }
   
   return   ANTLR3_TRUE;
}

static ANTLR3_UINT32
numWordsToHold(ANTLR3_UINT32 bit)
{
    return  (bit >> ANTLR3_BITSET_LOG_BITS) + 1;
}

static	ANTLR3_UINT32
wordNumber(ANTLR3_UINT32 bit)
{
    return  bit >> ANTLR3_BITSET_LOG_BITS;
}

static ANTLR3_UINT32
antlr3BitsetNumBits(pANTLR3_BITSET bitset)
{
    return  bitset->blist.length << ANTLR3_BITSET_LOG_BITS;
}

/** Produce an integer list of all the bits that are turned on
 *  in this bitset. Used for error processing in the main as the bitset
 *  reresents a number of integer tokens which we use for follow sets
 *  and so on.
 *
 *  The first entry is the number of elements following in the list.
 */
static	pANTLR3_INT32	
antlr3BitsetToIntList	(pANTLR3_BITSET bitset)
{
    ANTLR3_UINT32   numInts;	    // How many integers we will need
    ANTLR3_UINT32   numBits;	    // How many bits are in the set
    ANTLR3_UINT32   i;
    ANTLR3_UINT32   index;

    pANTLR3_INT32  intList;

    numInts = bitset->size(bitset) + 1;
    numBits = bitset->numBits(bitset);
 
    intList = (pANTLR3_INT32)ANTLR3_MALLOC(numInts * sizeof(ANTLR3_INT32));

    if	(intList == NULL)
    {
		return NULL;	// Out of memory
    }

    intList[0] = numInts;

    // Enumerate the bits that are turned on
    //
    for	(i = 0, index = 1; i<numBits; i++)
    {
		if  (bitset->isMember(bitset, i) == ANTLR3_TRUE)
		{
			intList[index++]    = i;
		}
    }

    // Result set
    //
    return  intList;
}

