/** \file
 * Simple string interface allows indiscriminate allocation of strings
 * such that they can be allocated all over the place and released in 
 * one chunk via a string factory - saves lots of hassle in remembering what
 * strings were allocated where.
 */
#ifndef	_ANTLR3_STRING_H
#define	_ANTLR3_STRING_H

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

#ifdef __cplusplus
extern "C" {
#endif

/** Base string class tracks the allocations and provides simple string
 *  tracking functions. Mostly you can work directly on the string for things
 *  that don't reallocate it, like strchr() etc. Perhaps someone will want to provide implementations for UTF8
 *  and so on.
 */
typedef	struct ANTLR3_STRING_struct
{

    /** The factory that created this string
     */
    pANTLR3_STRING_FACTORY	factory;

    /** Pointer to the current string value (starts at NULL unless
     *  the string allocator is told to create it with a pre known size.
     */
    pANTLR3_UINT8		chars;

    /** Current length of the string up to and not including, the trailing '\0'
     *  Note that the actual allocation (->size)
     *  is always at least one byte more than this to accommodate trailing '\0'
     */
    ANTLR3_UINT32		len;

    /** Current size of the string in bytes including the trailing '\0'
     */
    ANTLR3_UINT32		size;

    /** Index of string (allocation number) in case someone wants
     *  to explicitly release it.
     */
    ANTLR3_UINT32		index;

    /** Occasionally it is useful to know what the encoding of the string
     *  actually is, hence it is stored here as one the ANTLR3_ENCODING_ values
     */
    ANTLR3_UINT8		encoding;

    /** Pointer to function that sets the string value to a specific string in the default encoding
     *  for this string. For instance, if this is ASCII 8 bit, then this function is the same as set8
     *  but if the encoding is 16 bit, then the pointer is assumed to point to 16 bit characters not
     *  8 bit.
     */
    pANTLR3_UINT8   (*set)	(struct ANTLR3_STRING_struct * string, const char * chars);
   
    /** Pointer to function that sets the string value to a specific 8 bit string in the default encoding
     *  for this string. For instance, if this is a 16 bit string, then this function is the same as set8
     *  but if the encoding is 16 bit, then the pointer is assumed to point to 8 bit characters that must
     *  be converted to 16 bit characters on the fly.
     */
    pANTLR3_UINT8   (*set8)	(struct ANTLR3_STRING_struct * string, const char * chars);

    /** Pointer to function adds a raw char * type pointer in the default encoding
     *  for this string. For instance, if this is ASCII 8 bit, then this function is the same as append8
     *  but if the encoding is 16 bit, then the pointer is assumed to point to 16 bit characters not
     *  8 bit.
     */
    pANTLR3_UINT8   (*append)	(struct ANTLR3_STRING_struct * string, const char * newbit);

    /** Pointer to function adds a raw char * type pointer in the default encoding
     *  for this string. For instance, if this is a 16 bit string, then this function assumes the pointer
     *  points to 8 bit characters that must be converted on the fly.
     */
    pANTLR3_UINT8   (*append8)	(struct ANTLR3_STRING_struct * string, const char * newbit);

    /** Pointer to function that inserts the supplied string at the specified
     *  offset in the current string in the default encoding for this string. For instance, if this is an 8
     *  bit string, then this is the same as insert8, but if this is a 16 bit string, then the poitner
     *  must point to 16 bit characters.
     *  
     */
    pANTLR3_UINT8   (*insert)	(struct ANTLR3_STRING_struct * string, ANTLR3_UINT32 point, const char * newbit);

    /** Pointer to function that inserts the supplied string at the specified
     *  offset in the current string in the default encoding for this string. For instance, if this is a 16 bit string
     *  then the pointer is assumed to point at 8 bit characteres that must be converted on the fly.
     */
    pANTLR3_UINT8   (*insert8)	(struct ANTLR3_STRING_struct * string, ANTLR3_UINT32 point, const char * newbit);

    /** Pointer to function that sets the string value to a copy of the supplied string (strings must be in the 
     *  same encoding.
     */
    pANTLR3_UINT8   (*setS)	(struct ANTLR3_STRING_struct * string, struct ANTLR3_STRING_struct * chars);

    /** Pointer to function appends a copy of the characters contained in another string. Strings must be in the
     *  same encoding.
     */
    pANTLR3_UINT8   (*appendS)	(struct ANTLR3_STRING_struct * string, struct ANTLR3_STRING_struct * newbit);

    /** Pointer to function that inserts a copy of the characters in the supplied string at the specified
     *  offset in the current string. strings must be in the same encoding.
     */
    pANTLR3_UINT8   (*insertS)	(struct ANTLR3_STRING_struct * string, ANTLR3_UINT32 point, struct ANTLR3_STRING_struct * newbit);

    /** Pointer to function that inserts the supplied integer in string form at the specified
     *  offset in the current string.
     */
    pANTLR3_UINT8   (*inserti)	(struct ANTLR3_STRING_struct * string, ANTLR3_UINT32 point, ANTLR3_INT32 i);

    /** Pointer to function that adds a single character to the end of the string, in the encoding of the
     *  string - 8 bit, 16 bit, utf-8 etc. Input is a single UTF32 (32 bits wide integer) character.
     */
    pANTLR3_UINT8   (*addc)	(struct ANTLR3_STRING_struct * string, ANTLR3_UINT32 c);

    /** Pointer to function that adds the stringified representation of an integer
     *  to the string.
     */
    pANTLR3_UINT8   (*addi)	(struct ANTLR3_STRING_struct * string, ANTLR3_INT32 i);

    /** Pointer to function that compares the text of a string to the supplied
     *  8 bit character string and returns a result a la strcmp()
     */
    ANTLR3_UINT32   (*compare8)	(struct ANTLR3_STRING_struct * string, const char * compStr);

    /** Pointer to a function that compares the text of a string with the supplied character string
     *  (which is assumed to be in the same encoding as the string itself) and returns a result
     *  a la strcmp()
     */
    ANTLR3_UINT32   (*compare)	(struct ANTLR3_STRING_struct * string, const char * compStr);

    /** Pointer to a function that compares the text of a string with the supplied string
     *  (which is assumed to be in the same encoding as the string itself) and returns a result
     *  a la strcmp()
     */
    ANTLR3_UINT32   (*compareS)	(struct ANTLR3_STRING_struct * string, struct ANTLR3_STRING_struct * compStr);

    /** Pointer to a function that returns the character indexed at the supplied
     *  offset as a 32 bit character.
     */
    ANTLR3_UCHAR    (*charAt)	(struct ANTLR3_STRING_struct * string, ANTLR3_UINT32 offset);

    /** Pointer to a function that returns a substring of the supplied string a la .subString(s,e)
     *  in the Java language.
     */
    struct ANTLR3_STRING_struct *
					(*subString)    (struct ANTLR3_STRING_struct * string, ANTLR3_UINT32 startIndex, ANTLR3_UINT32 endIndex);

    /** Pointer to a function that returns the integer representation of any numeric characters
     *  at the beginning of the string
     */
    ANTLR3_INT32	(*toInt32)	    (struct ANTLR3_STRING_struct * string);

    /** Pointer to a function that yields an 8 bit string regardless of the encoding of the supplied
     *  string. This is useful when you want to use the text of a token in some way that requires an 8 bit
     *  value, such as the key for a hashtable. The function is required to produce a usable string even
     *  if the text given as input has characters that do not fit in 8 bit space, it will replace them
     *  with some arbitrary character such as '?'
     */
    struct ANTLR3_STRING_struct *
					(*to8)	    (struct ANTLR3_STRING_struct * string);

	/// Pointer to a function that yields a UT8 encoded string of the current string,
	/// regardless of the current encoding of the string. Because there is currently no UTF8
	/// handling in the string class, it creates therefore, a string that is useful only for read only 
	/// applications as it will not contain methods that deal with UTF8 at the moment.
	///
	struct ANTLR3_STRING_struct *
					(*toUTF8)	(struct ANTLR3_STRING_struct * string);
	
}
    ANTLR3_STRING;

/** Definition of the string factory interface, which creates and tracks
 *  strings for you of various shapes and sizes.
 */
typedef struct	ANTLR3_STRING_FACTORY_struct
{
    /** List of all the strings that have been allocated by the factory
     */
    pANTLR3_VECTOR    strings;

    /* Index of next string that we allocate
     */
    ANTLR3_UINT32   index;

    /** Pointer to function that manufactures an empty string
     */
    pANTLR3_STRING  (*newRaw)	(struct ANTLR3_STRING_FACTORY_struct * factory);

    /** Pointer to function that manufactures a raw string with no text in it but space for size
     *  characters.
     */
    pANTLR3_STRING  (*newSize)	(struct ANTLR3_STRING_FACTORY_struct * factory, ANTLR3_UINT32 size);

    /** Pointer to function that manufactures a string from a given pointer and length. The pointer is assumed
     *  to point to characters in the same encoding as the string type, hence if this is a 16 bit string the
     *  pointer should point to 16 bit characters.
     */
    pANTLR3_STRING  (*newPtr)	(struct ANTLR3_STRING_FACTORY_struct * factory, pANTLR3_UINT8 string, ANTLR3_UINT32 size);

    /** Pointer to function that manufactures a string from a given pointer and length. The pointer is assumed to
     *  point at 8 bit characters which must be converted on the fly to the encoding of the actual string.
     */
    pANTLR3_STRING  (*newPtr8)	(struct ANTLR3_STRING_FACTORY_struct * factory, pANTLR3_UINT8 string, ANTLR3_UINT32 size);

    /** Pointer to function that manufactures a string from a given pointer and works out the length. The pointer is 
     *  assumed to point to characters in the same encoding as the string itself, i.e. 16 bit if a 16 bit
     *  string and so on.
     */
    pANTLR3_STRING  (*newStr)	(struct ANTLR3_STRING_FACTORY_struct * factory, pANTLR3_UINT8 string);

    /** Pointer to function that manufactures a string from a given pointer and length. The pointer should
     *  point to 8 bit characters regardless of the actual encoding of the string. The 8 bit characters
     *  will be converted to the actual string encoding on the fly.
     */
    pANTLR3_STRING  (*newStr8)	(struct ANTLR3_STRING_FACTORY_struct * factory, pANTLR3_UINT8 string);

    /** Pointer to function that deletes the string altogether
     */
    void	    (*destroy)	(struct ANTLR3_STRING_FACTORY_struct * factory, pANTLR3_STRING string);

    /** Pointer to function that returns a copy of the string in printable form without any control
     *  characters in it.
     */
    pANTLR3_STRING  (*printable)(struct ANTLR3_STRING_FACTORY_struct * factory, pANTLR3_STRING string);

    /** Pointer to function that closes the factory
     */
    void	    (*close)	(struct ANTLR3_STRING_FACTORY_struct * factory);

}
    ANTLR3_STRING_FACTORY;

#ifdef __cplusplus
}
#endif

#endif

