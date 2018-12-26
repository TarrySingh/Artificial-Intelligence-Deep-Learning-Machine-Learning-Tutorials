/** \file
 * \brief The ANTLR3 C filestream is used when the source character stream
 * is a filesystem based input set and all the characters in the filestream
 * can be loaded at once into memory and away the lexer goes.
 *
 * A number of initializers are provided in order that various character
 * sets can be supported from input files. The ANTLR3 C runtime expects
 * to deal with UTF32 characters only (the reasons for this are to
 * do with the simplification of C code when using this form of Unicode 
 * encoding, though this is not a panacea. More information can be
 * found on this by consulting: 
 *   - http://www.unicode.org/versions/Unicode4.0.0/ch02.pdf#G11178
 * Where a well grounded discussion of the encoding formats available
 * may be found.
 *
 */

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


/** \brief Use the contents of an operating system file as the input
 *         for an input stream.
 *
 * \param fileName Name of operating system file to read.
 * \return
 *	- Pointer to new input stream context upon success
 *	- One of the ANTLR3_ERR_ defines on error.
 */
ANTLR3_API pANTLR3_INPUT_STREAM
antlr3AsciiFileStreamNew(pANTLR3_UINT8 fileName)
{
	// Pointer to the input stream we are going to create
	//
	pANTLR3_INPUT_STREAM    input;
	ANTLR3_UINT32	    status;

	if	(fileName == NULL)
	{
		return NULL;
	}

	// Allocate memory for the input stream structure
	//
	input   = (pANTLR3_INPUT_STREAM)
		ANTLR3_CALLOC(1, sizeof(ANTLR3_INPUT_STREAM));

	if	(input == NULL)
	{
		return	NULL;
	}

	// Structure was allocated correctly, now we can read the file.
	//
	status  = antlr3readAscii(input, fileName);

	// Call the common 8 bit ASCII input stream handler
	// Initializer type thingy doobry function.
	//
	antlr3AsciiSetupStream(input, ANTLR3_CHARSTREAM);

	// Now we can set up the file name
	//	
	input->istream->streamName	= input->strFactory->newStr(input->strFactory, fileName);
	input->fileName				= input->istream->streamName;

	if	(status != ANTLR3_SUCCESS)
	{
		input->close(input);
		return	NULL;
	}

	return  input;
}

ANTLR3_API ANTLR3_UINT32
antlr3readAscii(pANTLR3_INPUT_STREAM    input, pANTLR3_UINT8 fileName)
{
	ANTLR3_FDSC		    infile;
	ANTLR3_UINT32	    fSize;

	/* Open the OS file in read binary mode
	*/
	infile  = antlr3Fopen(fileName, "rb");

	/* Check that it was there
	*/
	if	(infile == NULL)
	{
		return	(ANTLR3_UINT32)ANTLR3_ERR_NOFILE;
	}

	/* It was there, so we can read the bytes now
	*/
	fSize   = antlr3Fsize(fileName);	/* Size of input file	*/

	/* Allocate buffer for this input set   
	*/
	input->data	    = ANTLR3_MALLOC((size_t)fSize);
	input->sizeBuf  = fSize;

	if	(input->data == NULL)
	{
		return	(ANTLR3_UINT32)ANTLR3_ERR_NOMEM;
	}

	input->isAllocated	= ANTLR3_TRUE;

	/* Now we read the file. Characters are not converted to
	* the internal ANTLR encoding until they are read from the buffer
	*/
	antlr3Fread(infile, fSize, input->data);

	/* And close the file handle
	*/
	antlr3Fclose(infile);

	return  ANTLR3_SUCCESS;
}

/** \brief Open an operating system file and return the descriptor
 * We just use the common open() and related functions here. 
 * Later we might find better ways on systems
 * such as Windows and OpenVMS for instance. But the idea is to read the 
 * while file at once anyway, so it may be irrelevant.
 */
ANTLR3_API ANTLR3_FDSC
antlr3Fopen(pANTLR3_UINT8 filename, const char * mode)
{
    return  (ANTLR3_FDSC)fopen((const char *)filename, mode);
}

/** \brief Close an operating system file and free any handles
 *  etc.
 */
ANTLR3_API void
antlr3Fclose(ANTLR3_FDSC fd)
{
    fclose(fd);
}
ANTLR3_API ANTLR3_UINT32
antlr3Fsize(pANTLR3_UINT8 fileName)
{   
    struct _stat	statbuf;

    _stat((const char *)fileName, &statbuf);

    return (ANTLR3_UINT32)statbuf.st_size;
}

ANTLR3_API ANTLR3_UINT32
antlr3Fread(ANTLR3_FDSC fdsc, ANTLR3_UINT32 count,  void * data)
{
    return  (ANTLR3_UINT32)fread(data, (size_t)count, 1, fdsc);
}
