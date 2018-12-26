/// \file
/// Provides a number of useful functions that are roughly equivalent
/// to java HashTable and List for the purposes of Antlr 3 C runtime.
/// Also useable by the C programmer for things like symbol tables pointers
/// and so on.
///
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

// Interface functions for hash table
//

// String based keys
//
static void					antlr3HashDelete    (pANTLR3_HASH_TABLE table, void * key);
static void *				antlr3HashGet	(pANTLR3_HASH_TABLE table, void * key);
static pANTLR3_HASH_ENTRY   antlr3HashRemove    (pANTLR3_HASH_TABLE table, void * key);
static ANTLR3_INT32			antlr3HashPut	(pANTLR3_HASH_TABLE table, void * key, void * element, void (ANTLR3_CDECL *freeptr)(void *));

// Integer based keys (Lists and so on)
//
static void					antlr3HashDeleteI   (pANTLR3_HASH_TABLE table, ANTLR3_INTKEY key);
static void *				antlr3HashGetI	(pANTLR3_HASH_TABLE table, ANTLR3_INTKEY key);
static pANTLR3_HASH_ENTRY   antlr3HashRemoveI   (pANTLR3_HASH_TABLE table, ANTLR3_INTKEY key);
static ANTLR3_INT32			antlr3HashPutI	(pANTLR3_HASH_TABLE table, ANTLR3_INTKEY key, void * element, void (ANTLR3_CDECL *freeptr)(void *));

static void					antlr3HashFree	(pANTLR3_HASH_TABLE table);
static ANTLR3_UINT32	    antlr3HashSize	(pANTLR3_HASH_TABLE table);

// -----------

// Interface functions for enumeration
//
static int	    antlr3EnumNext	    (pANTLR3_HASH_ENUM en, pANTLR3_HASH_KEY * key, void ** data);
static void	    antlr3EnumFree	    (pANTLR3_HASH_ENUM en);

// Interface functions for List
//
static void				antlr3ListFree	(pANTLR3_LIST list);
static void				antlr3ListDelete(pANTLR3_LIST list, ANTLR3_INTKEY key);
static void *			antlr3ListGet	(pANTLR3_LIST list, ANTLR3_INTKEY key);
static ANTLR3_INT32		antlr3ListPut	(pANTLR3_LIST list, ANTLR3_INTKEY key, void * element, void (ANTLR3_CDECL *freeptr)(void *));
static ANTLR3_INT32		antlr3ListAdd   (pANTLR3_LIST list, void * element, void (ANTLR3_CDECL *freeptr)(void *));
static void *			antlr3ListRemove(pANTLR3_LIST list, ANTLR3_INTKEY key);
static ANTLR3_UINT32	antlr3ListSize	(pANTLR3_LIST list);

// Interface functions for Stack
//
static void				antlr3StackFree	(pANTLR3_STACK  stack);
static void *			antlr3StackPop	(pANTLR3_STACK	stack);
static void *			antlr3StackGet	(pANTLR3_STACK	stack, ANTLR3_INTKEY key);
static ANTLR3_BOOLEAN	antlr3StackPush	(pANTLR3_STACK	stack, void * element, void (ANTLR3_CDECL *freeptr)(void *));
static ANTLR3_UINT32	antlr3StackSize	(pANTLR3_STACK	stack);
static void *			antlr3StackPeek	(pANTLR3_STACK	stack);

// Interface functions for vectors
//
static	void ANTLR3_CDECL	antlr3VectorFree	(pANTLR3_VECTOR vector);
static	void				antlr3VectorDel		(pANTLR3_VECTOR vector, ANTLR3_UINT32 entry);
static	void *				antlr3VectorGet		(pANTLR3_VECTOR vector, ANTLR3_UINT32 entry);
static	void *				antrl3VectorRemove	(pANTLR3_VECTOR vector, ANTLR3_UINT32 entry);
static	void				antlr3VectorClear	(pANTLR3_VECTOR vector);
static	ANTLR3_UINT32		antlr3VectorAdd		(pANTLR3_VECTOR vector, void * element, void (ANTLR3_CDECL *freeptr)(void *));
static	ANTLR3_UINT32		antlr3VectorSet		(pANTLR3_VECTOR vector, ANTLR3_UINT32 entry, void * element, void (ANTLR3_CDECL *freeptr)(void *), ANTLR3_BOOLEAN freeExisting);
static	ANTLR3_UINT32		antlr3VectorSize    (pANTLR3_VECTOR vector);

static  void                newPool             (pANTLR3_VECTOR_FACTORY factory);
static  void				closeVectorFactory  (pANTLR3_VECTOR_FACTORY factory);
static	pANTLR3_VECTOR		newVector			(pANTLR3_VECTOR_FACTORY factory);


// Interface functions for int TRIE
//
static	pANTLR3_TRIE_ENTRY	intTrieGet		(pANTLR3_INT_TRIE trie, ANTLR3_INTKEY key);
static	ANTLR3_BOOLEAN		intTrieDel		(pANTLR3_INT_TRIE trie, ANTLR3_INTKEY key);
static	ANTLR3_BOOLEAN		intTrieAdd		(pANTLR3_INT_TRIE trie, ANTLR3_INTKEY key, ANTLR3_UINT32 type, ANTLR3_INTKEY intType, void * data, void (ANTLR3_CDECL *freeptr)(void *));
static	void				intTrieFree		(pANTLR3_INT_TRIE trie);

// Local function to advance enumeration structure pointers
//
static void antlr3EnumNextEntry(pANTLR3_HASH_ENUM en);

pANTLR3_HASH_TABLE
antlr3HashTableNew(ANTLR3_UINT32 sizeHint)
{
	// All we have to do is create the hashtable tracking structure
	// and allocate memory for the requested number of buckets.
	//
	pANTLR3_HASH_TABLE	table;

	ANTLR3_UINT32	bucket;	// Used to traverse the buckets

	table   = ANTLR3_MALLOC(sizeof(ANTLR3_HASH_TABLE));

	// Error out if no memory left
	if	(table	== NULL)
	{
		return	NULL;
	}

	// Allocate memory for the buckets
	//
	table->buckets = (pANTLR3_HASH_BUCKET) ANTLR3_MALLOC((size_t) (sizeof(ANTLR3_HASH_BUCKET) * sizeHint)); 

	if	(table->buckets == NULL)
	{
		ANTLR3_FREE((void *)table);
		return	NULL;
	}

	// Modulo of the table, (bucket count).
	//
	table->modulo   = sizeHint;

	table->count    = 0;	    /* Nothing in there yet ( I hope)	*/

	/* Initialize the buckets to empty
	*/
	for	(bucket = 0; bucket < sizeHint; bucket++)
	{
		table->buckets[bucket].entries = NULL;
	}

	/* Exclude duplicate entries by default
	*/
	table->allowDups	= ANTLR3_FALSE;

    /* Assume that keys should by strduped before they are
     * entered in the table.
     */
    table->doStrdup     = ANTLR3_TRUE;

	/* Install the interface
	*/

	table->get		=  antlr3HashGet;
	table->put		=  antlr3HashPut;
	table->del		=  antlr3HashDelete;
	table->remove	=  antlr3HashRemove;

	table->getI		=  antlr3HashGetI;
	table->putI		=  antlr3HashPutI;
	table->delI		=  antlr3HashDeleteI;
	table->removeI	=  antlr3HashRemoveI;

	table->size		=  antlr3HashSize;
	table->free		=  antlr3HashFree;

	return  table;
}

static void
antlr3HashFree(pANTLR3_HASH_TABLE table)
{
    ANTLR3_UINT32	bucket;	/* Used to traverse the buckets	*/

    pANTLR3_HASH_BUCKET	thisBucket;
    pANTLR3_HASH_ENTRY	entry;
    pANTLR3_HASH_ENTRY	nextEntry;

    /* Free the table, all buckets and all entries, and all the
     * keys and data (if the table exists)
     */
    if	(table	!= NULL)
    {
	for	(bucket = 0; bucket < table->modulo; bucket++)
	{
	    thisBucket	= &(table->buckets[bucket]);

	    /* Allow sparse tables, though we don't create them as such at present
	     */
	    if	( thisBucket != NULL)
	    {
		entry	= thisBucket->entries;

		/* Search all entries in the bucket and free them up
		 */
		while	(entry != NULL)
		{
		    /* Save next entry - we do not want to access memory in entry after we
		     * have freed it.
		     */
		    nextEntry	= entry->nextEntry;

		    /* Free any data pointer, this only happens if the user supplied
		     * a pointer to a routine that knwos how to free the structure they
		     * added to the table.
		     */
		    if	(entry->free != NULL)
		    {
			entry->free(entry->data);
		    }

		    /* Free the key memory - we know that we allocated this
		     */
		    if	(entry->keybase.type == ANTLR3_HASH_TYPE_STR && entry->keybase.key.sKey != NULL)
		    {
			ANTLR3_FREE(entry->keybase.key.sKey);
		    }

		    /* Free this entry
		     */
		    ANTLR3_FREE(entry);
		    entry   = nextEntry;    /* Load next pointer to see if we shoud free it */
		}
		/* Invalidate the current pointer
		 */
		thisBucket->entries = NULL;
	    }
	}

	/* Now we can free the bucket memory
	 */
	ANTLR3_FREE(table->buckets);
    }

    /* Now we free teh memory for the table itself
     */
    ANTLR3_FREE(table);
}

/** return the current size of the hash table
 */
static ANTLR3_UINT32	antlr3HashSize	    (pANTLR3_HASH_TABLE table)
{
    return  table->count;
}

/** Remove a numeric keyed entry from a hash table if it exists,
 *  no error if it does not exist.
 */
static pANTLR3_HASH_ENTRY   antlr3HashRemoveI   (pANTLR3_HASH_TABLE table, ANTLR3_INTKEY key)
{
    ANTLR3_UINT32	    hash;
    pANTLR3_HASH_BUCKET	    bucket;
    pANTLR3_HASH_ENTRY	    entry;
    pANTLR3_HASH_ENTRY	    * nextPointer;

    /* First we need to know the hash of the provided key
     */
    hash    = (ANTLR3_UINT32)(key % (ANTLR3_INTKEY)(table->modulo));

    /* Knowing the hash, we can find the bucket
     */
    bucket  = table->buckets + hash;

    /* Now, we traverse the entries in the bucket until
     * we find the key or the end of the entries in the bucket. 
     * We track the element prior to the one we are examining
     * as we need to set its next pointer to the next pointer
     * of the entry we are deleting (if we find it).
     */
    entry	    =   bucket->entries;    /* Entry to examine					    */
    nextPointer	    = & bucket->entries;    /* Where to put the next pointer of the deleted entry   */

    while   (entry != NULL)
    {
	/* See if this is the entry we wish to delete
	 */
	if  (entry->keybase.key.iKey == key)
	{
	    /* It was the correct entry, so we set the next pointer
	     * of the previous entry to the next pointer of this
	     * located one, which takes it out of the chain.
	     */
	    (*nextPointer)		= entry->nextEntry;

	    table->count--;

	    return entry;
	}
	else
	{
	    /* We found an entry but it wasn't the one that was wanted, so
	     * move to the next one, if any.
	     */
	    nextPointer	= & (entry->nextEntry);	    /* Address of the next pointer in the current entry	    */
	    entry	= entry->nextEntry;	    /* Address of the next element in the bucket (if any)   */
	}
    }

    return NULL;  /* Not found */
}

/** Remove the element in the hash table for a particular
 *  key value, if it exists - no error if it does not.
 */
static pANTLR3_HASH_ENTRY
antlr3HashRemove(pANTLR3_HASH_TABLE table, void * key)
{
    ANTLR3_UINT32	    hash;
    pANTLR3_HASH_BUCKET	    bucket;
    pANTLR3_HASH_ENTRY	    entry;
    pANTLR3_HASH_ENTRY	    * nextPointer;

    /* First we need to know the hash of the provided key
     */
    hash    = antlr3Hash(key, (ANTLR3_UINT32)strlen((const char *)key));

    /* Knowing the hash, we can find the bucket
     */
    bucket  = table->buckets + (hash % table->modulo);

    /* Now, we traverse the entries in the bucket until
     * we find the key or the end of the entires in the bucket. 
     * We track the element prior to the one we are exmaining
     * as we need to set its next pointer to the next pointer
     * of the entry we are deleting (if we find it).
     */
    entry	    =   bucket->entries;    /* Entry to examine					    */
    nextPointer	    = & bucket->entries;    /* Where to put the next pointer of the deleted entry   */

    while   (entry != NULL)
    {
	/* See if this is the entry we wish to delete
	 */
	if  (strcmp((const char *)key, (const char *)entry->keybase.key.sKey) == 0)
	{
	    /* It was the correct entry, so we set the next pointer
	     * of the previous entry to the next pointer of this
	     * located one, which takes it out of the chain.
	     */
	    (*nextPointer)		= entry->nextEntry;

	    /* Release the key - if we allocated that
	     */
        if (table->doStrdup == ANTLR3_TRUE)
        {
            ANTLR3_FREE(entry->keybase.key.sKey);
        }
	    entry->keybase.key.sKey	= NULL;

	    table->count--;

	    return entry;
	}
	else
	{
	    /* We found an entry but it wasn't the one that was wanted, so
	     * move to the next one, if any.
	     */
	    nextPointer	= & (entry->nextEntry);	    /* Address of the next pointer in the current entry	    */
	    entry	= entry->nextEntry;	    /* Address of the next element in the bucket (if any)   */
	}
    }

    return NULL;  /* Not found */
}

/** Takes the element with the supplied key out of the list, and deletes the data
 *  calling the supplied free() routine if any. 
 */
static void
antlr3HashDeleteI    (pANTLR3_HASH_TABLE table, ANTLR3_INTKEY key)
{
    pANTLR3_HASH_ENTRY	entry;

    entry = antlr3HashRemoveI(table, key);
	
    /* Now we can free the elements and the entry in order
     */
    if	(entry != NULL && entry->free != NULL)
    {
	/* Call programmer supplied function to release this entry data
	 */
	entry->free(entry->data);
	entry->data = NULL;
    }
    /* Finally release the space for this entry block.
     */
    ANTLR3_FREE(entry);
}

/** Takes the element with the supplied key out of the list, and deletes the data
 *  calling the supplied free() routine if any. 
 */
static void
antlr3HashDelete    (pANTLR3_HASH_TABLE table, void * key)
{
    pANTLR3_HASH_ENTRY	entry;

    entry = antlr3HashRemove(table, key);
	
    /* Now we can free the elements and the entry in order
     */
    if	(entry != NULL && entry->free != NULL)
    {
	/* Call programmer supplied function to release this entry data
	 */
	entry->free(entry->data);
	entry->data = NULL;
    }
    /* Finally release the space for this entry block.
     */
    ANTLR3_FREE(entry);
}

/** Return the element pointer in the hash table for a particular
 *  key value, or NULL if it don't exist (or was itself NULL).
 */
static void *
antlr3HashGetI(pANTLR3_HASH_TABLE table, ANTLR3_INTKEY key)
{
    ANTLR3_UINT32	    hash;
    pANTLR3_HASH_BUCKET	    bucket;
    pANTLR3_HASH_ENTRY	    entry;

    /* First we need to know the hash of the provided key
     */
    hash    = (ANTLR3_UINT32)(key % (ANTLR3_INTKEY)(table->modulo));

    /* Knowing the hash, we can find the bucket
     */
    bucket  = table->buckets + hash;

    /* Now we can inspect the key at each entry in the bucket
     * and see if we have a match.
     */
    entry   = bucket->entries;

    while   (entry != NULL)
    {
	if  (entry->keybase.key.iKey == key)
	{
	    /* Match was found, return the data pointer for this entry
	     */
	    return  entry->data;
	}
	entry = entry->nextEntry;
    }

    /* If we got here, then we did not find the key
     */
    return  NULL;
}

/** Return the element pointer in the hash table for a particular
 *  key value, or NULL if it don't exist (or was itself NULL).
 */
static void *
antlr3HashGet(pANTLR3_HASH_TABLE table, void * key)
{
    ANTLR3_UINT32	    hash;
    pANTLR3_HASH_BUCKET	    bucket;
    pANTLR3_HASH_ENTRY	    entry;


    /* First we need to know the hash of the provided key
     */
    hash    = antlr3Hash(key, (ANTLR3_UINT32)strlen((const char *)key));

    /* Knowing the hash, we can find the bucket
     */
    bucket  = table->buckets + (hash % table->modulo);

    /* Now we can inspect the key at each entry in the bucket
     * and see if we have a match.
     */
    entry   = bucket->entries;

    while   (entry != NULL)
    {
	if  (strcmp((const char *)key, (const char *)entry->keybase.key.sKey) == 0)
	{
	    /* Match was found, return the data pointer for this entry
	     */
	    return  entry->data;
	}
	entry = entry->nextEntry;
    }

    /* If we got here, then we did not find the key
     */
    return  NULL;
}

/** Add the element pointer in to the table, based upon the 
 *  hash of the provided key.
 */
static	ANTLR3_INT32
antlr3HashPutI(pANTLR3_HASH_TABLE table, ANTLR3_INTKEY key, void * element, void (ANTLR3_CDECL *freeptr)(void *))
{
	ANTLR3_UINT32	    hash;
	pANTLR3_HASH_BUCKET	    bucket;
	pANTLR3_HASH_ENTRY	    entry;
	pANTLR3_HASH_ENTRY	    * newPointer;

	/* First we need to know the hash of the provided key
	*/
	hash    = (ANTLR3_UINT32)(key % (ANTLR3_INTKEY)(table->modulo));

	/* Knowing the hash, we can find the bucket
	*/
	bucket  = table->buckets + hash;

	/* Knowing the bucket, we can traverse the entries until we
	* we find a NULL pointer or we find that this is already 
	* in the table and duplicates were not allowed.
	*/
	newPointer	= &bucket->entries;

	while   (*newPointer !=  NULL)
	{
		/* The value at new pointer is pointing to an existing entry.
		* If duplicates are allowed then we don't care what it is, but
		* must reject this add if the key is the same as the one we are
		* supplied with.
		*/
		if  (table->allowDups == ANTLR3_FALSE)
		{
			if	((*newPointer)->keybase.key.iKey == key)
			{
				return	ANTLR3_ERR_HASHDUP;
			}
		}

		/* Point to the next entry pointer of the current entry we
		* are traversing, if it is NULL we will create our new
		* structure and point this to it.
		*/
		newPointer = &((*newPointer)->nextEntry);
	}

	/* newPointer is now pointing at the pointer where we need to
	* add our new entry, so let's crate the entry and add it in.
	*/
	entry   = (pANTLR3_HASH_ENTRY)ANTLR3_MALLOC((size_t)sizeof(ANTLR3_HASH_ENTRY));

	if	(entry == NULL)
	{
		return	ANTLR3_ERR_NOMEM;
	}

	entry->data			= element;		/* Install the data element supplied			*/
	entry->free			= freeptr;		/* Function that knows how to release the entry		*/
	entry->keybase.type		= ANTLR3_HASH_TYPE_INT;	/* Indicate the key type stored here for when we free	*/
	entry->keybase.key.iKey	= key;			/* Record the key value					*/
	entry->nextEntry		= NULL;			/* Ensure that the forward pointer ends the chain	*/

	*newPointer	= entry;    /* Install the next entry in this bucket	*/

	table->count++;

	return  ANTLR3_SUCCESS;
}


/** Add the element pointer in to the table, based upon the 
 *  hash of the provided key.
 */
static	ANTLR3_INT32
antlr3HashPut(pANTLR3_HASH_TABLE table, void * key, void * element, void (ANTLR3_CDECL *freeptr)(void *))
{
	ANTLR3_UINT32	    hash;
	pANTLR3_HASH_BUCKET	    bucket;
	pANTLR3_HASH_ENTRY	    entry;
	pANTLR3_HASH_ENTRY	    * newPointer;

	/* First we need to know the hash of the provided key
	*/
	hash    = antlr3Hash(key, (ANTLR3_UINT32)strlen((const char *)key));

	/* Knowing the hash, we can find the bucket
	*/
	bucket  = table->buckets + (hash % table->modulo);

	/* Knowign the bucket, we can traverse the entries until we
	* we find a NULL pointer ofr we find that this is already 
	* in the table and duplicates were not allowed.
	*/
	newPointer	= &bucket->entries;

	while   (*newPointer !=  NULL)
	{
		/* The value at new pointer is pointing to an existing entry.
		* If duplicates are allowed then we don't care what it is, but
		* must reject this add if the key is the same as the one we are
		* supplied with.
		*/
		if  (table->allowDups == ANTLR3_FALSE)
		{
			if	(strcmp((const char*) key, (const char *)(*newPointer)->keybase.key.sKey) == 0)
			{
				return	ANTLR3_ERR_HASHDUP;
			}
		}

		/* Point to the next entry pointer of the current entry we
		* are traversing, if it is NULL we will create our new
		* structure and point this to it.
		*/
		newPointer = &((*newPointer)->nextEntry);
	}

	/* newPointer is now poiting at the pointer where we need to
	* add our new entry, so let's crate the entry and add it in.
	*/
	entry   = (pANTLR3_HASH_ENTRY)ANTLR3_MALLOC((size_t)sizeof(ANTLR3_HASH_ENTRY));

	if	(entry == NULL)
	{
		return	ANTLR3_ERR_NOMEM;
	}

	entry->data			= element;					/* Install the data element supplied				*/
	entry->free			= freeptr;					/* Function that knows how to release the entry	    */
	entry->keybase.type	= ANTLR3_HASH_TYPE_STR;     /* Indicate the key type stored here for free()	    */
    if  (table->doStrdup == ANTLR3_TRUE)
    {
        entry->keybase.key.sKey	= ANTLR3_STRDUP(key);	/* Record the key value								*/
    }
    else
    {
        entry->keybase.key.sKey	= key;                  /* Record the key value								*/
    }
	entry->nextEntry		= NULL;					/* Ensure that the forward pointer ends the chain   */

	*newPointer	= entry;    /* Install the next entry in this bucket	*/

	table->count++;

	return  ANTLR3_SUCCESS;
}

/** \brief Creates an enumeration structure to traverse the hash table.
 *
 * \param table Table to enumerate
 * \return Pointer to enumeration structure.
 */
pANTLR3_HASH_ENUM
antlr3EnumNew	(pANTLR3_HASH_TABLE table)
{
    pANTLR3_HASH_ENUM	en;

    /* Allocate structure memory
     */
    en    = (pANTLR3_HASH_ENUM) ANTLR3_MALLOC((size_t)sizeof(ANTLR3_HASH_ENUM));

    /* Check that the allocation was good 
     */
    if	(en == NULL)
    {
	return	(pANTLR3_HASH_ENUM) ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM);
    }
    
    /* Initialize the start pointers
    */
    en->table	= table;
    en->bucket	= 0;				/* First bucket		    */
    en->entry	= en->table->buckets->entries;	/* First entry to return    */

    /* Special case in that the first bucket may not have anything in it
     * but the antlr3EnumNext() function expects that the en->entry is
     * set to the next valid pointer. Hence if it is not a valid element
     * pointer, attempt to find the next one that is, (table may be empty
     * of course.
     */
    if	(en->entry == NULL)
    {
	antlr3EnumNextEntry(en);
    }

    /* Install the interface
     */
    en->free	=  antlr3EnumFree;
    en->next	=  antlr3EnumNext;

    /* All is good
     */
    return  en;
}

/** \brief Return the next entry in the hashtable being traversed by the supplied
 *         enumeration.
 *
 * \param[in] en Pointer to the enumeration tracking structure
 * \param key	 Pointer to void pointer, where the key pointer is returned.
 * \param data	 Pointer to void pointer where the data pointer is returned.
 * \return 
 *	- ANTLR3_SUCCESS if there was a next key
 *	- ANTLR3_FAIL	 if there were no more keys
 *
 * \remark
 *  No checking of input structure is performed!
 */
static int
antlr3EnumNext	(pANTLR3_HASH_ENUM en, pANTLR3_HASH_KEY * key, void ** data)
{
    /* If the current entry is valid, then use it
     */
    if  (en->bucket >= en->table->modulo)
    {
        /* Already exhausted the table
         */
        return	ANTLR3_FAIL;
    }

    /* Pointers are already set to the current entry to return, or
     * we would not be at this point in the logic flow.
     */
    *key	= &(en->entry->keybase);
    *data	= en->entry->data;

    /* Return pointers are set up, so now we move the element
     * pointer to the next in the table (if any).
     */
    antlr3EnumNextEntry(en);

    return	ANTLR3_SUCCESS;
}

/** \brief Local function to advance the entry pointer of an enumeration 
 * structure to the next valid entry (if there is one).
 *
 * \param[in] enum Pointer to ANTLR3 enumeration structure returned by antlr3EnumNew()
 *
 * \remark
 *   - The function always leaves the pointers pointing at a valid entry if there
 *     is one, so if the entry pointer is NULL when this function exits, there were
 *     no more entries in the table.
 */
static void
antlr3EnumNextEntry(pANTLR3_HASH_ENUM en)
{
    pANTLR3_HASH_BUCKET	bucket;

    /* See if the current entry pointer is valid first of all
     */
    if	(en->entry != NULL)
    {
	/* Current entry was a valid point, see if there is another
	 * one in the chain.
	 */
	if  (en->entry->nextEntry != NULL)
	{
	    /* Next entry in the enumeration is just the next entry
	     * in the chain.
	     */
	    en->entry = en->entry->nextEntry;
	    return;
	}
    }

    /* There were no more entries in the current bucket, if there are
     * more buckets then chase them until we find an entry.
     */
    en->bucket++;

    while   (en->bucket < en->table->modulo)
    {
	/* There was one more bucket, see if it has any elements in it
	 */
	bucket	= en->table->buckets + en->bucket;

	if  (bucket->entries != NULL)
	{
	    /* There was an entry in this bucket, so we can use it
	     * for the next entry in the enumeration.
	     */
	    en->entry	= bucket->entries;
	    return;
	}

	/* There was nothing in the bucket we just examined, move to the
	 * next one.
	 */
	en->bucket++;
    }

    /* Here we have exhausted all buckets and the enumeration pointer will 
     * have its bucket count = table->modulo which signifies that we are done.
     */
}

/** \brief Frees up the memory structures that represent a hash table
 *  enumeration.
 * \param[in] enum Pointer to ANTLR3 enumeration structure returned by antlr3EnumNew()
 */
static void
antlr3EnumFree	(pANTLR3_HASH_ENUM en)
{
    /* Nothing to check, we just free it.
     */
    ANTLR3_FREE(en);
}

/** Given an input key of arbitrary length, return a hash value of
 *  it. This can then be used (with suitable modulo) to index other
 *  structures.
 */
ANTLR3_API ANTLR3_UINT32
antlr3Hash(void * key, ANTLR3_UINT32 keylen)
{
    /* Accumulate the hash value of the key
     */
    ANTLR3_UINT32   hash;
    pANTLR3_UINT8   keyPtr;
    ANTLR3_UINT32   i1;

    hash    = 0;
    keyPtr  = (pANTLR3_UINT8) key;

    /* Iterate the key and accumulate the hash
     */
    while(keylen > 0)
    {
	hash = (hash << 4) + (*(keyPtr++));

	if ((i1=hash&0xf0000000) != 0)
	{
		hash = hash ^ (i1 >> 24);
		hash = hash ^ i1;
	}
	keylen--;
    }

    return  hash;
}

ANTLR3_API  pANTLR3_LIST
antlr3ListNew	(ANTLR3_UINT32 sizeHint)
{
    pANTLR3_LIST    list;

    /* Allocate memory
     */
    list    = (pANTLR3_LIST)ANTLR3_MALLOC((size_t)sizeof(ANTLR3_LIST));

    if	(list == NULL)
    {
	return	(pANTLR3_LIST)ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM);
    }

    /* Now we need to add a new table
     */
    list->table	= antlr3HashTableNew(sizeHint);

    if	(list->table == (pANTLR3_HASH_TABLE)ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM))
    {
	return	(pANTLR3_LIST)ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM);
    }

    /* Allocation was good, install interface
     */
    list->free	    =  antlr3ListFree;
    list->del	    =  antlr3ListDelete;
    list->get	    =  antlr3ListGet;
    list->add	    =  antlr3ListAdd;
    list->remove    =  antlr3ListRemove;
    list->put	    =  antlr3ListPut;
    list->size	    =  antlr3ListSize;

    return  list;
}

static ANTLR3_UINT32	antlr3ListSize	    (pANTLR3_LIST list)
{
    return  list->table->size(list->table);
}

static void
antlr3ListFree	(pANTLR3_LIST list)
{
    /* Free the hashtable that stores the list
     */
    list->table->free(list->table);

    /* Free the allocation for the list itself
     */
    ANTLR3_FREE(list);
}

static void
antlr3ListDelete    (pANTLR3_LIST list, ANTLR3_INTKEY key)
{
    list->table->delI(list->table, key);
}

static void *
antlr3ListGet	    (pANTLR3_LIST list, ANTLR3_INTKEY key)
{
    return list->table->getI(list->table, key);
}

/** Add the supplied element to the list, at the next available key
 */
static ANTLR3_INT32	antlr3ListAdd   (pANTLR3_LIST list, void * element, void (ANTLR3_CDECL *freeptr)(void *))
{
    ANTLR3_INTKEY   key;

    key	    = list->table->size(list->table) + 1;
    return list->put(list, key, element, freeptr);
}

/** Remove from the list, but don't free the element, just send it back to the
 *  caller.
 */
static	void *
antlr3ListRemove	    (pANTLR3_LIST list, ANTLR3_INTKEY key)
{
    pANTLR3_HASH_ENTRY	    entry;

    entry = list->table->removeI(list->table, key);

    if	(entry != NULL)
    {
        return  entry->data;
    }
    else
    {
	return	NULL;
    }
}

static	ANTLR3_INT32
antlr3ListPut	    (pANTLR3_LIST list, ANTLR3_INTKEY key, void * element, void (ANTLR3_CDECL *freeptr)(void *))
{
    return  list->table->putI(list->table, key, element, freeptr);
}

ANTLR3_API  pANTLR3_STACK
antlr3StackNew	(ANTLR3_UINT32 sizeHint)
{
    pANTLR3_STACK   stack;

    /* Allocate memory
     */
    stack    = (pANTLR3_STACK)ANTLR3_MALLOC((size_t)sizeof(ANTLR3_STACK));

    if	(stack == NULL)
    {
	return	(pANTLR3_STACK)ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM);
    }

    /* Now we need to add a new table
     */
    stack->vector   = antlr3VectorNew(sizeHint);
    stack->top	    = NULL;

    if	(stack->vector == (pANTLR3_VECTOR)ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM))
    {
	return	(pANTLR3_STACK)ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM);
    }

    /* Looks good, now add the interface
     */
    stack->get	=  antlr3StackGet;
    stack->free	=  antlr3StackFree;
    stack->pop	=  antlr3StackPop;
    stack->push	=  antlr3StackPush;
    stack->size	=  antlr3StackSize;
    stack->peek	=  antlr3StackPeek;

    return  stack;
}

static ANTLR3_UINT32	antlr3StackSize	    (pANTLR3_STACK stack)
{
    return  stack->vector->count;
}


static void
antlr3StackFree	(pANTLR3_STACK  stack)
{
    /* Free the list that supports the stack
     */
    stack->vector->free(stack->vector);
    stack->vector   = NULL;
    stack->top	    = NULL;

    ANTLR3_FREE(stack);
}

static void *
antlr3StackPop	(pANTLR3_STACK	stack)
{
    // Delete the element that is currently at the top of the stack
    //
    stack->vector->del(stack->vector, stack->vector->count - 1);

    // And get the element that is the now the top of the stack (if anything)
    // NOTE! This is not quite like a 'real' stack, which would normally return you
    // the current top of the stack, then remove it from the stack.
    // TODO: Review this, it is correct for follow sets which is what this was done for
    //       but is not as obvious when using it as a 'real'stack.
    //
    stack->top = stack->vector->get(stack->vector, stack->vector->count - 1);
    return stack->top;
}

static void *
antlr3StackGet	(pANTLR3_STACK stack, ANTLR3_INTKEY key)
{
    return  stack->vector->get(stack->vector, (ANTLR3_UINT32)key);
}

static void *
antlr3StackPeek	(pANTLR3_STACK	stack)
{
    return  stack->top;
}

static ANTLR3_BOOLEAN 
antlr3StackPush	(pANTLR3_STACK stack, void * element, void (ANTLR3_CDECL *freeptr)(void *))
{
    stack->top	= element;
    return (ANTLR3_BOOLEAN)(stack->vector->add(stack->vector, element, freeptr));
}

ANTLR3_API  pANTLR3_VECTOR
antlr3VectorNew	(ANTLR3_UINT32 sizeHint)
{
	pANTLR3_VECTOR  vector;


	// Allocate memory for the vector structure itself
	//
	vector  = (pANTLR3_VECTOR) ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_VECTOR)));

	if	(vector == NULL)
	{
		return	(pANTLR3_VECTOR)ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM);
	}

	// Now fill in the defaults
	//
    antlr3SetVectorApi(vector, sizeHint);

	// And everything is hunky dory
	//
	return  vector;
}

ANTLR3_API void
antlr3SetVectorApi  (pANTLR3_VECTOR vector, ANTLR3_UINT32 sizeHint)
{
	ANTLR3_UINT32   initialSize;

  	// Allow vectors to be guessed by ourselves, so input size can be zero
	//
	if	(sizeHint > ANTLR3_VECTOR_INTERNAL_SIZE)
	{
		initialSize = sizeHint;
	}
	else
	{
		initialSize = ANTLR3_VECTOR_INTERNAL_SIZE;
	}

    if  (sizeHint > ANTLR3_VECTOR_INTERNAL_SIZE)
    {
        vector->elements	= (pANTLR3_VECTOR_ELEMENT)ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_VECTOR_ELEMENT) * initialSize));
    }
    else
    {
        vector->elements    = vector->internal;
    }

	if	(vector->elements == NULL)
	{
		ANTLR3_FREE(vector);
		return;
	}

	// Memory allocated successfully
	//
	vector->count			= 0;			// No entries yet of course
	vector->elementsSize    = initialSize;  // Available entries

	// Now we can install the API
	//
	vector->add	    = antlr3VectorAdd;
	vector->del	    = antlr3VectorDel;
	vector->get	    = antlr3VectorGet;
	vector->free    = antlr3VectorFree;
	vector->set	    = antlr3VectorSet;
	vector->remove  = antrl3VectorRemove;
	vector->clear	= antlr3VectorClear;
	vector->size    = antlr3VectorSize;

	// Assume that this is not a factory made vector
	//
	vector->factoryMade	= ANTLR3_FALSE;
}
// Clear the entries in a vector.
// Clearing the vector leaves its capacity the same but
// it walks the entries first to see if any of them
// have a free routine that must be called.
//
static	void				
antlr3VectorClear	(pANTLR3_VECTOR vector)
{
	ANTLR3_UINT32   entry;

	// We must traverse every entry in the vector and if it has
	// a pointer to a free function then we call it with the
	// the entry pointer
	//
	for	(entry = 0; entry < vector->count; entry++)
	{
		if  (vector->elements[entry].freeptr != NULL)
		{
			vector->elements[entry].freeptr(vector->elements[entry].element);
		}
		vector->elements[entry].freeptr    = NULL;
		vector->elements[entry].element    = NULL;
	}

	// Having called any free pointers, we just reset the entry count
	// back to zero.
	//
	vector->count	= 0;
}

static	
void	ANTLR3_CDECL	antlr3VectorFree    (pANTLR3_VECTOR vector)
{
	ANTLR3_UINT32   entry;

	// We must traverse every entry in the vector and if it has
	// a pointer to a free function then we call it with the
	// the entry pointer
	//
	for	(entry = 0; entry < vector->count; entry++)
	{
		if  (vector->elements[entry].freeptr != NULL)
		{
			vector->elements[entry].freeptr(vector->elements[entry].element);
		}
		vector->elements[entry].freeptr    = NULL;
		vector->elements[entry].element    = NULL;
	}

	if	(vector->factoryMade == ANTLR3_FALSE)
	{
		// The entries are freed, so free the element allocation
		//
        if  (vector->elementsSize > ANTLR3_VECTOR_INTERNAL_SIZE)
        {
            ANTLR3_FREE(vector->elements);
        }
		vector->elements = NULL;

		// Finally, free the allocation for the vector itself
		//
		ANTLR3_FREE(vector);
	}
}

static	void		antlr3VectorDel	    (pANTLR3_VECTOR vector, ANTLR3_UINT32 entry)
{
	// Check this is a valid request first
	//
	if	(entry >= vector->count)
	{
		return;
	}

	// Valid request, check for free pointer and call it if present
	//
	if	(vector->elements[entry].freeptr != NULL)
	{
		vector->elements[entry].freeptr(vector->elements[entry].element);
		vector->elements[entry].freeptr    = NULL;
	}

	if	(entry == vector->count - 1)
	{
		// Ensure the pointer is never reused by accident, but otherwise just 
		// decrement the pointer.
		//
		vector->elements[entry].element    = NULL;
	}
	else
	{
		// Need to shuffle trailing pointers back over the deleted entry
		//
		ANTLR3_MEMMOVE(vector->elements + entry, vector->elements + entry + 1, sizeof(ANTLR3_VECTOR_ELEMENT) * (vector->count - entry - 1));
	}

	// One less entry in the vector now
	//
	vector->count--;
}

static	void *		antlr3VectorGet     (pANTLR3_VECTOR vector, ANTLR3_UINT32 entry)
{
	// Ensure this is a valid request
	//
	if	(entry < vector->count)
	{
		return	vector->elements[entry].element;
	}
	else
	{
		// I know nothing, Mr. Fawlty!
		//
		return	NULL;
	}
}

/// Remove the entry from the vector, but do not free any entry, even if it has
/// a free pointer.
///
static	void *		antrl3VectorRemove  (pANTLR3_VECTOR vector, ANTLR3_UINT32 entry)
{
	void * element;

	// Check this is a valid request first 
	//
	if	(entry >= vector->count)
	{
		return NULL;
	}

	// Valid request, return the sorted pointer
	//

	element				    = vector->elements[entry].element;

	if	(entry == vector->count - 1)
	{
		// Ensure the pointer is never reused by accident, but otherwise just 
		// decrement the pointer.
		///
		vector->elements[entry].element    = NULL;
		vector->elements[entry].freeptr    = NULL;
	}
	else
	{
		// Need to shuffle trailing pointers back over the deleted entry
		//
		ANTLR3_MEMMOVE(vector->elements + entry, vector->elements + entry + 1, sizeof(ANTLR3_VECTOR_ELEMENT) * (vector->count - entry - 1));
	}

	// One less entry in the vector now
	//
	vector->count--;

	return  element;
}

static  void
antlr3VectorResize  (pANTLR3_VECTOR vector, ANTLR3_UINT32 hint)
{
	ANTLR3_UINT32	newSize;

	// Need to resize the element pointers. We double the allocation
	// we already have unless asked for a specific increase.
    //
    if (hint == 0 || hint < vector->elementsSize)
    {
        newSize = vector->elementsSize * 2;
    }
    else
    {
        newSize = hint * 2;
    }

    // Now we know how many we need, so we see if we have just expanded
    // past the built in vector elements or were already past that
    //
    if  (vector->elementsSize > ANTLR3_VECTOR_INTERNAL_SIZE)
    {
        // We were already larger than the internal size, so we just
        // use realloc so that the pointers are copied for us
        //
        vector->elements	= (pANTLR3_VECTOR_ELEMENT)ANTLR3_REALLOC(vector->elements, (sizeof(ANTLR3_VECTOR_ELEMENT)* newSize));
    }
    else
    {
        // The current size was less than or equal to the internal array size and as we always start
        // with a size that is at least the maximum internal size, then we must need to allocate new memory
        // for external pointers. We don't want to take the time to calculate if a requested element
        // is part of the internal or external entries, so we copy the internal ones to the new space
        //
        vector->elements	= (pANTLR3_VECTOR_ELEMENT)ANTLR3_MALLOC((sizeof(ANTLR3_VECTOR_ELEMENT)* newSize));
        ANTLR3_MEMCPY(vector->elements, vector->internal, ANTLR3_VECTOR_INTERNAL_SIZE * sizeof(ANTLR3_VECTOR_ELEMENT));
    }

	vector->elementsSize	= newSize;
}

/// Add the supplied pointer and freeing function pointer to the list,
/// expanding the vector if needed.
///
static	ANTLR3_UINT32    antlr3VectorAdd	    (pANTLR3_VECTOR vector, void * element, void (ANTLR3_CDECL *freeptr)(void *))
{
	// Do we need to resize the vector table?
	//
	if	(vector->count == vector->elementsSize)
	{
		antlr3VectorResize(vector, 0);	    // Give no hint, we let it add 1024 or double it
	}

	// Insert the new entry
	//
	vector->elements[vector->count].element	= element;
	vector->elements[vector->count].freeptr	= freeptr;

	vector->count++;	    // One more element counted

	return  (ANTLR3_UINT32)(vector->count);

}

/// Replace the element at the specified entry point with the supplied
/// entry.
///
static	ANTLR3_UINT32    
antlr3VectorSet	    (pANTLR3_VECTOR vector, ANTLR3_UINT32 entry, void * element, void (ANTLR3_CDECL *freeptr)(void *), ANTLR3_BOOLEAN freeExisting)
{

	// If the vector is currently not big enough, then we expand it
	//
	if (entry >= vector->elementsSize)
	{
		antlr3VectorResize(vector, entry);	// We will get at least this many 
	}

	// Valid request, replace the current one, freeing any prior entry if told to
	//
	if	(		entry < vector->count						// If actually replacing an element
			&&	freeExisting								// And told to free any existing element
			&&	vector->elements[entry].freeptr != NULL		// And the existing element has a free pointer
		)
	{
		vector->elements[entry].freeptr(vector->elements[entry].element);
	}

	// Install the new pointers
	//
	vector->elements[entry].freeptr	= freeptr;
	vector->elements[entry].element	= element;

	if (entry >= vector->count)
	{
		vector->count = entry + 1;
	}
	return  (ANTLR3_UINT32)(entry);	    // Indicates the replacement was successful

}

static	ANTLR3_UINT32   antlr3VectorSize    (pANTLR3_VECTOR vector)
{
    return  vector->count;
}

/// Vector factory creation
///
ANTLR3_API pANTLR3_VECTOR_FACTORY
antlr3VectorFactoryNew	    (ANTLR3_UINT32 sizeHint)
{
	pANTLR3_VECTOR_FACTORY  factory;

	// Allocate memory for the factory
	//
	factory = (pANTLR3_VECTOR_FACTORY)ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_VECTOR_FACTORY)));

	if	(factory == NULL)
	{
		return	NULL;
	}

	// Factory memory is good, so create a new vector pool
	//
    factory->pools      = NULL;
    factory->thisPool   = -1;

    newPool(factory);

    // Initialize the API, ignore the hint as this algorithm does
    // a better job really.
    //
    antlr3SetVectorApi(&(factory->unTruc), ANTLR3_VECTOR_INTERNAL_SIZE);
    
    factory->unTruc.factoryMade = ANTLR3_TRUE;

	// Install the factory API
	//
	factory->close		= closeVectorFactory;
	factory->newVector	= newVector;

	return  factory;
}

static void
newPool(pANTLR3_VECTOR_FACTORY factory)
{
    /* Increment factory count
     */
    factory->thisPool++;

    /* Ensure we have enough pointers allocated
     */
    factory->pools = (pANTLR3_VECTOR *)
		     ANTLR3_REALLOC(	(void *)factory->pools,	    /* Current pools pointer (starts at NULL)	*/
					(ANTLR3_UINT32)((factory->thisPool + 1) * sizeof(pANTLR3_VECTOR *))	/* Memory for new pool pointers */
					);

    /* Allocate a new pool for the factory
     */
    factory->pools[factory->thisPool]	=
			    (pANTLR3_VECTOR)
				ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_VECTOR) * ANTLR3_FACTORY_VPOOL_SIZE));


    /* Reset the counters
     */
    factory->nextVector	= 0;

    /* Done
     */
    return;
}

static  void		
closeVectorFactory  (pANTLR3_VECTOR_FACTORY factory)
{
    pANTLR3_VECTOR      pool;
    ANTLR3_INT32        poolCount;
    ANTLR3_UINT32       limit;
    ANTLR3_UINT32       vector;
    pANTLR3_VECTOR      check;

    /* We iterate the vector pools one at a time
     */
    for (poolCount = 0; poolCount <= factory->thisPool; poolCount++)
    {
        /* Pointer to current pool
         */
        pool = factory->pools[poolCount];

        /* Work out how many tokens we need to check in this pool.
         */
        limit = (poolCount == factory->thisPool ? factory->nextVector : ANTLR3_FACTORY_VPOOL_SIZE);

        /* Marginal condition, we might be at the start of a brand new pool
         * where the nextToken is 0 and nothing has been allocated.
         */
        if (limit > 0)
        {
            /* We have some vectors allocated from this pool
             */
            for (vector = 0; vector < limit; vector++)
            {
                /* Next one in the chain
                 */
                check = pool + vector;

                // Call the free function on each of the vectors in the pool,
                // which in turn will cause any elements it holds that also have a free
                // pointer to be freed. However, because any vector may be in any other
                // vector, we don't free the element allocations yet. We do that in a
                // a specific pass, coming up next. The vector free function knows that
                // this is a factory allocated pool vector and so it won't free things it
                // should not.
                //
                check->free(check);
            }
        }
    }

    /* We iterate the vector pools one at a time once again, but this time
     * we are going to free up any allocated element pointers. Note that we are doing this
     * so that we do not try to release vectors twice. When building ASTs we just copy
     * the vectors all over the place and they may be embedded in this vector pool
     * numerous times.
     */
    for (poolCount = 0; poolCount <= factory->thisPool; poolCount++)
    {
        /* Pointer to current pool
         */
        pool = factory->pools[poolCount];

        /* Work out how many tokens we need to check in this pool.
         */
        limit = (poolCount == factory->thisPool ? factory->nextVector : ANTLR3_FACTORY_VPOOL_SIZE);

        /* Marginal condition, we might be at the start of a brand new pool
         * where the nextToken is 0 and nothing has been allocated.
         */
        if (limit > 0)
        {
            /* We have some vectors allocated from this pool
             */
            for (vector = 0; vector < limit; vector++)
            {
                /* Next one in the chain
                 */
                check = pool + vector;

                // Anything in here should be factory made, but we do this just
                // to triple check. We just free up the elements if they were
                // allocated beyond the internal size.
                //
                if (check->factoryMade == ANTLR3_TRUE && check->elementsSize > ANTLR3_VECTOR_INTERNAL_SIZE)
                {
                    ANTLR3_FREE(check->elements);
                    check->elements = NULL;
                }
            }
        }

        // We can now free this pool allocation as we have called free on every element in every vector
        // and freed any memory for pointers the grew beyond the internal size limit.
        //
        ANTLR3_FREE(factory->pools[poolCount]);
        factory->pools[poolCount] = NULL;
    }

    /* All the pools are deallocated we can free the pointers to the pools
     * now.
     */
    ANTLR3_FREE(factory->pools);

    /* Finally, we can free the space for the factory itself
     */
    ANTLR3_FREE(factory);

}

static pANTLR3_VECTOR
newVector(pANTLR3_VECTOR_FACTORY factory)
{
    pANTLR3_VECTOR vector;

    // See if we need a new vector pool before allocating a new
    // one
    //
    if (factory->nextVector >= ANTLR3_FACTORY_VPOOL_SIZE)
    {
        // We ran out of vectors in the current pool, so we need a new pool
        //
        newPool(factory);
    }

    // Assuming everything went well (we are trying for performance here so doing minimal
    // error checking. Then we can work out what the pointer is to the next vector.
    //
    vector = factory->pools[factory->thisPool] + factory->nextVector;
    factory->nextVector++;

    // We have our token pointer now, so we can initialize it to the predefined model.
    //
    antlr3SetVectorApi(vector, ANTLR3_VECTOR_INTERNAL_SIZE);
    vector->factoryMade = ANTLR3_TRUE;

    // We know that the pool vectors are created at the default size, which means they
    // will start off using their internal entry pointers. We must intialize our pool vector
    // to point to its own internal entry table and not the pre-made one.
    //
    vector->elements = vector->internal;

    // And we are done
    //
    return vector;
}

/** Array of left most significant bit positions for an 8 bit
 *  element provides an efficient way to find the highest bit
 *  that is set in an n byte value (n>0). Assuming the values will all hit the data cache,
 *  coding without conditional elements should allow branch
 *  prediction to work well and of course a parallel instruction cache
 *  will whip through this. Otherwise we must loop shifting a one
 *  bit and masking. The values we tend to be placing in out integer
 *  patricia trie are usually a lot lower than the 64 bits we
 *  allow for the key allows. Hence there is a lot of redundant looping and
 *  shifting in a while loop. Whereas, the lookup table is just
 *  a few ands and indirect lookups, while testing for 0. This
 *  is likely to be done in parallel on many processors available
 *  when I wrote this. If this code survives as long as yacc, then
 *  I may already be dead by the time you read this and maybe there is
 *  a single machine instruction to perform the operation. What
 *  else are you going to do with all those transistors? Jim 2007
 *
 * The table is probably obvious but it is just the number 0..7
 * of the MSB in each integer value 0..256
 */
static ANTLR3_UINT8 bitIndex[256] = 
{ 
    0,													// 0 - Just for padding
    0,													// 1
    1, 1,												// 2..3
    2, 2, 2, 2,											// 4..7
    3, 3, 3, 3, 3, 3, 3, 3,								// 8+
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,	    // 16+
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,	    // 32+
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,	    
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,	    // 64+
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,	    // 128+
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
};

/** Rather than use the bit index of a trie node to shift
 *  0x01 left that many times, then & with the result, it is
 *  faster to use the bit index as an index into this table
 *  which holds precomputed masks for any of the 64 bits
 *  we need to mask off singly. The data values will stay in
 *  cache while ever a trie is in heavy use, such as in
 *  memoization. It is also pretty enough to be ASCII art.
 */
static ANTLR3_UINT64 bitMask[64] = 
{
    0x0000000000000001ULL, 0x0000000000000002ULL, 0x0000000000000004ULL, 0x0000000000000008ULL,
    0x0000000000000010ULL, 0x0000000000000020ULL, 0x0000000000000040ULL, 0x0000000000000080ULL,
    0x0000000000000100ULL, 0x0000000000000200ULL, 0x0000000000000400ULL, 0x0000000000000800ULL,
    0x0000000000001000ULL, 0x0000000000002000ULL, 0x0000000000004000ULL, 0x0000000000008000ULL,
    0x0000000000010000ULL, 0x0000000000020000ULL, 0x0000000000040000ULL, 0x0000000000080000ULL,
    0x0000000000100000ULL, 0x0000000000200000ULL, 0x0000000000400000ULL, 0x0000000000800000ULL,
    0x0000000001000000ULL, 0x0000000002000000ULL, 0x0000000004000000ULL, 0x0000000008000000ULL,
    0x0000000010000000ULL, 0x0000000020000000ULL, 0x0000000040000000ULL, 0x0000000080000000ULL,
    0x0000000100000000ULL, 0x0000000200000000ULL, 0x0000000400000000ULL, 0x0000000800000000ULL,
    0x0000001000000000ULL, 0x0000002000000000ULL, 0x0000004000000000ULL, 0x0000008000000000ULL,
    0x0000010000000000ULL, 0x0000020000000000ULL, 0x0000040000000000ULL, 0x0000080000000000ULL,
    0x0000100000000000ULL, 0x0000200000000000ULL, 0x0000400000000000ULL, 0x0000800000000000ULL,
    0x0001000000000000ULL, 0x0002000000000000ULL, 0x0004000000000000ULL, 0x0008000000000000ULL,
    0x0010000000000000ULL, 0x0020000000000000ULL, 0x0040000000000000ULL, 0x0080000000000000ULL,
    0x0100000000000000ULL, 0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL,
    0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL, 0x8000000000000000ULL
};

/* INT TRIE Implementation of depth 64 bits, being the number of bits
 * in a 64 bit integer. 
 */

pANTLR3_INT_TRIE
antlr3IntTrieNew(ANTLR3_UINT32 depth)
{
	pANTLR3_INT_TRIE	trie;

	trie    = (pANTLR3_INT_TRIE) ANTLR3_CALLOC(1, sizeof(ANTLR3_INT_TRIE));	/* Base memory required	*/

	if (trie == NULL)
	{
		return	(pANTLR3_INT_TRIE) ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM);
	}

	/* Now we need to allocate the root node. This makes it easier
	 * to use the tree as we don't have to do anything special 
	 * for the root node.
	 */
	trie->root	= (pANTLR3_INT_TRIE_NODE) ANTLR3_CALLOC(1, sizeof(ANTLR3_INT_TRIE));

	if (trie->root == NULL)
	{
		ANTLR3_FREE(trie);
		return	(pANTLR3_INT_TRIE) ANTLR3_FUNC_PTR(ANTLR3_ERR_NOMEM);
	}

	trie->add	= intTrieAdd;
	trie->del	= intTrieDel;
	trie->free	= intTrieFree;
	trie->get	= intTrieGet;

	/* Now we seed the root node with the index being the
	 * highest left most bit we want to test, which limits the
	 * keys in the trie. This is the trie 'depth'. The limit for
	 * this implementation is 63 (bits 0..63).
	 */
	trie->root->bitNum = depth;

	/* And as we have nothing in here yet, we set both child pointers
	 * of the root node to point back to itself.
	 */
	trie->root->leftN	= trie->root;
	trie->root->rightN	= trie->root;
	trie->count			= 0;

	/* Finally, note that the key for this root node is 0 because
	 * we use calloc() to initialise it.
	 */

	return trie;
}

/** Search the int Trie and return a pointer to the first bucket indexed
 *  by the key if it is contained in the trie, otherwise NULL.
 */
static	pANTLR3_TRIE_ENTRY   
intTrieGet	(pANTLR3_INT_TRIE trie, ANTLR3_INTKEY key)
{
	pANTLR3_INT_TRIE_NODE    thisNode; 
	pANTLR3_INT_TRIE_NODE    nextNode; 

	if (trie->count == 0)
	{
		return NULL;	    /* Nothing in this trie yet	*/
	}
	/* Starting at the root node in the trie, compare the bit index
	 * of the current node with its next child node (starts left from root).
	 * When the bit index of the child node is greater than the bit index of the current node
	 * then by definition (as the bit index decreases as we descent the trie)
	 * we have reached a 'backward' pointer. A backward pointer means we
	 * have reached the only node that can be reached by the bits given us so far
	 * and it must either be the key we are looking for, or if not then it
	 * means the entry was not in the trie, and we return NULL. A backward pointer
	 * points back in to the tree structure rather than down (deeper) within the
	 * tree branches.
	 */
	thisNode	= trie->root;		/* Start at the root node		*/
	nextNode	= thisNode->leftN;	/* Examine the left node from the root	*/

	/* While we are descending the tree nodes...
	 */
	while (thisNode->bitNum > nextNode->bitNum)
	{
		/* Next node now becomes the new 'current' node
		 */
		thisNode    = nextNode;

		/* We now test the bit indicated by the bitmap in the next node
		 * in the key we are searching for. The new next node is the
		 * right node if that bit is set and the left node it is not.
		 */
		if (key & bitMask[nextNode->bitNum])
		{
			nextNode = nextNode->rightN;	/* 1 is right	*/
		}
		else
		{
			nextNode = nextNode->leftN;		/* 0 is left	*/
		}
	}

	/* Here we have reached a node where the bitMap index is lower than
	 * its parent. This means it is pointing backward in the tree and
	 * must therefore be a terminal node, being the only point than can
	 * be reached with the bits seen so far. It is either the actual key
	 * we wanted, or if that key is not in the trie it is another key
	 * that is currently the only one that can be reached by those bits.
	 * That situation would obviously change if the key was to be added
	 * to the trie.
	 *
	 * Hence it only remains to test whether this is actually the key or not.
	 */
	if (nextNode->key == key)
	{
		/* This was the key, so return the entry pointer
		 */
		return	nextNode->buckets;
	}
	else
	{
		return	NULL;	/* That key is not in the trie (note that we set the pointer to -1 if no payload) */
	}
}


static	ANTLR3_BOOLEAN		
intTrieDel	(pANTLR3_INT_TRIE trie, ANTLR3_INTKEY key)
{
    pANTLR3_INT_TRIE_NODE   p;

    p=trie->root;
    key = key;

    return ANTLR3_FALSE;
}

/** Add an entry into the INT trie.
 *  Basically we descend the trie as we do when searching it, which will
 *  locate the only node in the trie that can be reached by the bit pattern of the
 *  key. If the key is actually at that node, then if the trie accepts duplicates
 *  we add the supplied data in a new chained bucket to that data node. If it does
 *  not accept duplicates then we merely return FALSE in case the caller wants to know
 *  whether the key was already in the trie.
 *  If the node we locate is not the key we are looking to add, then we insert a new node
 *  into the trie with a bit index of the leftmost differing bit and the left or right 
 *  node pointing to itself or the data node we are inserting 'before'. 
 */
static	ANTLR3_BOOLEAN		
intTrieAdd	(pANTLR3_INT_TRIE trie, ANTLR3_INTKEY key, ANTLR3_UINT32 type, ANTLR3_INTKEY intVal, void * data, void (ANTLR3_CDECL *freeptr)(void *))
{
	pANTLR3_INT_TRIE_NODE   thisNode;
	pANTLR3_INT_TRIE_NODE   nextNode;
	pANTLR3_INT_TRIE_NODE   entNode;
	ANTLR3_UINT32			depth;
	pANTLR3_TRIE_ENTRY	    newEnt;
	pANTLR3_TRIE_ENTRY	    nextEnt;
	ANTLR3_INTKEY		    xorKey;

	/* Cache the bit depth of this trie, which is always the highest index, 
	 * which is in the root node
	 */
	depth   = trie->root->bitNum;

	thisNode	= trie->root;		/* Start with the root node	    */
	nextNode	= trie->root->leftN;	/* And assume we start to the left  */

	/* Now find the only node that can be currently reached by the bits in the
	 * key we are being asked to insert.
	 */
	while (thisNode->bitNum  > nextNode->bitNum)
	{
		/* Still descending the structure, next node becomes current.
		 */
		thisNode = nextNode;

		if (key & bitMask[nextNode->bitNum])
		{
			/* Bit at the required index was 1, so travers the right node from here
			 */
			nextNode = nextNode->rightN;
		}
		else
		{
			/* Bit at the required index was 0, so we traverse to the left
			 */
			nextNode = nextNode->leftN;
		}
	}
	/* Here we have located the only node that can be reached by the
	 * bits in the requested key. It could in fact be that key or the node
	 * we need to use to insert the new key.
	 */
	if (nextNode->key == key)
	{
		/* We have located an exact match, but we will only append to the bucket chain
		 * if this trie accepts duplicate keys.
		 */
		if (trie->allowDups ==ANTLR3_TRUE)
		{
			/* Yes, we are accepting duplicates
			 */
			newEnt = (pANTLR3_TRIE_ENTRY)ANTLR3_CALLOC(1, sizeof(ANTLR3_TRIE_ENTRY));

			if (newEnt == NULL)
			{
				/* Out of memory, all we can do is return the fact that the insert failed.
				 */
				return	ANTLR3_FALSE;
			}

			/* Otherwise insert this in the chain
			*/
			newEnt->type	= type;
			newEnt->freeptr	= freeptr;
			if (type == ANTLR3_HASH_TYPE_STR)
			{
				newEnt->data.ptr = data;
			}
			else
			{
				newEnt->data.intVal = intVal;
			}

			/* We want to be able to traverse the stored elements in the order that they were
			 * added as duplicate keys. We might need to revise this opinion if we end up having many duplicate keys
			 * as perhaps reverse order is just as good, so long as it is ordered.
			 */
			nextEnt = nextNode->buckets;
			while (nextEnt->next != NULL)
			{
				nextEnt = nextEnt->next;    
			}
			nextEnt->next = newEnt;

			trie->count++;
			return  ANTLR3_TRUE;
		}
		else
		{
			/* We found the key is already there and we are not allowed duplicates in this
			 * trie.
			 */
			return  ANTLR3_FALSE;
		}
	}

	/* Here we have discovered the only node that can be reached by the bits in the key
	 * but we have found that this node is not the key we need to insert. We must find the
	 * the leftmost bit by which the current key for that node and the new key we are going 
	 * to insert, differ. While this nested series of ifs may look a bit strange, experimentation
	 * showed that it allows a machine code path that works well with predicated execution
	 */
	xorKey = (key ^ nextNode->key);   /* Gives 1 bits only where they differ then we find the left most 1 bit*/

	/* Most common case is a 32 bit key really
	 */
#ifdef	ANTLR3_USE_64BIT
	if	(xorKey & 0xFFFFFFFF00000000)
	{
		if  (xorKey & 0xFFFF000000000000)
		{
			if	(xorKey & 0xFF00000000000000)
			{
				depth = 56 + bitIndex[((xorKey & 0xFF00000000000000)>>56)];
			}
			else
			{
				depth = 48 + bitIndex[((xorKey & 0x00FF000000000000)>>48)];
			}
		}
		else
		{
			if	(xorKey & 0x0000FF0000000000)
			{
				depth = 40 + bitIndex[((xorKey & 0x0000FF0000000000)>>40)];
			}
			else
			{
				depth = 32 + bitIndex[((xorKey & 0x000000FF00000000)>>32)];
			}
		}
	}
	else
#endif
	{
		if  (xorKey & 0x00000000FFFF0000)
		{
			if	(xorKey & 0x00000000FF000000)
			{
				depth = 24 + bitIndex[((xorKey & 0x00000000FF000000)>>24)];
			}
			else
			{
				depth = 16 + bitIndex[((xorKey & 0x0000000000FF0000)>>16)];
			}
		}
		else
		{
			if	(xorKey & 0x000000000000FF00)
			{
				depth = 8 + bitIndex[((xorKey & 0x0000000000000FF00)>>8)];
			}
			else
			{
				depth = bitIndex[xorKey & 0x00000000000000FF];
			}
		}
	}

    /* We have located the leftmost differing bit, indicated by the depth variable. So, we know what
     * bit index we are to insert the new entry at. There are two cases, being where the two keys
     * differ at a bit position that is not currently part of the bit testing, where they differ on a bit
     * that is currently being skipped in the indexed comparisons, and where they differ on a bit
     * that is merely lower down in the current bit search. If the bit index went bit 4, bit 2 and they differ
     * at bit 3, then we have the "skipped" bit case. But if that chain was Bit 4, Bit 2 and they differ at bit 1
     * then we have the easy bit <pun>.
     *
     * So, set up to descend the tree again, but this time looking for the insert point
     * according to whether we skip the bit that differs or not.
     */
    thisNode	= trie->root;
    entNode	= trie->root->leftN;

    /* Note the slight difference in the checks here to cover both cases
     */
    while (thisNode->bitNum > entNode->bitNum && entNode->bitNum > depth)
    {
	/* Still descending the structure, next node becomes current.
	 */
	thisNode = entNode;

	if (key & bitMask[entNode->bitNum])
	{
	    /* Bit at the required index was 1, so traverse the right node from here
	     */
	    entNode = entNode->rightN;
	}
	else
	{
	    /* Bit at the required index was 0, so we traverse to the left
	     */
	    entNode = entNode->leftN;
	}
    }

    /* We have located the correct insert point for this new key, so we need
     * to allocate our entry and insert it etc.
     */
    nextNode	= (pANTLR3_INT_TRIE_NODE)ANTLR3_CALLOC(1, sizeof(ANTLR3_INT_TRIE_NODE));
    if (nextNode == NULL)
    {
	/* All that work and no memory - bummer.
	 */
	return	ANTLR3_FALSE;
    }

    /* Build a new entry block for the new node
     */
    newEnt = (pANTLR3_TRIE_ENTRY)ANTLR3_CALLOC(1, sizeof(ANTLR3_TRIE_ENTRY));

    if (newEnt == NULL)
    {
	/* Out of memory, all we can do is return the fact that the insert failed.
	 */
	return	ANTLR3_FALSE;
    }

    /* Otherwise enter this in our new node
    */
    newEnt->type	= type;
    newEnt->freeptr	= freeptr;
    if (type == ANTLR3_HASH_TYPE_STR)
    {
	newEnt->data.ptr = data;
    }
    else
    {
	newEnt->data.intVal = intVal;
    }
    /* Install it
     */
    nextNode->buckets	= newEnt;
    nextNode->key	= key;
    nextNode->bitNum	= depth;

    /* Work out the right and left pointers for this new node, which involve
     * terminating with the current found node either right or left according
     * to whether the current index bit is 1 or 0
     */
    if (key & bitMask[depth])
    {
	nextNode->leftN	    = entNode;	    /* Terminates at previous position	*/
	nextNode->rightN    = nextNode;	    /* Terminates with itself		*/
    }
    else
    {
	nextNode->rightN   = entNode;	    /* Terminates at previous position	*/
	nextNode->leftN    = nextNode;	    /* Terminates with itself		*/		
    }

    /* Finally, we need to change the pointers at the node we located
     * for inserting. If the key bit at its index is set then the right
     * pointer for that node becomes the newly created node, otherwise the left 
     * pointer does.
     */
    if (key & bitMask[thisNode->bitNum] )
    {
	thisNode->rightN    = nextNode;
    }
    else
    {
	thisNode->leftN	    = nextNode;
    }

    /* Et voila
     */
    trie->count++;
    return  ANTLR3_TRUE;

}
/** Release memory allocated to this tree.
 *  Basic algorithm is that we do a depth first left descent and free
 *  up any nodes that are not backward pointers.
 */
static void
freeIntNode(pANTLR3_INT_TRIE_NODE node)
{
    pANTLR3_TRIE_ENTRY	thisEntry;
    pANTLR3_TRIE_ENTRY	nextEntry;

    /* If this node has a left pointer that is not a back pointer
     * then recursively call to free this
     */
    if (node->bitNum > node->leftN->bitNum)
    {
	/* We have a left node that needs descending, so do it.
	 */
	freeIntNode(node->leftN);
    }

    /* The left nodes from here should now be dealt with, so 
     * we need to descend any right nodes that are not back pointers
     */
    if (node->bitNum > node->rightN->bitNum)
    {
	/* There are some right nodes to descend and deal with.
	 */
	freeIntNode(node->rightN);
    }

    /* Now all the children are dealt with, we can destroy
     * this node too
     */
    thisEntry	= node->buckets;

    while (thisEntry != NULL)
    {
	nextEntry   = thisEntry->next;

	/* Do we need to call a custom free pointer for this string entry?
	 */
	if (thisEntry->type == ANTLR3_HASH_TYPE_STR && thisEntry->freeptr != NULL)
	{
	    thisEntry->freeptr(thisEntry->data.ptr);
	}

	/* Now free the data for this bucket entry
	 */
	ANTLR3_FREE(thisEntry);
	thisEntry = nextEntry;	    /* See if there are any more to free    */
    }

    /* The bucket entry is now gone, so we can free the memory for
     * the entry itself.
     */
    ANTLR3_FREE(node);

    /* And that should be it for everything under this node and itself
     */
}

/** Called to free all nodes and the structure itself.
 */
static	void			
intTrieFree	(pANTLR3_INT_TRIE trie)
{
    /* Descend from the root and free all the nodes
     */
    freeIntNode(trie->root);

    /* the nodes are all gone now, so we need only free the memory
     * for the structure itself
     */
    ANTLR3_FREE(trie);
}
