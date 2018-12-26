#ifndef	ANTLR3COLLECTIONS_H
#define	ANTLR3COLLECTIONS_H

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


#define	ANTLR3_HASH_TYPE_INT	0   /**< Indicates the hashed file has integer keys */
#define	ANTLR3_HASH_TYPE_STR	1   /**< Indicates the hashed file has numeric keys */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANTLR3_HASH_KEY_struct
{
	ANTLR3_UINT8	type;	/**< One of ##ANTLR3_HASH_TYPE_INT or ##ANTLR3_HASH_TYPE_STR	*/

	union
	{
		pANTLR3_UINT8   sKey;	/**< Used if type is ANTLR3_HASH_TYPE_STR			*/
		ANTLR3_INTKEY   iKey;	/**< used if type is ANTLR3_HASH_TYPE_INT			*/
	}
	key;

} ANTLR3_HASH_KEY, *pANTLR3_HASH_KEY;

/** Internal structure representing an element in a hash bucket.
 *  Stores the original key so that duplicate keys can be rejected
 *  if necessary, and contains function can be supported. If the hash key
 *  could be unique I would have invented the perfect compression algorithm ;-)
 */
typedef	struct	ANTLR3_HASH_ENTRY_struct
{
    /** Key that created this particular entry
     */
    ANTLR3_HASH_KEY 	keybase;

    /** Pointer to the data for this particular entry
     */
    void	    * data;

    /** Pointer to routine that knows how to release the memory
     *  structure pointed at by data. If this is NULL then we assume
     *  that the data pointer does not need to be freed when the entry
     *  is deleted from the table.
     */
    void	    (ANTLR3_CDECL *free)(void * data);

    /** Pointer to the next entry in this bucket if there
     *  is one. Sometimes different keys will hash to the same bucket (especially
     *  if the number of buckets is small). We could implement dual hashing algorithms
     *  to minimize this, but that seems over the top for what this is needed for.
     */
    struct	ANTLR3_HASH_ENTRY_struct * nextEntry;
}
    ANTLR3_HASH_ENTRY;

/** Internal structure of a hash table bucket, which tracks
 *  all keys that hash to the same bucket.
 */
typedef struct	ANTLR3_HASH_BUCKET_struct
{
    /** Pointer to the first entry in the bucket (if any, it
     *  may be NULL). Duplicate entries are chained from
     * here.
     */
    pANTLR3_HASH_ENTRY	entries;
    
}
    ANTLR3_HASH_BUCKET;

/** Structure that tracks a hash table
 */
typedef	struct	ANTLR3_HASH_TABLE_struct
{
    /** Indicates whether the table allows duplicate keys
     */
    int					allowDups;

    /** Number of buckets available in this table
     */
    ANTLR3_UINT32		modulo;

    /** Points to the memory where the array of buckets
     * starts.
     */
    pANTLR3_HASH_BUCKET	buckets;

    /** How many elements currently exist in the table.
     */
    ANTLR3_UINT32		count;

    /** Whether the hash table should strdup the keys it is given or not.
     */
    ANTLR3_BOOLEAN              doStrdup;

    /** Pointer to function to completely delete this table
     */
    void				(*free)	    (struct ANTLR3_HASH_TABLE_struct * table);
    
    /* String keyed hashtable functions */
    void				(*del)	    (struct ANTLR3_HASH_TABLE_struct * table, void * key);
    pANTLR3_HASH_ENTRY	(*remove)   (struct ANTLR3_HASH_TABLE_struct * table, void * key);
    void *				(*get)	    (struct ANTLR3_HASH_TABLE_struct * table, void * key);
    ANTLR3_INT32		(*put)	    (struct ANTLR3_HASH_TABLE_struct * table, void * key, void * element, void (ANTLR3_CDECL *freeptr)(void *));

    /* Integer based hash functions */
    void				(*delI)	    (struct ANTLR3_HASH_TABLE_struct * table, ANTLR3_INTKEY key);
    pANTLR3_HASH_ENTRY	(*removeI)  (struct ANTLR3_HASH_TABLE_struct * table, ANTLR3_INTKEY key);
    void *				(*getI)	    (struct ANTLR3_HASH_TABLE_struct * table, ANTLR3_INTKEY key);
    ANTLR3_INT32		(*putI)	    (struct ANTLR3_HASH_TABLE_struct * table, ANTLR3_INTKEY key, void * element, void (ANTLR3_CDECL *freeptr)(void *));

    ANTLR3_UINT32		(*size)	    (struct ANTLR3_HASH_TABLE_struct * table);
}
    ANTLR3_HASH_TABLE;


/** Internal structure representing an enumeration of a table.
 *  This is returned by antlr3Enumeration()
 *  Allows the programmer to traverse the table in hash order without 
 *  knowing what is in the actual table.
 *
 *  Note that it is up to the caller to ensure that the table
 *  structure does not change in the hash bucket that is currently being
 *  enumerated as this structure just tracks the next pointers in the
 *  bucket series.
 */
typedef struct	ANTLR3_HASH_ENUM_struct
{
    /* Pointer to the table we are enumerating
     */
    pANTLR3_HASH_TABLE	table;

    /* Bucket we are currently enumerating (if NULL then we are done)
     */
    ANTLR3_UINT32	bucket;

    /* Next entry to return, if NULL, then move to next bucket if any
     */
    pANTLR3_HASH_ENTRY	entry;

    /* Interface
     */
    int		(*next)	    (struct ANTLR3_HASH_ENUM_struct * en, pANTLR3_HASH_KEY *key, void ** data);
    void	(*free)	    (struct ANTLR3_HASH_ENUM_struct * table);
}
    ANTLR3_HASH_ENUM;

/** Structure that represents a LIST collection
 */
typedef	struct	ANTLR3_LIST_struct
{
    /** Hash table that is storing the list elements
     */
    pANTLR3_HASH_TABLE	table;

    void			(*free)		(struct ANTLR3_LIST_struct * list);
    void			(*del)		(struct ANTLR3_LIST_struct * list, ANTLR3_INTKEY key);
    void *			(*get)		(struct ANTLR3_LIST_struct * list, ANTLR3_INTKEY key);
    void *			(*remove)	(struct ANTLR3_LIST_struct * list, ANTLR3_INTKEY key);
    ANTLR3_INT32    (*add)		(struct ANTLR3_LIST_struct * list, void * element, void (ANTLR3_CDECL *freeptr)(void *));
    ANTLR3_INT32    (*put)		(struct ANTLR3_LIST_struct * list, ANTLR3_INTKEY key, void * element, void (ANTLR3_CDECL *freeptr)(void *));
    ANTLR3_UINT32   (*size)		(struct ANTLR3_LIST_struct * list);
    
}
    ANTLR3_LIST;

/** Structure that represents a Stack collection
 */
typedef	struct	ANTLR3_STACK_struct
{
    /** List that supports the stack structure
     */
    pANTLR3_VECTOR  vector;

    /** Used for quick access to the top of the stack
     */
    void *	    top;
    void			(*free)	(struct ANTLR3_STACK_struct * stack);
    void *			(*pop)	(struct ANTLR3_STACK_struct * stack);
    void *			(*get)	(struct ANTLR3_STACK_struct * stack, ANTLR3_INTKEY key);
    ANTLR3_BOOLEAN  (*push)	(struct ANTLR3_STACK_struct * stack, void * element, void (ANTLR3_CDECL *freeptr)(void *));
    ANTLR3_UINT32   (*size)	(struct ANTLR3_STACK_struct * stack);
    void *			(*peek)	(struct ANTLR3_STACK_struct * stack);

}
    ANTLR3_STACK;

/* Structure that represents a vector element
 */
typedef struct ANTLR3_VECTOR_ELEMENT_struct
{
    void    * element;
    void (ANTLR3_CDECL *freeptr)(void *);
}
    ANTLR3_VECTOR_ELEMENT, *pANTLR3_VECTOR_ELEMENT;

#define ANTLR3_VECTOR_INTERNAL_SIZE     16
/* Structure that represents a vector collection. A vector is a simple list
 * that contains a pointer to the element and a pointer to a function that
 * that can free the element if it is removed. It auto resizes but does not
 * use hash techniques as it is referenced by a simple numeric index. It is not a 
 * sparse list, so if any element is deleted, then the ones following are moved
 * down in memory and the count is adjusted.
 */
typedef struct ANTLR3_VECTOR_struct
{
    /** Array of pointers to vector elements
     */
    pANTLR3_VECTOR_ELEMENT  elements;

    /** Number of entries currently in the list;
     */
    ANTLR3_UINT32   count;

    /** Many times, a vector holds just a few nodes in an AST and it
     * is too much overhead to malloc the space for elements so
     * at the expense of a few bytes of memory, we hold the first
     * few elements internally. It means we must copy them when
     * we grow beyond this initial size, but that is less overhead than
     * the malloc/free callas we would otherwise require.
     */
    ANTLR3_VECTOR_ELEMENT   internal[ANTLR3_VECTOR_INTERNAL_SIZE];

    /** Indicates if the structure was made by a factory, in which
     *  case only the factory can free the memory for the actual vector,
     *  though the vector free function is called and will recurse through its
     *  entries calling any free pointers for each entry.
     */
    ANTLR3_BOOLEAN  factoryMade;

    /** Total number of entries in elements at any point in time
     */
    ANTLR3_UINT32   elementsSize;

    void			(ANTLR3_CDECL *free)	(struct ANTLR3_VECTOR_struct * vector);
    void			(*del)					(struct ANTLR3_VECTOR_struct * vector, ANTLR3_UINT32 entry);
    void *			(*get)					(struct ANTLR3_VECTOR_struct * vector, ANTLR3_UINT32 entry);
    void *			(*remove)				(struct ANTLR3_VECTOR_struct * vector, ANTLR3_UINT32 entry);
	void			(*clear)				(struct ANTLR3_VECTOR_struct * vector);
    ANTLR3_UINT32   (*add)					(struct ANTLR3_VECTOR_struct * vector, void * element, void (ANTLR3_CDECL *freeptr)(void *));
    ANTLR3_UINT32   (*set)					(struct ANTLR3_VECTOR_struct * vector, ANTLR3_UINT32 entry, void * element, void (ANTLR3_CDECL *freeptr)(void *), ANTLR3_BOOLEAN freeExisting);
    ANTLR3_UINT32   (*size)					(struct ANTLR3_VECTOR_struct * vector);
}
    ANTLR3_VECTOR;

/** Default vector pool size if otherwise unspecified
 */
#define ANTLR3_FACTORY_VPOOL_SIZE 256

/** Structure that tracks vectors in a vector and auto deletes the vectors
 *  in the vector factory when closed.
 */
typedef struct ANTLR3_VECTOR_FACTORY_struct
{

        /** List of all vector pools allocated so far
         */
        pANTLR3_VECTOR      *pools;

        /** Count of the vector pools allocated so far (current active pool)
         */
        ANTLR3_INT32         thisPool;

        /** The next vector available in the pool
         */
        ANTLR3_UINT32        nextVector;

        /** Trick to quickly initialize a new vector via memcpy and not a function call
         */
        ANTLR3_VECTOR        unTruc;

       	/** Function to close the vector factory
	 */
	void                (*close)	    (struct ANTLR3_VECTOR_FACTORY_struct * factory);

	/** Function to supply a new vector
	 */
	pANTLR3_VECTOR      (*newVector)    (struct ANTLR3_VECTOR_FACTORY_struct * factory);

}
ANTLR3_VECTOR_FACTORY; 
    
    
/* -------------- TRIE Interfaces ---------------- */


/** Structure that holds the payload entry in an ANTLR3_INT_TRIE or ANTLR3_STRING_TRIE
 */
typedef struct ANTLR3_TRIE_ENTRY_struct
{
	ANTLR3_UINT32   type;
	void (ANTLR3_CDECL *freeptr)(void *);
	union
	{
		ANTLR3_INTKEY     intVal;
		void		* ptr;
	} data;

	struct ANTLR3_TRIE_ENTRY_struct	* next;	    /* Allows duplicate entries for same key in insertion order	*/
}
ANTLR3_TRIE_ENTRY, * pANTLR3_TRIE_ENTRY;


/** Structure that defines an element/node in an ANTLR3_INT_TRIE
 */
typedef struct ANTLR3_INT_TRIE_NODE_struct
{
    ANTLR3_UINT32							  bitNum;	/**< This is the left/right bit index for traversal along the nodes				*/
    ANTLR3_INTKEY							  key;		/**< This is the actual key that the entry represents if it is a terminal node  */
    pANTLR3_TRIE_ENTRY						  buckets;	/**< This is the data bucket(s) that the key indexes, which may be NULL			*/
    struct ANTLR3_INT_TRIE_NODE_struct	    * leftN;	/**< Pointer to the left node from here when sKey & bitNum = 0					*/
    struct ANTLR3_INT_TRIE_NODE_struct	    * rightN;	/**< Pointer to the right node from here when sKey & bitNum, = 1				*/
}
    ANTLR3_INT_TRIE_NODE, * pANTLR3_INT_TRIE_NODE;

/** Structure that defines an ANTLR3_INT_TRIE. For this particular implementation,
 *  as you might expect, the key is turned into a "string" by looking at bit(key, depth)
 *  of the integer key. Using 64 bit keys gives us a depth limit of 64 (or bit 0..63)
 *  and potentially a huge trie. This is the algorithm for a Patricia Trie.
 *  Note also that this trie [can] accept multiple entries for the same key and is
 *  therefore a kind of elastic bucket patricia trie.
 *
 *  If you find this code useful, please feel free to 'steal' it for any purpose
 *  as covered by the BSD license under which ANTLR is issued. You can cut the code
 *  but as the ANTLR library is only about 50K (Windows Vista), you might find it 
 *  easier to just link the library. Please keep all comments and licenses and so on
 *  in any version of this you create of course.
 *
 *  Jim Idle.
 *  
 */
typedef struct ANTLR3_INT_TRIE_struct
{
    pANTLR3_INT_TRIE_NODE   root;			/* Root node of this integer trie					*/
    pANTLR3_INT_TRIE_NODE   current;		/* Used to traverse the TRIE with the next() method	*/
    ANTLR3_UINT32			count;			/* Current entry count								*/
    ANTLR3_BOOLEAN			allowDups;		/* Whether this trie accepts duplicate keys			*/

    
    pANTLR3_TRIE_ENTRY	(*get)	(struct ANTLR3_INT_TRIE_struct * trie, ANTLR3_INTKEY key);
    ANTLR3_BOOLEAN		(*del)	(struct ANTLR3_INT_TRIE_struct * trie, ANTLR3_INTKEY key);
    ANTLR3_BOOLEAN		(*add)	(struct ANTLR3_INT_TRIE_struct * trie, ANTLR3_INTKEY key, ANTLR3_UINT32 type, ANTLR3_INTKEY intVal, void * data, void (ANTLR3_CDECL *freeptr)(void *));
    void				(*free)	(struct ANTLR3_INT_TRIE_struct * trie);

}
    ANTLR3_INT_TRIE;

#ifdef __cplusplus
}
#endif

#endif


