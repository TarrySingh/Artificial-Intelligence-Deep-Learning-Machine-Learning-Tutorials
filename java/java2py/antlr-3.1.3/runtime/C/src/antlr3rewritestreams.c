/// \file
/// Implementation of token/tree streams that are used by the
/// tree re-write rules to manipulate the tokens and trees produced
/// by rules that are subject to rewrite directives.
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

#include    <antlr3rewritestreams.h>

// Static support function forward declarations for the stream types.
//
static	void				reset			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream); 
static	void				add				(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * el, void (ANTLR3_CDECL *freePtr)(void *));
static	void *				next			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	pANTLR3_BASE_TREE	nextTree		(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	void *				nextToken		(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	void *				_next			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	void *				dupTok			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * el);
static	void *				dupTree			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * el);
static	void *				dupTreeNode		(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * el);
static	pANTLR3_BASE_TREE	toTree			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * element);
static	pANTLR3_BASE_TREE	toTreeNode		(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * element);
static	ANTLR3_BOOLEAN		hasNext			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	pANTLR3_BASE_TREE	nextNode		(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	pANTLR3_BASE_TREE	nextNodeNode	(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	pANTLR3_BASE_TREE	nextNodeToken	(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	ANTLR3_UINT32		size			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	void *				getDescription	(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	void				freeRS			(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);
static	void				expungeRS		(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream);


// Place a now unused rewrite stream back on the rewrite stream pool
// so we can reuse it if we need to.
//
static void
freeRS	(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	// Before placing the stream back in the pool, we
	// need to clear any vector it has. This is so any
	// free pointers that are associated with the
	// entires are called.
	//
	if	(stream->elements != NULL)
	{
		// Protext factory generated nodes as we cannot clear them,
		// the factory is responsible for that.
		//
		if	(stream->elements->factoryMade == ANTLR3_TRUE)
		{
			stream->elements = NULL;
		} 
		else
		{
			stream->elements->clear(stream->elements);
			stream->freeElements = ANTLR3_TRUE;
		}
	}
	else
	{
		stream->freeElements = ANTLR3_FALSE; // Just in case
	}

	// Add the stream into the recognizer stream stack vector
	// adding the stream memory free routine so that
	// it is thrown away when the stack vector is destroyed
	//
	stream->rec->state->rStreams->add(stream->rec->state->rStreams, stream, (void(*)(void *))expungeRS);
}

/** Do special nilNode reuse detection for node streams.
 */
static void
freeNodeRS(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
    pANTLR3_BASE_TREE tree;

    // Before placing the stream back in the pool, we
	// need to clear any vector it has. This is so any
	// free pointers that are associated with the
	// entires are called. However, if this particular function is called
    // then we know that the entries in the stream are definately
    // tree nodes. Hence we check to see if any of them were nilNodes as
    // if they were, we can reuse them.
	//
	if	(stream->elements != NULL)
	{
        // We have some elements to traverse
        //
        ANTLR3_UINT32 i;

        for (i = 1; i<= stream->elements->count; i++)
        {
            tree = (pANTLR3_BASE_TREE)(stream->elements->elements[i-1].element);
            if  (tree->isNilNode(tree))
            {
                 tree->reuse(tree);
            }

        }
		// Protext factory generated nodes as we cannot clear them,
		// the factory is responsible for that.
		//
		if	(stream->elements->factoryMade == ANTLR3_TRUE)
		{
			stream->elements = NULL;
		}
		else
		{
			stream->elements->clear(stream->elements);
			stream->freeElements = ANTLR3_TRUE;
		}
	}
	else
	{
        if  (stream->singleElement != NULL)
        {
            tree = (pANTLR3_BASE_TREE)(stream->singleElement);
            if  (tree->isNilNode(tree))
            {
                 tree->reuse(tree);
            }
        }
		stream->freeElements = ANTLR3_FALSE; // Just in case
	}

	// Add the stream into the recognizer stream stack vector
	// adding the stream memory free routine so that
	// it is thrown away when the stack vector is destroyed
	//
	stream->rec->state->rStreams->add(stream->rec->state->rStreams, stream, (void(*)(void *))expungeRS);
}
static void
expungeRS(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{

	if (stream->freeElements == ANTLR3_TRUE && stream->elements != NULL)
	{
		stream->elements->free(stream->elements);
	}
	ANTLR3_FREE(stream);
}

// Functions for creating streams
//
static  pANTLR3_REWRITE_RULE_ELEMENT_STREAM 
antlr3RewriteRuleElementStreamNewAE(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description)
{
	pANTLR3_REWRITE_RULE_ELEMENT_STREAM	stream;

	// First - do we already have a rewrite stream that was returned
	// to the pool? If we do, then we will just reuse it by resetting
	// the generic interface.
	//
	if	(rec->state->rStreams->count > 0)
	{
		// Remove the entry from the vector. We do not
		// cause it to be freed by using remove.
		//
		stream = rec->state->rStreams->remove(rec->state->rStreams, rec->state->rStreams->count - 1);

		// We found a stream we can reuse.
		// If the stream had a vector, then it will have been cleared
		// when the freeRS was called that put it in this stack
		//
	}
	else
	{
		// Ok, we need to allocate a new one as there were none on the stack.
		// First job is to create the memory we need.
		//
		stream	= (pANTLR3_REWRITE_RULE_ELEMENT_STREAM) ANTLR3_MALLOC((size_t)(sizeof(ANTLR3_REWRITE_RULE_ELEMENT_STREAM)));

		if	(stream == NULL)
		{
			return	NULL;
		}
		stream->elements		= NULL;
		stream->freeElements	= ANTLR3_FALSE;
	}

	// Populate the generic interface
	//
	stream->rec				= rec;
	stream->reset			= reset;
	stream->add				= add;
	stream->next			= next;
	stream->nextTree		= nextTree;
	stream->nextNode		= nextNode;
	stream->nextToken		= nextToken;
	stream->_next			= _next;
	stream->hasNext			= hasNext;
	stream->size			= size;
	stream->getDescription  = getDescription;
	stream->toTree			= toTree;
	stream->free			= freeRS;
	stream->singleElement	= NULL;

	// Reset the stream to empty.
	//

	stream->cursor			= 0;
	stream->dirty			= ANTLR3_FALSE;

	// Install the description
	//
	stream->elementDescription	= description;

	// Install the adaptor
	//
	stream->adaptor		= adaptor;

	return stream;
}

static pANTLR3_REWRITE_RULE_ELEMENT_STREAM 
antlr3RewriteRuleElementStreamNewAEE(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description, void * oneElement)
{
	pANTLR3_REWRITE_RULE_ELEMENT_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAE(adaptor, rec, description);

	if (stream == NULL)
	{
		return NULL;
	}

	// Stream seems good so we need to add the supplied element
	//
	if	(oneElement != NULL)
	{
		stream->add(stream, oneElement, NULL);
	}
	return stream;
}

static pANTLR3_REWRITE_RULE_ELEMENT_STREAM 
antlr3RewriteRuleElementStreamNewAEV(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description, pANTLR3_VECTOR vector)
{
	pANTLR3_REWRITE_RULE_ELEMENT_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAE(adaptor, rec, description);

	if (stream == NULL)
	{
		return stream;
	}

	// Stream seems good so we need to install the vector we were
	// given. We assume that someone else is going to free the
	// vector.
	//
	if	(stream->elements != NULL && stream->elements->factoryMade == ANTLR3_FALSE && stream->freeElements == ANTLR3_TRUE )
	{
		stream->elements->free(stream->elements);
	}
	stream->elements		= vector;
	stream->freeElements	= ANTLR3_FALSE;
	return stream;
}

// Token rewrite stream ...
//
ANTLR3_API pANTLR3_REWRITE_RULE_TOKEN_STREAM 
antlr3RewriteRuleTOKENStreamNewAE(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description)
{
	pANTLR3_REWRITE_RULE_TOKEN_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAE(adaptor, rec, description);

	if (stream == NULL)
	{
		return stream;
	}

	// Install the token based overrides
	//
	stream->dup			= dupTok;
	stream->nextNode	= nextNodeToken;

	// No nextNode implementation for a token rewrite stream
	//
	return stream;
}

ANTLR3_API pANTLR3_REWRITE_RULE_TOKEN_STREAM 
antlr3RewriteRuleTOKENStreamNewAEE(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description, void * oneElement)
{
	pANTLR3_REWRITE_RULE_TOKEN_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAEE(adaptor, rec, description, oneElement);

	// Install the token based overrides
	//
	stream->dup			= dupTok;
	stream->nextNode	= nextNodeToken;

	// No nextNode implementation for a token rewrite stream
	//
	return stream;
}

ANTLR3_API pANTLR3_REWRITE_RULE_TOKEN_STREAM 
antlr3RewriteRuleTOKENStreamNewAEV(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description, pANTLR3_VECTOR vector)
{
	pANTLR3_REWRITE_RULE_TOKEN_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAEV(adaptor, rec, description, vector);

	// Install the token based overrides
	//
	stream->dup			= dupTok;
	stream->nextNode	= nextNodeToken;

	// No nextNode implementation for a token rewrite stream
	//
	return stream;
}

// Subtree rewrite stream
//
ANTLR3_API pANTLR3_REWRITE_RULE_SUBTREE_STREAM 
antlr3RewriteRuleSubtreeStreamNewAE(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description)
{
	pANTLR3_REWRITE_RULE_SUBTREE_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAE(adaptor, rec, description);

	if (stream == NULL)
	{
		return stream;
	}

	// Install the subtree based overrides
	//
	stream->dup			= dupTree;
	stream->nextNode	= nextNode;
    stream->free        = freeNodeRS;
	return stream;

}
ANTLR3_API pANTLR3_REWRITE_RULE_SUBTREE_STREAM 
antlr3RewriteRuleSubtreeStreamNewAEE(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description, void * oneElement)
{
	pANTLR3_REWRITE_RULE_SUBTREE_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAEE(adaptor, rec, description, oneElement);

	if (stream == NULL)
	{
		return stream;
	}

	// Install the subtree based overrides
	//
	stream->dup			= dupTree;
	stream->nextNode	= nextNode;
    stream->free        = freeNodeRS;

	return stream;
}

ANTLR3_API pANTLR3_REWRITE_RULE_SUBTREE_STREAM 
antlr3RewriteRuleSubtreeStreamNewAEV(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description, pANTLR3_VECTOR vector)
{
	pANTLR3_REWRITE_RULE_SUBTREE_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAEV(adaptor, rec, description, vector);

	if (stream == NULL)
	{
		return NULL;
	}

	// Install the subtree based overrides
	//
	stream->dup			= dupTree;
	stream->nextNode	= nextNode;
    stream->free        = freeNodeRS;

	return stream;
}
// Node rewrite stream ...
//
ANTLR3_API pANTLR3_REWRITE_RULE_NODE_STREAM 
antlr3RewriteRuleNODEStreamNewAE(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description)
{
	pANTLR3_REWRITE_RULE_NODE_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAE(adaptor, rec, description);

	if (stream == NULL)
	{
		return stream;
	}

	// Install the node based overrides
	//
	stream->dup			= dupTreeNode;
	stream->toTree		= toTreeNode;
	stream->nextNode	= nextNodeNode;
    stream->free        = freeNodeRS;

	return stream;
}

ANTLR3_API pANTLR3_REWRITE_RULE_NODE_STREAM 
antlr3RewriteRuleNODEStreamNewAEE(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description, void * oneElement)
{
	pANTLR3_REWRITE_RULE_NODE_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAEE(adaptor, rec, description, oneElement);

	// Install the node based overrides
	//
	stream->dup			= dupTreeNode;
	stream->toTree		= toTreeNode;
	stream->nextNode	= nextNodeNode;
    stream->free        = freeNodeRS;

	return stream;
}

ANTLR3_API pANTLR3_REWRITE_RULE_NODE_STREAM 
antlr3RewriteRuleNODEStreamNewAEV(pANTLR3_BASE_TREE_ADAPTOR adaptor, pANTLR3_BASE_RECOGNIZER rec, pANTLR3_UINT8 description, pANTLR3_VECTOR vector)
{
	pANTLR3_REWRITE_RULE_NODE_STREAM	stream;

	// First job is to create the memory we need.
	//
	stream	= antlr3RewriteRuleElementStreamNewAEV(adaptor, rec, description, vector);

	// Install the Node based overrides
	//
	stream->dup			= dupTreeNode;
	stream->toTree		= toTreeNode;
	stream->nextNode	= nextNodeNode;
    stream->free        = freeNodeRS;
    
	return stream;
}

//----------------------------------------------------------------------
// Static support functions 

/// Reset the condition of this stream so that it appears we have
/// not consumed any of its elements.  Elements themselves are untouched.
///
static void		
reset    (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	stream->dirty	= ANTLR3_TRUE;
	stream->cursor	= 0;
}

// Add a new pANTLR3_BASE_TREE to this stream
//
static void		
add	    (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * el, void (ANTLR3_CDECL *freePtr)(void *))
{
	if (el== NULL)
	{
		return;
	}
	// As we may be reusing a stream, we may already have allocated
	// a rewrite stream vector. If we have then is will be empty if
	// we have either zero or just one element in the rewrite stream
	//
	if (stream->elements != NULL && stream->elements->count > 0)
	{
		// We already have >1 entries in the stream. So we can just add this new element to the existing
		// collection. 
		//
		stream->elements->add(stream->elements, el, freePtr);
		return;
	}
	if (stream->singleElement == NULL)
	{
		stream->singleElement = el;
		return;
	}

	// If we got here then we had only the one element so far
	// and we must now create a vector to hold a collection of them
	//
	if	(stream->elements == NULL)
	{
        pANTLR3_VECTOR_FACTORY factory = ((pANTLR3_COMMON_TREE_ADAPTOR)(stream->adaptor->super))->arboretum->vFactory;

        
		stream->elements		= factory->newVector(factory);
		stream->freeElements	= ANTLR3_TRUE;			// We 'ummed it, so we play it son.
	}
    
	stream->elements->add	(stream->elements, stream->singleElement, freePtr);
	stream->elements->add	(stream->elements, el, freePtr);
	stream->singleElement	= NULL;

	return;
}

/// Return the next element in the stream.  If out of elements, throw
/// an exception unless size()==1.  If size is 1, then return elements[0].
/// Return a duplicate node/subtree if stream is out of elements and
/// size==1.  If we've already used the element, dup (dirty bit set).
///
static pANTLR3_BASE_TREE
nextTree(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream) 
{
	ANTLR3_UINT32		n;
	void			*  el;

	n = stream->size(stream);

	if ( stream->dirty || (stream->cursor >=n && n==1) ) 
	{
		// if out of elements and size is 1, dup
		//
		el = stream->_next(stream);
		return stream->dup(stream, el);
	}

	// test size above then fetch
	//
	el = stream->_next(stream);
	return el;
}

/// Return the next element for a caller that wants just the token
///
static	void *
nextToken		(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	return stream->_next(stream);
}

/// Return the next element in the stream.  If out of elements, throw
/// an exception unless size()==1.  If size is 1, then return elements[0].
///
static void *	
next	    (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	ANTLR3_UINT32   s;

	s = stream->size(stream);
	if (stream->cursor >= s && s == 1)
	{
		pANTLR3_BASE_TREE el;

		el = stream->_next(stream);

		return	stream->dup(stream, el);
	}

	return stream->_next(stream);
}

/// Do the work of getting the next element, making sure that it's
/// a tree node or subtree.  Deal with the optimization of single-
/// element list versus list of size > 1.  Throw an exception (or something similar)
/// if the stream is empty or we're out of elements and size>1.
/// You can override in a 'subclass' if necessary.
///
static void *
_next    (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	ANTLR3_UINT32		n;
	pANTLR3_BASE_TREE	t;

	n = stream->size(stream);

	if (n == 0)
	{
		// This means that the stream is empty
		//
		return NULL;	// Caller must cope with this
	}

	// Traversed all the available elements already?
	//
	if (stream->cursor >= n)
	{
		if (n == 1)
		{
			// Special case when size is single element, it will just dup a lot
			//
			return stream->toTree(stream, stream->singleElement);
		}

		// Out of elements and the size is not 1, so we cannot assume
		// that we just duplicate the entry n times (such as ID ent+ -> ^(ID ent)+)
		// This means we ran out of elements earlier than was expected.
		//
		return NULL;	// Caller must cope with this
	}

	// Elements available either for duping or just available
	//
	if (stream->singleElement != NULL)
	{
		stream->cursor++;   // Cursor advances even for single element as this tells us to dup()
		return stream->toTree(stream, stream->singleElement);
	}

	// More than just a single element so we extract it from the 
	// vector.
	//
	t = stream->toTree(stream, stream->elements->get(stream->elements, stream->cursor));
	stream->cursor++;
	return t;
}

#ifdef ANTLR3_WINDOWS
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
/// When constructing trees, sometimes we need to dup a token or AST
/// subtree.  Dup'ing a token means just creating another AST node
/// around it.  For trees, you must call the adaptor.dupTree().
///
static void *	
dupTok	    (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * el)
{
	ANTLR3_FPRINTF(stderr, "dup() cannot be called on a token rewrite stream!!");
	return NULL;
}
#ifdef ANTLR3_WINDOWS
#pragma warning(pop)
#endif

/// When constructing trees, sometimes we need to dup a token or AST
/// subtree.  Dup'ing a token means just creating another AST node
/// around it.  For trees, you must call the adaptor.dupTree().
///
static void *	
dupTree	    (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * element)
{
	return stream->adaptor->dupNode(stream->adaptor, (pANTLR3_BASE_TREE)element);
}

#ifdef ANTLR3_WINDOWS
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
/// When constructing trees, sometimes we need to dup a token or AST
/// subtree.  Dup'ing a token means just creating another AST node
/// around it.  For trees, you must call the adaptor.dupTree().
///
static void *	
dupTreeNode	    (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * element)
{
	ANTLR3_FPRINTF(stderr, "dup() cannot be called on a node rewrite stream!!!");
	return NULL;
}


/// We don;t explicitly convert to a tree unless the call goes to 
/// nextTree, which means rewrites are heterogeneous 
///
static pANTLR3_BASE_TREE	
toTree   (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * element)
{
	return (pANTLR3_BASE_TREE)element;
}
#ifdef ANTLR3_WINDOWS
#pragma warning(pop)
#endif

/// Ensure stream emits trees; tokens must be converted to AST nodes.
/// AST nodes can be passed through unmolested.
///
#ifdef ANTLR3_WINDOWS
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

static pANTLR3_BASE_TREE	
toTreeNode   (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream, void * element)
{
	return stream->adaptor->dupNode(stream->adaptor, (pANTLR3_BASE_TREE)element);
}

#ifdef ANTLR3_WINDOWS
#pragma warning(pop)
#endif

/// Returns ANTLR3_TRUE if there is a next element available
///
static ANTLR3_BOOLEAN	
hasNext  (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	if (	(stream->singleElement != NULL && stream->cursor < 1)
		||	(stream->elements != NULL && stream->cursor < stream->elements->size(stream->elements)))
	{
		return ANTLR3_TRUE;
	}
	else
	{
		return ANTLR3_FALSE;
	}
}

/// Get the next token from the list and create a node for it
/// This is the implementation for token streams.
///
static pANTLR3_BASE_TREE
nextNodeToken(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	return stream->adaptor->create(stream->adaptor, stream->_next(stream));
}

static pANTLR3_BASE_TREE
nextNodeNode(pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	return stream->_next(stream);
}

/// Treat next element as a single node even if it's a subtree.
/// This is used instead of next() when the result has to be a
/// tree root node.  Also prevents us from duplicating recently-added
/// children; e.g., ^(type ID)+ adds ID to type and then 2nd iteration
/// must dup the type node, but ID has been added.
///
/// Referencing to a rule result twice is ok; dup entire tree as
/// we can't be adding trees; e.g., expr expr. 
///
static pANTLR3_BASE_TREE	
nextNode (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{

	ANTLR3_UINT32	n;
	pANTLR3_BASE_TREE	el = stream->_next(stream);

	n = stream->size(stream);
	if (stream->dirty == ANTLR3_TRUE || (stream->cursor > n && n == 1))
	{
		// We are out of elements and the size is 1, which means we just 
		// dup the node that we have
		//
		return	stream->adaptor->dupNode(stream->adaptor, el);
	}

	// We were not out of nodes, so the one we received is the one to return
	//
	return  el;
}

/// Number of elements available in the stream
///
static ANTLR3_UINT32	
size	    (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	ANTLR3_UINT32   n = 0;

	/// Should be a count of one if singleElement is set. I copied this
	/// logic from the java implementation, which I suspect is just guarding
	/// against someone setting singleElement and forgetting to NULL it out
	///
	if (stream->singleElement != NULL)
	{
		n = 1;
	}
	else
	{
		if (stream->elements != NULL)
		{
			return (ANTLR3_UINT32)(stream->elements->count);
		}
	}
	return n;
}

/// Returns the description string if there is one available (check for NULL).
///
static void *	
getDescription  (pANTLR3_REWRITE_RULE_ELEMENT_STREAM stream)
{
	if (stream->elementDescription == NULL)
	{
		stream->elementDescription = "<unknown source>";
	}

	return  stream->elementDescription;
}
