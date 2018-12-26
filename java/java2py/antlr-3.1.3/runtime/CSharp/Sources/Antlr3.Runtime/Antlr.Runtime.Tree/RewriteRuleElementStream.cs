/*
[The "BSD licence"]
Copyright (c) 2005-2007 Kunle Odutola
Copyright (c) 2007 Johannes Luber
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code MUST RETAIN the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form MUST REPRODUCE the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior WRITTEN permission.
4. Unless explicitly state otherwise, any contribution intentionally 
   submitted for inclusion in this work to the copyright owner or licensor
   shall be under the terms and conditions of this license, without any 
   additional terms or conditions.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#if DOTNET1
namespace Antlr.Runtime.Tree
{
	using System;
	using IList			= System.Collections.IList;
	using ArrayList		= System.Collections.ArrayList;
	using IToken		= Antlr.Runtime.IToken;
	using CommonToken	= Antlr.Runtime.CommonToken;
	
	/// <summary>
	/// A generic list of elements tracked in an alternative to be used in
	/// a -> rewrite rule.  We need to subclass to fill in the next() method,
	/// which returns either an AST node wrapped around a token payload or
	/// an existing subtree.
	///
	/// Once you start next()ing, do not try to add more elements.  It will
	/// break the cursor tracking I believe.
	///
	/// <see cref="RewriteRuleSubtreeStream"/>
	/// <see cref="RewriteRuleTokenStream"/>
	///
	/// TODO: add mechanism to detect/puke on modification after reading from stream
	/// </summary>
	public abstract class RewriteRuleElementStream
	{
		/// <summary>
		/// Cursor 0..n-1.  If singleElement!=null, cursor is 0 until you next(),
		/// which bumps it to 1 meaning no more elements.
		/// </summary>
		protected int cursor = 0;

		/// <summary>
		/// Track single elements w/o creating a list.  Upon 2nd add, alloc list
		/// </summary>
		protected object singleElement;

		/// <summary>
		/// The list of tokens or subtrees we are tracking
		/// </summary>
		protected IList elements;

		/// <summary>
		/// Tracks whether a node or subtree has been used in a stream
		/// </summary>
		/// <remarks>
		/// Once a node or subtree has been used in a stream, it must be dup'd
		/// from then on.  Streams are reset after subrules so that the streams
		/// can be reused in future subrules.  So, reset must set a dirty bit.
		/// If dirty, then next() always returns a dup.
		/// </remarks>
		protected bool dirty = false;

		/// <summary>
		/// The element or stream description; usually has name of the token or
		/// rule reference that this list tracks.  Can include rulename too, but
		/// the exception would track that info.
		/// </summary>
		protected string elementDescription;
		protected ITreeAdaptor adaptor;

		public RewriteRuleElementStream(ITreeAdaptor adaptor, string elementDescription) 
		{
			this.elementDescription = elementDescription;
			this.adaptor = adaptor;
		}

		/// <summary>
		/// Create a stream with one element
		/// </summary>
		public RewriteRuleElementStream(ITreeAdaptor adaptor, string elementDescription, object oneElement)
			: this(adaptor, elementDescription)
		{
			Add(oneElement);
		}

		/// <summary>
		/// Create a stream, but feed off an existing list
		/// </summary>
		public RewriteRuleElementStream(ITreeAdaptor adaptor, string elementDescription, IList elements)
			: this(adaptor, elementDescription)
		{
			this.singleElement = null;
			this.elements = elements;
		}

		public void Add(object el) 
		{
			if ( el == null ) 
			{
				return;
			}
			if ( elements != null ) 
			{ // if in list, just add
				elements.Add(el);
				return;
			}
			if ( singleElement == null ) 
			{ // no elements yet, track w/o list
				singleElement = el;
				return;
			}
			// adding 2nd element, move to list
			elements = new ArrayList(5);
			elements.Add(singleElement);
			singleElement = null;
			elements.Add(el);
		}

		/// <summary>
		/// Reset the condition of this stream so that it appears we have
		/// not consumed any of its elements.  Elements themselves are untouched.
		/// </summary>
		/// <remarks>
		/// Once we reset the stream, any future use will need duplicates.  Set
		/// the dirty bit.
		/// </remarks>
		public virtual void Reset()
		{
			cursor = 0;
			dirty = true;
		}

		public bool HasNext()
		{
			return ( ((singleElement != null) && (cursor < 1)) ||
				  ((elements != null) && (cursor < elements.Count)) );
		}

		/// <summary>
		/// Return the next element in the stream.
		/// </summary>
		/// <remarks>
		/// If out of elements, throw an exception unless Count==1.
		/// If Count is 1, then return elements[0].
		/// Return a duplicate node/subtree if stream is out of 
		/// elements and Count==1.
		/// If we've already used the element, dup (dirty bit set).
		/// </remarks>
		public virtual object NextTree()
		{
			int size = Count;
			if ( dirty || ((cursor >= size) && (size == 1)) )
			{
				// if out of elements and size is 1, dup
				return Dup(_Next());
			}
			// test size above then fetch
			return _Next();
		}

		/// <summary>
		/// Do the work of getting the next element, making sure that
		/// it's a tree node or subtree.
		/// </summary>
		/// <remarks>
		/// Deal with the optimization of single-element list versus 
		/// list of size > 1.  Throw an exception if the stream is 
		/// empty or we're out of elements and size>1.
		/// </remarks>
		protected object _Next() 
		{
			int size = Count;
			if ( size == 0 ) 
			{
				throw new RewriteEmptyStreamException(elementDescription);
			}
			if ( cursor >= size ) 
			{ // out of elements?
				if ( size == 1 ) 
				{  // if size is 1, it's ok; return and we'll dup 
					return ToTree(singleElement);
				}
				// out of elements and size was not 1, so we can't dup
				throw new RewriteCardinalityException(elementDescription);
			}
			// we have elements
			if ( singleElement != null ) 
			{
				cursor++; // move cursor even for single element list
				return ToTree(singleElement);
			}
			// must have more than one in list, pull from elements
			return ToTree(elements[cursor++]);
		}

		/// <summary>
		/// When constructing trees, sometimes we need to dup a token or AST
		/// subtree.  Dup'ing a token means just creating another AST node
		/// around it.  For trees, you must call the adaptor.dupTree()
		/// unless the element is for a tree root; then it must be a node dup
		/// </summary>
		protected abstract object Dup(object el);

		/// <summary>
		/// Ensure stream emits trees; tokens must be converted to AST nodes.
		/// AST nodes can be passed through unmolested.
		/// </summary>
		protected virtual object ToTree(object el) 
		{
			return el;
		}

#warning Size() should be converted into a property named Count.
		public int Size() 
		{
			if (singleElement != null) {
				return 1;
			}
			if (elements != null) {
				return elements.Count;
			}
			return 0;
		}

		public string Description
		{
			get { return elementDescription; }
		}
	}
}
#elif DOTNET2
namespace Antlr.Runtime.Tree {
	using System;
	using System.Collections.Generic;
	using IToken = Antlr.Runtime.IToken;
	using CommonToken = Antlr.Runtime.CommonToken;

	/// <summary>
	/// A generic list of elements tracked in an alternative to be used in
	/// a -> rewrite rule.  We need to subclass to fill in the next() method,
	/// which returns either an AST node wrapped around a token payload or
	/// an existing subtree.
	///
	/// Once you start next()ing, do not try to add more elements.  It will
	/// break the cursor tracking I believe.
	///
	/// <see cref="RewriteRuleSubtreeStream"/>
	/// <see cref="RewriteRuleTokenStream"/>
	///
	/// TODO: add mechanism to detect/puke on modification after reading from stream
	/// </summary>
	public abstract class RewriteRuleElementStream<T> {
		/// <summary>
		/// Cursor 0..n-1.  If singleElement!=null, cursor is 0 until you next(),
		/// which bumps it to 1 meaning no more elements.
		/// </summary>
		protected int cursor = 0;

		/// <summary>
		/// Track single elements w/o creating a list.  Upon 2nd add, alloc list
		/// </summary>
		protected T singleElement;

		/// <summary>
		/// The list of tokens or subtrees we are tracking
		/// </summary>
		protected IList<T> elements;

		/// <summary>
		/// Tracks whether a node or subtree has been used in a stream
		/// </summary>
		/// <remarks>
		/// Once a node or subtree has been used in a stream, it must be dup'd
		/// from then on.  Streams are reset after subrules so that the streams
		/// can be reused in future subrules.  So, reset must set a dirty bit.
		/// If dirty, then next() always returns a dup.
		/// </remarks>
		protected bool dirty = false;

		/// <summary>
		/// The element or stream description; usually has name of the token or
		/// rule reference that this list tracks.  Can include rulename too, but
		/// the exception would track that info.
		/// </summary>
		protected string elementDescription;
		protected ITreeAdaptor adaptor;

		public RewriteRuleElementStream(ITreeAdaptor adaptor, string elementDescription) {
			this.elementDescription = elementDescription;
			this.adaptor = adaptor;
		}

		/// <summary>
		/// Create a stream with one element
		/// </summary>
		public RewriteRuleElementStream(
			ITreeAdaptor adaptor,
			string elementDescription,
			T oneElement
		)
			: this(adaptor, elementDescription) {
			Add(oneElement);
		}

		/// <summary>
		/// Create a stream, but feed off an existing list
		/// </summary>
		public RewriteRuleElementStream(
			ITreeAdaptor adaptor,
		    string elementDescription,
		    IList<T> elements
		)
			: this(adaptor, elementDescription) {
			this.singleElement = default(T);
			this.elements = elements;
		}

		/// <summary>
		/// Create a stream, but feed off an existing list
		/// </summary>
		[Obsolete("This constructor is for internal use only and might be phased out soon. Use instead the one with IList<T>.")]
		public RewriteRuleElementStream(
			ITreeAdaptor adaptor,
			string elementDescription,
		    System.Collections.IList elements
		)
			: this(adaptor, elementDescription) {
			this.singleElement = default(T);
			this.elements = new List<T>();
			if (elements != null) {
				foreach (T t in elements) {
					this.elements.Add(t);
				}
			}
		}

		public void Add(T el) {
			if (el == null) {
				return;
			}
			if (elements != null) { // if in list, just add
				elements.Add(el);
				return;
			}
			if (singleElement == null) { // no elements yet, track w/o list
				singleElement = el;
				return;
			}
			// adding 2nd element, move to list
			elements = new List<T>(5);
			elements.Add(singleElement);
			singleElement = default(T);
			elements.Add(el);
		}

		/// <summary>
		/// Reset the condition of this stream so that it appears we have
		/// not consumed any of its elements.  Elements themselves are untouched.
		/// </summary>
		/// <remarks>
		/// Once we reset the stream, any future use will need duplicates.  Set
		/// the dirty bit.
		/// </remarks>
		public virtual void Reset() {
			cursor = 0;
			dirty = true;
		}

		public bool HasNext() {
			return (((singleElement != null) && (cursor < 1)) ||
				  ((elements != null) && (cursor < elements.Count)));
		}

		/// <summary>
		/// Return the next element in the stream.
		/// </summary>
		public virtual object NextTree() {
			return _Next();
		}

		/// <summary>
		/// Do the work of getting the next element, making sure that
		/// it's a tree node or subtree.
		/// </summary>
		/// <remarks>
		/// Deal with the optimization of single-element list versus 
		/// list of size > 1.  Throw an exception if the stream is 
		/// empty or we're out of elements and size>1.
		/// </remarks>
		protected object _Next() {
			int size = Count;
			if (size == 0) {
				throw new RewriteEmptyStreamException(elementDescription);
			}
			if (cursor >= size) { // out of elements?
				if (size == 1) {  // if size is 1, it's ok; return and we'll dup 
					return ToTree(singleElement);
				}
				// out of elements and size was not 1, so we can't dup
				throw new RewriteCardinalityException(elementDescription);
			}
			// we have elements
			if (singleElement != null) {
				cursor++; // move cursor even for single element list
				return ToTree(singleElement);
			}
			// must have more than one in list, pull from elements
			return ToTree(elements[cursor++]);
		}

		/// <summary>
		/// Ensure stream emits trees; tokens must be converted to AST nodes.
		/// AST nodes can be passed through unmolested.
		/// </summary>
		protected virtual object ToTree(T el) {
			return el;
		}

		public int Count {
			get {
				if (singleElement != null) {
					return 1;
				}
				if (elements != null) {
					return elements.Count;
				}
				return 0;
			}
		}
		
		[Obsolete("Please use property Count instead.")]
		public int Size() {
			return Count;
		}

		public string Description {
			get { return elementDescription; }
		}
	}
}
#endif