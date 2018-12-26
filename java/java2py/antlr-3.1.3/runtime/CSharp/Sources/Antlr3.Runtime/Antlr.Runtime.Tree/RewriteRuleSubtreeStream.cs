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
	using IList = System.Collections.IList;
	
	public class RewriteRuleSubtreeStream : RewriteRuleElementStream
	{
		public RewriteRuleSubtreeStream(ITreeAdaptor adaptor, string elementDescription)
			: base(adaptor, elementDescription)
		{
		}
		
		/// <summary>
		/// Create a stream with one element
		/// </summary>
		public RewriteRuleSubtreeStream(ITreeAdaptor adaptor, string elementDescription, object oneElement)
			: base(adaptor, elementDescription, oneElement)
		{
		}
		
		/// <summary>
		/// Create a stream, but feed off an existing list
		/// </summary>
		public RewriteRuleSubtreeStream(ITreeAdaptor adaptor, string elementDescription, IList elements)
			: base(adaptor, elementDescription, elements)
		{
		}
		
		/// <summary>
		/// Treat next element as a single node even if it's a subtree.
		/// </summary>
		/// <remarks>
		/// This is used instead of next() when the result has to be a
		/// tree root node.  Also prevents us from duplicating recently-added
		/// children; e.g., ^(type ID)+ adds ID to type and then 2nd iteration
		/// must dup the type node, but ID has been added.
		///
		/// Referencing a rule result twice is ok; dup entire tree as
		/// we can't be adding trees as root; e.g., expr expr. 
		/// 
		/// Hideous code duplication here with respect to base.NextTree() inherited from Java version.
		/// Can't think of a proper way to refactor.  This needs to always call dup node
		/// and base.NextTree() doesn't know which to call: dup node or dup tree.
		/// </remarks>
		public object NextNode()
		{
			int size = Count;
			if (dirty || ((cursor >= size) && (size == 1)))
			{
				// if out of elements and size is 1, dup (at most a single node
				// since this is for making root nodes).
				object el = _Next();
				return adaptor.DupNode(el);
			}
			// test size above then fetch
			object elem = _Next();
			return elem;
		}
		
		override protected object Dup(object el) 
		{
			return adaptor.DupTree(el);
		}
	}
}
#elif DOTNET2
namespace Antlr.Runtime.Tree {
	using System;
	using System.Collections.Generic;
	using SpecializingType = System.Object;

#warning Check, if RewriteRuleSubtreeStream can be changed to take advantage of something more specific than object.
	/// <summary>
	/// </summary>
	/// <remarks></remarks>
	/// <example></example>
	public class RewriteRuleSubtreeStream : RewriteRuleElementStream<SpecializingType> {

		#region private delegate object ProcessHandler(object o)
		/// <summary>
		/// This delegate is used to allow the outfactoring of some common code.
		/// </summary>
		/// <param name="o">The to be processed object</param>
		private delegate object ProcessHandler(object o);
		#endregion

		public RewriteRuleSubtreeStream(ITreeAdaptor adaptor, string elementDescription)
			: base(adaptor, elementDescription) {
		}

		/// <summary>
		/// Create a stream with one element
		/// </summary>
		public RewriteRuleSubtreeStream(
			ITreeAdaptor adaptor,
			string elementDescription,
			SpecializingType oneElement
		) : base(adaptor, elementDescription, oneElement) {
		}

		/// <summary>Create a stream, but feed off an existing list</summary>
		public RewriteRuleSubtreeStream(
			ITreeAdaptor adaptor,
			string elementDescription,
			IList<SpecializingType> elements
		) : base(adaptor, elementDescription, elements) {
		}

		/// <summary>Create a stream, but feed off an existing list</summary>
		[Obsolete("This constructor is for internal use only and might be phased out soon. Use instead the one with IList<T>.")]
		public RewriteRuleSubtreeStream(
			ITreeAdaptor adaptor,
			string elementDescription,
			System.Collections.IList elements
		) : base(adaptor, elementDescription, elements) {
		}

		/// <summary>
		/// Treat next element as a single node even if it's a subtree.
		/// </summary>
		/// <remarks>
		/// This is used instead of next() when the result has to be a
		/// tree root node.  Also prevents us from duplicating recently-added
		/// children; e.g., ^(type ID)+ adds ID to type and then 2nd iteration
		/// must dup the type node, but ID has been added.
		///
		/// Referencing a rule result twice is ok; dup entire tree as
		/// we can't be adding trees as root; e.g., expr expr. 
		/// </remarks>
		public object NextNode() {
			// if necessary, dup (at most a single node since this is for making root nodes).
			return FetchObject(delegate(object o) { return adaptor.DupNode(o); });
		}

		#region private object FetchObject(ProcessHandler ph)
		/// <summary>
		/// This method has the common code of two other methods, which differed in only one
		/// function call.
		/// </summary>
		/// <param name="ph">The delegate, which has the chosen function</param>
		/// <returns>The required object</returns>
		private object FetchObject(ProcessHandler ph) {
			if (RequiresDuplication()) {
				// process the object
				return ph(_Next());
			}
			// test above then fetch
			return _Next();
		}
		#endregion

		/// <summary>
		/// Tests, if the to be returned object requires duplication
		/// </summary>
		/// <returns><code>true</code>, if positive, <code>false</code>, if negative.</returns>
		private bool RequiresDuplication() {
			int size = Count;
			// if dirty or if out of elements and size is 1
			return dirty || ((cursor >= size) && (size == 1));
		}

		#region public override object NextTree()
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
		public override object NextTree() {
			// if out of elements and size is 1, dup
			return FetchObject(delegate(object o) { return Dup(o); });
		}
		#endregion


		/// <summary>
		/// When constructing trees, sometimes we need to dup a token or AST
		/// subtree. Dup'ing a token means just creating another AST node
		/// around it. For trees, you must call the adaptor.dupTree()
		/// unless the element is for a tree root; then it must be a node dup
		/// </summary>
		private SpecializingType Dup(SpecializingType el) {
			return adaptor.DupTree(el);
		}
	}
}
#endif