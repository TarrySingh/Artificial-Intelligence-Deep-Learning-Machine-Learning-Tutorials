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

	/// <summary>
	/// Queues up nodes matched on left side of -> in a tree parser. This is
	/// the analog of RewriteRuleTokenStream for normal parsers. 
	/// </summary>
	public class RewriteRuleNodeStream : RewriteRuleElementStream
	{
		public RewriteRuleNodeStream(ITreeAdaptor adaptor, string elementDescription)
			: base(adaptor, elementDescription)
		{
		}

		/// <summary>Create a stream with one element</summary>
		public RewriteRuleNodeStream(ITreeAdaptor adaptor, string elementDescription, object oneElement)
			: base(adaptor, elementDescription, oneElement)
		{
		}

		/// <summary>Create a stream, but feed off an existing list</summary>
		public RewriteRuleNodeStream(ITreeAdaptor adaptor, string elementDescription, IList elements)
			: base(adaptor, elementDescription, elements)
		{
		}

		 public object NextNode()
		{
			return _Next();
		}

		override protected object ToTree(object el)
		{
			return adaptor.DupNode(el);
		}

		override protected object Dup(object el)
		{
			// we dup every node, so don't have to worry about calling dup; 
			// short-circuited NextTree() so it doesn't call.
			throw new NotSupportedException("dup can't be called for a node stream.");
		}
	}
}
#elif DOTNET2
namespace Antlr.Runtime.Tree {
	using System;
	using System.Collections.Generic;
	using SpecializingType = System.Object;

	/// <summary>
	/// Queues up nodes matched on left side of -> in a tree parser. This is
	/// the analog of RewriteRuleTokenStream for normal parsers. 
	/// </summary>
#warning Check, if RewriteRuleNodeStream can be changed to take advantage of something more specific than object.
	public class RewriteRuleNodeStream : RewriteRuleElementStream<SpecializingType> {
		public RewriteRuleNodeStream(ITreeAdaptor adaptor, string elementDescription)
			: base(adaptor, elementDescription) {
		}

		/// <summary>Create a stream with one element</summary>
		public RewriteRuleNodeStream(
			ITreeAdaptor adaptor,
			string elementDescription,
			SpecializingType oneElement
		) : base(adaptor, elementDescription, oneElement) {
		}

		/// <summary>Create a stream, but feed off an existing list</summary>
		public RewriteRuleNodeStream(
			ITreeAdaptor adaptor,
			string elementDescription,
			IList<SpecializingType> elements
		) : base(adaptor, elementDescription, elements) {
		}

		/// <summary>Create a stream, but feed off an existing list</summary>
		[Obsolete("This constructor is for internal use only and might be phased out soon. Use instead the one with IList<T>.")]
		public RewriteRuleNodeStream(
			ITreeAdaptor adaptor,
			string elementDescription,
			System.Collections.IList elements
		) : base(adaptor, elementDescription, elements) {
		}

		public object NextNode() {
			return _Next();
		}

		override protected object ToTree(object el) {
			return adaptor.DupNode(el);
		}
	}
}
#endif