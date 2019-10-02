/*
[The "BSD licence"]
Copyright (c) 2005-2007 Kunle Odutola
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


namespace Antlr.Runtime.Collections
{
	using System;
	using ArrayList = System.Collections.ArrayList;
	
	/// <summary>
	/// Stack abstraction that also supports the IList interface
	/// </summary>
	public class StackList : ArrayList
	{
		public StackList() : base()
		{
		}

		/// <summary>
		/// Adds an element to the top of the stack list.
		/// </summary>
		public void Push(object item)
		{
			Add(item);
		}
		
		/// <summary>
		/// Removes the element at the top of the stack list and returns it.
		/// </summary>
		/// <returns>The element at the top of the stack.</returns>
		public object Pop()
		{
			object poppedItem = this[this.Count - 1];
			RemoveAt(this.Count - 1);
			return poppedItem;
		}
		
		/// <summary>
		/// Removes the element at the top of the stack list without removing it.
		/// </summary>
		/// <returns>The element at the top of the stack.</returns>
		public object Peek()
		{
			return this[this.Count - 1];
		}
	}
}