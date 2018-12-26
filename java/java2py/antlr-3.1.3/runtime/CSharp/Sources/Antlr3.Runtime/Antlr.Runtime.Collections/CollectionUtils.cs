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
	using IList				= System.Collections.IList;
	using IDictionary		= System.Collections.IDictionary;
	using DictionaryEntry	= System.Collections.DictionaryEntry;
	using IEnumerator		= System.Collections.IEnumerator;
	using StringBuilder		= System.Text.StringBuilder;
	
	public class CollectionUtils
	{
		/// <summary>
		/// Returns a string representation of this IList.
		/// </summary>
		/// <remarks>
		/// The string representation is a list of the collection's elements in the order 
		/// they are returned by its IEnumerator, enclosed in square brackets ("[]").
		/// The separator is a comma followed by a space i.e. ", ".
		/// </remarks>
		/// <param name="coll">Collection whose string representation will be returned</param>
		/// <returns>A string representation of the specified collection or "null"</returns>
		public static string ListToString(IList coll)
		{
			StringBuilder sb = new StringBuilder();
		
			if (coll != null)
			{
				sb.Append("[");
				for (int i = 0; i < coll.Count; i++) 
				{
					if (i > 0)
						sb.Append(", ");

					object element = coll[i];
					if (element == null)
						sb.Append("null");
					else if (element is IDictionary)
						sb.Append(DictionaryToString((IDictionary)element));
					else if (element is IList)
						sb.Append(ListToString((IList)element));
					else
						sb.Append(element.ToString());

				}
				sb.Append("]");
			}
			else
				sb.Insert(0, "null");

			return sb.ToString();
		}

	
		/// <summary>
		/// Returns a string representation of this IDictionary.
		/// </summary>
		/// <remarks>
		/// The string representation is a list of the collection's elements in the order 
		/// they are returned by its IEnumerator, enclosed in curly brackets ("{}").
		/// The separator is a comma followed by a space i.e. ", ".
		/// </remarks>
		/// <param name="dict">Dictionary whose string representation will be returned</param>
		/// <returns>A string representation of the specified dictionary or "null"</returns>
		public static string DictionaryToString(IDictionary dict)
		{
			StringBuilder sb = new StringBuilder();
		
			if (dict != null)
			{
				sb.Append("{");
				int i = 0;
				foreach (DictionaryEntry e in dict) 
				{
					if (i > 0)
					{
						sb.Append(", ");
					}

					if (e.Value is IDictionary)
						sb.AppendFormat("{0}={1}", e.Key.ToString(), DictionaryToString((IDictionary)e.Value));
					else if (e.Value is IList)
						sb.AppendFormat("{0}={1}", e.Key.ToString(), ListToString((IList)e.Value));
					else
						sb.AppendFormat("{0}={1}", e.Key.ToString(), e.Value.ToString());
					i++;
				}
				sb.Append("}");
			}
			else
				sb.Insert(0, "null");

			return sb.ToString();
		}
	}
}