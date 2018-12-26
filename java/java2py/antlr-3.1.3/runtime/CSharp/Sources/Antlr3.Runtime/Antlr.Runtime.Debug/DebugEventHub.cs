/*
[The "BSD licence"]
Copyright (c) 2007-2008 Johannes Luber
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


namespace Antlr.Runtime.Debug
{
	using System;
	using IList = System.Collections.IList;
	using ArrayList = System.Collections.ArrayList;
	using IToken = Antlr.Runtime.IToken;
	using RecognitionException = Antlr.Runtime.RecognitionException;

	/// <summary>
	/// Broadcast debug events to multiple listeners.
	/// </summary>
	/// <remarks>
	/// Lets you debug and still use the event mechanism to build 
	/// parse trees etc...
	/// Not thread-safe. Don't add events in one thread while parser 
	/// fires events in another.
	/// </remarks>
	public class DebugEventHub : IDebugEventListener
	{
		protected IList listeners = new ArrayList();

		public DebugEventHub(IDebugEventListener listener)
		{
			listeners.Add(listener);
		}

		public DebugEventHub(params IDebugEventListener[] listeners)
		{

			foreach (IDebugEventListener listener in listeners)
			{
				this.listeners.Add(listener);
			}
		}

		/// <summary>
		/// Add another listener to broadcast events too. 
		/// </summary>
		/// <remarks>
		/// Not thread-safe. Don't add events in one thread while parser 
		/// fires events in another.
		/// </remarks>
		public void AddListener(IDebugEventListener listener)
		{
			listeners.Add(listener);
		}

		public void EnterRule(string grammarFileName, string ruleName)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.EnterRule(grammarFileName, ruleName);
			}
		}

		public void ExitRule(string grammarFileName, string ruleName)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.ExitRule(grammarFileName, ruleName);
			}
		}

		public void EnterAlt(int alt)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.EnterAlt(alt);
			}
		}

		public void EnterSubRule(int decisionNumber)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.EnterSubRule(decisionNumber);
			}
		}

		public void ExitSubRule(int decisionNumber)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.ExitSubRule(decisionNumber);
			}
		}

		public void EnterDecision(int decisionNumber)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.EnterDecision(decisionNumber);
			}
		}

		public void ExitDecision(int decisionNumber)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.ExitDecision(decisionNumber);
			}
		}

		public void Location(int line, int pos)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.Location(line, pos);
			}
		}

		public void ConsumeToken(IToken token)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.ConsumeToken(token);
			}
		}

		public void ConsumeHiddenToken(IToken token)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.ConsumeHiddenToken(token);
			}
		}

		public void LT(int index, IToken t)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.LT(index, t);
			}
		}

		public void Mark(int index)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.Mark(index);
			}
		}

		public void Rewind(int index)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.Rewind(index);
			}
		}

		public void Rewind()
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.Rewind();
			}
		}

		public void BeginBacktrack(int level)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.BeginBacktrack(level);
			}
		}

		public void EndBacktrack(int level, bool successful)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.EndBacktrack(level, successful);
			}
		}

		public void RecognitionException(RecognitionException e)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.RecognitionException(e);
			}
		}

		public void BeginResync()
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.BeginResync();
			}
		}

		public void EndResync()
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.EndResync();
			}
		}

		public void SemanticPredicate(bool result, string predicate)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.SemanticPredicate(result, predicate);
			}
		}

		public void Commence()
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.Commence();
			}
		}

		public void Terminate()
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.Terminate();
			}
		}


		#region Tree parsing stuff

		public void ConsumeNode(object t)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.ConsumeNode(t);
			}
		}

		public void LT(int index, object t)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.LT(index, t);
			}
		}

		#endregion


		#region AST Stuff

		public void GetNilNode(object t)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.GetNilNode(t);
			}
		}

		public void ErrorNode(object t) {
			for (int i = 0; i < listeners.Count; i++) {
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.ErrorNode(t);
			}
		}

		public void CreateNode(object t)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.CreateNode(t);
			}
		}

		public void CreateNode(object node, IToken token)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.CreateNode(node, token);
			}
		}

		public void BecomeRoot(object newRoot, object oldRoot)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.BecomeRoot(newRoot, oldRoot);
			}
		}

		public void AddChild(object root, object child)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.AddChild(root, child);
			}
		}

		public void SetTokenBoundaries(object t, int tokenStartIndex, int tokenStopIndex)
		{
			for (int i = 0; i < listeners.Count; i++)
			{
				IDebugEventListener listener = (IDebugEventListener)listeners[i];
				listener.SetTokenBoundaries(t, tokenStartIndex, tokenStopIndex);
			}
		}

		#endregion
	}
}