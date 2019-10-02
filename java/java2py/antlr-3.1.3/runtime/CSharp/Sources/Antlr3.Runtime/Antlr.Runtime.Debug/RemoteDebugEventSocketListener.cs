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
	using System.Globalization;
	using System.Threading;
	using StreamReader = System.IO.StreamReader;
	using StreamWriter = System.IO.StreamWriter;
	using IOException = System.IO.IOException;
	using Encoding = System.Text.Encoding;
	using StringBuilder = System.Text.StringBuilder;
	using TcpClient = System.Net.Sockets.TcpClient;
	using TcpListener = System.Net.Sockets.TcpListener;
	using Antlr.Runtime;
	using ITree = Antlr.Runtime.Tree.ITree;
	using BaseTree = Antlr.Runtime.Tree.BaseTree;


	public class RemoteDebugEventSocketListener
	{
		internal const int MAX_EVENT_ELEMENTS = 8;
		internal IDebugEventListener listener;
		internal string hostName;
		internal int port;
		internal TcpClient channel = null;
		internal StreamWriter writer;
		internal StreamReader reader;
		internal string eventLabel;
		/// <summary>Version of ANTLR (dictates events)</summary>
		public string version;
		public string grammarFileName;
		/// <summary>
		/// Track the last token index we saw during a consume.  If same, then
		/// set a flag that we have a problem.
		/// </summary>
		int previousTokenIndex = -1;
		bool tokenIndexesInvalid = false;

		#region ProxyToken Class

		public class ProxyToken : IToken
		{
			internal int index;
			internal int type;
			internal int channel;
			internal int line;
			internal int charPos;
			internal string text;

			public ProxyToken(int index)
			{
				this.index = index;
			}
			
			public ProxyToken(int index, int type, int channel, int line, int charPos, string text)
			{
				this.index = index;
				this.type = type;
				this.channel = channel;
				this.line = line;
				this.charPos = charPos;
				this.text = text;
			}

			public int Type
			{
				get { return this.type; }
				set { this.type = value; }
			}

			public int Line
			{
				get { return this.line; }
				set { this.line = value; }
			}

			public int CharPositionInLine
			{
				get { return this.charPos; }
				set { this.charPos = value; }
			}

			public int Channel
			{
				get { return this.channel; }
				set { this.channel = value; }
			}

			public int TokenIndex
			{
				get { return this.index; }
				set { this.index = value; }
			}

			public string Text
			{
				get { return this.text; }
				set { this.text = value; }
			}

			public ICharStream InputStream
			{
				get { return null; }
				set { ; }
			}

			public override string ToString()
			{
				string channelStr = "";
				if (channel != Token.DEFAULT_CHANNEL)
				{
					channelStr = ",channel=" + channel;
				}
				return "[" + Text + "/<" + type + ">" + channelStr + "," + line + ":" + CharPositionInLine + ",@" + index + "]";
			}
		}

		#endregion

		#region ProxyTree Class

		public class ProxyTree : BaseTree
		{
			public int ID;
			public int type;
			public int line = 0;
			public int charPos = -1;
			public int tokenIndex = -1;
			public string text;

			public ProxyTree(int ID)
			{
				this.ID = ID;
			}

			public ProxyTree(int ID, int type, int line, int charPos, int tokenIndex, string text)
			{
				this.ID = ID;
				this.type = type;
				this.line = line;
				this.charPos = charPos;
				this.tokenIndex = tokenIndex;
				this.text = text;
			}

			override public int TokenStartIndex
			{
				get { return tokenIndex; }
				set { ; }
			}
			override public int TokenStopIndex
			{
				get { return 0; }
				set { ; }
			}
			override public ITree DupNode()
			{
				return null;
			}
			override public int Type
			{
				get { return type; }
			}
			override public string Text
			{
				get { return text; }
			}

			override public string ToString()
			{
				return "fix this";
			}
		}

		#endregion

		public RemoteDebugEventSocketListener(IDebugEventListener listener, string hostName, int port)
		{
			this.listener = listener;
			this.hostName = hostName;
			this.port = port;
			
			if (!OpenConnection())
			{
				throw new System.Exception();
			}
		}
		
		protected virtual void EventHandler()
		{
			try
			{
				Handshake();
				eventLabel = reader.ReadLine();
				while (eventLabel != null)
				{
					Dispatch(eventLabel);
					Ack();
					eventLabel = reader.ReadLine();
				}
			}
			catch (System.Exception e)
			{
				Console.Error.WriteLine(e);
				Console.Error.WriteLine(e.StackTrace);
			}
			finally
			{
				CloseConnection();
			}
		}
		
		protected virtual bool OpenConnection()
		{
			bool success = false;
			try
			{
				channel = new TcpClient(hostName, port);
				channel.NoDelay = true;
				writer = new StreamWriter(channel.GetStream(), Encoding.UTF8);
				reader = new StreamReader(channel.GetStream(), Encoding.UTF8);
				success = true;
			}
			catch (Exception e)
			{
				Console.Error.WriteLine(e);
			}
			return success;
		}
		
		protected virtual void CloseConnection()
		{
			try
			{
				reader.Close(); reader = null;
				writer.Close();
				writer = null;
				channel.Close();
				channel = null;
			}
			catch (System.Exception e)
			{
				Console.Error.WriteLine(e);
				Console.Error.WriteLine(e.StackTrace);
			}
			finally
			{
				if (reader != null)
				{
					try
					{
						reader.Close();
					}
					catch (IOException ioe)
					{
						Console.Error.WriteLine(ioe);
					}
				}
				if (writer != null)
				{
					writer.Close();
				}
				if (channel != null)
				{
					try
					{
						channel.Close();
					}
					catch (IOException ioe)
					{
						Console.Error.WriteLine(ioe);
					}
				}
			}
		}

		protected virtual void Handshake()
		{
			string antlrLine = reader.ReadLine();
			string[] antlrElements = GetEventElements(antlrLine);
			version = antlrElements[1];
			string grammarLine = reader.ReadLine();
			string[] grammarElements = GetEventElements(grammarLine);
			grammarFileName = grammarElements[1];
			Ack();
			listener.Commence(); // inform listener after handshake
		}
		
		protected virtual void Ack()
		{
			writer.WriteLine("ack");
			writer.Flush();
		}
		
		protected virtual void Dispatch(string line)
		{
			string[] elements = GetEventElements(line);
			if (elements == null || elements[0] == null)
			{
				Console.Error.WriteLine("unknown debug event: " + line);
				return ;
			}
			if (elements[0].Equals("enterRule"))
			{
				listener.EnterRule(elements[1], elements[2]);
			}
			else if (elements[0].Equals("exitRule"))
			{
				listener.ExitRule(elements[1], elements[2]);
			}
			else if (elements[0].Equals("enterAlt"))
			{
				listener.EnterAlt(int.Parse(elements[1], CultureInfo.InvariantCulture));
			}
			else if (elements[0].Equals("enterSubRule"))
			{
				listener.EnterSubRule(int.Parse(elements[1], CultureInfo.InvariantCulture));
			}
			else if (elements[0].Equals("exitSubRule"))
			{
				listener.ExitSubRule(int.Parse(elements[1], CultureInfo.InvariantCulture));
			}
			else if (elements[0].Equals("enterDecision"))
			{
				listener.EnterDecision(int.Parse(elements[1], CultureInfo.InvariantCulture));
			}
			else if (elements[0].Equals("exitDecision"))
			{
				listener.ExitDecision(int.Parse(elements[1], CultureInfo.InvariantCulture));
			}
			else if (elements[0].Equals("location"))
			{
				listener.Location(int.Parse(elements[1], CultureInfo.InvariantCulture),
				    int.Parse(elements[2], CultureInfo.InvariantCulture));
			}
			else if (elements[0].Equals("consumeToken"))
			{
				ProxyToken t = DeserializeToken(elements, 1);
				if (t.TokenIndex == previousTokenIndex)
				{
					tokenIndexesInvalid = true;
				}
				previousTokenIndex = t.TokenIndex;
				listener.ConsumeToken(t);
			}
			else if (elements[0].Equals("consumeHiddenToken"))
			{
				ProxyToken t = DeserializeToken(elements, 1);
				if (t.TokenIndex == previousTokenIndex)
				{
					tokenIndexesInvalid = true;
				}
				previousTokenIndex = t.TokenIndex;
				listener.ConsumeHiddenToken(t);
			}
			else if (elements[0].Equals("LT"))
			{
				IToken t = DeserializeToken(elements, 2);
				listener.LT(int.Parse(elements[1], CultureInfo.InvariantCulture), t);
			}
			else if (elements[0].Equals("mark"))
			{
				listener.Mark(int.Parse(elements[1], CultureInfo.InvariantCulture));
			}
			else if (elements[0].Equals("rewind"))
			{
				if (elements[1] != null)
				{
					listener.Rewind(int.Parse(elements[1], CultureInfo.InvariantCulture));
				}
				else
				{
					listener.Rewind();
				}
			}
			else if (elements[0].Equals("beginBacktrack"))
			{
				listener.BeginBacktrack(int.Parse(elements[1], CultureInfo.InvariantCulture));
			}
			else if (elements[0].Equals("endBacktrack"))
			{
				int level = int.Parse(elements[1], CultureInfo.InvariantCulture);
				int successI = int.Parse(elements[2], CultureInfo.InvariantCulture);
				//listener.EndBacktrack(level, successI == (int)true);
				listener.EndBacktrack(level, successI == 1 /*1=TRUE*/);
			}
			else if (elements[0].Equals("exception"))
			{
				string excName = elements[1];
				string indexS = elements[2];
				string lineS = elements[3];
				string posS = elements[4];
				Type excClass = null;
				try
				{
					excClass = System.Type.GetType(excName);
					RecognitionException e = (RecognitionException) System.Activator.CreateInstance(excClass);
					e.Index = int.Parse(indexS, CultureInfo.InvariantCulture);
					e.Line = int.Parse(lineS, CultureInfo.InvariantCulture);
					e.CharPositionInLine = int.Parse(posS, CultureInfo.InvariantCulture);
					listener.RecognitionException(e);
				}
				catch (System.UnauthorizedAccessException iae)
				{
					Console.Error.WriteLine("can't access class " + iae);
					Console.Error.WriteLine(iae.StackTrace);
				}
			}
			else if (elements[0].Equals("beginResync"))
			{
				listener.BeginResync();
			}
			else if (elements[0].Equals("endResync"))
			{
				listener.EndResync();
			}
			else if (elements[0].Equals("terminate"))
			{
				listener.Terminate();
			}
			else if (elements[0].Equals("semanticPredicate"))
			{
				bool result = bool.Parse(elements[1]);
				string predicateText = elements[2];
				predicateText = UnEscapeNewlines(predicateText);
				listener.SemanticPredicate(result, predicateText);
			}
			else if (elements[0].Equals("consumeNode"))
			{
				ProxyTree node = DeserializeNode(elements, 1);
				listener.ConsumeNode(node);
			}
			else if (elements[0].Equals("LN"))
			{
				int i = int.Parse(elements[1], CultureInfo.InvariantCulture);
				ProxyTree node = DeserializeNode(elements, 2);
				listener.LT(i, node);
			}
			else if (elements[0].Equals("createNodeFromTokenElements"))
			{
				int ID = int.Parse(elements[1], CultureInfo.InvariantCulture);
				int type = int.Parse(elements[2], CultureInfo.InvariantCulture);
				string text = elements[3];
				text = UnEscapeNewlines(text);
				ProxyTree node = new ProxyTree(ID, type, -1, -1, -1, text);
				listener.CreateNode(node);
			}
			else if (elements[0].Equals("createNode"))
			{
				int ID = int.Parse(elements[1], CultureInfo.InvariantCulture);
				int tokenIndex = int.Parse(elements[2], CultureInfo.InvariantCulture);
				// create dummy node/token filled with ID, tokenIndex
				ProxyTree node = new ProxyTree(ID);
				ProxyToken token = new ProxyToken(tokenIndex);
				listener.CreateNode(node, token);
			}
			else if (elements[0].Equals("nilNode"))
			{
				int ID = int.Parse(elements[1], CultureInfo.InvariantCulture);
				ProxyTree node = new ProxyTree(ID);
				listener.GetNilNode(node);
			}
			else if ( elements[0].Equals("errorNode") ) {
				// TODO: do we need a special tree here?
				int ID = int.Parse(elements[1], CultureInfo.InvariantCulture);
				int type = int.Parse(elements[2], CultureInfo.InvariantCulture);
				String text = elements[3];
				text = UnEscapeNewlines(text);
				ProxyTree node = new ProxyTree(ID, type, -1, -1, -1, text);
				listener.ErrorNode(node);
			}
			else if (elements[0].Equals("becomeRoot"))
			{
				int newRootID = int.Parse(elements[1], CultureInfo.InvariantCulture);
				int oldRootID = int.Parse(elements[2], CultureInfo.InvariantCulture);
				ProxyTree newRoot = new ProxyTree(newRootID);
				ProxyTree oldRoot = new ProxyTree(oldRootID);
				listener.BecomeRoot(newRoot, oldRoot);
			}
			else if (elements[0].Equals("addChild"))
			{
				int rootID = int.Parse(elements[1], CultureInfo.InvariantCulture);
				int childID = int.Parse(elements[2], CultureInfo.InvariantCulture);
				ProxyTree root = new ProxyTree(rootID);
				ProxyTree child = new ProxyTree(childID);
				listener.AddChild(root, child);
			}
			else if (elements[0].Equals("setTokenBoundaries"))
			{
				int ID = int.Parse(elements[1], CultureInfo.InvariantCulture);
				ProxyTree node = new ProxyTree(ID);
				listener.SetTokenBoundaries(node,
											int.Parse(elements[2], CultureInfo.InvariantCulture),
											int.Parse(elements[3], CultureInfo.InvariantCulture));
			}
			else
			{
				Console.Error.WriteLine("unknown debug event: " + line);
			}
		}

		protected internal ProxyTree DeserializeNode(string[] elements, int offset)
		{
			int ID = int.Parse(elements[offset + 0], CultureInfo.InvariantCulture);
			int type = int.Parse(elements[offset + 1], CultureInfo.InvariantCulture);
			int tokenLine = int.Parse(elements[offset + 2], CultureInfo.InvariantCulture);
			int charPositionInLine = int.Parse(elements[offset + 3], CultureInfo.InvariantCulture);
			int tokenIndex = int.Parse(elements[offset + 4], CultureInfo.InvariantCulture);
			string text = elements[offset + 5];
			text = UnEscapeNewlines(text);
			return new ProxyTree(ID, type, tokenLine, charPositionInLine, tokenIndex, text);
		}

		protected internal virtual ProxyToken DeserializeToken(string[] elements, int offset)
		{
			string indexS = elements[offset + 0];
			string typeS = elements[offset + 1];
			string channelS = elements[offset + 2];
			string lineS = elements[offset + 3];
			string posS = elements[offset + 4];
			string text = elements[offset + 5];
			text = UnEscapeNewlines(text);
			int index = int.Parse(indexS, CultureInfo.InvariantCulture);
			ProxyToken t = new ProxyToken(index, int.Parse(typeS, CultureInfo.InvariantCulture), int.Parse(channelS, CultureInfo.InvariantCulture), int.Parse(lineS, CultureInfo.InvariantCulture), int.Parse(posS, CultureInfo.InvariantCulture), text);
			return t;
		}
		
		/// <summary>Create a thread to listen to the remote running recognizer </summary>
		public virtual void start()
		{
			Thread t = new Thread(new ThreadStart(this.Run));
			t.Start();
		}
		
		public virtual void Run()
		{
			EventHandler();
		}
		
		// M i s c
		
		public virtual string[] GetEventElements(string eventLabel)
		{
			if (eventLabel == null)
				return null;

			string[] elements = new string[MAX_EVENT_ELEMENTS];
			string str = null; // a string element if present (must be last)
			try
			{
				int firstQuoteIndex = eventLabel.IndexOf('"');
				if (firstQuoteIndex >= 0)
				{
					// treat specially; has a string argument like "a comment\n
					// Note that the string is terminated by \n not end quote.
					// Easier to parse that way.
					string eventWithoutString = eventLabel.Substring(0, (firstQuoteIndex) - (0));
					str = eventLabel.Substring(firstQuoteIndex + 1, (eventLabel.Length) - (firstQuoteIndex + 1));
					eventLabel = eventWithoutString;
				}

				string[] strings = eventLabel.Split('\t');
				int i = 0;
				for ( ; i < strings.Length; i++)
				{
					if (i >= MAX_EVENT_ELEMENTS)
					{
						return elements;
					}
					elements[i] = strings[i];
				}

				if (str != null)
				{
					elements[i] = str;
				}
			}
			catch (System.Exception e)
			{
				Console.Error.WriteLine(e.StackTrace);
			}
			return elements;
		}
		
		protected string UnEscapeNewlines(string txt)
		{
			// this unescape is slow but easy to understand
			txt = txt.Replace("%0A", "\n"); // unescape \n
			txt = txt.Replace("%0D", "\r"); // unescape \r
			txt = txt.Replace("%25", "%"); // undo escaped escape chars
			return txt;
		}

		public bool TokenIndexesAreInvalid
		{
			get { return false; /*tokenIndexesInvalid;*/ }
		}
	}
}
