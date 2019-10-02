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


namespace Antlr.Runtime.Misc
{
	using System;
	using StringBuilder = System.Text.StringBuilder;
	using StackTrace = System.Diagnostics.StackTrace;
	using StackFrame = System.Diagnostics.StackFrame;
	using IList = System.Collections.IList;

	/// <summary>A minimal ANTLR3 error [message] manager with the ST bits</summary>
	public class ErrorManager
	{
		public static void InternalError(object error, Exception e)
		{
			StackFrame location = GetLastNonErrorManagerCodeLocation(e);
			string msg = "Exception " + e + "@" + location + ": " + error;
			//Error(MSG_INTERNAL_ERROR, msg);
			Error(msg);
		}

		public static void InternalError(object error)
		{
			StackFrame location = GetLastNonErrorManagerCodeLocation(new Exception());
			string msg = location + ": " + error;
			//Error(MSG_INTERNAL_ERROR, msg);
			Error(msg);
		}

		/// <summary>
		/// Return first non ErrorManager code location for generating messages
		/// </summary>
		/// <param name="e">Current exception</param>
		/// <returns></returns>
		private static StackFrame GetLastNonErrorManagerCodeLocation(Exception e)
		{
			StackTrace stackTrace = new StackTrace(e);
			int i = 0;
			for (; i < stackTrace.FrameCount; i++)
			{
				StackFrame f = stackTrace.GetFrame(i);
				if (f.ToString().IndexOf("ErrorManager") < 0)
				{
					break;
				}
			}
			StackFrame location = stackTrace.GetFrame(i);

			return location;
		}

		public static void Error(/*int msgID,*/ object arg)
		{
			//getErrorCount().errors++;
			//getErrorListener().error(new ToolMessage(msgID, arg));

			StringBuilder sb = new StringBuilder();
			//sb.AppendFormat("internal error: {0} {1}", arg);
			sb.AppendFormat("internal error: {0} ", arg);
		}

		/*
			INTERNAL_ERROR(arg,arg2,exception,stackTrace) ::= <<
			internal error: <arg> <arg2><if(exception)>: <exception><endif>
			<stackTrace; separator="\n">
			>>
		 */
	}
}