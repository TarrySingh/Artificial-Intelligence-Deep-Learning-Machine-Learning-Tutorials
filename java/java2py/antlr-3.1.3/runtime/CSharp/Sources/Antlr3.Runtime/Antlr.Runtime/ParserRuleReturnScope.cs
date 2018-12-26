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


namespace Antlr.Runtime
{
	using System;
	
	/// <summary>
	/// Rules that return more than a single value must return an object
	/// containing all the values.  Besides the properties defined in
	/// RuleLabelScope.PredefinedRulePropertiesScope there may be user-defined
	/// return values.  This class simply defines the minimum properties that
	/// are always defined and methods to access the others that might be
	/// available depending on output option such as template and tree.
	/// 
	/// Note text is not an actual property of the return value, it is computed
	/// from start and stop using the input stream's ToString() method.  I
	/// could add a ctor to this so that we can pass in and store the input
	/// stream, but I'm not sure we want to do that.  It would seem to be undefined
	/// to get the .text property anyway if the rule matches tokens from multiple
	/// input streams.
	/// 
	/// I do not use getters for fields of objects that are used simply to
	/// group values such as this aggregate.
	/// </summary>
	public class ParserRuleReturnScope : RuleReturnScope
	{
		private IToken start, stop;

		/// <summary>Return the start token or tree </summary>
		override public object Start
		{
			get { return start; }
			set { start = (IToken) value; }
		}

		/// <summary>Return the stop token or tree </summary>
		override public object Stop
		{
			get { return stop; }
			set { stop = (IToken) value; }
		}

	}
}