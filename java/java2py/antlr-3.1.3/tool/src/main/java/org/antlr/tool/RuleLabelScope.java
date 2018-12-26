/*
 [The "BSD licence"]
 Copyright (c) 2005-2008 Terence Parr
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

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
package org.antlr.tool;

import antlr.Token;

public class RuleLabelScope extends AttributeScope {
	/** Rules have a predefined set of attributes as well as
	 *  the return values.  'text' needs to be computed though so.
	 */
	public static AttributeScope predefinedRulePropertiesScope =
		new AttributeScope("RulePredefined",null) {{
			addAttribute("text", null);
			addAttribute("start", null);
			addAttribute("stop", null);
			addAttribute("tree", null);
			addAttribute("st", null);
			isPredefinedRuleScope = true;
		}};

	public static AttributeScope predefinedTreeRulePropertiesScope =
		new AttributeScope("RulePredefined",null) {{
			addAttribute("text", null);
			addAttribute("start", null); // note: no stop; not meaningful
			addAttribute("tree", null);
			addAttribute("st", null);
			isPredefinedRuleScope = true;
		}};

	public static AttributeScope predefinedLexerRulePropertiesScope =
		new AttributeScope("LexerRulePredefined",null) {{
			addAttribute("text", null);
			addAttribute("type", null);
			addAttribute("line", null);
			addAttribute("index", null);
			addAttribute("pos", null);
			addAttribute("channel", null);
			addAttribute("start", null);
			addAttribute("stop", null);
			addAttribute("int", null);
			isPredefinedLexerRuleScope = true;
		}};

	public static AttributeScope[] grammarTypeToRulePropertiesScope =
		{
			null,
			predefinedLexerRulePropertiesScope,	// LEXER
			predefinedRulePropertiesScope,		// PARSER
			predefinedTreeRulePropertiesScope,	// TREE_PARSER
			predefinedRulePropertiesScope,		// COMBINED
		};

	public Rule referencedRule;

	public RuleLabelScope(Rule referencedRule, Token actionToken) {
		super("ref_"+referencedRule.name,actionToken);
		this.referencedRule = referencedRule;
	}

	/** If you label a rule reference, you can access that rule's
	 *  return values as well as any predefined attributes.
	 */
	public Attribute getAttribute(String name) {
		AttributeScope rulePropertiesScope =
			RuleLabelScope.grammarTypeToRulePropertiesScope[grammar.type];
		if ( rulePropertiesScope.getAttribute(name)!=null ) {
			return rulePropertiesScope.getAttribute(name);
		}

		if ( referencedRule.returnScope!=null ) {
			return referencedRule.returnScope.getAttribute(name);
		}
		return null;
	}
}
