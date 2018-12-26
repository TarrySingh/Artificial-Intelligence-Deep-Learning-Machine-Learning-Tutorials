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
import org.antlr.codegen.CodeGenerator;

import java.util.*;

/** Track the attributes within a scope.  A named scoped has just its list
 *  of attributes.  Each rule has potentially 3 scopes: return values,
 *  parameters, and an implicitly-named scope (i.e., a scope defined in a rule).
 *  Implicitly-defined scopes are named after the rule; rules and scopes then
 *  must live in the same name space--no collisions allowed.
 */
public class AttributeScope {

	/** All token scopes (token labels) share the same fixed scope of
	 *  of predefined attributes.  I keep this out of the runtime.Token
	 *  object to avoid a runtime space burden.
	 */
	public static AttributeScope tokenScope = new AttributeScope("Token",null);
	static {
		tokenScope.addAttribute("text", null);
		tokenScope.addAttribute("type", null);
		tokenScope.addAttribute("line", null);
		tokenScope.addAttribute("index", null);
		tokenScope.addAttribute("pos", null);
		tokenScope.addAttribute("channel", null);
		tokenScope.addAttribute("tree", null);
		tokenScope.addAttribute("int", null);
	}

	/** This scope is associated with which input token (for error handling)? */
	public Token derivedFromToken;

	public Grammar grammar;

	/** The scope name */
	private String name;

	/** Not a rule scope, but visible to all rules "scope symbols { ...}" */
	public boolean isDynamicGlobalScope;

	/** Visible to all rules, but defined in rule "scope { int i; }" */
	public boolean isDynamicRuleScope;

	public boolean isParameterScope;

	public boolean isReturnScope;

	public boolean isPredefinedRuleScope;

	public boolean isPredefinedLexerRuleScope;

	/** The list of Attribute objects */

	protected LinkedHashMap<String,Attribute> attributes = new LinkedHashMap();

	public AttributeScope(String name, Token derivedFromToken) {
		this(null,name,derivedFromToken);
	}

	public AttributeScope(Grammar grammar, String name, Token derivedFromToken) {
		this.grammar = grammar;
		this.name = name;
		this.derivedFromToken = derivedFromToken;
	}

	public String getName() {
		if ( isParameterScope ) {
			return name+"_parameter";
		}
		else if ( isReturnScope ) {
			return name+"_return";
		}
		return name;
	}

	/** From a chunk of text holding the definitions of the attributes,
	 *  pull them apart and create an Attribute for each one.  Add to
	 *  the list of attributes for this scope.  Pass in the character
	 *  that terminates a definition such as ',' or ';'.  For example,
	 *
	 *  scope symbols {
	 *  	int n;
	 *  	List names;
	 *  }
	 *
	 *  would pass in definitions equal to the text in between {...} and
	 *  separator=';'.  It results in two Attribute objects.
	 */
	public void addAttributes(String definitions, int separator) {
		List<String> attrs = new ArrayList<String>();
		CodeGenerator.getListOfArgumentsFromAction(definitions,0,-1,separator,attrs);
		for (String a : attrs) {
			Attribute attr = new Attribute(a);
			if ( !isReturnScope && attr.initValue!=null ) {
				ErrorManager.grammarError(ErrorManager.MSG_ARG_INIT_VALUES_ILLEGAL,
										  grammar,
										  derivedFromToken,
										  attr.name);
				attr.initValue=null; // wipe it out
			}
			attributes.put(attr.name, attr);
		}
	}

	public void addAttribute(String name, String decl) {
		attributes.put(name, new Attribute(name,decl));
	}

	public Attribute getAttribute(String name) {
		return (Attribute)attributes.get(name);
	}

	/** Used by templates to get all attributes */
	public List<Attribute> getAttributes() {
		List<Attribute> a = new ArrayList<Attribute>();
		a.addAll(attributes.values());
		return a;
	}

	/** Return the set of keys that collide from
	 *  this and other.
	 */
	public Set intersection(AttributeScope other) {
		if ( other==null || other.size()==0 || size()==0 ) {
			return null;
		}
		Set inter = new HashSet();
		Set thisKeys = attributes.keySet();
		for (Iterator it = thisKeys.iterator(); it.hasNext();) {
			String key = (String) it.next();
			if ( other.attributes.get(key)!=null ) {
				inter.add(key);
			}
		}
		if ( inter.size()==0 ) {
			return null;
		}
		return inter;
	}

	public int size() {
		return attributes==null?0:attributes.size();
	}

	public String toString() {
		return (isDynamicGlobalScope?"global ":"")+getName()+":"+attributes;
	}
}
