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
package org.antlr.runtime.tree;

import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;

public class TreePatternParser {
	protected TreePatternLexer tokenizer;
	protected int ttype;
	protected TreeWizard wizard;
	protected TreeAdaptor adaptor;

	public TreePatternParser(TreePatternLexer tokenizer, TreeWizard wizard, TreeAdaptor adaptor) {
		this.tokenizer = tokenizer;
		this.wizard = wizard;
		this.adaptor = adaptor;
		ttype = tokenizer.nextToken(); // kickstart
	}

	public Object pattern() {
		if ( ttype==TreePatternLexer.BEGIN ) {
			return parseTree();
		}
		else if ( ttype==TreePatternLexer.ID ) {
			Object node = parseNode();
			if ( ttype==TreePatternLexer.EOF ) {
				return node;
			}
			return null; // extra junk on end
		}
		return null;
	}

	public Object parseTree() {
		if ( ttype != TreePatternLexer.BEGIN ) {
			System.out.println("no BEGIN");
			return null;
		}
		ttype = tokenizer.nextToken();
		Object root = parseNode();
		if ( root==null ) {
			return null;
		}
		while ( ttype==TreePatternLexer.BEGIN ||
				ttype==TreePatternLexer.ID ||
				ttype==TreePatternLexer.PERCENT ||
				ttype==TreePatternLexer.DOT )
		{
			if ( ttype==TreePatternLexer.BEGIN ) {
				Object subtree = parseTree();
				adaptor.addChild(root, subtree);
			}
			else {
				Object child = parseNode();
				if ( child==null ) {
					return null;
				}
				adaptor.addChild(root, child);
			}
		}
		if ( ttype != TreePatternLexer.END ) {
			System.out.println("no END");
			return null;
		}
		ttype = tokenizer.nextToken();
		return root;
	}

	public Object parseNode() {
		// "%label:" prefix
		String label = null;
		if ( ttype == TreePatternLexer.PERCENT ) {
			ttype = tokenizer.nextToken();
			if ( ttype != TreePatternLexer.ID ) {
				return null;
			}
			label = tokenizer.sval.toString();
			ttype = tokenizer.nextToken();
			if ( ttype != TreePatternLexer.COLON ) {
				return null;
			}
			ttype = tokenizer.nextToken(); // move to ID following colon
		}

		// Wildcard?
		if ( ttype == TreePatternLexer.DOT ) {
			ttype = tokenizer.nextToken();
			Token wildcardPayload = new CommonToken(0, ".");
			TreeWizard.TreePattern node =
				new TreeWizard.WildcardTreePattern(wildcardPayload);
			if ( label!=null ) {
				node.label = label;
			}
			return node;
		}

		// "ID" or "ID[arg]"
		if ( ttype != TreePatternLexer.ID ) {
			return null;
		}
		String tokenName = tokenizer.sval.toString();
		ttype = tokenizer.nextToken();
		if ( tokenName.equals("nil") ) {
			return adaptor.nil();
		}
		String text = tokenName;
		// check for arg
		String arg = null;
		if ( ttype == TreePatternLexer.ARG ) {
			arg = tokenizer.sval.toString();
			text = arg;
			ttype = tokenizer.nextToken();
		}
		
		// create node
		int treeNodeType = wizard.getTokenType(tokenName);
		if ( treeNodeType==Token.INVALID_TOKEN_TYPE ) {
			return null;
		}
		Object node;
		node = adaptor.create(treeNodeType, text);
		if ( label!=null && node.getClass()==TreeWizard.TreePattern.class ) {
			((TreeWizard.TreePattern)node).label = label;
		}
		if ( arg!=null && node.getClass()==TreeWizard.TreePattern.class ) {
			((TreeWizard.TreePattern)node).hasTextArg = true;
		}
		return node;
	}
}
