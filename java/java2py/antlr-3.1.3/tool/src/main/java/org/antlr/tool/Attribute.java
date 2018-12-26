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

/** Track the names of attributes define in arg lists, return values,
 *  scope blocks etc...
 */
public class Attribute {
	/** The entire declaration such as "String foo;" */
	public String decl;

	/** The type; might be empty such as for Python which has no static typing */
	public String type;

	/** The name of the attribute "foo" */
	public String name;

	/** The optional attribute intialization expression */
	public String initValue;

	public Attribute(String decl) {
		extractAttribute(decl);
	}

	public Attribute(String name, String decl) {
		this.name = name;
		this.decl = decl;
	}

	/** For decls like "String foo" or "char *foo32[3]" compute the ID
	 *  and type declarations.  Also handle "int x=3" and 'T t = new T("foo")'
	 *  but if the separator is ',' you cannot use ',' in the initvalue.
	 *  AttributeScope.addAttributes takes care of the separation so we are
	 *  free here to use from '=' to end of string as the expression.
	 *
	 *  Set name, type, initvalue, and full decl instance vars.
	 */
	protected void extractAttribute(String decl) {
		if ( decl==null ) {
			return;
		}
		boolean inID = false;
		int start = -1;
		int rightEdgeOfDeclarator = decl.length()-1;
		int equalsIndex = decl.indexOf('=');
		if ( equalsIndex>0 ) {
			// everything after the '=' is the init value
			this.initValue = decl.substring(equalsIndex+1,decl.length());
			rightEdgeOfDeclarator = equalsIndex-1;
		}
		// walk backwards looking for start of an ID
		for (int i=rightEdgeOfDeclarator; i>=0; i--) {
			// if we haven't found the end yet, keep going
			if ( !inID && Character.isLetterOrDigit(decl.charAt(i)) ) {
			    inID = true;
			}
			else if ( inID &&
				      !(Character.isLetterOrDigit(decl.charAt(i))||
				       decl.charAt(i)=='_') ) {
				start = i+1;
				break;
			}
		}
		if ( start<0 && inID ) {
			start = 0;
		}
		if ( start<0 ) {
			ErrorManager.error(ErrorManager.MSG_CANNOT_FIND_ATTRIBUTE_NAME_IN_DECL,decl);
		}
		// walk forwards looking for end of an ID
		int stop=-1;
		for (int i=start; i<=rightEdgeOfDeclarator; i++) {
			// if we haven't found the end yet, keep going
			if ( !(Character.isLetterOrDigit(decl.charAt(i))||
				decl.charAt(i)=='_') )
			{
				stop = i;
				break;
			}
			if ( i==rightEdgeOfDeclarator ) {
				stop = i+1;
			}
		}

		// the name is the last ID
		this.name = decl.substring(start,stop);

		// the type is the decl minus the ID (could be empty)
		this.type = decl.substring(0,start);
		if ( stop<=rightEdgeOfDeclarator ) {
			this.type += decl.substring(stop,rightEdgeOfDeclarator+1);
		}
		this.type = type.trim();
		if ( this.type.length()==0 ) {
			this.type = null;
		}

		this.decl = decl;
	}

	public String toString() {
		if ( initValue!=null ) {
			return type+" "+name+"="+initValue;
		}
		return type+" "+name;
	}
}

