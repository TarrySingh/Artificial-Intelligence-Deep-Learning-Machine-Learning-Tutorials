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


namespace Antlr.Runtime.Tree
{
	using System;
	using IToken = Antlr.Runtime.IToken;
	
	/// <summary>
	/// How to create and navigate trees.  Rather than have a separate factory
	/// and adaptor, I've merged them.  Makes sense to encapsulate.
	/// 
	/// This takes the place of the tree construction code generated in the
	/// generated code in 2.x and the ASTFactory.
	/// 
	/// I do not need to know the type of a tree at all so they are all
	/// generic Objects.  This may increase the amount of typecasting needed. :(
	/// </summary>
	public interface ITreeAdaptor
	{
		// C o n s t r u c t i o n
		
		/// <summary>
		/// Create a tree node from Token object; for CommonTree type trees,
		/// then the token just becomes the payload.
		/// </summary>
		/// <remarks>
		/// This is the most common create call. Override if you want another kind of node to be built.
		/// </remarks>
		object Create(IToken payload);

		/// <summary>Duplicate a single tree node </summary>
		/// <remarks> Override if you want another kind of node to be built.</remarks>
		object DupNode(object treeNode);

		/// <summary>Duplicate tree recursively, using DupNode() for each node </summary>
		object DupTree(object tree);
		
		/// <summary>
		/// Return a nil node (an empty but non-null node) that can hold
		/// a list of element as the children.  If you want a flat tree (a list)
		/// use "t=adaptor.nil(); t.AddChild(x); t.AddChild(y);"
		/// </summary>
		object GetNilNode();
		
		/// <summary>
		/// Return a tree node representing an error. This node records the
		/// tokens consumed during error recovery. The start token indicates the
		/// input symbol at which the error was detected. The stop token indicates
		/// the last symbol consumed during recovery.
		/// </summary> 
		/// <remarks>
		/// <para>You must specify the input stream so that the erroneous text can
		/// be packaged up in the error node. The exception could be useful
		/// to some applications; default implementation stores ptr to it in
		/// the CommonErrorNode.</para>
		///
		/// <para>This only makes sense during token parsing, not tree parsing.
		/// Tree parsing should happen only when parsing and tree construction
		/// succeed.</para>
		/// </remarks>
		object ErrorNode(ITokenStream input, IToken start, IToken stop, RecognitionException e);

		/// <summary>
		/// Is tree considered a nil node used to make lists of child nodes?
		/// </summary>
		bool IsNil(object tree);

		/// <summary>
		/// Add a child to the tree t.  If child is a flat tree (a list), make all
		/// in list children of t.
		/// </summary>
		/// <remarks>
		/// <para>
		/// Warning: if t has no children, but child does and child isNil then you 
		/// can decide it is ok to move children to t via t.children = child.children;
		/// i.e., without copying the array.  Just make sure that this is consistent 
		/// with have the user will build ASTs. Do nothing if t or child is null.
		/// </para>
		/// <para>
		/// This is for construction and I'm not sure it's completely general for
		/// a tree's addChild method to work this way.  Make sure you differentiate
		/// between your tree's addChild and this parser tree construction addChild
		/// if it's not ok to move children to t with a simple assignment.
		/// </para>
		/// </remarks>
		void  AddChild(object t, object child);

		/// <summary>
		/// If oldRoot is a nil root, just copy or move the children to newRoot.
		/// If not a nil root, make oldRoot a child of newRoot.
		/// </summary>
		/// <remarks>
		/// 
		///   old=^(nil a b c), new=r yields ^(r a b c)
		///   old=^(a b c), new=r yields ^(r ^(a b c))
		/// 
		/// If newRoot is a nil-rooted single child tree, use the single
		/// child as the new root node.
		/// 
		///   old=^(nil a b c), new=^(nil r) yields ^(r a b c)
		///   old=^(a b c), new=^(nil r) yields ^(r ^(a b c))
		///  
		/// If oldRoot was null, it's ok, just return newRoot (even if isNil).
		/// 
		///   old=null, new=r yields r
		///   old=null, new=^(nil r) yields ^(nil r)
		/// 
		/// Return newRoot.  Throw an exception if newRoot is not a
		/// simple node or nil root with a single child node--it must be a root
		/// node.  If newRoot is ^(nil x) return x as newRoot.
		/// 
		/// Be advised that it's ok for newRoot to point at oldRoot's
		/// children; i.e., you don't have to copy the list.  We are
		/// constructing these nodes so we should have this control for
		/// efficiency.
		/// </remarks>
		object BecomeRoot(object newRoot, object oldRoot);
		
		/// <summary>
		/// Given the root of the subtree created for this rule, post process
		/// it to do any simplifications or whatever you want.  A required
		/// behavior is to convert ^(nil singleSubtree) to singleSubtree
		/// as the setting of start/stop indexes relies on a single non-nil root
		/// for non-flat trees.
		/// 
		/// Flat trees such as for lists like "idlist : ID+ ;" are left alone
		/// unless there is only one ID.  For a list, the start/stop indexes
		/// are set in the nil node.
		/// 
		/// This method is executed after all rule tree construction and right
		/// before SetTokenBoundaries().
		/// </summary>
		object RulePostProcessing(object root);

		/// <summary>
		/// For identifying trees. How to identify nodes so we can say "add node 
		/// to a prior node"?
		/// </summary>
		/// <remarks>
		/// Even BecomeRoot is an issue. Ok, we could:
		/// <list type="number">
		///		<item>Number the nodes as they are created?</item>
		///		<item>
		///			Use the original framework assigned hashcode that's unique 
		///			across instances of a given type.
		///			WARNING: This is usually implemented either as IL to make a 
		///			non-virt call to object.GetHashCode() or by via a call to
		///			System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode().
		///			Both have issues especially on .NET 1.x and Mono.
		///		</item>
		/// </list>
		/// </remarks>
		int GetUniqueID(object node);

		
		#region R e w r i t e  R u l e s
		
		/// <summary>
		/// Create a node for newRoot make it the root of oldRoot.
		/// If oldRoot is a nil root, just copy or move the children to newRoot.
		/// If not a nil root, make oldRoot a child of newRoot.
		/// 
		/// Return node created for newRoot.
		/// </summary>
		object BecomeRoot(IToken newRoot, object oldRoot);
		
		/// <summary>Create a new node derived from a token, with a new token type.
		/// This is invoked from an imaginary node ref on right side of a
		/// rewrite rule as IMAG[$tokenLabel].
		/// 
		/// This should invoke createToken(Token).
		/// </summary>
		object Create(int tokenType, IToken fromToken);
		
		/// <summary>Same as Create(tokenType,fromToken) except set the text too.
		/// This is invoked from an imaginary node ref on right side of a
		/// rewrite rule as IMAG[$tokenLabel, "IMAG"].
		/// 
		/// This should invoke createToken(Token).
		/// </summary>
		object Create(int tokenType, IToken fromToken, string text);
		
		/// <summary>Create a new node derived from a token, with a new token type.
		/// This is invoked from an imaginary node ref on right side of a
		/// rewrite rule as IMAG["IMAG"].
		/// 
		/// This should invoke createToken(int,String).
		/// </summary>
		object Create(int tokenType, string text);

		#endregion

		#region C o n t e n t

		/// <summary>For tree parsing, I need to know the token type of a node </summary>
		int GetNodeType(object t);
		
		/// <summary>Node constructors can set the type of a node </summary>
		void  SetNodeType(object t, int type);

		string GetNodeText(object t);

		/// <summary>Node constructors can set the text of a node </summary>
		void  SetNodeText(object t, string text);
		

		/// <summary>
		/// Return the token object from which this node was created.
		/// </summary>
		/// <remarks>
		/// Currently used only for printing an error message. The error 
		/// display routine in BaseRecognizer needs to display where the 
		/// input the error occurred. If your tree of limitation does not 
		/// store information that can lead you to the token, you can create 
		/// a token filled with the appropriate information and pass that back.
		/// <see cref="BaseRecognizer.GetErrorMessage"/>
		/// </remarks>
		IToken GetToken(object treeNode);

		/// <summary>
		/// Where are the bounds in the input token stream for this node and
		/// all children?
		/// </summary>
		/// <remarks>
		/// Each rule that creates AST nodes will call this
		/// method right before returning.  Flat trees (i.e., lists) will
		/// still usually have a nil root node just to hold the children list.
		/// That node would contain the start/stop indexes then.
		/// </remarks>
		void  SetTokenBoundaries(object t, IToken startToken, IToken stopToken);

		/// <summary>
		/// Get the token start index for this subtree; return -1 if no such index
		/// </summary>
		int GetTokenStartIndex(object t);

		/// <summary>
		/// Get the token stop index for this subtree; return -1 if no such index
		/// </summary>
		int GetTokenStopIndex(object t);

		#endregion

		#region N a v i g a t i o n  /  T r e e  P a r s i n g

		/// <summary>Get a child 0..n-1 node </summary>
		object GetChild(object t, int i);

		/// <summary>Set ith child (0..n-1) to t; t must be non-null and non-nil node</summary>
		void SetChild(object t, int i, object child);

		/// <summary>Remove ith child and shift children down from right.</summary>
		object DeleteChild(object t, int i);

		/// <summary>How many children?  If 0, then this is a leaf node </summary>
		int GetChildCount(object t);

		/// <summary>
		/// Who is the parent node of this node; if null, implies node is root.
		/// </summary>
		/// <remarks>
		/// If your node type doesn't handle this, it's ok but the tree rewrites
		/// in tree parsers need this functionality.
		/// </remarks>
		object GetParent(object t);
		void   SetParent(object t, object parent);

		/// <summary>
		/// What index is this node in the child list? Range: 0..n-1
		/// </summary>
		/// <remarks>
		/// If your node type doesn't handle this, it's ok but the tree rewrites
		/// in tree parsers need this functionality.
		/// </remarks>
		int  GetChildIndex(object t);
		void SetChildIndex(object t, int index);

		/// <summary>
		/// Replace from start to stop child index of parent with t, which might
		/// be a list.  Number of children may be different after this call.
		/// </summary>
		/// <remarks>
		/// If parent is null, don't do anything; must be at root of overall tree.
		/// Can't replace whatever points to the parent externally.  Do nothing.
		/// </remarks>
		void ReplaceChildren(object parent, int startChildIndex, int stopChildIndex, object t);
		
		#endregion
	}
}