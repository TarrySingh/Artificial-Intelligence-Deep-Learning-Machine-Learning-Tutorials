unit Antlr.Runtime.Tree;
(*
[The "BSD licence"]
Copyright (c) 2008 Erik van Bilsen
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
*)

interface

{$IF CompilerVersion < 20}
{$MESSAGE ERROR 'You need Delphi 2009 or higher to use the Antlr runtime'}
{$IFEND}

uses
  Classes,
  SysUtils,
  Antlr.Runtime,
  Antlr.Runtime.Tools,
  Antlr.Runtime.Collections;

type
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
  ITreeAdaptor = interface(IANTLRInterface)
  ['{F9DEB286-F555-4CC8-A51A-93F3F649B248}']
    { Methods }

    // C o n s t r u c t i o n

    /// <summary>
    /// Create a tree node from Token object; for CommonTree type trees,
    /// then the token just becomes the payload.
    /// </summary>
    /// <remarks>
    /// This is the most common create call. Override if you want another kind of node to be built.
    /// </remarks>
    function CreateNode(const Payload: IToken): IANTLRInterface; overload;

    /// <summary>Duplicate a single tree node </summary>
    /// <remarks> Override if you want another kind of node to be built.</remarks>
    function DupNode(const TreeNode: IANTLRInterface): IANTLRInterface;

    /// <summary>Duplicate tree recursively, using DupNode() for each node </summary>
    function DupTree(const Tree: IANTLRInterface): IANTLRInterface;

    /// <summary>
    /// Return a nil node (an empty but non-null node) that can hold
    /// a list of element as the children.  If you want a flat tree (a list)
    /// use "t=adaptor.nil(); t.AddChild(x); t.AddChild(y);"
    /// </summary>
    function GetNilNode: IANTLRInterface;

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
    function ErrorNode(const Input: ITokenStream; const Start, Stop: IToken;
      const E: ERecognitionException): IANTLRInterface;

    /// <summary>
    /// Is tree considered a nil node used to make lists of child nodes?
    /// </summary>
    function IsNil(const Tree: IANTLRInterface): Boolean;

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
    procedure AddChild(const T, Child: IANTLRInterface);

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
    function BecomeRoot(const NewRoot, OldRoot: IANTLRInterface): IANTLRInterface; overload;

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
    function RulePostProcessing(const Root: IANTLRInterface): IANTLRInterface;

    /// <summary>
    /// For identifying trees. How to identify nodes so we can say "add node
    /// to a prior node"?
    /// </summary>
    /// <remarks>
    /// Even BecomeRoot is an issue. Ok, we could:
    /// <list type="number">
    ///   <item>Number the nodes as they are created?</item>
    ///   <item>
    ///     Use the original framework assigned hashcode that's unique
    ///     across instances of a given type.
    ///     WARNING: This is usually implemented either as IL to make a
    ///     non-virt call to object.GetHashCode() or by via a call to
    ///     System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode().
    ///     Both have issues especially on .NET 1.x and Mono.
    ///   </item>
    /// </list>
    /// </remarks>
    function GetUniqueID(const Node: IANTLRInterface): Integer;

    // R e w r i t e  R u l e s

    /// <summary>
    /// Create a node for newRoot make it the root of oldRoot.
    /// If oldRoot is a nil root, just copy or move the children to newRoot.
    /// If not a nil root, make oldRoot a child of newRoot.
    ///
    /// Return node created for newRoot.
    /// </summary>
    function BecomeRoot(const NewRoot: IToken; const OldRoot: IANTLRInterface): IANTLRInterface; overload;

    /// <summary>Create a new node derived from a token, with a new token type.
    /// This is invoked from an imaginary node ref on right side of a
    /// rewrite rule as IMAG[$tokenLabel].
    ///
    /// This should invoke createToken(Token).
    /// </summary>
    function CreateNode(const TokenType: Integer; const FromToken: IToken): IANTLRInterface; overload;

    /// <summary>Same as Create(tokenType,fromToken) except set the text too.
    /// This is invoked from an imaginary node ref on right side of a
    /// rewrite rule as IMAG[$tokenLabel, "IMAG"].
    ///
    /// This should invoke createToken(Token).
    /// </summary>
    function CreateNode(const TokenType: Integer; const FromToken: IToken;
      const Text: String): IANTLRInterface; overload;

    /// <summary>Create a new node derived from a token, with a new token type.
    /// This is invoked from an imaginary node ref on right side of a
    /// rewrite rule as IMAG["IMAG"].
    ///
    /// This should invoke createToken(int,String).
    /// </summary>
    function CreateNode(const TokenType: Integer; const Text: String): IANTLRInterface; overload;

    // C o n t e n t

    /// <summary>For tree parsing, I need to know the token type of a node </summary>
    function GetNodeType(const T: IANTLRInterface): Integer;

    /// <summary>Node constructors can set the type of a node </summary>
    procedure SetNodeType(const T: IANTLRInterface; const NodeType: Integer);

    function GetNodeText(const T: IANTLRInterface): String;

    /// <summary>Node constructors can set the text of a node </summary>
    procedure SetNodeText(const T: IANTLRInterface; const Text: String);

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
    function GetToken(const TreeNode: IANTLRInterface): IToken;

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
    procedure SetTokenBoundaries(const T: IANTLRInterface; const StartToken,
      StopToken: IToken);

    /// <summary>
    /// Get the token start index for this subtree; return -1 if no such index
    /// </summary>
    function GetTokenStartIndex(const T: IANTLRInterface): Integer;

    /// <summary>
    /// Get the token stop index for this subtree; return -1 if no such index
    /// </summary>
    function GetTokenStopIndex(const T: IANTLRInterface): Integer;

    // N a v i g a t i o n  /  T r e e  P a r s i n g

    /// <summary>Get a child 0..n-1 node </summary>
    function GetChild(const T: IANTLRInterface; const I: Integer): IANTLRInterface;

    /// <summary>Set ith child (0..n-1) to t; t must be non-null and non-nil node</summary>
    procedure SetChild(const T: IANTLRInterface; const I: Integer; const Child: IANTLRInterface);

    /// <summary>Remove ith child and shift children down from right.</summary>
    function DeleteChild(const T: IANTLRInterface; const I: Integer): IANTLRInterface;

    /// <summary>How many children?  If 0, then this is a leaf node </summary>
    function GetChildCount(const T: IANTLRInterface): Integer;

    /// <summary>
    /// Who is the parent node of this node; if null, implies node is root.
    /// </summary>
    /// <remarks>
    /// If your node type doesn't handle this, it's ok but the tree rewrites
    /// in tree parsers need this functionality.
    /// </remarks>
    function GetParent(const T: IANTLRInterface): IANTLRInterface;
    procedure SetParent(const T, Parent: IANTLRInterface);

    /// <summary>
    /// What index is this node in the child list? Range: 0..n-1
    /// </summary>
    /// <remarks>
    /// If your node type doesn't handle this, it's ok but the tree rewrites
    /// in tree parsers need this functionality.
    /// </remarks>
    function GetChildIndex(const T: IANTLRInterface): Integer;
    procedure SetChildIdex(const T: IANTLRInterface; const Index: Integer);

    /// <summary>
    /// Replace from start to stop child index of parent with t, which might
    /// be a list.  Number of children may be different after this call.
    /// </summary>
    /// <remarks>
    /// If parent is null, don't do anything; must be at root of overall tree.
    /// Can't replace whatever points to the parent externally.  Do nothing.
    /// </remarks>
    procedure ReplaceChildren(const Parent: IANTLRInterface; const StartChildIndex,
      StopChildIndex: Integer; const T: IANTLRInterface);
  end;

  /// <summary>A stream of tree nodes, accessing nodes from a tree of some kind </summary>
  ITreeNodeStream = interface(IIntStream)
  ['{75EA5C06-8145-48F5-9A56-43E481CE86C6}']
    { Property accessors }
    function GetTreeSource: IANTLRInterface;
    function GetTokenStream: ITokenStream;
    function GetTreeAdaptor: ITreeAdaptor;
    procedure SetHasUniqueNavigationNodes(const Value: Boolean);

    { Methods }

    /// <summary>Get a tree node at an absolute index i; 0..n-1.</summary>
    /// <remarks>
    /// If you don't want to buffer up nodes, then this method makes no
    /// sense for you.
    /// </remarks>
    function Get(const I: Integer): IANTLRInterface;

    /// <summary>
    /// Get tree node at current input pointer + i ahead where i=1 is next node.
    /// i&lt;0 indicates nodes in the past.  So LT(-1) is previous node, but
    /// implementations are not required to provide results for k &lt; -1.
    /// LT(0) is undefined.  For i&gt;=n, return null.
    /// Return null for LT(0) and any index that results in an absolute address
    /// that is negative.
    ///
    /// This is analogus to the LT() method of the TokenStream, but this
    /// returns a tree node instead of a token.  Makes code gen identical
    /// for both parser and tree grammars. :)
    /// </summary>
    function LT(const K: Integer): IANTLRInterface;

    /// <summary>Return the text of all nodes from start to stop, inclusive.
    /// If the stream does not buffer all the nodes then it can still
    /// walk recursively from start until stop.  You can always return
    /// null or "" too, but users should not access $ruleLabel.text in
    /// an action of course in that case.
    /// </summary>
    function ToString(const Start, Stop: IANTLRInterface): String; overload;
    function ToString: String; overload;

    // REWRITING TREES (used by tree parser)

    /// <summary>
    /// Replace from start to stop child index of parent with t, which might
    /// be a list.  Number of children may be different after this call.
    /// </summary>
    /// <remarks>
    /// The stream is notified because it is walking the tree and might need
    /// to know you are monkeying with the underlying tree.  Also, it might be
    /// able to modify the node stream to avoid restreaming for future phases.
    ///
    /// If parent is null, don't do anything; must be at root of overall tree.
    /// Can't replace whatever points to the parent externally.  Do nothing.
    /// </remarks>
    procedure ReplaceChildren(const Parent: IANTLRInterface; const StartChildIndex,
      StopChildIndex: Integer; const T: IANTLRInterface);

    { Properties }

    /// <summary>
    /// Where is this stream pulling nodes from?  This is not the name, but
    /// the object that provides node objects.
    ///
    /// TODO: do we really need this?
    /// </summary>
    property TreeSource: IANTLRInterface read GetTreeSource;

    /// <summary>
    /// Get the ITokenStream from which this stream's Tree was created
    /// (may be null)
    /// </summary>
    /// <remarks>
    /// If the tree associated with this stream was created from a
    /// TokenStream, you can specify it here.  Used to do rule $text
    /// attribute in tree parser.  Optional unless you use tree parser
    /// rule text attribute or output=template and rewrite=true options.
    /// </remarks>
    property TokenStream: ITokenStream read GetTokenStream;

    /// <summary>
    /// What adaptor can tell me how to interpret/navigate nodes and trees.
    /// E.g., get text of a node.
    /// </summary>
    property TreeAdaptor: ITreeAdaptor read GetTreeAdaptor;

    /// <summary>
    /// As we flatten the tree, we use UP, DOWN nodes to represent
    /// the tree structure.  When debugging we need unique nodes
    /// so we have to instantiate new ones.  When doing normal tree
    /// parsing, it's slow and a waste of memory to create unique
    /// navigation nodes.  Default should be false;
    /// </summary>
    property HasUniqueNavigationNodes: Boolean write SetHasUniqueNavigationNodes;
  end;

  /// <summary>
  /// What does a tree look like?  ANTLR has a number of support classes
  /// such as CommonTreeNodeStream that work on these kinds of trees.  You
  /// don't have to make your trees implement this interface, but if you do,
  /// you'll be able to use more support code.
  ///
  /// NOTE: When constructing trees, ANTLR can build any kind of tree; it can
  /// even use Token objects as trees if you add a child list to your tokens.
  ///
  /// This is a tree node without any payload; just navigation and factory stuff.
  /// </summary>
  ITree = interface(IANTLRInterface)
  ['{4B6EFB53-EBF6-4647-BA4D-48B68134DC2A}']
    { Property accessors }
    function GetChildCount: Integer;
    function GetParent: ITree;
    procedure SetParent(const Value: ITree);
    function GetChildIndex: Integer;
    procedure SetChildIndex(const Value: Integer);
    function GetIsNil: Boolean;
    function GetTokenType: Integer;
    function GetText: String;
    function GetLine: Integer;
    function GetCharPositionInLine: Integer;
    function GetTokenStartIndex: Integer;
    procedure SetTokenStartIndex(const Value: Integer);
    function GetTokenStopIndex: Integer;
    procedure SetTokenStopIndex(const Value: Integer);

    { Methods }

    /// <summary>Set (or reset) the parent and child index values for all children</summary>
    procedure FreshenParentAndChildIndexes;

    function GetChild(const I: Integer): ITree;

    /// <summary>
    /// Add t as a child to this node.  If t is null, do nothing.  If t
    /// is nil, add all children of t to this' children.
    /// </summary>
    /// <param name="t">Tree to add</param>
    procedure AddChild(const T: ITree);

    /// <summary>Set ith child (0..n-1) to t; t must be non-null and non-nil node</summary>
    procedure SetChild(const I: Integer; const T: ITree);

    function DeleteChild(const I: Integer): IANTLRInterface;

    /// <summary>
    /// Delete children from start to stop and replace with t even if t is
    /// a list (nil-root tree).  num of children can increase or decrease.
    /// For huge child lists, inserting children can force walking rest of
    /// children to set their childindex; could be slow.
    /// </summary>
    procedure ReplaceChildren(const StartChildIndex, StopChildIndex: Integer;
      const T: IANTLRInterface);

    function DupNode: ITree;

    function ToStringTree: String;

    function ToString: String;

    { Properties }

    property ChildCount: Integer read GetChildCount;

    // Tree tracks parent and child index now > 3.0
    property Parent: ITree read GetParent write SetParent;

    /// <summary>This node is what child index? 0..n-1</summary>
    property ChildIndex: Integer read GetChildIndex write SetChildIndex;

    /// <summary>
    /// Indicates the node is a nil node but may still have children, meaning
    /// the tree is a flat list.
    /// </summary>
    property IsNil: Boolean read GetIsNil;

    /// <summary>Return a token type; needed for tree parsing </summary>
    property TokenType: Integer read GetTokenType;

    property Text: String read GetText;

    /// <summary>In case we don't have a token payload, what is the line for errors? </summary>
    property Line: Integer read GetLine;
    property CharPositionInLine: Integer read GetCharPositionInLine;

    /// <summary>
    /// What is the smallest token index (indexing from 0) for this node
    /// and its children?
    /// </summary>
    property TokenStartIndex: Integer read GetTokenStartIndex write SetTokenStartIndex;

    /// <summary>
    /// What is the largest token index (indexing from 0) for this node
    /// and its children?
    /// </summary>
    property TokenStopIndex: Integer read GetTokenStopIndex write SetTokenStopIndex;
  end;

  /// <summary>
  /// A generic tree implementation with no payload.  You must subclass to
  /// actually have any user data.  ANTLR v3 uses a list of children approach
  /// instead of the child-sibling approach in v2.  A flat tree (a list) is
  /// an empty node whose children represent the list.  An empty, but
  /// non-null node is called "nil".
  /// </summary>
  IBaseTree = interface(ITree)
  ['{6772F6EA-5FE0-40C6-BE5C-800AB2540E55}']
    { Property accessors }
    function GetChildren: IList<IBaseTree>;
    function GetChildIndex: Integer;
    procedure SetChildIndex(const Value: Integer);
    function GetParent: ITree;
    procedure SetParent(const Value: ITree);
    function GetTokenType: Integer;
    function GetTokenStartIndex: Integer;
    procedure SetTokenStartIndex(const Value: Integer);
    function GetTokenStopIndex: Integer;
    procedure SetTokenStopIndex(const Value: Integer);
    function GetText: String;

    { Methods }

    /// <summary>
    /// Add all elements of kids list as children of this node
    /// </summary>
    /// <param name="kids"></param>
    procedure AddChildren(const Kids: IList<IBaseTree>);

    procedure SetChild(const I: Integer; const T: ITree);
    procedure FreshenParentAndChildIndexes(const Offset: Integer);

    procedure SanityCheckParentAndChildIndexes; overload;
    procedure SanityCheckParentAndChildIndexes(const Parent: ITree;
      const I: Integer); overload;

    /// <summary>
    /// Print out a whole tree not just a node
    /// </summary>
    function ToStringTree: String;

    function DupNode: ITree;

    { Properties }

    /// <summary>
    /// Get the children internal list of children. Manipulating the list
    /// directly is not a supported operation (i.e. you do so at your own risk)
    /// </summary>
    property Children: IList<IBaseTree> read GetChildren;

    /// <summary>BaseTree doesn't track child indexes.</summary>
    property ChildIndex: Integer read GetChildIndex write SetChildIndex;

    /// <summary>BaseTree doesn't track parent pointers.</summary>
    property Parent: ITree read GetParent write SetParent;

    /// <summary>Return a token type; needed for tree parsing </summary>
    property TokenType: Integer read GetTokenType;

    /// <summary>
    /// What is the smallest token index (indexing from 0) for this node
    /// and its children?
    /// </summary>
    property TokenStartIndex: Integer read GetTokenStartIndex write SetTokenStartIndex;

    /// <summary>
    /// What is the largest token index (indexing from 0) for this node
    /// and its children?
    /// </summary>
    property TokenStopIndex: Integer read GetTokenStopIndex write SetTokenStopIndex;

    property Text: String read GetText;
  end;

  /// <summary>A tree node that is wrapper for a Token object. </summary>
  /// <remarks>
  /// After 3.0 release while building tree rewrite stuff, it became clear
  /// that computing parent and child index is very difficult and cumbersome.
  /// Better to spend the space in every tree node.  If you don't want these
  /// extra fields, it's easy to cut them out in your own BaseTree subclass.
  /// </remarks>
  ICommonTree = interface(IBaseTree)
  ['{791C0EA6-1E4D-443E-83E2-CC1EFEAECC8B}']
    { Property accessors }
    function GetToken: IToken;
    function GetStartIndex: Integer;
    procedure SetStartIndex(const Value: Integer);
    function GetStopIndex: Integer;
    procedure SetStopIndex(const Value: Integer);

    { Properties }
    property Token: IToken read GetToken;
    property StartIndex: Integer read GetStartIndex write SetStartIndex;
    property StopIndex: Integer read GetStopIndex write SetStopIndex;
  end;

  // A node representing erroneous token range in token stream
  ICommonErrorNode = interface(ICommonTree)
  ['{20FF30BA-C055-4E8F-B3E7-7FFF6313853E}']
  end;

  /// <summary>
  /// A TreeAdaptor that works with any Tree implementation
  /// </summary>
  IBaseTreeAdaptor = interface(ITreeAdaptor)
  ['{B9CE670A-E53F-494C-B700-E4A3DF42D482}']
    /// <summary>
    /// This is generic in the sense that it will work with any kind of
    /// tree (not just the ITree interface).  It invokes the adaptor routines
    /// not the tree node routines to do the construction.
    /// </summary>
    function DupTree(const Tree: IANTLRInterface): IANTLRInterface; overload;
    function DupTree(const T, Parent: IANTLRInterface): IANTLRInterface; overload;

    /// <summary>
    /// Tell me how to create a token for use with imaginary token nodes.
    /// For example, there is probably no input symbol associated with imaginary
    /// token DECL, but you need to create it as a payload or whatever for
    /// the DECL node as in ^(DECL type ID).
    ///
    /// If you care what the token payload objects' type is, you should
    /// override this method and any other createToken variant.
    /// </summary>
    function CreateToken(const TokenType: Integer; const Text: String): IToken; overload;

    /// <summary>
    /// Tell me how to create a token for use with imaginary token nodes.
    /// For example, there is probably no input symbol associated with imaginary
    /// token DECL, but you need to create it as a payload or whatever for
    /// the DECL node as in ^(DECL type ID).
    ///
    /// This is a variant of createToken where the new token is derived from
    /// an actual real input token.  Typically this is for converting '{'
    /// tokens to BLOCK etc...  You'll see
    ///
    ///    r : lc='{' ID+ '}' -> ^(BLOCK[$lc] ID+) ;
    ///
    /// If you care what the token payload objects' type is, you should
    /// override this method and any other createToken variant.
    /// </summary>
    function CreateToken(const FromToken: IToken): IToken; overload;
  end;

  /// <summary>
  /// A TreeAdaptor that works with any Tree implementation.  It provides
  /// really just factory methods; all the work is done by BaseTreeAdaptor.
  /// If you would like to have different tokens created than ClassicToken
  /// objects, you need to override this and then set the parser tree adaptor to
  /// use your subclass.
  ///
  /// To get your parser to build nodes of a different type, override
  /// Create(Token).
  /// </summary>
  ICommonTreeAdaptor = interface(IBaseTreeAdaptor)
  ['{B067EE7A-38EB-4156-9447-CDD6DDD6D13B}']
  end;

  /// <summary>
  /// A buffered stream of tree nodes.  Nodes can be from a tree of ANY kind.
  /// </summary>
  /// <remarks>
  /// This node stream sucks all nodes out of the tree specified in the
  /// constructor during construction and makes pointers into the tree
  /// using an array of Object pointers. The stream necessarily includes
  /// pointers to DOWN and UP and EOF nodes.
  ///
  /// This stream knows how to mark/release for backtracking.
  ///
  /// This stream is most suitable for tree interpreters that need to
  /// jump around a lot or for tree parsers requiring speed (at cost of memory).
  /// There is some duplicated functionality here with UnBufferedTreeNodeStream
  /// but just in bookkeeping, not tree walking etc...
  ///
  /// <see cref="UnBufferedTreeNodeStream"/>
  ///
  /// </remarks>
  ICommonTreeNodeStream = interface(ITreeNodeStream)
  ['{0112FB31-AA1E-471C-ADC3-D97AC5D77E05}']
    { Property accessors }
    function GetCurrentSymbol: IANTLRInterface;
    function GetTreeSource: IANTLRInterface;
    function GetSourceName: String;
    function GetTokenStream: ITokenStream;
    procedure SetTokenStream(const Value: ITokenStream);
    function GetTreeAdaptor: ITreeAdaptor;
    procedure SetTreeAdaptor(const Value: ITreeAdaptor);
    function GetHasUniqueNavigationNodes: Boolean;
    procedure SetHasUniqueNavigationNodes(const Value: Boolean);

    { Methods }
    /// <summary>
    /// Walk tree with depth-first-search and fill nodes buffer.
    /// Don't do DOWN, UP nodes if its a list (t is isNil).
    /// </summary>
    procedure FillBuffer(const T: IANTLRInterface);

    function Get(const I: Integer): IANTLRInterface;

    function LT(const K: Integer): IANTLRInterface;

    /// <summary>
    /// Look backwards k nodes
    /// </summary>
    function LB(const K: Integer): IANTLRInterface;

    /// <summary>
    /// Make stream jump to a new location, saving old location.
    /// Switch back with pop().
    /// </summary>
    procedure Push(const Index: Integer);

    /// <summary>
    /// Seek back to previous index saved during last Push() call.
    /// Return top of stack (return index).
    /// </summary>
    function Pop: Integer;

    procedure Reset;

    // Debugging
    function ToTokenString(const Start, Stop: Integer): String;
    function ToString(const Start, Stop: IANTLRInterface): String; overload;
    function ToString: String; overload;

    { Properties }
    property CurrentSymbol: IANTLRInterface read GetCurrentSymbol;

    /// <summary>
    /// Where is this stream pulling nodes from?  This is not the name, but
    /// the object that provides node objects.
    /// </summary>
    property TreeSource: IANTLRInterface read GetTreeSource;

    property SourceName: String read GetSourceName;
    property TokenStream: ITokenStream read GetTokenStream write SetTokenStream;
    property TreeAdaptor: ITreeAdaptor read GetTreeAdaptor write SetTreeAdaptor;
    property HasUniqueNavigationNodes: Boolean read GetHasUniqueNavigationNodes write SetHasUniqueNavigationNodes;
  end;

  /// <summary>
  /// A record of the rules used to Match a token sequence.  The tokens
  /// end up as the leaves of this tree and rule nodes are the interior nodes.
  /// This really adds no functionality, it is just an alias for CommonTree
  /// that is more meaningful (specific) and holds a String to display for a node.
  /// </summary>
  IParseTree = interface(IANTLRInterface)
  ['{1558F260-CAF8-4488-A242-3559BCE4E573}']
    { Methods }

    // Emit a token and all hidden nodes before.  EOF node holds all
    // hidden tokens after last real token.
    function ToStringWithHiddenTokens: String;

    // Print out the leaves of this tree, which means printing original
    // input back out.
    function ToInputString: String;

    procedure _ToStringLeaves(const Buf: TStringBuilder);
  end;

  /// <summary>
  /// A generic list of elements tracked in an alternative to be used in
  /// a -> rewrite rule.  We need to subclass to fill in the next() method,
  /// which returns either an AST node wrapped around a token payload or
  /// an existing subtree.
  ///
  /// Once you start next()ing, do not try to add more elements.  It will
  /// break the cursor tracking I believe.
  ///
  /// <see cref="RewriteRuleSubtreeStream"/>
  /// <see cref="RewriteRuleTokenStream"/>
  ///
  /// TODO: add mechanism to detect/puke on modification after reading from stream
  /// </summary>
  IRewriteRuleElementStream = interface(IANTLRInterface)
  ['{3CB6C521-F583-40DC-A1E3-4D7D57B98C74}']
    { Property accessors }
    function GetDescription: String;

    { Methods }
    procedure Add(const El: IANTLRInterface);

    /// <summary>
    /// Reset the condition of this stream so that it appears we have
    /// not consumed any of its elements.  Elements themselves are untouched.
    /// </summary>
    /// <remarks>
    /// Once we reset the stream, any future use will need duplicates.  Set
    /// the dirty bit.
    /// </remarks>
    procedure Reset;

    function HasNext: Boolean;

    /// <summary>
    /// Return the next element in the stream.
    /// </summary>
    function NextTree: IANTLRInterface;
    function NextNode: IANTLRInterface;

    function Size: Integer;

    { Properties }
    property Description: String read GetDescription;
  end;

  /// <summary>
  /// Queues up nodes matched on left side of -> in a tree parser. This is
  /// the analog of RewriteRuleTokenStream for normal parsers.
  /// </summary>
  IRewriteRuleNodeStream = interface(IRewriteRuleElementStream)
  ['{F60D1D36-FE13-4312-99DA-11E5F4BEBB66}']
    { Methods }
    function NextNode: IANTLRInterface;
  end;

  IRewriteRuleSubtreeStream = interface(IRewriteRuleElementStream)
  ['{C6BDA145-D926-45BC-B293-67490D72829B}']
    { Methods }

    /// <summary>
    /// Treat next element as a single node even if it's a subtree.
    /// </summary>
    /// <remarks>
    /// This is used instead of next() when the result has to be a
    /// tree root node.  Also prevents us from duplicating recently-added
    /// children; e.g., ^(type ID)+ adds ID to type and then 2nd iteration
    /// must dup the type node, but ID has been added.
    ///
    /// Referencing a rule result twice is ok; dup entire tree as
    /// we can't be adding trees as root; e.g., expr expr.
    /// </remarks>
    function NextNode: IANTLRInterface;
  end;

  IRewriteRuleTokenStream = interface(IRewriteRuleElementStream)
  ['{4D46AB00-7A19-4F69-B159-1EF09DB8C09C}']
    /// <summary>
    /// Get next token from stream and make a node for it.
    /// </summary>
    /// <remarks>
    /// ITreeAdaptor.Create() returns an object, so no further restrictions possible.
    /// </remarks>
    function NextNode: IANTLRInterface;

    function NextToken: IToken;
  end;

  /// <summary>
  /// A parser for a stream of tree nodes.  "tree grammars" result in a subclass
  /// of this.  All the error reporting and recovery is shared with Parser via
  /// the BaseRecognizer superclass.
  /// </summary>
  ITreeParser = interface(IBaseRecognizer)
  ['{20611FB3-9830-444D-B385-E8C2D094484B}']
    { Property accessors }
    function GetTreeNodeStream: ITreeNodeStream;
    procedure SetTreeNodeStream(const Value: ITreeNodeStream);

    { Methods }
    procedure TraceIn(const RuleName: String; const RuleIndex: Integer);
    procedure TraceOut(const RuleName: String; const RuleIndex: Integer);

    { Properties }
    property TreeNodeStream: ITreeNodeStream read GetTreeNodeStream write SetTreeNodeStream;
  end;

  ITreePatternLexer = interface(IANTLRInterface)
  ['{C3FEC614-9E6F-48D2-ABAB-59FC83D8BC2F}']
    { Methods }
    function NextToken: Integer;
    function SVal: String;
  end;

  IContextVisitor = interface(IANTLRInterface)
  ['{92B80D23-C63E-48B4-A9CD-EC2639317E43}']
    { Methods }
    procedure Visit(const T, Parent: IANTLRInterface; const ChildIndex: Integer;
      const Labels: IDictionary<String, IANTLRInterface>);
  end;

  /// <summary>
  /// Build and navigate trees with this object.  Must know about the names
  /// of tokens so you have to pass in a map or array of token names (from which
  /// this class can build the map).  I.e., Token DECL means nothing unless the
  /// class can translate it to a token type.
  /// </summary>
  /// <remarks>
  /// In order to create nodes and navigate, this class needs a TreeAdaptor.
  ///
  /// This class can build a token type -> node index for repeated use or for
  /// iterating over the various nodes with a particular type.
  ///
  /// This class works in conjunction with the TreeAdaptor rather than moving
  /// all this functionality into the adaptor.  An adaptor helps build and
  /// navigate trees using methods.  This class helps you do it with string
  /// patterns like "(A B C)".  You can create a tree from that pattern or
  /// match subtrees against it.
  /// </remarks>
  ITreeWizard = interface(IANTLRInterface)
  ['{4F440E19-893A-4E52-A979-E5377EAFA3B8}']
    { Methods }
    /// <summary>
    /// Compute a Map&lt;String, Integer&gt; that is an inverted index of
    /// tokenNames (which maps int token types to names).
    /// </summary>
    function ComputeTokenTypes(const TokenNames: TStringArray): IDictionary<String, Integer>;

    /// <summary>
    /// Using the map of token names to token types, return the type.
    /// </summary>
    function GetTokenType(const TokenName: String): Integer;

    /// <summary>
    /// Walk the entire tree and make a node name to nodes mapping.
    /// </summary>
    /// <remarks>
    /// For now, use recursion but later nonrecursive version may be
    /// more efficient.  Returns Map&lt;Integer, List&gt; where the List is
    /// of your AST node type.  The Integer is the token type of the node.
    ///
    /// TODO: save this index so that find and visit are faster
    /// </remarks>
    function Index(const T: IANTLRInterface): IDictionary<Integer, IList<IANTLRInterface>>;

    /// <summary>Return a List of tree nodes with token type ttype</summary>
    function Find(const T: IANTLRInterface; const TokenType: Integer): IList<IANTLRInterface>; overload;

    /// <summary>Return a List of subtrees matching pattern</summary>
    function Find(const T: IANTLRInterface; const Pattern: String): IList<IANTLRInterface>; overload;

    function FindFirst(const T: IANTLRInterface; const TokenType: Integer): IANTLRInterface; overload;
    function FindFirst(const T: IANTLRInterface; const Pattern: String): IANTLRInterface; overload;

    /// <summary>
    /// Visit every ttype node in t, invoking the visitor.
    /// </summary>
    /// <remarks>
    /// This is a quicker
    /// version of the general visit(t, pattern) method.  The labels arg
    /// of the visitor action method is never set (it's null) since using
    /// a token type rather than a pattern doesn't let us set a label.
    /// </remarks>
    procedure Visit(const T: IANTLRInterface; const TokenType: Integer;
      const Visitor: IContextVisitor); overload;

    /// <summary>
    /// For all subtrees that match the pattern, execute the visit action.
    /// </summary>
    /// <remarks>
    /// The implementation uses the root node of the pattern in combination
    /// with visit(t, ttype, visitor) so nil-rooted patterns are not allowed.
    /// Patterns with wildcard roots are also not allowed.
    /// </remarks>
    procedure Visit(const T: IANTLRInterface; const Pattern: String;
      const Visitor: IContextVisitor); overload;

    /// <summary>
    /// Given a pattern like (ASSIGN %lhs:ID %rhs:.) with optional labels
    /// on the various nodes and '.' (dot) as the node/subtree wildcard,
    /// return true if the pattern matches and fill the labels Map with
    /// the labels pointing at the appropriate nodes.  Return false if
    /// the pattern is malformed or the tree does not match.
    /// </summary>
    /// <remarks>
    /// If a node specifies a text arg in pattern, then that must match
    /// for that node in t.
    ///
    /// TODO: what's a better way to indicate bad pattern? Exceptions are a hassle
    /// </remarks>
    function Parse(const T: IANTLRInterface; const Pattern: String;
      const Labels: IDictionary<String, IANTLRInterface>): Boolean; overload;
    function Parse(const T: IANTLRInterface; const Pattern: String): Boolean; overload;

    /// <summary>
    /// Create a tree or node from the indicated tree pattern that closely
    /// follows ANTLR tree grammar tree element syntax:
    ///
    ///   (root child1 ... child2).
    ///
    /// </summary>
    /// <remarks>
    /// You can also just pass in a node: ID
    ///
    /// Any node can have a text argument: ID[foo]
    /// (notice there are no quotes around foo--it's clear it's a string).
    ///
    /// nil is a special name meaning "give me a nil node".  Useful for
    /// making lists: (nil A B C) is a list of A B C.
    /// </remarks>
    function CreateTreeOrNode(const Pattern: String): IANTLRInterface;

    /// <summary>
    /// Compare type, structure, and text of two trees, assuming adaptor in
    /// this instance of a TreeWizard.
    /// </summary>
    function Equals(const T1, T2: IANTLRInterface): Boolean; overload;

    /// <summary>
    /// Compare t1 and t2; return true if token types/text, structure match exactly.
    /// The trees are examined in their entirety so that (A B) does not match
    /// (A B C) nor (A (B C)).
    /// </summary>
    /// <remarks>
    /// TODO: allow them to pass in a comparator
    /// TODO: have a version that is nonstatic so it can use instance adaptor
    ///
    /// I cannot rely on the tree node's equals() implementation as I make
    /// no constraints at all on the node types nor interface etc...
    /// </remarks>
    function Equals(const T1, T2: IANTLRInterface; const Adaptor: ITreeAdaptor): Boolean; overload;
  end;

  ITreePatternParser = interface(IANTLRInterface)
  ['{0CE3DF2A-7E4C-4A7C-8FE8-F1D7AFF97CAE}']
    { Methods }
    function Pattern: IANTLRInterface;
    function ParseTree: IANTLRInterface;
    function ParseNode: IANTLRInterface;
  end;

  /// <summary>
  /// This is identical to the ParserRuleReturnScope except that
  /// the start property is a tree node and not a Token object
  /// when you are parsing trees.  To be generic the tree node types
  /// have to be Object :(
  /// </summary>
  ITreeRuleReturnScope = interface(IRuleReturnScope)
  ['{FA2B1766-34E5-4D92-8996-371D5CFED999}']
  end;

  /// <summary>
  /// A stream of tree nodes, accessing nodes from a tree of ANY kind.
  /// </summary>
  /// <remarks>
  /// No new nodes should be created in tree during the walk.  A small buffer
  /// of tokens is kept to efficiently and easily handle LT(i) calls, though
  /// the lookahead mechanism is fairly complicated.
  ///
  /// For tree rewriting during tree parsing, this must also be able
  /// to replace a set of children without "losing its place".
  /// That part is not yet implemented.  Will permit a rule to return
  /// a different tree and have it stitched into the output tree probably.
  ///
  /// <see cref="CommonTreeNodeStream"/>
  ///
  /// </remarks>
  IUnBufferedTreeNodeStream = interface(ITreeNodeStream)
  ['{E46367AD-ED41-4D97-824E-575A48F7435D}']
    { Property accessors }
    function GetHasUniqueNavigationNodes: Boolean;
    procedure SetHasUniqueNavigationNodes(const Value: Boolean);
    function GetCurrent: IANTLRInterface;
    function GetTokenStream: ITokenStream;
    procedure SetTokenStream(const Value: ITokenStream);

    { Methods }
    procedure Reset;
    function MoveNext: Boolean;

    { Properties }
    property HasUniqueNavigationNodes: Boolean read GetHasUniqueNavigationNodes write SetHasUniqueNavigationNodes;
    property Current: IANTLRInterface read GetCurrent;
    property TokenStream: ITokenStream read GetTokenStream write SetTokenStream;
  end;

  /// <summary>Base class for all exceptions thrown during AST rewrite construction.</summary>
  /// <remarks>
  /// This signifies a case where the cardinality of two or more elements
  /// in a subrule are different: (ID INT)+ where |ID|!=|INT|
  /// </remarks>
  ERewriteCardinalityException = class(Exception)
  strict private
    FElementDescription: String;
  public
    constructor Create(const AElementDescription: String);

    property ElementDescription: String read FElementDescription write FElementDescription;
  end;

  /// <summary>
  /// No elements within a (...)+ in a rewrite rule
  /// </summary>
  ERewriteEarlyExitException = class(ERewriteCardinalityException)
    // No new declarations
  end;

  /// <summary>
  /// Ref to ID or expr but no tokens in ID stream or subtrees in expr stream
  /// </summary>
  ERewriteEmptyStreamException = class(ERewriteCardinalityException)
    // No new declarations
  end;

type
  TTree = class sealed
  strict private
    class var
      FINVALID_NODE: ITree;
  private
    class procedure Initialize; static;
  public
    class property INVALID_NODE: ITree read FINVALID_NODE;
  end;

  TBaseTree = class abstract(TANTLRObject, IBaseTree, ITree)
  protected
    { ITree / IBaseTree }
    function GetParent: ITree; virtual;
    procedure SetParent(const Value: ITree); virtual;
    function GetChildIndex: Integer; virtual;
    procedure SetChildIndex(const Value: Integer); virtual;
    function GetTokenType: Integer; virtual; abstract;
    function GetText: String; virtual; abstract;
    function GetTokenStartIndex: Integer; virtual; abstract;
    procedure SetTokenStartIndex(const Value: Integer); virtual; abstract;
    function GetTokenStopIndex: Integer; virtual; abstract;
    procedure SetTokenStopIndex(const Value: Integer); virtual; abstract;
    function DupNode: ITree; virtual; abstract;
    function ToStringTree: String; virtual;
    function GetChildCount: Integer; virtual;
    function GetIsNil: Boolean; virtual;
    function GetLine: Integer; virtual;
    function GetCharPositionInLine: Integer; virtual;
    function GetChild(const I: Integer): ITree; virtual;
    procedure AddChild(const T: ITree);
    function DeleteChild(const I: Integer): IANTLRInterface;
    procedure FreshenParentAndChildIndexes; overload;
    procedure ReplaceChildren(const StartChildIndex, StopChildIndex: Integer;
      const T: IANTLRInterface);
  protected
    { IBaseTree }
    function GetChildren: IList<IBaseTree>;
    procedure AddChildren(const Kids: IList<IBaseTree>);
    procedure SetChild(const I: Integer; const T: ITree); virtual;
    procedure FreshenParentAndChildIndexes(const Offset: Integer); overload;
    procedure SanityCheckParentAndChildIndexes; overload; virtual;
    procedure SanityCheckParentAndChildIndexes(const Parent: ITree;
      const I: Integer); overload; virtual;
  strict protected
    FChildren: IList<IBaseTree>;

    /// <summary>Override in a subclass to change the impl of children list </summary>
    function CreateChildrenList: IList<IBaseTree>; virtual;

  public
    constructor Create; overload;

    /// <summary>Create a new node from an existing node does nothing for BaseTree
    /// as there are no fields other than the children list, which cannot
    /// be copied as the children are not considered part of this node.
    /// </summary>
    constructor Create(const ANode: ITree); overload;

    function ToString: String; override; abstract;
  end;

  TCommonTree = class(TBaseTree, ICommonTree)
  strict protected
    /// <summary>A single token is the payload </summary>
    FToken: IToken;

    /// <summary>
    /// What token indexes bracket all tokens associated with this node
    /// and below?
    /// </summary>
    FStartIndex: Integer;
    FStopIndex: Integer;

    /// <summary>Who is the parent node of this node; if null, implies node is root</summary>
    /// <remarks>
    /// FParent should be of type ICommonTree, but that would introduce a
    /// circular reference because the tree also maintains links to it's
    /// children. This circular reference would cause a memory leak because
    /// the reference count will never reach 0. This is avoided by making
    /// FParent a regular pointer and letting the GetParent and SetParent
    /// property accessors do the conversion to/from ICommonTree.
    /// </remarks>
    FParent: Pointer; { ICommonTree ; }

    /// <summary>What index is this node in the child list? Range: 0..n-1</summary>
    FChildIndex: Integer;
  protected
    { ITree / IBaseTree }
    function GetIsNil: Boolean; override;
    function GetTokenType: Integer; override;
    function GetText: String; override;
    function GetLine: Integer; override;
    function GetCharPositionInLine: Integer; override;
    function GetTokenStartIndex: Integer; override;
    procedure SetTokenStartIndex(const Value: Integer); override;
    function GetTokenStopIndex: Integer; override;
    procedure SetTokenStopIndex(const Value: Integer); override;
    function GetChildIndex: Integer; override;
    procedure SetChildIndex(const Value: Integer); override;
    function GetParent: ITree; override;
    procedure SetParent(const Value: ITree); override;
    function DupNode: ITree; override;
  protected
    { ICommonTree }
    function GetToken: IToken;
    function GetStartIndex: Integer;
    procedure SetStartIndex(const Value: Integer);
    function GetStopIndex: Integer;
    procedure SetStopIndex(const Value: Integer);
  public
    constructor Create; overload;
    constructor Create(const ANode: ICommonTree); overload;
    constructor Create(const AToken: IToken); overload;

    function ToString: String; override;
  end;

  TCommonErrorNode = class(TCommonTree, ICommonErrorNode)
  strict private
    FInput: IIntStream;
    FStart: IToken;
    FStop: IToken;
    FTrappedException: ERecognitionException;
  protected
    { ITree / IBaseTree }
    function GetIsNil: Boolean; override;
    function GetTokenType: Integer; override;
    function GetText: String; override;
  public
    constructor Create(const AInput: ITokenStream; const AStart, AStop: IToken;
      const AException: ERecognitionException);

    function ToString: String; override;
  end;

  TBaseTreeAdaptor = class abstract(TANTLRObject, IBaseTreeAdaptor, ITreeAdaptor)
  strict private
    /// <summary>A map of tree node to unique IDs.</summary>
    FTreeToUniqueIDMap: IDictionary<IANTLRInterface, Integer>;

    /// <summary>Next available unique ID.</summary>
    FUniqueNodeID: Integer;
  protected
    { ITreeAdaptor }
    function CreateNode(const Payload: IToken): IANTLRInterface; overload; virtual; abstract;
    function DupNode(const TreeNode: IANTLRInterface): IANTLRInterface; virtual; abstract;
    function DupTree(const Tree: IANTLRInterface): IANTLRInterface; overload; virtual;
    function GetNilNode: IANTLRInterface; virtual;
    function ErrorNode(const Input: ITokenStream; const Start, Stop: IToken;
      const E: ERecognitionException): IANTLRInterface; virtual;
    function IsNil(const Tree: IANTLRInterface): Boolean; virtual;
    procedure AddChild(const T, Child: IANTLRInterface); virtual;
    function BecomeRoot(const NewRoot, OldRoot: IANTLRInterface): IANTLRInterface; overload; virtual;
    function RulePostProcessing(const Root: IANTLRInterface): IANTLRInterface; virtual;
    function GetUniqueID(const Node: IANTLRInterface): Integer;
    function BecomeRoot(const NewRoot: IToken; const OldRoot: IANTLRInterface): IANTLRInterface; overload; virtual;
    function CreateNode(const TokenType: Integer; const FromToken: IToken): IANTLRInterface; overload; virtual;
    function CreateNode(const TokenType: Integer; const FromToken: IToken;
      const Text: String): IANTLRInterface; overload; virtual;
    function CreateNode(const TokenType: Integer; const Text: String): IANTLRInterface; overload; virtual;
    function GetNodeType(const T: IANTLRInterface): Integer; virtual;
    procedure SetNodeType(const T: IANTLRInterface; const NodeType: Integer); virtual;
    function GetNodeText(const T: IANTLRInterface): String; virtual;
    procedure SetNodeText(const T: IANTLRInterface; const Text: String); virtual;
    function GetToken(const TreeNode: IANTLRInterface): IToken; virtual; abstract;
    procedure SetTokenBoundaries(const T: IANTLRInterface; const StartToken,
      StopToken: IToken); virtual; abstract;
    function GetTokenStartIndex(const T: IANTLRInterface): Integer; virtual; abstract;
    function GetTokenStopIndex(const T: IANTLRInterface): Integer; virtual; abstract;
    function GetChild(const T: IANTLRInterface; const I: Integer): IANTLRInterface; virtual;
    procedure SetChild(const T: IANTLRInterface; const I: Integer; const Child: IANTLRInterface); virtual;
    function DeleteChild(const T: IANTLRInterface; const I: Integer): IANTLRInterface; virtual;
    function GetChildCount(const T: IANTLRInterface): Integer; virtual;
    function GetParent(const T: IANTLRInterface): IANTLRInterface; virtual; abstract;
    procedure SetParent(const T, Parent: IANTLRInterface); virtual; abstract;
    function GetChildIndex(const T: IANTLRInterface): Integer; virtual; abstract;
    procedure SetChildIdex(const T: IANTLRInterface; const Index: Integer); virtual; abstract;
    procedure ReplaceChildren(const Parent: IANTLRInterface; const StartChildIndex,
      StopChildIndex: Integer; const T: IANTLRInterface); virtual; abstract;
  protected
    { IBaseTreeAdaptor }
    function DupTree(const T, Parent: IANTLRInterface): IANTLRInterface; overload; virtual;
    function CreateToken(const TokenType: Integer; const Text: String): IToken; overload; virtual; abstract;
    function CreateToken(const FromToken: IToken): IToken; overload; virtual; abstract;
  public
    constructor Create;
  end;

  TCommonTreeAdaptor = class(TBaseTreeAdaptor, ICommonTreeAdaptor)
  protected
    { ITreeAdaptor }
    function DupNode(const TreeNode: IANTLRInterface): IANTLRInterface; override;
    function CreateNode(const Payload: IToken): IANTLRInterface; overload; override;
    procedure SetTokenBoundaries(const T: IANTLRInterface; const StartToken,
      StopToken: IToken); override;
    function GetTokenStartIndex(const T: IANTLRInterface): Integer; override;
    function GetTokenStopIndex(const T: IANTLRInterface): Integer; override;
    function GetNodeText(const T: IANTLRInterface): String; override;
    function GetToken(const TreeNode: IANTLRInterface): IToken; override;
    function GetNodeType(const T: IANTLRInterface): Integer; override;
    function GetChild(const T: IANTLRInterface; const I: Integer): IANTLRInterface; override;
    function GetChildCount(const T: IANTLRInterface): Integer; override;
    function GetParent(const T: IANTLRInterface): IANTLRInterface; override;
    procedure SetParent(const T, Parent: IANTLRInterface); override;
    function GetChildIndex(const T: IANTLRInterface): Integer; override;
    procedure SetChildIdex(const T: IANTLRInterface; const Index: Integer); override;
    procedure ReplaceChildren(const Parent: IANTLRInterface; const StartChildIndex,
      StopChildIndex: Integer; const T: IANTLRInterface); override;
  protected
    { IBaseTreeAdaptor }
    function CreateToken(const TokenType: Integer; const Text: String): IToken; overload; override;
    function CreateToken(const FromToken: IToken): IToken; overload; override;
  end;

  TCommonTreeNodeStream = class(TANTLRObject, ICommonTreeNodeStream, ITreeNodeStream)
  public
    const
      DEFAULT_INITIAL_BUFFER_SIZE = 100;
      INITIAL_CALL_STACK_SIZE = 10;
  strict private
    // all these navigation nodes are shared and hence they
    // cannot contain any line/column info
    FDown: IANTLRInterface;
    FUp: IANTLRInterface;
    FEof: IANTLRInterface;

    /// <summary>
    /// The complete mapping from stream index to tree node. This buffer
    /// includes pointers to DOWN, UP, and EOF nodes.
    ///
    /// It is built upon ctor invocation.  The elements are type Object
    /// as we don't what the trees look like. Load upon first need of
    /// the buffer so we can set token types of interest for reverseIndexing.
    /// Slows us down a wee bit  to do all of the if p==-1 testing everywhere though.
    /// </summary>
    FNodes: IList<IANTLRInterface>;

    /// <summary>Pull nodes from which tree? </summary>
    FRoot: IANTLRInterface;

    /// <summary>IF this tree (root) was created from a token stream, track it</summary>
    FTokens: ITokenStream;

    /// <summary>What tree adaptor was used to build these trees</summary>
    FAdaptor: ITreeAdaptor;

    /// <summary>
    /// Reuse same DOWN, UP navigation nodes unless this is true
    /// </summary>
    FUniqueNavigationNodes: Boolean;

    /// <summary>
    /// The index into the nodes list of the current node (next node
    /// to consume).  If -1, nodes array not filled yet.
    /// </summary>
    FP: Integer;

    /// <summary>
    /// Track the last mark() call result value for use in rewind().
    /// </summary>
    FLastMarker: Integer;

    /// <summary>
    /// Stack of indexes used for push/pop calls
    /// </summary>
    FCalls: IStackList<Integer>;
  protected
    { IIntStream }
    function GetSourceName: String; virtual;

    procedure Consume; virtual;
    function LA(I: Integer): Integer; virtual;
    function LAChar(I: Integer): Char;
    function Mark: Integer; virtual;
    function Index: Integer; virtual;
    procedure Rewind(const Marker: Integer); overload; virtual;
    procedure Rewind; overload;
    procedure Release(const Marker: Integer); virtual;
    procedure Seek(const Index: Integer); virtual;
    function Size: Integer; virtual;
  protected
    { ITreeNodeStream }
    function GetTreeSource: IANTLRInterface; virtual;
    function GetTokenStream: ITokenStream; virtual;
    function GetTreeAdaptor: ITreeAdaptor;
    procedure SetHasUniqueNavigationNodes(const Value: Boolean);

    function Get(const I: Integer): IANTLRInterface;
    function LT(const K: Integer): IANTLRInterface;
    function ToString(const Start, Stop: IANTLRInterface): String; reintroduce; overload;
    procedure ReplaceChildren(const Parent: IANTLRInterface; const StartChildIndex,
      StopChildIndex: Integer; const T: IANTLRInterface);
  protected
    { ICommonTreeNodeStream }
    function GetCurrentSymbol: IANTLRInterface; virtual;
    procedure SetTokenStream(const Value: ITokenStream); virtual;
    procedure SetTreeAdaptor(const Value: ITreeAdaptor);
    function GetHasUniqueNavigationNodes: Boolean;

    procedure FillBuffer(const T: IANTLRInterface); overload;
    function LB(const K: Integer): IANTLRInterface;
    procedure Push(const Index: Integer);
    function Pop: Integer;
    procedure Reset;
    function ToTokenString(const Start, Stop: Integer): String;
  strict protected
    /// <summary>
    /// Walk tree with depth-first-search and fill nodes buffer.
    /// Don't do DOWN, UP nodes if its a list (t is isNil).
    /// </summary>
    procedure FillBuffer; overload;

    /// <summary>
    /// As we flatten the tree, we use UP, DOWN nodes to represent
    /// the tree structure.  When debugging we need unique nodes
    /// so instantiate new ones when uniqueNavigationNodes is true.
    /// </summary>
    procedure AddNavigationNode(const TokenType: Integer);

    /// <summary>
    /// Returns the stream index for the spcified node in the range 0..n-1 or,
    /// -1 if node not found.
    /// </summary>
    function GetNodeIndex(const Node: IANTLRInterface): Integer;
  public
    constructor Create; overload;
    constructor Create(const ATree: IANTLRInterface); overload;
    constructor Create(const AAdaptor: ITreeAdaptor;
      const ATree: IANTLRInterface); overload;
    constructor Create(const AAdaptor: ITreeAdaptor;
      const ATree: IANTLRInterface; const AInitialBufferSize: Integer); overload;

    function ToString: String; overload; override;
  end;

  TParseTree = class(TBaseTree, IParseTree)
  strict private
    FPayload: IANTLRInterface;
    FHiddenTokens: IList<IToken>;
  protected
    { ITree / IBaseTree }
    function GetTokenType: Integer; override;
    function GetText: String; override;
    function GetTokenStartIndex: Integer; override;
    procedure SetTokenStartIndex(const Value: Integer); override;
    function GetTokenStopIndex: Integer; override;
    procedure SetTokenStopIndex(const Value: Integer); override;
    function DupNode: ITree; override;
  protected
    { IParseTree }
    function ToStringWithHiddenTokens: String;
    function ToInputString: String;
    procedure _ToStringLeaves(const Buf: TStringBuilder);
  public
    constructor Create(const ALabel: IANTLRInterface);

    function ToString: String; override;
  end;

  TRewriteRuleElementStream = class abstract(TANTLRObject, IRewriteRuleElementStream)
  private
    /// <summary>
    /// Cursor 0..n-1.  If singleElement!=null, cursor is 0 until you next(),
    /// which bumps it to 1 meaning no more elements.
    /// </summary>
    FCursor: Integer;

    /// <summary>
    /// Track single elements w/o creating a list.  Upon 2nd add, alloc list
    /// </summary>
    FSingleElement: IANTLRInterface;

    /// <summary>
    /// The list of tokens or subtrees we are tracking
    /// </summary>
    FElements: IList<IANTLRInterface>;

    /// <summary>
    /// Tracks whether a node or subtree has been used in a stream
    /// </summary>
    /// <remarks>
    /// Once a node or subtree has been used in a stream, it must be dup'd
    /// from then on.  Streams are reset after subrules so that the streams
    /// can be reused in future subrules.  So, reset must set a dirty bit.
    /// If dirty, then next() always returns a dup.
    /// </remarks>
    FDirty: Boolean;

    /// <summary>
    /// The element or stream description; usually has name of the token or
    /// rule reference that this list tracks.  Can include rulename too, but
    /// the exception would track that info.
    /// </summary>
    FElementDescription: String;
    FAdaptor: ITreeAdaptor;
  protected
    { IRewriteRuleElementStream }
    function GetDescription: String;

    procedure Add(const El: IANTLRInterface);
    procedure Reset; virtual;
    function HasNext: Boolean;
    function NextTree: IANTLRInterface; virtual;
    function NextNode: IANTLRInterface; virtual; abstract;
    function Size: Integer;
  strict protected
    /// <summary>
    /// Do the work of getting the next element, making sure that
    /// it's a tree node or subtree.
    /// </summary>
    /// <remarks>
    /// Deal with the optimization of single-element list versus
    /// list of size > 1.  Throw an exception if the stream is
    /// empty or we're out of elements and size>1.
    /// </remarks>
    function _Next: IANTLRInterface;

    /// <summary>
    /// Ensure stream emits trees; tokens must be converted to AST nodes.
    /// AST nodes can be passed through unmolested.
    /// </summary>
    function ToTree(const El: IANTLRInterface): IANTLRInterface; virtual;
  public
    constructor Create(const AAdaptor: ITreeAdaptor;
      const AElementDescription: String); overload;

    /// <summary>
    /// Create a stream with one element
    /// </summary>
    constructor Create(const AAdaptor: ITreeAdaptor;
      const AElementDescription: String; const AOneElement: IANTLRInterface); overload;

    /// <summary>
    /// Create a stream, but feed off an existing list
    /// </summary>
    constructor Create(const AAdaptor: ITreeAdaptor;
      const AElementDescription: String; const AElements: IList<IANTLRInterface>); overload;
  end;

  TRewriteRuleNodeStream = class(TRewriteRuleElementStream, IRewriteRuleNodeStream)
  protected
    { IRewriteRuleElementStream }
    function NextNode: IANTLRInterface; override;
    function ToTree(const El: IANTLRInterface): IANTLRInterface; override;
  end;

  TRewriteRuleSubtreeStream = class(TRewriteRuleElementStream, IRewriteRuleSubtreeStream)
  public
    type
      /// <summary>
      /// This delegate is used to allow the outfactoring of some common code.
      /// </summary>
      /// <param name="o">The to be processed object</param>
      TProcessHandler = function(const O: IANTLRInterface): IANTLRInterface of Object;
  strict private
    /// <summary>
    /// This method has the common code of two other methods, which differed in only one
    /// function call.
    /// </summary>
    /// <param name="ph">The delegate, which has the chosen function</param>
    /// <returns>The required object</returns>
    function FetchObject(const PH: TProcessHandler): IANTLRInterface;
    function DupNode(const O: IANTLRInterface): IANTLRInterface;

    /// <summary>
    /// Tests, if the to be returned object requires duplication
    /// </summary>
    /// <returns><code>true</code>, if positive, <code>false</code>, if negative.</returns>
    function RequiresDuplication: Boolean;

    /// <summary>
    /// When constructing trees, sometimes we need to dup a token or AST
    /// subtree. Dup'ing a token means just creating another AST node
    /// around it. For trees, you must call the adaptor.dupTree()
    /// unless the element is for a tree root; then it must be a node dup
    /// </summary>
    function Dup(const O: IANTLRInterface): IANTLRInterface;
  protected
    { IRewriteRuleElementStream }
    function NextNode: IANTLRInterface; override;
    function NextTree: IANTLRInterface; override;
  end;

  TRewriteRuleTokenStream = class(TRewriteRuleElementStream, IRewriteRuleTokenStream)
  protected
    { IRewriteRuleElementStream }
    function NextNode: IANTLRInterface; override;
    function NextToken: IToken;
    function ToTree(const El: IANTLRInterface): IANTLRInterface; override;
  end;

  TTreeParser = class(TBaseRecognizer, ITreeParser)
  public
    const
      DOWN = TToken.DOWN;
      UP = TToken.UP;
  strict private
    FInput: ITreeNodeStream;
  strict protected
    property Input: ITreeNodeStream read FInput;
  protected
    { IBaseRecognizer }
    function GetSourceName: String; override;
    procedure Reset; override;
    procedure MatchAny(const Input: IIntStream); override;
    function GetInput: IIntStream; override;
    function GetErrorHeader(const E: ERecognitionException): String; override;
    function GetErrorMessage(const E: ERecognitionException;
      const TokenNames: TStringArray): String; override;
  protected
    { ITreeParser }
    function GetTreeNodeStream: ITreeNodeStream; virtual;
    procedure SetTreeNodeStream(const Value: ITreeNodeStream); virtual;

    procedure TraceIn(const RuleName: String; const RuleIndex: Integer); reintroduce; overload; virtual;
    procedure TraceOut(const RuleName: String; const RuleIndex: Integer); reintroduce; overload; virtual;
  strict protected
    function GetCurrentInputSymbol(const Input: IIntStream): IANTLRInterface; override;
    function GetMissingSymbol(const Input: IIntStream;
      const E: ERecognitionException; const ExpectedTokenType: Integer;
      const Follow: IBitSet): IANTLRInterface; override;
    procedure Mismatch(const Input: IIntStream; const TokenType: Integer;
      const Follow: IBitSet); override;
  public
    constructor Create(const AInput: ITreeNodeStream); overload;
    constructor Create(const AInput: ITreeNodeStream;
      const AState: IRecognizerSharedState); overload;
  end;

  TTreePatternLexer = class(TANTLRObject, ITreePatternLexer)
  public
    const
      EOF = -1;
      START = 1;
      STOP = 2;
      ID = 3;
      ARG = 4;
      PERCENT = 5;
      COLON = 6;
      DOT = 7;
  strict private
    /// <summary>The tree pattern to lex like "(A B C)"</summary>
    FPattern: String;

    /// <summary>Index into input string</summary>
    FP: Integer;

    /// <summary>Current char</summary>
    FC: Integer;

    /// <summary>How long is the pattern in char?</summary>
    FN: Integer;

    /// <summary>
    /// Set when token type is ID or ARG (name mimics Java's StreamTokenizer)
    /// </summary>
    FSVal: TStringBuilder;

    FError: Boolean;
  protected
    { ITreePatternLexer }
    function NextToken: Integer;
    function SVal: String;
  strict protected
    procedure Consume;
  public
    constructor Create; overload;
    constructor Create(const APattern: String); overload;
    destructor Destroy; override;
  end;

  TTreeWizard = class(TANTLRObject, ITreeWizard)
  strict private
    FAdaptor: ITreeAdaptor;
    FTokenNameToTypeMap: IDictionary<String, Integer>;
  public
    type
      /// <summary>
      /// When using %label:TOKENNAME in a tree for parse(), we must track the label.
      /// </summary>
      ITreePattern = interface(ICommonTree)
      ['{893C6B4E-8474-4A1E-BEAA-8B704868401B}']
        { Property accessors }
        function GetHasTextArg: Boolean;
        procedure SetHasTextArg(const Value: Boolean);
        function GetTokenLabel: String;
        procedure SetTokenLabel(const Value: String);

        { Properties }
        property HasTextArg: Boolean read GetHasTextArg write SetHasTextArg;
        property TokenLabel: String read GetTokenLabel write SetTokenLabel;
      end;

      IWildcardTreePattern = interface(ITreePattern)
      ['{4778789A-5EAB-47E3-A05B-7F35CD87ECE4}']
      end;
    type
      TVisitor = class abstract(TANTLRObject, IContextVisitor)
      protected
        { IContextVisitor }
        procedure Visit(const T, Parent: IANTLRInterface; const ChildIndex: Integer;
          const Labels: IDictionary<String, IANTLRInterface>); overload;
      strict protected
        procedure Visit(const T: IANTLRInterface); overload; virtual; abstract;
      end;

      TTreePattern = class(TCommonTree, ITreePattern)
      strict private
        FLabel: String;
        FHasTextArg: Boolean;
      protected
        { ITreePattern }
        function GetHasTextArg: Boolean;
        procedure SetHasTextArg(const Value: Boolean);
        function GetTokenLabel: String;
        procedure SetTokenLabel(const Value: String);
      public
        function ToString: String; override;
      end;

      TWildcardTreePattern = class(TTreePattern, IWildcardTreePattern)

      end;

      /// <summary>
      /// This adaptor creates TreePattern objects for use during scan()
      /// </summary>
      TTreePatternTreeAdaptor = class(TCommonTreeAdaptor)
      protected
        { ITreeAdaptor }
        function CreateNode(const Payload: IToken): IANTLRInterface; overload; override;
      end;
  strict private
    type
      TRecordAllElementsVisitor = class sealed(TVisitor)
      strict private
        FList: IList<IANTLRInterface>;
      strict protected
        procedure Visit(const T: IANTLRInterface); override;
      public
        constructor Create(const AList: IList<IANTLRInterface>);
      end;

    type
      TPatternMatchingContextVisitor = class sealed(TANTLRObject, IContextVisitor)
      strict private
        FOwner: TTreeWizard;
        FPattern: ITreePattern;
        FList: IList<IANTLRInterface>;
      protected
        { IContextVisitor }
        procedure Visit(const T, Parent: IANTLRInterface; const ChildIndex: Integer;
          const Labels: IDictionary<String, IANTLRInterface>); overload;
      public
        constructor Create(const AOwner: TTreeWizard; const APattern: ITreePattern;
          const AList: IList<IANTLRInterface>);
      end;

    type
      TInvokeVisitorOnPatternMatchContextVisitor = class sealed(TANTLRObject, IContextVisitor)
      strict private
        FOwner: TTreeWizard;
        FPattern: ITreePattern;
        FVisitor: IContextVisitor;
        FLabels: IDictionary<String, IANTLRInterface>;
      protected
        { IContextVisitor }
        procedure Visit(const T, Parent: IANTLRInterface; const ChildIndex: Integer;
          const UnusedLabels: IDictionary<String, IANTLRInterface>); overload;
      public
        constructor Create(const AOwner: TTreeWizard; const APattern: ITreePattern;
          const AVisitor: IContextVisitor);
      end;
  protected
    { ITreeWizard }
    function ComputeTokenTypes(const TokenNames: TStringArray): IDictionary<String, Integer>;
    function GetTokenType(const TokenName: String): Integer;
    function Index(const T: IANTLRInterface): IDictionary<Integer, IList<IANTLRInterface>>;
    function Find(const T: IANTLRInterface; const TokenType: Integer): IList<IANTLRInterface>; overload;
    function Find(const T: IANTLRInterface; const Pattern: String): IList<IANTLRInterface>; overload;
    function FindFirst(const T: IANTLRInterface; const TokenType: Integer): IANTLRInterface; overload;
    function FindFirst(const T: IANTLRInterface; const Pattern: String): IANTLRInterface; overload;
    procedure Visit(const T: IANTLRInterface; const TokenType: Integer;
      const Visitor: IContextVisitor); overload;
    procedure Visit(const T: IANTLRInterface; const Pattern: String;
      const Visitor: IContextVisitor); overload;
    function Parse(const T: IANTLRInterface; const Pattern: String;
      const Labels: IDictionary<String, IANTLRInterface>): Boolean; overload;
    function Parse(const T: IANTLRInterface; const Pattern: String): Boolean; overload;
    function CreateTreeOrNode(const Pattern: String): IANTLRInterface;
    function Equals(const T1, T2: IANTLRInterface): Boolean; reintroduce; overload;
    function Equals(const T1, T2: IANTLRInterface;
      const Adaptor: ITreeAdaptor): Boolean; reintroduce; overload;
  strict protected
    function _Parse(const T1: IANTLRInterface; const T2: ITreePattern;
      const Labels: IDictionary<String, IANTLRInterface>): Boolean;

    /// <summary>Do the work for index</summary>
    procedure _Index(const T: IANTLRInterface;
      const M: IDictionary<Integer, IList<IANTLRInterface>>);

    /// <summary>Do the recursive work for visit</summary>
    procedure _Visit(const T, Parent: IANTLRInterface; const ChildIndex,
      TokenType: Integer; const Visitor: IContextVisitor);

    class function _Equals(const T1, T2: IANTLRInterface;
      const Adaptor: ITreeAdaptor): Boolean; static;
  public
    constructor Create(const AAdaptor: ITreeAdaptor); overload;
    constructor Create(const AAdaptor: ITreeAdaptor;
      const ATokenNameToTypeMap: IDictionary<String, Integer>); overload;
    constructor Create(const AAdaptor: ITreeAdaptor;
      const TokenNames: TStringArray); overload;
    constructor Create(const TokenNames: TStringArray); overload;
  end;

  TTreePatternParser = class(TANTLRObject, ITreePatternParser)
  strict private
    FTokenizer: ITreePatternLexer;
    FTokenType: Integer;
    FWizard: ITreeWizard;
    FAdaptor: ITreeAdaptor;
  protected
    { ITreePatternParser }
    function Pattern: IANTLRInterface;
    function ParseTree: IANTLRInterface;
    function ParseNode: IANTLRInterface;
  public
    constructor Create(const ATokenizer: ITreePatternLexer;
      const AWizard: ITreeWizard; const AAdaptor: ITreeAdaptor);
  end;

  TTreeRuleReturnScope = class(TRuleReturnScope, ITreeRuleReturnScope)
  strict private
    /// <summary>First node or root node of tree matched for this rule.</summary>
    FStart: IANTLRInterface;
  protected
    { IRuleReturnScope }
    function GetStart: IANTLRInterface; override;
    procedure SetStart(const Value: IANTLRInterface); override;
  end;

  TUnBufferedTreeNodeStream = class(TANTLRObject, IUnBufferedTreeNodeStream, ITreeNodeStream)
  public
    const
      INITIAL_LOOKAHEAD_BUFFER_SIZE = 5;
  strict protected
    type
      /// <summary>
      /// When walking ahead with cyclic DFA or for syntactic predicates,
      /// we need to record the state of the tree node stream.  This
      /// class wraps up the current state of the UnBufferedTreeNodeStream.
      /// Calling Mark() will push another of these on the markers stack.
      /// </summary>
      ITreeWalkState = interface(IANTLRInterface)
      ['{506D1014-53CF-4B9D-BE0E-1666E9C22091}']
        { Property accessors }
        function GetCurrentChildIndex: Integer;
        procedure SetCurrentChildIndex(const Value: Integer);
        function GetAbsoluteNodeIndex: Integer;
        procedure SetAbsoluteNodeIndex(const Value: Integer);
        function GetCurrentNode: IANTLRInterface;
        procedure SetCurrentNode(const Value: IANTLRInterface);
        function GetPreviousNode: IANTLRInterface;
        procedure SetPreviousNode(const Value: IANTLRInterface);
        function GetNodeStackSize: Integer;
        procedure SetNodeStackSize(const Value: Integer);
        function GetIndexStackSize: integer;
        procedure SetIndexStackSize(const Value: integer);
        function GetLookAhead: TANTLRInterfaceArray;
        procedure SetLookAhead(const Value: TANTLRInterfaceArray);

        { Properties }
        property CurrentChildIndex: Integer read GetCurrentChildIndex write SetCurrentChildIndex;
        property AbsoluteNodeIndex: Integer read GetAbsoluteNodeIndex write SetAbsoluteNodeIndex;
        property CurrentNode: IANTLRInterface read GetCurrentNode write SetCurrentNode;
        property PreviousNode: IANTLRInterface read GetPreviousNode write SetPreviousNode;
        ///<summary>Record state of the nodeStack</summary>
        property NodeStackSize: Integer read GetNodeStackSize write SetNodeStackSize;
        ///<summary>Record state of the indexStack</summary>
        property IndexStackSize: integer read GetIndexStackSize write SetIndexStackSize;
        property LookAhead: TANTLRInterfaceArray read GetLookAhead write SetLookAhead;
      end;

      TTreeWalkState = class(TANTLRObject, ITreeWalkState)
      strict private
        FCurrentChildIndex: Integer;
        FAbsoluteNodeIndex: Integer;
        FCurrentNode: IANTLRInterface;
        FPreviousNode: IANTLRInterface;
        ///<summary>Record state of the nodeStack</summary>
        FNodeStackSize: Integer;
        ///<summary>Record state of the indexStack</summary>
        FIndexStackSize: integer;
        FLookAhead: TANTLRInterfaceArray;
      protected
        { ITreeWalkState }
        function GetCurrentChildIndex: Integer;
        procedure SetCurrentChildIndex(const Value: Integer);
        function GetAbsoluteNodeIndex: Integer;
        procedure SetAbsoluteNodeIndex(const Value: Integer);
        function GetCurrentNode: IANTLRInterface;
        procedure SetCurrentNode(const Value: IANTLRInterface);
        function GetPreviousNode: IANTLRInterface;
        procedure SetPreviousNode(const Value: IANTLRInterface);
        function GetNodeStackSize: Integer;
        procedure SetNodeStackSize(const Value: Integer);
        function GetIndexStackSize: integer;
        procedure SetIndexStackSize(const Value: integer);
        function GetLookAhead: TANTLRInterfaceArray;
        procedure SetLookAhead(const Value: TANTLRInterfaceArray);
      end;
  strict private
    /// <summary>Reuse same DOWN, UP navigation nodes unless this is true</summary>
    FUniqueNavigationNodes: Boolean;

    /// <summary>Pull nodes from which tree? </summary>
    FRoot: IANTLRInterface;

    /// <summary>IF this tree (root) was created from a token stream, track it.</summary>
    FTokens: ITokenStream;

    /// <summary>What tree adaptor was used to build these trees</summary>
    FAdaptor: ITreeAdaptor;

    /// <summary>
    /// As we walk down the nodes, we must track parent nodes so we know
    /// where to go after walking the last child of a node.  When visiting
    /// a child, push current node and current index.
    /// </summary>
    FNodeStack: IStackList<IANTLRInterface>;

    /// <summary>
    /// Track which child index you are visiting for each node we push.
    /// TODO: pretty inefficient...use int[] when you have time
    /// </summary>
    FIndexStack: IStackList<Integer>;

    /// <summary>Which node are we currently visiting? </summary>
    FCurrentNode: IANTLRInterface;

    /// <summary>Which node did we visit last?  Used for LT(-1) calls. </summary>
    FPreviousNode: IANTLRInterface;

    /// <summary>
    /// Which child are we currently visiting?  If -1 we have not visited
    /// this node yet; next Consume() request will set currentIndex to 0.
    /// </summary>
    FCurrentChildIndex: Integer;

    /// <summary>
    /// What node index did we just consume?  i=0..n-1 for n node trees.
    /// IntStream.next is hence 1 + this value.  Size will be same.
    /// </summary>
    FAbsoluteNodeIndex: Integer;

    /// <summary>
    /// Buffer tree node stream for use with LT(i).  This list grows
    /// to fit new lookahead depths, but Consume() wraps like a circular
    /// buffer.
    /// </summary>
    FLookahead: TANTLRInterfaceArray;

    /// <summary>lookahead[head] is the first symbol of lookahead, LT(1). </summary>
    FHead: Integer;

    /// <summary>
    /// Add new lookahead at lookahead[tail].  tail wraps around at the
    /// end of the lookahead buffer so tail could be less than head.
    /// </summary>
    FTail: Integer;

    /// <summary>
    /// Calls to Mark() may be nested so we have to track a stack of them.
    /// The marker is an index into this stack. This is a List&lt;TreeWalkState&gt;.
    /// Indexed from 1..markDepth. A null is kept at index 0. It is created
    /// upon first call to Mark().
    /// </summary>
    FMarkers: IList<ITreeWalkState>;

    ///<summary>
    /// tracks how deep Mark() calls are nested
    /// </summary>
    FMarkDepth: Integer;

    ///<summary>
    /// Track the last Mark() call result value for use in Rewind().
    /// </summary>
    FLastMarker: Integer;

    // navigation nodes
    FDown: IANTLRInterface;
    FUp: IANTLRInterface;
    FEof: IANTLRInterface;

    FCurrentEnumerationNode: ITree;
  protected
    { IIntStream }
    function GetSourceName: String;

    procedure Consume; virtual;
    function LA(I: Integer): Integer; virtual;
    function LAChar(I: Integer): Char;
    function Mark: Integer; virtual;
    function Index: Integer; virtual;
    procedure Rewind(const Marker: Integer); overload; virtual;
    procedure Rewind; overload;
    procedure Release(const Marker: Integer); virtual;
    procedure Seek(const Index: Integer); virtual;
    function Size: Integer; virtual;
  protected
    { ITreeNodeStream }
    function GetTreeSource: IANTLRInterface; virtual;
    function GetTokenStream: ITokenStream;
    function GetTreeAdaptor: ITreeAdaptor;

    function Get(const I: Integer): IANTLRInterface; virtual;
    function LT(const K: Integer): IANTLRInterface; virtual;
    function ToString(const Start, Stop: IANTLRInterface): String; reintroduce; overload; virtual;
    procedure ReplaceChildren(const Parent: IANTLRInterface; const StartChildIndex,
      StopChildIndex: Integer; const T: IANTLRInterface);
  protected
    { IUnBufferedTreeNodeStream }
    function GetHasUniqueNavigationNodes: Boolean;
    procedure SetHasUniqueNavigationNodes(const Value: Boolean);
    function GetCurrent: IANTLRInterface; virtual;
    procedure SetTokenStream(const Value: ITokenStream);

    procedure Reset; virtual;

    /// <summary>
    /// Navigates to the next node found during a depth-first walk of root.
    /// Also, adds these nodes and DOWN/UP imaginary nodes into the lokoahead
    /// buffer as a side-effect.  Normally side-effects are bad, but because
    /// we can Emit many tokens for every MoveNext() call, it's pretty hard to
    /// use a single return value for that.  We must add these tokens to
    /// the lookahead buffer.
    ///
    /// This routine does *not* cause the 'Current' property to ever return the
    /// DOWN/UP nodes; those are only returned by the LT() method.
    ///
    /// Ugh.  This mechanism is much more complicated than a recursive
    /// solution, but it's the only way to provide nodes on-demand instead
    /// of walking once completely through and buffering up the nodes. :(
    /// </summary>
    function MoveNext: Boolean; virtual;
  strict protected
    /// <summary>Make sure we have at least k symbols in lookahead buffer </summary>
    procedure Fill(const K: Integer); virtual;
    function LookaheadSize: Integer;

    /// <summary>
    /// Add a node to the lookahead buffer.  Add at lookahead[tail].
    /// If you tail+1 == head, then we must create a bigger buffer
    /// and copy all the nodes over plus reset head, tail.  After
    /// this method, LT(1) will be lookahead[0].
    /// </summary>
    procedure AddLookahead(const Node: IANTLRInterface); virtual;

    procedure ToStringWork(const P, Stop: IANTLRInterface;
      const Buf: TStringBuilder); virtual;

    function HandleRootNode: IANTLRInterface; virtual;
    function VisitChild(const Child: Integer): IANTLRInterface; virtual;

    /// <summary>
    ///  Walk upwards looking for a node with more children to walk.
    /// </summary>
    procedure WalkBackToMostRecentNodeWithUnvisitedChildren; virtual;

    /// <summary>
    /// As we flatten the tree, we use UP, DOWN nodes to represent
    /// the tree structure.  When debugging we need unique nodes
    /// so instantiate new ones when uniqueNavigationNodes is true.
    /// </summary>
    procedure AddNavigationNode(const TokenType: Integer); virtual;
  public
    constructor Create; overload;
    constructor Create(const ATree: IANTLRInterface); overload;
    constructor Create(const AAdaptor: ITreeAdaptor; const ATree: IANTLRInterface); overload;

    function ToString: String; overload; override;
  end;

{ These functions return X or, if X = nil, an empty default instance }
function Def(const X: ICommonTree): ICommonTree; overload;

implementation

uses
  Math;

{ TTree }

class procedure TTree.Initialize;
begin
  FINVALID_NODE := TCommonTree.Create(TToken.INVALID_TOKEN);
end;

{ TBaseTree }

constructor TBaseTree.Create;
begin
  inherited;
end;

procedure TBaseTree.AddChild(const T: ITree);
var
  ChildTree: IBaseTree;
  C: IBaseTree;
begin
  if (T = nil) then
    Exit;

  ChildTree := T as IBaseTree;
  if ChildTree.IsNil then // t is an empty node possibly with children
  begin
    if Assigned(FChildren) and SameObj(FChildren, ChildTree.Children) then
      raise EInvalidOperation.Create('Attempt to add child list to itself');

    // just add all of childTree's children to this
    if Assigned(ChildTree.Children) then
    begin
      if Assigned(FChildren) then // must copy, this has children already
      begin
        for C in ChildTree.Children do
        begin
          FChildren.Add(C);
          // handle double-link stuff for each child of nil root
          C.Parent := Self;
          C.ChildIndex := FChildren.Count - 1;
        end;
      end
      else begin
        // no children for this but t has children; just set pointer
        // call general freshener routine
        FChildren := ChildTree.Children;
        FreshenParentAndChildIndexes;
      end;
    end;
  end
  else
  begin
    // child is not nil (don't care about children)
    if (FChildren = nil) then
    begin
      FChildren := CreateChildrenList; // create children list on demand
    end;
    FChildren.Add(ChildTree);
    ChildTree.Parent := Self;
    ChildTree.ChildIndex := FChildren.Count - 1;
  end;
end;

procedure TBaseTree.AddChildren(const Kids: IList<IBaseTree>);
var
  T: IBaseTree;
begin
  for T in Kids do
    AddChild(T);
end;

constructor TBaseTree.Create(const ANode: ITree);
begin
  Create;
  // No default implementation
end;

function TBaseTree.CreateChildrenList: IList<IBaseTree>;
begin
  Result := TList<IBaseTree>.Create;
end;

function TBaseTree.DeleteChild(const I: Integer): IANTLRInterface;
begin
  if (FChildren = nil) then
    Result := nil
  else
  begin
    Result := FChildren[I];
    FChildren.Delete(I);
    // walk rest and decrement their child indexes
    FreshenParentAndChildIndexes(I);
  end;
end;

procedure TBaseTree.FreshenParentAndChildIndexes(const Offset: Integer);
var
  N, C: Integer;
  Child: ITree;
begin
  N := GetChildCount;
  for C := Offset to N - 1 do
  begin
    Child := GetChild(C);
    Child.ChildIndex := C;
    Child.Parent := Self;
  end;
end;

procedure TBaseTree.FreshenParentAndChildIndexes;
begin
  FreshenParentAndChildIndexes(0);
end;

function TBaseTree.GetCharPositionInLine: Integer;
begin
  Result := 0;
end;

function TBaseTree.GetChild(const I: Integer): ITree;
begin
  if (FChildren = nil) or (I >= FChildren.Count) then
    Result := nil
  else
    Result := FChildren[I];
end;

function TBaseTree.GetChildCount: Integer;
begin
  if Assigned(FChildren) then
    Result := FChildren.Count
  else
    Result := 0;
end;

function TBaseTree.GetChildIndex: Integer;
begin
  // No default implementation
  Result := 0;
end;

function TBaseTree.GetChildren: IList<IBaseTree>;
begin
  Result := FChildren;
end;

function TBaseTree.GetIsNil: Boolean;
begin
  Result := False;
end;

function TBaseTree.GetLine: Integer;
begin
  Result := 0;
end;

function TBaseTree.GetParent: ITree;
begin
  // No default implementation
  Result := nil;
end;

procedure TBaseTree.ReplaceChildren(const StartChildIndex,
  StopChildIndex: Integer; const T: IANTLRInterface);
var
  ReplacingHowMany, ReplacingWithHowMany, NumNewChildren, Delta, I, J: Integer;
  IndexToDelete, C, ReplacedSoFar: Integer;
  NewTree, Killed: IBaseTree;
  NewChildren: IList<IBaseTree>;
  Child: IBaseTree;
begin
  if (FChildren = nil) then
    raise EArgumentException.Create('indexes invalid; no children in list');
  ReplacingHowMany := StopChildIndex - StartChildIndex + 1;
  NewTree := T as IBaseTree;

  // normalize to a list of children to add: newChildren
  if (NewTree.IsNil) then
    NewChildren := NewTree.Children
  else
  begin
    NewChildren := TList<IBaseTree>.Create;
    NewChildren.Add(NewTree);
  end;

  ReplacingWithHowMany := NewChildren.Count;
  NumNewChildren := NewChildren.Count;
  Delta := ReplacingHowMany - ReplacingWithHowMany;

  // if same number of nodes, do direct replace
  if (Delta = 0) then
  begin
    J := 0; // index into new children
    for I := StartChildIndex to StopChildIndex do
    begin
      Child := NewChildren[J];
      FChildren[I] := Child;
      Child.Parent := Self;
      Child.ChildIndex := I;
      Inc(J);
    end;
  end
  else
    if (Delta > 0) then
    begin
      // fewer new nodes than there were
      // set children and then delete extra
      for J := 0 to NumNewChildren - 1 do
        FChildren[StartChildIndex + J] := NewChildren[J];
      IndexToDelete := StartChildIndex + NumNewChildren;
      for C := IndexToDelete to StopChildIndex do
      begin
        // delete same index, shifting everybody down each time
        Killed := FChildren[IndexToDelete];
        FChildren.Delete(IndexToDelete);
      end;
      FreshenParentAndChildIndexes(StartChildIndex);
    end
    else
      begin
        // more new nodes than were there before
        // fill in as many children as we can (replacingHowMany) w/o moving data
        ReplacedSoFar := 0;
        while (ReplacedSoFar < ReplacingHowMany) do
        begin
          FChildren[StartChildIndex + ReplacedSoFar] := NewChildren[ReplacedSoFar];
          Inc(ReplacedSoFar);
        end;

        // replacedSoFar has correct index for children to add
        while (ReplacedSoFar < ReplacingWithHowMany) do
        begin
          FChildren.Insert(StartChildIndex + ReplacedSoFar,NewChildren[ReplacedSoFar]);
          Inc(ReplacedSoFar);
        end;

        FreshenParentAndChildIndexes(StartChildIndex);
      end;
end;

procedure TBaseTree.SanityCheckParentAndChildIndexes;
begin
  SanityCheckParentAndChildIndexes(nil, -1);
end;

procedure TBaseTree.SanityCheckParentAndChildIndexes(const Parent: ITree;
  const I: Integer);
var
  N, C: Integer;
  Child: ICommonTree;
begin
  if not SameObj(Parent, GetParent) then
    raise EArgumentException.Create('parents don''t match; expected '
      + Parent.ToString + ' found ' + GetParent.ToString);

  if (I <> GetChildIndex) then
    raise EArgumentException.Create('child indexes don''t match; expected '
      + IntToStr(I) + ' found ' + IntToStr(GetChildIndex));

  N := GetChildCount;
  for C := 0 to N - 1 do
  begin
    Child := GetChild(C) as ICommonTree;
    Child.SanityCheckParentAndChildIndexes(Self, C);
  end;
end;

procedure TBaseTree.SetChild(const I: Integer; const T: ITree);
begin
  if (T = nil) then
    Exit;

  if T.IsNil then
    raise EArgumentException.Create('Cannot set single child to a list');

  if (FChildren = nil) then
  begin
    FChildren := CreateChildrenList;
  end;

  FChildren[I] := T as IBaseTree;
  T.Parent := Self;
  T.ChildIndex := I;
end;

procedure TBaseTree.SetChildIndex(const Value: Integer);
begin
  // No default implementation
end;

procedure TBaseTree.SetParent(const Value: ITree);
begin
  // No default implementation
end;

function TBaseTree.ToStringTree: String;
var
  Buf: TStringBuilder;
  I: Integer;
  T: IBaseTree;
begin
  if (FChildren = nil) or (FChildren.Count = 0) then
    Result := ToString
  else
  begin
    Buf := TStringBuilder.Create;
    try
      if (not GetIsNil) then
      begin
        Buf.Append('(');
        Buf.Append(ToString);
        Buf.Append(' ');
      end;

      for I := 0 to FChildren.Count - 1 do
      begin
        T := FChildren[I];
        if (I > 0) then
          Buf.Append(' ');
        Buf.Append(T.ToStringTree);
      end;

      if (not GetIsNil) then
        Buf.Append(')');

      Result := Buf.ToString;
    finally
      Buf.Free;
    end;
  end;
end;

{ TCommonTree }

constructor TCommonTree.Create;
begin
  inherited;
  FStartIndex := -1;
  FStopIndex := -1;
  FChildIndex := -1;
end;

constructor TCommonTree.Create(const ANode: ICommonTree);
begin
  inherited Create(ANode);
  FToken := ANode.Token;
  FStartIndex := ANode.StartIndex;
  FStopIndex := ANode.StopIndex;
  FChildIndex := -1;
end;

constructor TCommonTree.Create(const AToken: IToken);
begin
  Create;
  FToken := AToken;
end;

function TCommonTree.DupNode: ITree;
begin
  Result := TCommonTree.Create(Self) as ICommonTree;
end;

function TCommonTree.GetCharPositionInLine: Integer;
begin
  if (FToken = nil) or (FToken.CharPositionInLine = -1) then
  begin
    if (GetChildCount > 0) then
      Result := GetChild(0).CharPositionInLine
    else
      Result := 0;
  end
  else
    Result := FToken.CharPositionInLine;
end;

function TCommonTree.GetChildIndex: Integer;
begin
  Result := FChildIndex;
end;

function TCommonTree.GetIsNil: Boolean;
begin
  Result := (FToken = nil);
end;

function TCommonTree.GetLine: Integer;
begin
  if (FToken = nil) or (FToken.Line = 0) then
  begin
    if (GetChildCount > 0) then
      Result := GetChild(0).Line
    else
      Result := 0
  end
  else
    Result := FToken.Line;
end;

function TCommonTree.GetParent: ITree;
begin
  Result := ITree(FParent);
end;

function TCommonTree.GetStartIndex: Integer;
begin
  Result := FStartIndex;
end;

function TCommonTree.GetStopIndex: Integer;
begin
  Result := FStopIndex;
end;

function TCommonTree.GetText: String;
begin
  if (FToken = nil) then
    Result := ''
  else
    Result := FToken.Text;
end;

function TCommonTree.GetToken: IToken;
begin
  Result := FToken;
end;

function TCommonTree.GetTokenStartIndex: Integer;
begin
  if (FStartIndex = -1) and (FToken <> nil) then
    Result := FToken.TokenIndex
  else
    Result := FStartIndex;
end;

function TCommonTree.GetTokenStopIndex: Integer;
begin
  if (FStopIndex = -1) and (FToken <> nil) then
    Result := FToken.TokenIndex
  else
    Result := FStopIndex;
end;

function TCommonTree.GetTokenType: Integer;
begin
  if (FToken = nil) then
    Result := TToken.INVALID_TOKEN_TYPE
  else
    Result := FToken.TokenType;
end;

procedure TCommonTree.SetChildIndex(const Value: Integer);
begin
  FChildIndex := Value;
end;

procedure TCommonTree.SetParent(const Value: ITree);
begin
  FParent := Pointer(Value as ICommonTree);
end;

procedure TCommonTree.SetStartIndex(const Value: Integer);
begin
  FStartIndex := Value;
end;

procedure TCommonTree.SetStopIndex(const Value: Integer);
begin
  FStopIndex := Value;
end;

procedure TCommonTree.SetTokenStartIndex(const Value: Integer);
begin
  FStartIndex := Value;
end;

procedure TCommonTree.SetTokenStopIndex(const Value: Integer);
begin
  FStopIndex := Value;
end;

function TCommonTree.ToString: String;
begin
  if (GetIsNil) then
    Result := 'nil'
  else
    if (GetTokenType = TToken.INVALID_TOKEN_TYPE) then
      Result := '<errornode>'
    else
      if (FToken = nil) then
        Result := ''
      else
        Result := FToken.Text;
end;

{ TCommonErrorNode }

constructor TCommonErrorNode.Create(const AInput: ITokenStream; const AStart,
  AStop: IToken; const AException: ERecognitionException);
begin
  inherited Create;
  if (AStop = nil) or ((AStop.TokenIndex < AStart.TokenIndex)
    and (AStop.TokenType <> TToken.EOF))
  then
    // sometimes resync does not consume a token (when LT(1) is
    // in follow set). So, stop will be 1 to left to start. adjust.
    // Also handle case where start is the first token and no token
    // is consumed during recovery; LT(-1) will return null.
    FStop := AStart
  else
    FStop := AStop;
  FInput := AInput;
  FStart := AStart;
  FTrappedException := AException;
end;

function TCommonErrorNode.GetIsNil: Boolean;
begin
  Result := False;
end;

function TCommonErrorNode.GetText: String;
var
  I, J: Integer;
begin
  I := FStart.TokenIndex;
  if (FStop.TokenType = TToken.EOF) then
    J := (FInput as ITokenStream).Size
  else
    J := FStop.TokenIndex;
  Result := (FInput as ITokenStream).ToString(I, J);
end;

function TCommonErrorNode.GetTokenType: Integer;
begin
  Result := TToken.INVALID_TOKEN_TYPE;
end;

function TCommonErrorNode.ToString: String;
begin
  if (FTrappedException is EMissingTokenException) then
    Result := '<missing type: '
      + IntToStr(EMissingTokenException(FTrappedException).MissingType) + '>'
  else
    if (FTrappedException is EUnwantedTokenException) then
      Result := '<extraneous: '
        + EUnwantedTokenException(FTrappedException).UnexpectedToken.ToString
        + ', resync=' + GetText + '>'
    else
      if (FTrappedException is EMismatchedTokenException) then
        Result := '<mismatched token: ' + FTrappedException.Token.ToString
          + ', resync=' + GetText + '>'
      else
        if (FTrappedException is ENoViableAltException) then
          Result := '<unexpected: ' + FTrappedException.Token.ToString
            + ', resync=' + GetText + '>'
        else
          Result := '<error: ' + GetText + '>';
end;

{ TBaseTreeAdaptor }

procedure TBaseTreeAdaptor.AddChild(const T, Child: IANTLRInterface);
begin
  if Assigned(T) and Assigned(Child) then
    (T as ITree).AddChild(Child as ITree);
end;

function TBaseTreeAdaptor.BecomeRoot(const NewRoot,
  OldRoot: IANTLRInterface): IANTLRInterface;
var
  NewRootTree, OldRootTree: ITree;
  NC: Integer;
begin
  NewRootTree := NewRoot as ITree;
  OldRootTree := OldRoot as ITree;
  if (OldRoot = nil) then
    Result := NewRoot
  else
  begin
    // handle ^(nil real-node)
    if (NewRootTree.IsNil) then
    begin
      NC := NewRootTree.ChildCount;
      if (NC = 1) then
        NewRootTree := NewRootTree.GetChild(0)
      else
        if (NC > 1) then
          raise Exception.Create('more than one node as root');
    end;
    // add oldRoot to newRoot; AddChild takes care of case where oldRoot
    // is a flat list (i.e., nil-rooted tree).  All children of oldRoot
    // are added to newRoot.
    NewRootTree.AddChild(OldRootTree);
    Result := NewRootTree;
  end;
end;

function TBaseTreeAdaptor.BecomeRoot(const NewRoot: IToken;
  const OldRoot: IANTLRInterface): IANTLRInterface;
begin
  Result := BecomeRoot(CreateNode(NewRoot), OldRoot);
end;

function TBaseTreeAdaptor.CreateNode(const TokenType: Integer;
  const FromToken: IToken): IANTLRInterface;
var
  Token: IToken;
begin
  Token := CreateToken(FromToken);
  Token.TokenType := TokenType;
  Result := CreateNode(Token);
end;

function TBaseTreeAdaptor.CreateNode(const TokenType: Integer;
  const Text: String): IANTLRInterface;
var
  Token: IToken;
begin
  Token := CreateToken(TokenType, Text);
  Result := CreateNode(Token);
end;

function TBaseTreeAdaptor.CreateNode(const TokenType: Integer;
  const FromToken: IToken; const Text: String): IANTLRInterface;
var
  Token: IToken;
begin
  Token := CreateToken(FromToken);
  Token.TokenType := TokenType;
  Token.Text := Text;
  Result := CreateNode(Token);
end;

constructor TBaseTreeAdaptor.Create;
begin
  inherited Create;
  FUniqueNodeID := 1;
end;

function TBaseTreeAdaptor.DeleteChild(const T: IANTLRInterface;
  const I: Integer): IANTLRInterface;
begin
  Result := (T as ITree).DeleteChild(I);
end;

function TBaseTreeAdaptor.DupTree(const T,
  Parent: IANTLRInterface): IANTLRInterface;
var
  I, N: Integer;
  Child, NewSubTree: IANTLRInterface;
begin
  if (T = nil) then
    Result := nil
  else
  begin
    Result := DupNode(T);
    // ensure new subtree root has parent/child index set
    SetChildIdex(Result, GetChildIndex(T));
    SetParent(Result, Parent);
    N := GetChildCount(T);
    for I := 0 to N - 1 do
    begin
      Child := GetChild(T, I);
      NewSubTree := DupTree(Child, T);
      AddChild(Result, NewSubTree);
    end;
  end;
end;

function TBaseTreeAdaptor.DupTree(const Tree: IANTLRInterface): IANTLRInterface;
begin
  Result := DupTree(Tree, nil);
end;

function TBaseTreeAdaptor.ErrorNode(const Input: ITokenStream; const Start,
  Stop: IToken; const E: ERecognitionException): IANTLRInterface;
begin
  Result := TCommonErrorNode.Create(Input, Start, Stop, E);
end;

function TBaseTreeAdaptor.GetChild(const T: IANTLRInterface;
  const I: Integer): IANTLRInterface;
begin
  Result := (T as ITree).GetChild(I);
end;

function TBaseTreeAdaptor.GetChildCount(const T: IANTLRInterface): Integer;
begin
  Result := (T as ITree).ChildCount;
end;

function TBaseTreeAdaptor.GetNilNode: IANTLRInterface;
begin
  Result := CreateNode(nil);
end;

function TBaseTreeAdaptor.GetNodeText(const T: IANTLRInterface): String;
begin
  Result := (T as ITree).Text;
end;

function TBaseTreeAdaptor.GetNodeType(const T: IANTLRInterface): Integer;
begin
  Result := 0;
end;

function TBaseTreeAdaptor.GetUniqueID(const Node: IANTLRInterface): Integer;
begin
  if (FTreeToUniqueIDMap = nil) then
    FTreeToUniqueIDMap := TDictionary<IANTLRInterface, Integer>.Create;
  if (not FTreeToUniqueIDMap.TryGetValue(Node, Result)) then
  begin
    Result := FUniqueNodeID;
    FTreeToUniqueIDMap[Node] := Result;
    Inc(FUniqueNodeID);
  end;
end;

function TBaseTreeAdaptor.IsNil(const Tree: IANTLRInterface): Boolean;
begin
  Result := (Tree as ITree).IsNil;
end;

function TBaseTreeAdaptor.RulePostProcessing(
  const Root: IANTLRInterface): IANTLRInterface;
var
  R: ITree;
begin
  R := Root as ITree;
  if Assigned(R) and (R.IsNil) then
  begin
    if (R.ChildCount = 0) then
      R := nil
    else
      if (R.ChildCount = 1) then
      begin
        R := R.GetChild(0);
        // whoever invokes rule will set parent and child index
        R.Parent := nil;
        R.ChildIndex := -1;
      end;
  end;
  Result := R;
end;

procedure TBaseTreeAdaptor.SetChild(const T: IANTLRInterface; const I: Integer;
  const Child: IANTLRInterface);
begin
  (T as ITree).SetChild(I, Child as ITree);
end;

procedure TBaseTreeAdaptor.SetNodeText(const T: IANTLRInterface;
  const Text: String);
begin
  raise EInvalidOperation.Create('don''t know enough about Tree node');
end;

procedure TBaseTreeAdaptor.SetNodeType(const T: IANTLRInterface;
  const NodeType: Integer);
begin
  raise EInvalidOperation.Create('don''t know enough about Tree node');
end;

{ TCommonTreeAdaptor }

function TCommonTreeAdaptor.CreateNode(const Payload: IToken): IANTLRInterface;
begin
  Result := TCommonTree.Create(Payload);
end;

function TCommonTreeAdaptor.CreateToken(const TokenType: Integer;
  const Text: String): IToken;
begin
  Result := TCommonToken.Create(TokenType, Text);
end;

function TCommonTreeAdaptor.CreateToken(const FromToken: IToken): IToken;
begin
  Result := TCommonToken.Create(FromToken);
end;

function TCommonTreeAdaptor.DupNode(
  const TreeNode: IANTLRInterface): IANTLRInterface;
begin
  if (TreeNode = nil) then
    Result := nil
  else
    Result := (TreeNode as ITree).DupNode;
end;

function TCommonTreeAdaptor.GetChild(const T: IANTLRInterface;
  const I: Integer): IANTLRInterface;
begin
  if (T = nil) then
    Result := nil
  else
    Result := (T as ITree).GetChild(I);
end;

function TCommonTreeAdaptor.GetChildCount(const T: IANTLRInterface): Integer;
begin
  if (T = nil) then
    Result := 0
  else
    Result := (T as ITree).ChildCount;
end;

function TCommonTreeAdaptor.GetChildIndex(const T: IANTLRInterface): Integer;
begin
  Result := (T as ITree).ChildIndex;
end;

function TCommonTreeAdaptor.GetNodeText(const T: IANTLRInterface): String;
begin
  if (T = nil) then
    Result := ''
  else
    Result := (T as ITree).Text;
end;

function TCommonTreeAdaptor.GetNodeType(const T: IANTLRInterface): Integer;
begin
  if (T = nil) then
    Result := TToken.INVALID_TOKEN_TYPE
  else
    Result := (T as ITree).TokenType;
end;

function TCommonTreeAdaptor.GetParent(
  const T: IANTLRInterface): IANTLRInterface;
begin
  Result := (T as ITree).Parent;
end;

function TCommonTreeAdaptor.GetToken(const TreeNode: IANTLRInterface): IToken;
var
  CommonTree: ICommonTree;
begin
  if Supports(TreeNode, ICommonTree, CommonTree) then
    Result := CommonTree.Token
  else
    Result := nil; // no idea what to do
end;

function TCommonTreeAdaptor.GetTokenStartIndex(
  const T: IANTLRInterface): Integer;
begin
  if (T = nil) then
    Result := -1
  else
    Result := (T as ITree).TokenStartIndex;
end;

function TCommonTreeAdaptor.GetTokenStopIndex(
  const T: IANTLRInterface): Integer;
begin
  if (T = nil) then
    Result := -1
  else
    Result := (T as ITree).TokenStopIndex;
end;

procedure TCommonTreeAdaptor.ReplaceChildren(const Parent: IANTLRInterface;
  const StartChildIndex, StopChildIndex: Integer; const T: IANTLRInterface);
begin
  if Assigned(Parent) then
    (Parent as ITree).ReplaceChildren(StartChildIndex, StopChildIndex, T);
end;

procedure TCommonTreeAdaptor.SetChildIdex(const T: IANTLRInterface;
  const Index: Integer);
begin
  (T as ITree).ChildIndex := Index;
end;

procedure TCommonTreeAdaptor.SetParent(const T, Parent: IANTLRInterface);
begin
  (T as ITree).Parent := (Parent as ITree);
end;

procedure TCommonTreeAdaptor.SetTokenBoundaries(const T: IANTLRInterface;
  const StartToken, StopToken: IToken);
var
  Start, Stop: Integer;
begin
  if Assigned(T) then
  begin
    if Assigned(StartToken) then
      Start := StartToken.TokenIndex
    else
      Start := 0;

    if Assigned(StopToken) then
      Stop := StopToken.TokenIndex
    else
      Stop := 0;

    (T as ITree).TokenStartIndex := Start;
    (T as ITree).TokenStopIndex := Stop;
  end;
end;

{ TCommonTreeNodeStream }

procedure TCommonTreeNodeStream.AddNavigationNode(const TokenType: Integer);
var
  NavNode: IANTLRInterface;
begin
  if (TokenType = TToken.DOWN) then
  begin
    if (GetHasUniqueNavigationNodes) then
      NavNode := FAdaptor.CreateNode(TToken.DOWN, 'DOWN')
    else
      NavNode := FDown;
  end
  else
  begin
    if (GetHasUniqueNavigationNodes) then
      NavNode := FAdaptor.CreateNode(TToken.UP, 'UP')
    else
      NavNode := FUp;
  end;
  FNodes.Add(NavNode);
end;

procedure TCommonTreeNodeStream.Consume;
begin
  if (FP = -1) then
    FillBuffer;
  Inc(FP);
end;

constructor TCommonTreeNodeStream.Create;
begin
  inherited;
  FP := -1;
end;

constructor TCommonTreeNodeStream.Create(const ATree: IANTLRInterface);
begin
  Create(TCommonTreeAdaptor.Create, ATree);
end;

constructor TCommonTreeNodeStream.Create(const AAdaptor: ITreeAdaptor;
  const ATree: IANTLRInterface);
begin
  Create(AAdaptor, ATree, DEFAULT_INITIAL_BUFFER_SIZE);
end;

constructor TCommonTreeNodeStream.Create(const AAdaptor: ITreeAdaptor;
  const ATree: IANTLRInterface; const AInitialBufferSize: Integer);
begin
  Create;
  FRoot := ATree;
  FAdaptor := AAdaptor;
  FNodes := TList<IANTLRInterface>.Create;
  FNodes.Capacity := AInitialBufferSize;
  FDown := FAdaptor.CreateNode(TToken.DOWN, 'DOWN');
  FUp := FAdaptor.CreateNode(TToken.UP, 'UP');
  FEof := FAdaptor.CreateNode(TToken.EOF, 'EOF');
end;

procedure TCommonTreeNodeStream.FillBuffer;
begin
  FillBuffer(FRoot);
  FP := 0; // buffer of nodes intialized now
end;

procedure TCommonTreeNodeStream.FillBuffer(const T: IANTLRInterface);
var
  IsNil: Boolean;
  C, N: Integer;
begin
  IsNil := FAdaptor.IsNil(T);
  if (not IsNil) then
    FNodes.Add(T); // add this node

  // add DOWN node if t has children
  N := FAdaptor.GetChildCount(T);
  if (not IsNil) and (N > 0) then
    AddNavigationNode(TToken.DOWN);

  // and now add all its children
  for C := 0 to N - 1 do
    FillBuffer(FAdaptor.GetChild(T, C));

  // add UP node if t has children
  if (not IsNil) and (N > 0) then
    AddNavigationNode(TToken.UP);
end;

function TCommonTreeNodeStream.Get(const I: Integer): IANTLRInterface;
begin
  if (FP = -1) then
    FillBuffer;
  Result := FNodes[I];
end;

function TCommonTreeNodeStream.GetCurrentSymbol: IANTLRInterface;
begin
  Result := LT(1);
end;

function TCommonTreeNodeStream.GetHasUniqueNavigationNodes: Boolean;
begin
  Result := FUniqueNavigationNodes;
end;

function TCommonTreeNodeStream.GetNodeIndex(
  const Node: IANTLRInterface): Integer;
var
  T: IANTLRInterface;
begin
  if (FP = -1) then
    FillBuffer;
  for Result := 0 to FNodes.Count - 1 do
  begin
    T := FNodes[Result];
    if (T = Node) then
      Exit;
  end;
  Result := -1;
end;

function TCommonTreeNodeStream.GetSourceName: String;
begin
  Result := GetTokenStream.SourceName;
end;

function TCommonTreeNodeStream.GetTokenStream: ITokenStream;
begin
  Result := FTokens;
end;

function TCommonTreeNodeStream.GetTreeAdaptor: ITreeAdaptor;
begin
  Result := FAdaptor;
end;

function TCommonTreeNodeStream.GetTreeSource: IANTLRInterface;
begin
  Result := FRoot;
end;

function TCommonTreeNodeStream.Index: Integer;
begin
  Result := FP;
end;

function TCommonTreeNodeStream.LA(I: Integer): Integer;
begin
  Result := FAdaptor.GetNodeType(LT(I));
end;

function TCommonTreeNodeStream.LAChar(I: Integer): Char;
begin
  Result := Char(LA(I));
end;

function TCommonTreeNodeStream.LB(const K: Integer): IANTLRInterface;
begin
  if (K = 0) then
    Result := nil
  else
    if ((FP - K) < 0) then
      Result := nil
    else
      Result := FNodes[FP - K];
end;

function TCommonTreeNodeStream.LT(const K: Integer): IANTLRInterface;
begin
  if (FP = -1) then
    FillBuffer;
  if (K = 0) then
    Result := nil
  else
    if (K < 0) then
      Result := LB(-K)
    else
      if ((FP + K - 1) >= FNodes.Count) then
        Result := FEof
      else
        Result := FNodes[FP + K - 1];
end;

function TCommonTreeNodeStream.Mark: Integer;
begin
  if (FP = -1) then
    FillBuffer;
  FLastMarker := Index;
  Result := FLastMarker;
end;

function TCommonTreeNodeStream.Pop: Integer;
begin
  Result := FCalls.Pop;
  Seek(Result);
end;

procedure TCommonTreeNodeStream.Push(const Index: Integer);
begin
  if (FCalls = nil) then
    FCalls := TStackList<Integer>.Create;
  FCalls.Push(FP); // save current index
  Seek(Index);
end;

procedure TCommonTreeNodeStream.Release(const Marker: Integer);
begin
  // no resources to release
end;

procedure TCommonTreeNodeStream.ReplaceChildren(const Parent: IANTLRInterface;
  const StartChildIndex, StopChildIndex: Integer; const T: IANTLRInterface);
begin
  if Assigned(Parent) then
    FAdaptor.ReplaceChildren(Parent, StartChildIndex, StopChildIndex, T);
end;

procedure TCommonTreeNodeStream.Reset;
begin
  FP := -1;
  FLastMarker := 0;
  if Assigned(FCalls) then
    FCalls.Clear;
end;

procedure TCommonTreeNodeStream.Rewind(const Marker: Integer);
begin
  Seek(Marker);
end;

procedure TCommonTreeNodeStream.Rewind;
begin
  Seek(FLastMarker);
end;

procedure TCommonTreeNodeStream.Seek(const Index: Integer);
begin
  if (FP = -1) then
    FillBuffer;
  FP := Index;
end;

procedure TCommonTreeNodeStream.SetHasUniqueNavigationNodes(
  const Value: Boolean);
begin
  FUniqueNavigationNodes := Value;
end;

procedure TCommonTreeNodeStream.SetTokenStream(const Value: ITokenStream);
begin
  FTokens := Value;
end;

procedure TCommonTreeNodeStream.SetTreeAdaptor(const Value: ITreeAdaptor);
begin
  FAdaptor := Value;
end;

function TCommonTreeNodeStream.Size: Integer;
begin
  if (FP = -1) then
    FillBuffer;
  Result := FNodes.Count;
end;

function TCommonTreeNodeStream.ToString(const Start,
  Stop: IANTLRInterface): String;
var
  CommonTree: ICommonTree;
  I, BeginTokenIndex, EndTokenIndex: Integer;
  T: IANTLRInterface;
  Buf: TStringBuilder;
  Text: String;
begin
  WriteLn('ToString');
  if (Start = nil) or (Stop = nil) then
    Exit;
  if (FP = -1) then
    FillBuffer;

  if Supports(Start, ICommonTree, CommonTree) then
    Write('ToString: ' + CommonTree.Token.ToString + ', ')
  else
    WriteLn(Start.ToString);

  if Supports(Stop, ICommonTree, CommonTree) then
    WriteLn(CommonTree.Token.ToString)
  else
    WriteLn(Stop.ToString);

  // if we have the token stream, use that to dump text in order
  if Assigned(FTokens) then
  begin
    BeginTokenIndex := FAdaptor.GetTokenStartIndex(Start);
    EndTokenIndex := FAdaptor.GetTokenStartIndex(Stop);
    // if it's a tree, use start/stop index from start node
    // else use token range from start/stop nodes
    if (FAdaptor.GetNodeType(Stop) = TToken.UP) then
      EndTokenIndex := FAdaptor.GetTokenStopIndex(Start)
    else
      if (FAdaptor.GetNodeType(Stop) = TToken.EOF) then
        EndTokenIndex := Size - 2; // don't use EOF
    Result := FTokens.ToString(BeginTokenIndex, EndTokenIndex);
    Exit;
  end;

  // walk nodes looking for start
  T := nil;
  I := 0;
  while (I < FNodes.Count) do
  begin
    T := FNodes[I];
    if SameObj(T, Start) then
      Break;
    Inc(I);
  end;

  // now walk until we see stop, filling string buffer with text
  Buf := TStringBuilder.Create;
  try
    T := FNodes[I];
    while (T <> Stop) do
    begin
      Text := FAdaptor.GetNodeText(T);
      if (Text = '') then
        Text := ' ' + IntToStr(FAdaptor.GetNodeType(T));
      Buf.Append(Text);
      Inc(I);
      T := FNodes[I];
    end;

    // include stop node too
    Text := FAdaptor.GetNodeText(Stop);
    if (Text = '') then
      Text := ' ' + IntToStr(FAdaptor.GetNodeType(Stop));
    Buf.Append(Text);
    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

function TCommonTreeNodeStream.ToString: String;
var
  Buf: TStringBuilder;
  T: IANTLRInterface;
begin
  if (FP = -1) then
    FillBuffer;
  Buf := TStringBuilder.Create;
  try
    for T in FNodes do
    begin
      Buf.Append(' ');
      Buf.Append(FAdaptor.GetNodeType(T));
    end;
    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

function TCommonTreeNodeStream.ToTokenString(const Start,
  Stop: Integer): String;
var
  I: Integer;
  T: IANTLRInterface;
  Buf: TStringBuilder;
begin
  if (FP = -1) then
    FillBuffer;
  Buf := TStringBuilder.Create;
  try
    for I := Stop to Min(FNodes.Count - 1, Stop) do
    begin
      T := FNodes[I];
      Buf.Append(' ');
      Buf.Append(FAdaptor.GetToken(T).ToString);
    end;

    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

{ TParseTree }

constructor TParseTree.Create(const ALabel: IANTLRInterface);
begin
  inherited Create;
  FPayload := ALabel;
end;

function TParseTree.DupNode: ITree;
begin
  Result := nil;
end;

function TParseTree.GetText: String;
begin
  Result := ToString;
end;

function TParseTree.GetTokenStartIndex: Integer;
begin
  Result := 0;
end;

function TParseTree.GetTokenStopIndex: Integer;
begin
  Result := 0;
end;

function TParseTree.GetTokenType: Integer;
begin
  Result := 0;
end;

procedure TParseTree.SetTokenStartIndex(const Value: Integer);
begin
  // No implementation
end;

procedure TParseTree.SetTokenStopIndex(const Value: Integer);
begin
  // No implementation
end;

function TParseTree.ToInputString: String;
var
  Buf: TStringBuilder;
begin
  Buf := TStringBuilder.Create;
  try
    _ToStringLeaves(Buf);
    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

function TParseTree.ToString: String;
var
  T: IToken;
begin
  if Supports(FPayload, IToken, T) then
  begin
    if (T.TokenType = TToken.EOF) then
      Result := '<EOF>'
    else
      Result := T.Text;
  end
  else
    Result := FPayload.ToString;
end;

function TParseTree.ToStringWithHiddenTokens: String;
var
  Buf: TStringBuilder;
  Hidden: IToken;
  NodeText: String;
begin
  Buf := TStringBuilder.Create;
  try
    if Assigned(FHiddenTokens) then
    begin
      for Hidden in FHiddenTokens do
        Buf.Append(Hidden.Text);
    end;
    NodeText := ToString;
    if (NodeText <> '<EOF>') then
      Buf.Append(NodeText);
    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

procedure TParseTree._ToStringLeaves(const Buf: TStringBuilder);
var
  T: IBaseTree;
begin
  if Supports(FPayload, IToken) then
  begin
    // leaf node token?
    Buf.Append(ToStringWithHiddenTokens);
    Exit;
  end;
  if Assigned(FChildren) then
    for T in FChildren do
      (T as IParseTree)._ToStringLeaves(Buf);
end;

{ ERewriteCardinalityException }

constructor ERewriteCardinalityException.Create(
  const AElementDescription: String);
begin
  inherited Create(AElementDescription);
  FElementDescription := AElementDescription;
end;

{ TRewriteRuleElementStream }

procedure TRewriteRuleElementStream.Add(const El: IANTLRInterface);
begin
  if (El = nil) then
    Exit;
  if Assigned(FElements) then
     // if in list, just add
    FElements.Add(El)
  else
    if (FSingleElement = nil) then
      // no elements yet, track w/o list
      FSingleElement := El
    else
    begin
      // adding 2nd element, move to list
      FElements := TList<IANTLRInterface>.Create;
      FElements.Capacity := 5;
      FElements.Add(FSingleElement);
      FSingleElement := nil;
      FElements.Add(El);
    end;
end;

constructor TRewriteRuleElementStream.Create(const AAdaptor: ITreeAdaptor;
  const AElementDescription: String);
begin
  inherited Create;
  FAdaptor := AAdaptor;
  FElementDescription := AElementDescription;
end;

constructor TRewriteRuleElementStream.Create(const AAdaptor: ITreeAdaptor;
  const AElementDescription: String; const AOneElement: IANTLRInterface);
begin
  Create(AAdaptor, AElementDescription);
  Add(AOneElement);
end;

constructor TRewriteRuleElementStream.Create(const AAdaptor: ITreeAdaptor;
  const AElementDescription: String; const AElements: IList<IANTLRInterface>);
begin
  Create(AAdaptor, AElementDescription);
  FElements := AElements;
end;

function TRewriteRuleElementStream.GetDescription: String;
begin
  Result := FElementDescription;
end;

function TRewriteRuleElementStream.HasNext: Boolean;
begin
  Result := ((FSingleElement <> nil) and (FCursor < 1))
    or ((FElements <> nil) and (FCursor < FElements.Count));
end;

function TRewriteRuleElementStream.NextTree: IANTLRInterface;
begin
  Result := _Next;
end;

procedure TRewriteRuleElementStream.Reset;
begin
  FCursor := 0;
  FDirty := True;
end;

function TRewriteRuleElementStream.Size: Integer;
begin
  if Assigned(FSingleElement) then
    Result := 1
  else
    if Assigned(FElements) then
      Result := FElements.Count
    else
      Result := 0;
end;

function TRewriteRuleElementStream.ToTree(const El: IANTLRInterface): IANTLRInterface;
begin
  Result := El;
end;

function TRewriteRuleElementStream._Next: IANTLRInterface;
var
  Size: Integer;
begin
  Size := Self.Size;
  if (Size = 0) then
    raise ERewriteEmptyStreamException.Create(FElementDescription);

  if (FCursor >= Size) then
  begin
     // out of elements?
     if (Size = 1) then
       // if size is 1, it's ok; return and we'll dup
       Result := ToTree(FSingleElement)
     else
       // out of elements and size was not 1, so we can't dup
       raise ERewriteCardinalityException.Create(FElementDescription);
  end
  else
  begin
    // we have elements
    if Assigned(FSingleElement) then
    begin
      Inc(FCursor); // move cursor even for single element list
      Result := ToTree(FSingleElement);
    end
    else
    begin
      // must have more than one in list, pull from elements
      Result := ToTree(FElements[FCursor]);
      Inc(FCursor);
    end;
  end;
end;

{ TRewriteRuleNodeStream }

function TRewriteRuleNodeStream.NextNode: IANTLRInterface;
begin
  Result := _Next;
end;

function TRewriteRuleNodeStream.ToTree(
  const El: IANTLRInterface): IANTLRInterface;
begin
  Result := FAdaptor.DupNode(El);
end;

{ TRewriteRuleSubtreeStream }

function TRewriteRuleSubtreeStream.Dup(
  const O: IANTLRInterface): IANTLRInterface;
begin
  Result := FAdaptor.DupTree(O);
end;

function TRewriteRuleSubtreeStream.DupNode(
  const O: IANTLRInterface): IANTLRInterface;
begin
  Result := FAdaptor.DupNode(O);
end;

function TRewriteRuleSubtreeStream.FetchObject(
  const PH: TProcessHandler): IANTLRInterface;
begin
  if (RequiresDuplication) then
    // process the object
    Result := PH(_Next)
  else
    // test above then fetch
    Result := _Next;
end;

function TRewriteRuleSubtreeStream.NextNode: IANTLRInterface;
begin
  // if necessary, dup (at most a single node since this is for making root nodes).
  Result := FetchObject(DupNode);
end;

function TRewriteRuleSubtreeStream.NextTree: IANTLRInterface;
begin
  // if out of elements and size is 1, dup
  Result := FetchObject(Dup);
end;

function TRewriteRuleSubtreeStream.RequiresDuplication: Boolean;
var
  Size: Integer;
begin
  Size := Self.Size;
  // if dirty or if out of elements and size is 1
  Result := FDirty or ((FCursor >= Size) and (Size = 1));
end;

{ TRewriteRuleTokenStream }

function TRewriteRuleTokenStream.NextNode: IANTLRInterface;
begin
  Result := FAdaptor.CreateNode(_Next as IToken)
end;

function TRewriteRuleTokenStream.NextToken: IToken;
begin
  Result := _Next as IToken;
end;

function TRewriteRuleTokenStream.ToTree(
  const El: IANTLRInterface): IANTLRInterface;
begin
  Result := El;
end;

{ TTreeParser }

constructor TTreeParser.Create(const AInput: ITreeNodeStream);
begin
  inherited Create; // highlight that we go to super to set state object
  SetTreeNodeStream(AInput);
end;

constructor TTreeParser.Create(const AInput: ITreeNodeStream;
  const AState: IRecognizerSharedState);
begin
  inherited Create(AState); // share the state object with another parser
  SetTreeNodeStream(AInput);
end;

function TTreeParser.GetCurrentInputSymbol(
  const Input: IIntStream): IANTLRInterface;
begin
  Result := FInput.LT(1);
end;

function TTreeParser.GetErrorHeader(const E: ERecognitionException): String;
begin
  Result := GetGrammarFileName + ': node from ';
  if (E.ApproximateLineInfo) then
    Result := Result + 'after ';
  Result := Result + 'line ' + IntToStr(E.Line) + ':' + IntToStr(E.CharPositionInLine);
end;

function TTreeParser.GetErrorMessage(const E: ERecognitionException;
  const TokenNames: TStringArray): String;
var
  Adaptor: ITreeAdaptor;
begin
  if (Self is TTreeParser) then
  begin
    Adaptor := (E.Input as ITreeNodeStream).TreeAdaptor;
    E.Token := Adaptor.GetToken(E.Node);
    if (E.Token = nil) then
      // could be an UP/DOWN node
      E.Token := TCommonToken.Create(Adaptor.GetNodeType(E.Node),
        Adaptor.GetNodeText(E.Node));
  end;
  Result := inherited GetErrorMessage(E, TokenNames);
end;

function TTreeParser.GetInput: IIntStream;
begin
  Result := FInput;
end;

function TTreeParser.GetMissingSymbol(const Input: IIntStream;
  const E: ERecognitionException; const ExpectedTokenType: Integer;
  const Follow: IBitSet): IANTLRInterface;
var
  TokenText: String;
begin
  TokenText := '<missing ' + GetTokenNames[ExpectedTokenType] + '>';
  Result := TCommonTree.Create(TCommonToken.Create(ExpectedTokenType, TokenText));
end;

function TTreeParser.GetSourceName: String;
begin
  Result := FInput.SourceName;
end;

function TTreeParser.GetTreeNodeStream: ITreeNodeStream;
begin
  Result := FInput;
end;

procedure TTreeParser.MatchAny(const Input: IIntStream);
var
  Look: IANTLRInterface;
  Level, TokenType: Integer;
begin
  FState.ErrorRecovery := False;
  FState.Failed := False;
  Look := FInput.LT(1);
  if (FInput.TreeAdaptor.GetChildCount(Look) = 0) then
  begin
    FInput.Consume; // not subtree, consume 1 node and return
    Exit;
  end;

  // current node is a subtree, skip to corresponding UP.
  // must count nesting level to get right UP
  Level := 0;
  TokenType := FInput.TreeAdaptor.GetNodeType(Look);
  while (TokenType <> TToken.EOF) and not ((TokenType = UP) and (Level = 0)) do
  begin
    FInput.Consume;
    Look := FInput.LT(1);
    TokenType := FInput.TreeAdaptor.GetNodeType(Look);
    if (TokenType = DOWN) then
      Inc(Level)
    else
      if (TokenType = UP) then
        Dec(Level);
  end;
  FInput.Consume; // consume UP
end;

procedure TTreeParser.Mismatch(const Input: IIntStream;
  const TokenType: Integer; const Follow: IBitSet);
begin
  raise EMismatchedTreeNodeException.Create(TokenType, FInput);
end;

procedure TTreeParser.Reset;
begin
  inherited; // reset all recognizer state variables
  if Assigned(FInput) then
    FInput.Seek(0); // rewind the input
end;

procedure TTreeParser.SetTreeNodeStream(const Value: ITreeNodeStream);
begin
  FInput := Value;
end;

procedure TTreeParser.TraceIn(const RuleName: String; const RuleIndex: Integer);
begin
  inherited TraceIn(RuleName, RuleIndex, FInput.LT(1).ToString);
end;

procedure TTreeParser.TraceOut(const RuleName: String;
  const RuleIndex: Integer);
begin
  inherited TraceOut(RuleName, RuleIndex, FInput.LT(1).ToString);
end;

{ TTreePatternLexer }

constructor TTreePatternLexer.Create;
begin
  inherited;
  FSVal := TStringBuilder.Create;
end;

procedure TTreePatternLexer.Consume;
begin
  Inc(FP);
  if (FP > FN) then
    FC := EOF
  else
    FC := Integer(FPattern[FP]);
end;

constructor TTreePatternLexer.Create(const APattern: String);
begin
  Create;
  FPattern := APattern;
  FN := Length(FPattern);
  Consume;
end;

destructor TTreePatternLexer.Destroy;
begin
  FSVal.Free;
  inherited;
end;

function TTreePatternLexer.NextToken: Integer;
begin
  FSVal.Length := 0; // reset, but reuse buffer
  while (FC <> EOF) do
  begin
    if (FC = 32) or (FC = 10) or (FC = 13) or (FC = 9) then
    begin
      Consume;
      Continue;
    end;

    if ((FC >= Ord('a')) and (FC <= Ord('z')))
      or ((FC >= Ord('A')) and (FC <= Ord('Z')))
      or (FC = Ord('_'))
    then begin
      FSVal.Append(Char(FC));
      Consume;
      while ((FC >= Ord('a')) and (FC <= Ord('z')))
        or ((FC >= Ord('A')) and (FC <= Ord('Z')))
        or ((FC >= Ord('0')) and (FC <= Ord('9')))
        or (FC = Ord('_')) do
      begin
        FSVal.Append(Char(FC));
        Consume;
      end;
      Exit(ID);
    end;

    if (FC = Ord('(')) then
    begin
      Consume;
      Exit(START);
    end;

    if (FC = Ord(')')) then
    begin
      Consume;
      Exit(STOP);
    end;

    if (FC = Ord('%')) then
    begin
      Consume;
      Exit(PERCENT);
    end;

    if (FC = Ord(':')) then
    begin
      Consume;
      Exit(COLON);
    end;

    if (FC = Ord('.')) then
    begin
      Consume;
      Exit(DOT);
    end;

    if (FC = Ord('[')) then
    begin
      // grab [x] as a string, returning x
      Consume;
      while (FC <> Ord(']')) do
      begin
        if (FC = Ord('\')) then
        begin
          Consume;
          if (FC <> Ord(']')) then
            FSVal.Append('\');
          FSVal.Append(Char(FC));
        end
        else
          FSVal.Append(Char(FC));
        Consume;
      end;
      Consume;
      Exit(ARG);
    end;

    Consume;
    FError := True;
    Exit(EOF);
  end;
  Result := EOF;
end;

function TTreePatternLexer.SVal: String;
begin
  Result := FSVal.ToString;
end;

{ TTreeWizard }

function TTreeWizard.ComputeTokenTypes(
  const TokenNames: TStringArray): IDictionary<String, Integer>;
var
  TokenType: Integer;
begin
  Result := TDictionary<String, Integer>.Create;
  if (Length(TokenNames) > 0)then
  begin
    for TokenType := TToken.MIN_TOKEN_TYPE to Length(TokenNames) - 1 do
      Result.Add(TokenNames[TokenType], TokenType);
  end;
end;

constructor TTreeWizard.Create(const AAdaptor: ITreeAdaptor);
begin
  inherited Create;
  FAdaptor := AAdaptor;
end;

constructor TTreeWizard.Create(const AAdaptor: ITreeAdaptor;
  const ATokenNameToTypeMap: IDictionary<String, Integer>);
begin
  inherited Create;
  FAdaptor := AAdaptor;
  FTokenNameToTypeMap := ATokenNameToTypeMap;
end;

constructor TTreeWizard.Create(const AAdaptor: ITreeAdaptor;
  const TokenNames: TStringArray);
begin
  inherited Create;
  FAdaptor := AAdaptor;
  FTokenNameToTypeMap := ComputeTokenTypes(TokenNames);
end;

function TTreeWizard.CreateTreeOrNode(const Pattern: String): IANTLRInterface;
var
  Tokenizer: ITreePatternLexer;
  Parser: ITreePatternParser;
begin
  Tokenizer := TTreePatternLexer.Create(Pattern);
  Parser := TTreePatternParser.Create(Tokenizer, Self, FAdaptor);
  Result := Parser.Pattern;
end;

function TTreeWizard.Equals(const T1, T2: IANTLRInterface;
  const Adaptor: ITreeAdaptor): Boolean;
begin
  Result := _Equals(T1, T2, Adaptor);
end;

function TTreeWizard.Equals(const T1, T2: IANTLRInterface): Boolean;
begin
  Result := _Equals(T1, T2, FAdaptor);
end;

function TTreeWizard.Find(const T: IANTLRInterface;
  const Pattern: String): IList<IANTLRInterface>;
var
  Tokenizer: ITreePatternLexer;
  Parser: ITreePatternParser;
  TreePattern: ITreePattern;
  RootTokenType: Integer;
  Visitor: IContextVisitor;
begin
  Result := TList<IANTLRInterface>.Create;

  // Create a TreePattern from the pattern
  Tokenizer := TTreePatternLexer.Create(Pattern);
  Parser := TTreePatternParser.Create(Tokenizer, Self, TTreePatternTreeAdaptor.Create);
  TreePattern := Parser.Pattern as ITreePattern;

  // don't allow invalid patterns
  if (TreePattern = nil) or (TreePattern.IsNil)
    or Supports(TreePattern, IWildcardTreePattern)
  then
    Exit(nil);

  RootTokenType := TreePattern.TokenType;
  Visitor := TPatternMatchingContextVisitor.Create(Self, TreePattern, Result);
  Visit(T, RootTokenType, Visitor);
end;

function TTreeWizard.Find(const T: IANTLRInterface;
  const TokenType: Integer): IList<IANTLRInterface>;
begin
  Result := TList<IANTLRInterface>.Create;
  Visit(T, TokenType, TRecordAllElementsVisitor.Create(Result));
end;

function TTreeWizard.FindFirst(const T: IANTLRInterface;
  const TokenType: Integer): IANTLRInterface;
begin
  Result := nil;
end;

function TTreeWizard.FindFirst(const T: IANTLRInterface;
  const Pattern: String): IANTLRInterface;
begin
  Result := nil;
end;

function TTreeWizard.GetTokenType(const TokenName: String): Integer;
begin
  if (FTokenNameToTypeMap = nil) then
    Exit(TToken.INVALID_TOKEN_TYPE);
  if (not FTokenNameToTypeMap.TryGetValue(TokenName, Result)) then
    Result := TToken.INVALID_TOKEN_TYPE;
end;

function TTreeWizard.Index(
  const T: IANTLRInterface): IDictionary<Integer, IList<IANTLRInterface>>;
begin
  Result := TDictionary<Integer, IList<IANTLRInterface>>.Create;
  _Index(T, Result);
end;

function TTreeWizard.Parse(const T: IANTLRInterface;
  const Pattern: String): Boolean;
begin
  Result := Parse(T, Pattern, nil);
end;

function TTreeWizard.Parse(const T: IANTLRInterface; const Pattern: String;
  const Labels: IDictionary<String, IANTLRInterface>): Boolean;
var
  Tokenizer: ITreePatternLexer;
  Parser: ITreePatternParser;
  TreePattern: ITreePattern;
begin
  Tokenizer := TTreePatternLexer.Create(Pattern);
  Parser := TTreePatternParser.Create(Tokenizer, Self, TTreePatternTreeAdaptor.Create);
  TreePattern := Parser.Pattern as ITreePattern;
  Result := _Parse(T, TreePattern, Labels);
end;

procedure TTreeWizard.Visit(const T: IANTLRInterface; const Pattern: String;
  const Visitor: IContextVisitor);
var
  Tokenizer: ITreePatternLexer;
  Parser: ITreePatternParser;
  TreePattern: ITreePattern;
  RootTokenType: Integer;
  PatternVisitor: IContextVisitor;
begin
  // Create a TreePattern from the pattern
  Tokenizer := TTreePatternLexer.Create(Pattern);
  Parser := TTreePatternParser.Create(Tokenizer, Self, TTreePatternTreeAdaptor.Create);
  TreePattern := Parser.Pattern as ITreePattern;
  if (TreePattern = nil) or (TreePattern.IsNil)
    or Supports(TreePattern, IWildcardTreePattern)
  then
    Exit;
  RootTokenType := TreePattern.TokenType;
  PatternVisitor := TInvokeVisitorOnPatternMatchContextVisitor.Create(Self, TreePattern, Visitor);
  Visit(T, RootTokenType, PatternVisitor);
end;

class function TTreeWizard._Equals(const T1, T2: IANTLRInterface;
  const Adaptor: ITreeAdaptor): Boolean;
var
  I, N1, N2: Integer;
  Child1, Child2: IANTLRInterface;
begin
  // make sure both are non-null
  if (T1 = nil) or (T2 = nil) then
    Exit(False);

  // check roots
  if (Adaptor.GetNodeType(T1) <> Adaptor.GetNodeType(T2)) then
    Exit(False);
  if (Adaptor.GetNodeText(T1) <> Adaptor.GetNodeText(T2)) then
    Exit(False);

  // check children
  N1 := Adaptor.GetChildCount(T1);
  N2 := Adaptor.GetChildCount(T2);
  if (N1 <> N2) then
    Exit(False);
  for I := 0 to N1 - 1 do
  begin
    Child1 := Adaptor.GetChild(T1, I);
    Child2 := Adaptor.GetChild(T2, I);
    if (not _Equals(Child1, Child2, Adaptor)) then
      Exit(False);
  end;

  Result := True;
end;

procedure TTreeWizard._Index(const T: IANTLRInterface;
  const M: IDictionary<Integer, IList<IANTLRInterface>>);
var
  I, N, TType: Integer;
  Elements: IList<IANTLRInterface>;
begin
  if (T = nil) then
    Exit;
  TType := FAdaptor.GetNodeType(T);
  if (not M.TryGetValue(TType, Elements)) then
    Elements := nil;
  if (Elements = nil) then
  begin
    Elements := TList<IANTLRInterface>.Create;
    M.Add(TType, Elements);
  end;
  Elements.Add(T);
  N := FAdaptor.GetChildCount(T);
  for I := 0 to N - 1 do
    _Index(FAdaptor.GetChild(T, I), M);
end;

function TTreeWizard._Parse(const T1: IANTLRInterface; const T2: ITreePattern;
  const Labels: IDictionary<String, IANTLRInterface>): Boolean;
var
  I, N1, N2: Integer;
  Child1: IANTLRInterface;
  Child2: ITreePattern;
begin
  // make sure both are non-null
  if (T1 = nil) or (T2 = nil) then
    Exit(False);

  // check roots (wildcard matches anything)
  if (not Supports(T2, IWildcardTreePattern)) then
  begin
    if (FAdaptor.GetNodeType(T1) <> T2.TokenType) then
      Exit(False);
    if (T2.HasTextArg) and (FAdaptor.GetNodeText(T1) <> T2.Text) then
      Exit(False);
  end;

  if (T2.TokenLabel <> '') and Assigned(Labels) then
    // map label in pattern to node in t1
    Labels.AddOrSetValue(T2.TokenLabel, T1);

  // check children
  N1 := FAdaptor.GetChildCount(T1);
  N2 := T2.ChildCount;
  if (N1 <> N2) then
    Exit(False);

  for I := 0 to N1 - 1 do
  begin
    Child1 := FAdaptor.GetChild(T1, I);
    Child2 := T2.GetChild(I) as ITreePattern;
    if (not _Parse(Child1, Child2, Labels)) then
      Exit(False);
  end;

  Result := True;
end;

procedure TTreeWizard._Visit(const T, Parent: IANTLRInterface; const ChildIndex,
  TokenType: Integer; const Visitor: IContextVisitor);
var
  I, N: Integer;
begin
  if (T = nil) then
    Exit;
  if (FAdaptor.GetNodeType(T) = TokenType) then
    Visitor.Visit(T, Parent, ChildIndex, nil);

  N := FAdaptor.GetChildCount(T);
  for I := 0 to N - 1 do
    _Visit(FAdaptor.GetChild(T, I), T, I, TokenType, Visitor);
end;

procedure TTreeWizard.Visit(const T: IANTLRInterface; const TokenType: Integer;
  const Visitor: IContextVisitor);
begin
  _Visit(T, nil, 0, TokenType, Visitor);
end;

constructor TTreeWizard.Create(const TokenNames: TStringArray);
begin
  Create(nil, TokenNames);
end;

{ TTreePatternParser }

constructor TTreePatternParser.Create(const ATokenizer: ITreePatternLexer;
  const AWizard: ITreeWizard; const AAdaptor: ITreeAdaptor);
begin
  inherited Create;
  FTokenizer := ATokenizer;
  FWizard := AWizard;
  FAdaptor := AAdaptor;
  FTokenType := FTokenizer.NextToken; // kickstart
end;

function TTreePatternParser.ParseNode: IANTLRInterface;
var
  Lbl, TokenName, Text, Arg: String;
  WildcardPayload: IToken;
  Node: TTreeWizard.ITreePattern;
  TreeNodeType: Integer;
begin
  // "%label:" prefix
  Lbl := '';
  if (FTokenType = TTreePatternLexer.PERCENT) then
  begin
    FTokenType := FTokenizer.NextToken;
    if (FTokenType <> TTreePatternLexer.ID) then
      Exit(nil);
    Lbl := FTokenizer.SVal;
    FTokenType := FTokenizer.NextToken;
    if (FTokenType <> TTreePatternLexer.COLON) then
      Exit(nil);
    FTokenType := FTokenizer.NextToken; // move to ID following colon
  end;

  // Wildcard?
  if (FTokenType = TTreePatternLexer.DOT) then
  begin
    FTokenType := FTokenizer.NextToken;
    WildcardPayload := TCommonToken.Create(0, '.');
    Node := TTreeWizard.TWildcardTreePattern.Create(WildcardPayload);
    if (Lbl <> '') then
      Node.TokenLabel := Lbl;
    Exit(Node);
  end;

  // "ID" or "ID[arg]"
  if (FTokenType <> TTreePatternLexer.ID) then
    Exit(nil);
  TokenName := FTokenizer.SVal;
  FTokenType := FTokenizer.NextToken;
  if (TokenName = 'nil') then
    Exit(FAdaptor.GetNilNode);
  Text := TokenName;

  // check for arg
  Arg := '';
  if (FTokenType = TTreePatternLexer.ARG) then
  begin
    Arg := FTokenizer.SVal;
    Text := Arg;
    FTokenType := FTokenizer.NextToken;
  end;

  // create node
  TreeNodeType := FWizard.GetTokenType(TokenName);
  if (TreeNodeType = TToken.INVALID_TOKEN_TYPE) then
    Exit(nil);

  Result := FAdaptor.CreateNode(TreeNodeType, Text);
  if (Lbl <> '') and Supports(Result, TTreeWizard.ITreePattern, Node) then
    Node.TokenLabel := Lbl;
  if (Arg <> '') and Supports(Result, TTreeWizard.ITreePattern, Node) then
    Node.HasTextArg := True;
end;

function TTreePatternParser.ParseTree: IANTLRInterface;
var
  Subtree, Child: IANTLRInterface;
begin
  if (FTokenType <> TTreePatternLexer.START) then
  begin
    WriteLn('no BEGIN');
    Exit(nil);
  end;

  FTokenType := FTokenizer.NextToken;
  Result := ParseNode;
  if (Result = nil) then
    Exit;

  while (FTokenType in [TTreePatternLexer.START, TTreePatternLexer.ID,
    TTreePatternLexer.PERCENT, TTreePatternLexer.DOT]) do
  begin
    if (FTokenType = TTreePatternLexer.START) then
    begin
      Subtree := ParseTree;
      FAdaptor.AddChild(Result, Subtree);
    end
    else
    begin
      Child := ParseNode;
      if (Child = nil) then
        Exit(nil);
      FAdaptor.AddChild(Result, Child);
    end;
  end;

  if (FTokenType <> TTreePatternLexer.STOP) then
  begin
    WriteLn('no END');
    Exit(nil);
  end;

  FTokenType := FTokenizer.NextToken;
end;

function TTreePatternParser.Pattern: IANTLRInterface;
var
  Node: IANTLRInterface;
begin
  if (FTokenType = TTreePatternLexer.START) then
    Exit(ParseTree);

  if (FTokenType = TTreePatternLexer.ID) then
  begin
    Node := ParseNode;
    if (FTokenType = TTreePatternLexer.EOF) then
      Result := Node
    else
      Result := nil; // extra junk on end
  end
  else
    Result := nil;
end;

{ TTreeWizard.TVisitor }

procedure TTreeWizard.TVisitor.Visit(const T, Parent: IANTLRInterface;
  const ChildIndex: Integer;
  const Labels: IDictionary<String, IANTLRInterface>);
begin
  Visit(T);
end;

{ TTreeWizard.TRecordAllElementsVisitor }

constructor TTreeWizard.TRecordAllElementsVisitor.Create(
  const AList: IList<IANTLRInterface>);
begin
  inherited Create;
  FList := AList;
end;

procedure TTreeWizard.TRecordAllElementsVisitor.Visit(const T: IANTLRInterface);
begin
  FList.Add(T);
end;

{ TTreeWizard.TPatternMatchingContextVisitor }

constructor TTreeWizard.TPatternMatchingContextVisitor.Create(
  const AOwner: TTreeWizard; const APattern: ITreePattern;
  const AList: IList<IANTLRInterface>);
begin
  inherited Create;
  FOwner := AOwner;
  FPattern := APattern;
  FList := AList;
end;

procedure TTreeWizard.TPatternMatchingContextVisitor.Visit(const T,
  Parent: IANTLRInterface; const ChildIndex: Integer;
  const Labels: IDictionary<String, IANTLRInterface>);
begin
  if (FOwner._Parse(T, FPattern, nil)) then
    FList.Add(T);
end;

{ TTreeWizard.TInvokeVisitorOnPatternMatchContextVisitor }

constructor TTreeWizard.TInvokeVisitorOnPatternMatchContextVisitor.Create(
  const AOwner: TTreeWizard; const APattern: ITreePattern;
  const AVisitor: IContextVisitor);
begin
  inherited Create;
  FOwner := AOwner;
  FPattern := APattern;
  FVisitor := AVisitor;
  FLabels := TDictionary<String, IANTLRInterface>.Create;
end;

procedure TTreeWizard.TInvokeVisitorOnPatternMatchContextVisitor.Visit(const T,
  Parent: IANTLRInterface; const ChildIndex: Integer;
  const UnusedLabels: IDictionary<String, IANTLRInterface>);
begin
  // the unusedlabels arg is null as visit on token type doesn't set.
  FLabels.Clear;
  if (FOwner._Parse(T, FPattern, FLabels)) then
    FVisitor.Visit(T, Parent, ChildIndex, FLabels);
end;

{ TTreeWizard.TTreePattern }

function TTreeWizard.TTreePattern.GetHasTextArg: Boolean;
begin
  Result := FHasTextArg;
end;

function TTreeWizard.TTreePattern.GetTokenLabel: String;
begin
  Result := FLabel;
end;

procedure TTreeWizard.TTreePattern.SetHasTextArg(const Value: Boolean);
begin
  FHasTextArg := Value;
end;

procedure TTreeWizard.TTreePattern.SetTokenLabel(const Value: String);
begin
  FLabel := Value;
end;

function TTreeWizard.TTreePattern.ToString: String;
begin
  if (FLabel <> '') then
    Result := '%' + FLabel + ':' + inherited ToString
  else
    Result := inherited ToString;
end;

{ TTreeWizard.TTreePatternTreeAdaptor }

function TTreeWizard.TTreePatternTreeAdaptor.CreateNode(
  const Payload: IToken): IANTLRInterface;
begin
  Result := TTreePattern.Create(Payload);
end;

{ TTreeRuleReturnScope }

function TTreeRuleReturnScope.GetStart: IANTLRInterface;
begin
  Result := FStart;
end;

procedure TTreeRuleReturnScope.SetStart(const Value: IANTLRInterface);
begin
  FStart := Value;
end;

{ TUnBufferedTreeNodeStream }

procedure TUnBufferedTreeNodeStream.AddLookahead(const Node: IANTLRInterface);
var
  Bigger: TANTLRInterfaceArray;
  I, RemainderHeadToEnd: Integer;
begin
  FLookahead[FTail] := Node;
  FTail := (FTail + 1) mod Length(FLookahead);
  if (FTail = FHead) then
  begin
    // buffer overflow: tail caught up with head
    // allocate a buffer 2x as big
    SetLength(Bigger,2 * Length(FLookahead));
    // copy head to end of buffer to beginning of bigger buffer
    RemainderHeadToEnd := Length(FLookahead) - FHead;
    for I := 0 to RemainderHeadToEnd - 1 do
      Bigger[I] := FLookahead[FHead + I];
    // copy 0..tail to after that
    for I := 0 to FTail - 1 do
      Bigger[RemainderHeadToEnd + I] := FLookahead[I];
    FLookahead := Bigger; // reset to bigger buffer
    FHead := 0;
    Inc(FTail,RemainderHeadToEnd);
  end;
end;

procedure TUnBufferedTreeNodeStream.AddNavigationNode(const TokenType: Integer);
var
  NavNode: IANTLRInterface;
begin
  if (TokenType = TToken.DOWN) then
  begin
    if (GetHasUniqueNavigationNodes) then
      NavNode := FAdaptor.CreateNode(TToken.DOWN,'DOWN')
    else
      NavNode := FDown;
  end
  else
  begin
    if (GetHasUniqueNavigationNodes) then
      NavNode := FAdaptor.CreateNode(TToken.UP,'UP')
    else
      NavNode := FUp;
  end;
  AddLookahead(NavNode);
end;

procedure TUnBufferedTreeNodeStream.Consume;
begin
  // make sure there is something in lookahead buf, which might call next()
  Fill(1);
  Inc(FAbsoluteNodeIndex);
  FPreviousNode := FLookahead[FHead]; // track previous node before moving on
  FHead := (FHead + 1) mod Length(FLookahead);
end;

constructor TUnBufferedTreeNodeStream.Create;
begin
  inherited;
  SetLength(FLookAhead,INITIAL_LOOKAHEAD_BUFFER_SIZE);
  FNodeStack := TStackList<IANTLRInterface>.Create;
  FIndexStack := TStackList<Integer>.Create;
end;

constructor TUnBufferedTreeNodeStream.Create(const ATree: IANTLRInterface);
begin
  Create(TCommonTreeAdaptor.Create, ATree);
end;

constructor TUnBufferedTreeNodeStream.Create(const AAdaptor: ITreeAdaptor;
  const ATree: IANTLRInterface);
begin
  Create;
  FRoot := ATree;
  FAdaptor := AAdaptor;
  Reset;
  FDown := FAdaptor.CreateNode(TToken.DOWN, 'DOWN');
  FUp := FAdaptor.CreateNode(TToken.UP, 'UP');
  FEof := FAdaptor.CreateNode(TToken.EOF, 'EOF');
end;

procedure TUnBufferedTreeNodeStream.Fill(const K: Integer);
var
  I, N: Integer;
begin
  N := LookaheadSize;
  for I := 1 to K - N do
    MoveNext; // get at least k-depth lookahead nodes
end;

function TUnBufferedTreeNodeStream.Get(const I: Integer): IANTLRInterface;
begin
  raise EInvalidOperation.Create('stream is unbuffered');
end;

function TUnBufferedTreeNodeStream.GetCurrent: IANTLRInterface;
begin
  Result := FCurrentEnumerationNode;
end;

function TUnBufferedTreeNodeStream.GetHasUniqueNavigationNodes: Boolean;
begin
  Result := FUniqueNavigationNodes;
end;

function TUnBufferedTreeNodeStream.GetSourceName: String;
begin
  Result := GetTokenStream.SourceName;
end;

function TUnBufferedTreeNodeStream.GetTokenStream: ITokenStream;
begin
  Result := FTokens;
end;

function TUnBufferedTreeNodeStream.GetTreeAdaptor: ITreeAdaptor;
begin
  Result := FAdaptor;
end;

function TUnBufferedTreeNodeStream.GetTreeSource: IANTLRInterface;
begin
  Result := FRoot;
end;

function TUnBufferedTreeNodeStream.HandleRootNode: IANTLRInterface;
begin
  Result := FCurrentNode;
  // point to first child in prep for subsequent next()
  FCurrentChildIndex := 0;
  if (FAdaptor.IsNil(Result)) then
    // don't count this root nil node
    Result := VisitChild(FCurrentChildIndex)
  else
  begin
    AddLookahead(Result);
    if (FAdaptor.GetChildCount(FCurrentNode) = 0) then
      // single node case
      Result := nil; // say we're done
  end;
end;

function TUnBufferedTreeNodeStream.Index: Integer;
begin
  Result := FAbsoluteNodeIndex + 1;
end;

function TUnBufferedTreeNodeStream.LA(I: Integer): Integer;
var
  T: IANTLRInterface;
begin
  T := LT(I);
  if (T = nil) then
    Result := TToken.INVALID_TOKEN_TYPE
  else
    Result := FAdaptor.GetNodeType(T);
end;

function TUnBufferedTreeNodeStream.LAChar(I: Integer): Char;
begin
  Result := Char(LA(I));
end;

function TUnBufferedTreeNodeStream.LookaheadSize: Integer;
begin
  if (FTail < FHead) then
    Result := Length(FLookahead) - FHead + FTail
  else
    Result := FTail - FHead;
end;

function TUnBufferedTreeNodeStream.LT(const K: Integer): IANTLRInterface;
begin
  if (K = -1) then
    Exit(FPreviousNode);

  if (K < 0) then
    raise EArgumentException.Create('tree node streams cannot look backwards more than 1 node');

  if (K = 0) then
    Exit(TTree.INVALID_NODE);

  Fill(K);
  Result := FLookahead[(FHead + K - 1) mod Length(FLookahead)];
end;

function TUnBufferedTreeNodeStream.Mark: Integer;
var
  State: ITreeWalkState;
  I, N, K: Integer;
  LA: TANTLRInterfaceArray;
begin
  if (FMarkers = nil) then
  begin
    FMarkers := TList<ITreeWalkState>.Create;
    FMarkers.Add(nil); // depth 0 means no backtracking, leave blank
  end;

  Inc(FMarkDepth);
  State := nil;
  if (FMarkDepth >= FMarkers.Count) then
  begin
    State := TTreeWalkState.Create;
    FMarkers.Add(State);
  end
  else
    State := FMarkers[FMarkDepth];

  State.AbsoluteNodeIndex := FAbsoluteNodeIndex;
  State.CurrentChildIndex := FCurrentChildIndex;
  State.CurrentNode := FCurrentNode;
  State.PreviousNode := FPreviousNode;
  State.NodeStackSize := FNodeStack.Count;
  State.IndexStackSize := FIndexStack.Count;

  // take snapshot of lookahead buffer
  N := LookaheadSize;
  I := 0;
  SetLength(LA,N);
  for K := 1 to N do
  begin
    LA[I] := LT(K);
    Inc(I);
  end;
  State.LookAhead := LA;
  FLastMarker := FMarkDepth;
  Result := FMarkDepth;
end;

function TUnBufferedTreeNodeStream.MoveNext: Boolean;
begin
  // already walked entire tree; nothing to return
  if (FCurrentNode = nil) then
  begin
    AddLookahead(FEof);
    FCurrentEnumerationNode := nil;
    // this is infinite stream returning EOF at end forever
    // so don't throw NoSuchElementException
    Exit(False);
  end;

  // initial condition (first time method is called)
  if (FCurrentChildIndex = -1) then
  begin
    FCurrentEnumerationNode := HandleRootNode as ITree;
    Exit(True);
  end;

  // index is in the child list?
  if (FCurrentChildIndex < FAdaptor.GetChildCount(FCurrentNode)) then
  begin
    FCurrentEnumerationNode := VisitChild(FCurrentChildIndex) as ITree;
    Exit(True);
  end;

  // hit end of child list, return to parent node or its parent ...
  WalkBackToMostRecentNodeWithUnvisitedChildren;
  if (FCurrentNode <> nil) then
  begin
    FCurrentEnumerationNode := VisitChild(FCurrentChildIndex) as ITree;
    Result := True;
  end
  else
    Result := False;
end;

procedure TUnBufferedTreeNodeStream.Release(const Marker: Integer);
begin
  // unwind any other markers made after marker and release marker
  FMarkDepth := Marker;
  // release this marker
  Dec(FMarkDepth);
end;

procedure TUnBufferedTreeNodeStream.ReplaceChildren(
  const Parent: IANTLRInterface; const StartChildIndex, StopChildIndex: Integer;
  const T: IANTLRInterface);
begin
  raise EInvalidOperation.Create('can''t do stream rewrites yet');
end;

procedure TUnBufferedTreeNodeStream.Reset;
begin
  FCurrentNode := FRoot;
  FPreviousNode := nil;
  FCurrentChildIndex := -1;
  FAbsoluteNodeIndex := -1;
  FHead := 0;
  FTail := 0;
end;

procedure TUnBufferedTreeNodeStream.Rewind(const Marker: Integer);
var
  State: ITreeWalkState;
begin
  if (FMarkers = nil) then
    Exit;
  State := FMarkers[Marker];
  FAbsoluteNodeIndex := State.AbsoluteNodeIndex;
  FCurrentChildIndex := State.CurrentChildIndex;
  FCurrentNode := State.CurrentNode;
  FPreviousNode := State.PreviousNode;
  // drop node and index stacks back to old size
  FNodeStack.Capacity := State.NodeStackSize;
  FIndexStack.Capacity := State.IndexStackSize;
  FHead := 0; // wack lookahead buffer and then refill
  FTail := 0;
  while (FTail < Length(State.LookAhead)) do
  begin
    FLookahead[FTail] := State.LookAhead[FTail];
    Inc(FTail);
  end;
  Release(Marker);
end;

procedure TUnBufferedTreeNodeStream.Rewind;
begin
  Rewind(FLastMarker);
end;

procedure TUnBufferedTreeNodeStream.Seek(const Index: Integer);
begin
  if (Index < Self.Index) then
    raise EArgumentOutOfRangeException.Create('can''t seek backwards in node stream');

  // seek forward, consume until we hit index
  while (Self.Index < Index) do
    Consume;
end;

procedure TUnBufferedTreeNodeStream.SetHasUniqueNavigationNodes(
  const Value: Boolean);
begin
  FUniqueNavigationNodes := Value;
end;

procedure TUnBufferedTreeNodeStream.SetTokenStream(const Value: ITokenStream);
begin
  FTokens := Value;
end;

function TUnBufferedTreeNodeStream.Size: Integer;
var
  S: ICommonTreeNodeStream;
begin
  S := TCommonTreeNodeStream.Create(FRoot);
  Result := S.Size;
end;

function TUnBufferedTreeNodeStream.ToString: String;
begin
  Result := ToString(FRoot, nil);
end;

procedure TUnBufferedTreeNodeStream.ToStringWork(const P, Stop: IANTLRInterface;
  const Buf: TStringBuilder);
var
  Text: String;
  C, N: Integer;
begin
  if (not FAdaptor.IsNil(P)) then
  begin
    Text := FAdaptor.GetNodeText(P);
    if (Text = '') then
      Text := ' ' + IntToStr(FAdaptor.GetNodeType(P));
    Buf.Append(Text); // ask the node to go to string
  end;

  if SameObj(P, Stop) then
    Exit;

  N := FAdaptor.GetChildCount(P);
  if (N > 0) and (not FAdaptor.IsNil(P)) then
  begin
    Buf.Append(' ');
    Buf.Append(TToken.DOWN);
  end;

  for C := 0 to N - 1 do
    ToStringWork(FAdaptor.GetChild(P, C), Stop, Buf);

  if (N > 0) and (not FAdaptor.IsNil(P)) then
  begin
    Buf.Append(' ');
    Buf.Append(TToken.UP);
  end;
end;

function TUnBufferedTreeNodeStream.VisitChild(
  const Child: Integer): IANTLRInterface;
begin
  Result := nil;
  // save state
  FNodeStack.Push(FCurrentNode);
  FIndexStack.Push(Child);
  if (Child = 0) and (not FAdaptor.IsNil(FCurrentNode)) then
    AddNavigationNode(TToken.DOWN);
  // visit child
  FCurrentNode := FAdaptor.GetChild(FCurrentNode, Child);
  FCurrentChildIndex := 0;
  Result := FCurrentNode;
  AddLookahead(Result);
  WalkBackToMostRecentNodeWithUnvisitedChildren;
end;

procedure TUnBufferedTreeNodeStream.WalkBackToMostRecentNodeWithUnvisitedChildren;
begin
  while (FCurrentNode <> nil) and (FCurrentChildIndex >= FAdaptor.GetChildCount(FCurrentNode)) do
  begin
    FCurrentNode := FNodeStack.Pop;
    if (FCurrentNode = nil) then
      // hit the root?
      Exit;

    FCurrentChildIndex := FIndexStack.Pop;
    Inc(FCurrentChildIndex); // move to next child
    if (FCurrentChildIndex >= FAdaptor.GetChildCount(FCurrentNode)) then
    begin
      if (not FAdaptor.IsNil(FCurrentNode)) then
        AddNavigationNode(TToken.UP);
      if SameObj(FCurrentNode, FRoot) then
        // we done yet?
        FCurrentNode := nil;
    end;
  end;
end;

function TUnBufferedTreeNodeStream.ToString(const Start,
  Stop: IANTLRInterface): String;
var
  BeginTokenIndex, EndTokenIndex: Integer;
  Buf: TStringBuilder;
begin
  if (Start = nil) then
    Exit('');

  // if we have the token stream, use that to dump text in order
  if (FTokens <> nil) then
  begin
    // don't trust stop node as it's often an UP node etc...
    // walk backwards until you find a non-UP, non-DOWN node
    // and ask for it's token index.
    BeginTokenIndex := FAdaptor.GetTokenStartIndex(Start);
    if (Stop <> nil) and (FAdaptor.GetNodeType(Stop) = TToken.UP) then
      EndTokenIndex := FAdaptor.GetTokenStopIndex(Start)
    else
      EndTokenIndex := Size - 1;
    Exit(FTokens.ToString(BeginTokenIndex, EndTokenIndex));
  end;

  Buf := TStringBuilder.Create;
  try
    ToStringWork(Start, Stop, Buf);
    Result := Buf.ToString;
  finally
    Buf.Free;
  end;
end;

{ TUnBufferedTreeNodeStream.TTreeWalkState }

function TUnBufferedTreeNodeStream.TTreeWalkState.GetAbsoluteNodeIndex: Integer;
begin
  Result := FAbsoluteNodeIndex;
end;

function TUnBufferedTreeNodeStream.TTreeWalkState.GetCurrentChildIndex: Integer;
begin
  Result := FCurrentChildIndex;
end;

function TUnBufferedTreeNodeStream.TTreeWalkState.GetCurrentNode: IANTLRInterface;
begin
  Result := FCurrentNode;
end;

function TUnBufferedTreeNodeStream.TTreeWalkState.GetIndexStackSize: integer;
begin
  Result := FIndexStackSize;
end;

function TUnBufferedTreeNodeStream.TTreeWalkState.GetLookAhead: TANTLRInterfaceArray;
begin
  Result := FLookAhead;
end;

function TUnBufferedTreeNodeStream.TTreeWalkState.GetNodeStackSize: Integer;
begin
  Result := FNodeStackSize;
end;

function TUnBufferedTreeNodeStream.TTreeWalkState.GetPreviousNode: IANTLRInterface;
begin
  Result := FPreviousNode;
end;

procedure TUnBufferedTreeNodeStream.TTreeWalkState.SetAbsoluteNodeIndex(
  const Value: Integer);
begin
  FAbsoluteNodeIndex := Value;
end;

procedure TUnBufferedTreeNodeStream.TTreeWalkState.SetCurrentChildIndex(
  const Value: Integer);
begin
  FCurrentChildIndex := Value;
end;

procedure TUnBufferedTreeNodeStream.TTreeWalkState.SetCurrentNode(
  const Value: IANTLRInterface);
begin
  FCurrentNode := Value;
end;

procedure TUnBufferedTreeNodeStream.TTreeWalkState.SetIndexStackSize(
  const Value: integer);
begin
  FIndexStackSize := Value;
end;

procedure TUnBufferedTreeNodeStream.TTreeWalkState.SetLookAhead(
  const Value: TANTLRInterfaceArray);
begin
  FLookAhead := Value;
end;

procedure TUnBufferedTreeNodeStream.TTreeWalkState.SetNodeStackSize(
  const Value: Integer);
begin
  FNodeStackSize := Value;
end;

procedure TUnBufferedTreeNodeStream.TTreeWalkState.SetPreviousNode(
  const Value: IANTLRInterface);
begin
  FPreviousNode := Value;
end;

{ Utilities }

var
  EmptyCommonTree: ICommonTree = nil;

function Def(const X: ICommonTree): ICommonTree; overload;
begin
  if Assigned(X) then
    Result := X
  else
  begin
    if (EmptyCommonTree = nil) then
      EmptyCommonTree := TCommonTree.Create;
    Result := EmptyCommonTree;
  end;
end;

initialization
  TTree.Initialize;

end.
