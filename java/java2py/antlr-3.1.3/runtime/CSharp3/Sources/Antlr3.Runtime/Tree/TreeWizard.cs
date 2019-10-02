/*
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Conversion to C#:
 * Copyright (c) 2008-2009 Sam Harwell, Pixel Mine, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// TODO: build indexes for wizard
//#define BUILD_INDEXES

namespace Antlr.Runtime.Tree
{
    using System.Collections.Generic;

    using IList = System.Collections.IList;
#if BUILD_INDEXES
    using IDictionary = System.Collections.IDictionary;
#endif

    /** <summary>
     *  Build and navigate trees with this object.  Must know about the names
     *  of tokens so you have to pass in a map or array of token names (from which
     *  this class can build the map).  I.e., Token DECL means nothing unless the
     *  class can translate it to a token type.
     *  </summary>
     *
     *  <remarks>
     *  In order to create nodes and navigate, this class needs a TreeAdaptor.
     *
     *  This class can build a token type -> node index for repeated use or for
     *  iterating over the various nodes with a particular type.
     *
     *  This class works in conjunction with the TreeAdaptor rather than moving
     *  all this functionality into the adaptor.  An adaptor helps build and
     *  navigate trees using methods.  This class helps you do it with string
     *  patterns like "(A B C)".  You can create a tree from that pattern or
     *  match subtrees against it.
     *  </remarks>
     */
    public class TreeWizard
    {
        protected ITreeAdaptor adaptor;
        protected IDictionary<string, int> tokenNameToTypeMap;

        public interface IContextVisitor
        {
            // TODO: should this be called visit or something else?
            void Visit( object t, object parent, int childIndex, IDictionary<string, object> labels );
        }

        public abstract class Visitor : IContextVisitor
        {
            public virtual void Visit( object t, object parent, int childIndex, IDictionary<string, object> labels )
            {
                Visit( t );
            }
            public abstract void Visit( object t );
        }

        class ActionVisitor : Visitor
        {
            System.Action<object> _action;

            public ActionVisitor( System.Action<object> action )
            {
                _action = action;
            }

            public override void Visit( object t )
            {
                _action( t );
            }
        }

        /** <summary>
         *  When using %label:TOKENNAME in a tree for parse(), we must
         *  track the label.
         *  </summary>
         */
        public class TreePattern : CommonTree
        {
            public string label;
            public bool hasTextArg;
            public TreePattern( IToken payload ) :
                base( payload )
            {
            }
            public override string ToString()
            {
                if ( label != null )
                {
                    return "%" + label + ":"; //+ base.ToString();
                }
                else
                {
                    return base.ToString();
                }
            }
        }

        public class WildcardTreePattern : TreePattern
        {
            public WildcardTreePattern( IToken payload ) :
                base( payload )
            {
            }
        }

        /** <summary>This adaptor creates TreePattern objects for use during scan()</summary> */
        public class TreePatternTreeAdaptor : CommonTreeAdaptor
        {
            public override object Create( IToken payload )
            {
                return new TreePattern( payload );
            }
        }

#if BUILD_INDEXES
        // TODO: build indexes for the wizard

        /** <summary>
         *  During fillBuffer(), we can make a reverse index from a set
         *  of token types of interest to the list of indexes into the
         *  node stream.  This lets us convert a node pointer to a
         *  stream index semi-efficiently for a list of interesting
         *  nodes such as function definition nodes (you'll want to seek
         *  to their bodies for an interpreter).  Also useful for doing
         *  dynamic searches; i.e., go find me all PLUS nodes.
         *  </summary>
         */
        protected IDictionary<int, IList<int>> tokenTypeToStreamIndexesMap;

        /** <summary>
         *  If tokenTypesToReverseIndex set to INDEX_ALL then indexing
         *  occurs for all token types.
         *  </summary>
         */
        public static readonly HashSet<int> INDEX_ALL = new HashSet<int>();

        /** <summary>
         *  A set of token types user would like to index for faster lookup.
         *  If this is INDEX_ALL, then all token types are tracked.  If null,
         *  then none are indexed.
         *  </summary>
         */
        protected HashSet<int> tokenTypesToReverseIndex = null;
#endif

        public TreeWizard( ITreeAdaptor adaptor )
        {
            this.adaptor = adaptor;
        }

        public TreeWizard( ITreeAdaptor adaptor, IDictionary<string, int> tokenNameToTypeMap )
        {
            this.adaptor = adaptor;
            this.tokenNameToTypeMap = tokenNameToTypeMap;
        }

        public TreeWizard( ITreeAdaptor adaptor, string[] tokenNames )
        {
            this.adaptor = adaptor;
            this.tokenNameToTypeMap = ComputeTokenTypes( tokenNames );
        }

        public TreeWizard( string[] tokenNames ) :
            this( null, tokenNames )
        {
        }

        /** <summary>
         *  Compute a Map&lt;String, Integer&gt; that is an inverted index of
         *  tokenNames (which maps int token types to names).
         *  </summary>
         */
        public virtual IDictionary<string, int> ComputeTokenTypes( string[] tokenNames )
        {
            IDictionary<string, int> m = new Dictionary<string, int>();
            if ( tokenNames == null )
            {
                return m;
            }
            for ( int ttype = TokenConstants.MIN_TOKEN_TYPE; ttype < tokenNames.Length; ttype++ )
            {
                string name = tokenNames[ttype];
                m[name] = ttype;
            }
            return m;
        }

        /** <summary>Using the map of token names to token types, return the type.</summary> */
        public virtual int GetTokenType( string tokenName )
        {
            if ( tokenNameToTypeMap == null )
            {
                return TokenConstants.INVALID_TOKEN_TYPE;
            }

            int value;
            if ( tokenNameToTypeMap.TryGetValue( tokenName, out value ) )
                return value;

            return TokenConstants.INVALID_TOKEN_TYPE;
        }

        /** <summary>
         *  Walk the entire tree and make a node name to nodes mapping.
         *  For now, use recursion but later nonrecursive version may be
         *  more efficient.  Returns Map&lt;Integer, List&gt; where the List is
         *  of your AST node type.  The Integer is the token type of the node.
         *  </summary>
         *
         *  <remarks>
         *  TODO: save this index so that find and visit are faster
         *  </remarks>
         */
        public virtual IDictionary<int, IList> Index( object t )
        {
            IDictionary<int, IList> m = new Dictionary<int, IList>();
            _Index( t, m );
            return m;
        }

        /** <summary>Do the work for index</summary> */
        protected virtual void _Index( object t, IDictionary<int, IList> m )
        {
            if ( t == null )
            {
                return;
            }
            int ttype = adaptor.GetType( t );
            IList elements;
            if ( !m.TryGetValue( ttype, out elements ) || elements == null )
            {
                elements = new List<object>();
                m[ttype] = elements;
            }
            elements.Add( t );
            int n = adaptor.GetChildCount( t );
            for ( int i = 0; i < n; i++ )
            {
                object child = adaptor.GetChild( t, i );
                _Index( child, m );
            }
        }

        class FindTreeWizardVisitor : TreeWizard.Visitor
        {
            IList _nodes;
            public FindTreeWizardVisitor( IList nodes )
            {
                _nodes = nodes;
            }
            public override void Visit( object t )
            {
                _nodes.Add( t );
            }
        }
        class FindTreeWizardContextVisitor : TreeWizard.IContextVisitor
        {
            TreeWizard _outer;
            TreePattern _tpattern;
            IList _subtrees;
            public FindTreeWizardContextVisitor( TreeWizard outer, TreePattern tpattern, IList subtrees )
            {
                _outer = outer;
                _tpattern = tpattern;
                _subtrees = subtrees;
            }

            public void Visit( object t, object parent, int childIndex, IDictionary<string, object> labels )
            {
                if ( _outer._Parse( t, _tpattern, null ) )
                {
                    _subtrees.Add( t );
                }
            }
        }

        /** <summary>Return a List of tree nodes with token type ttype</summary> */
        public virtual IList Find( object t, int ttype )
        {
            IList nodes = new List<object>();
            Visit( t, ttype, new FindTreeWizardVisitor( nodes ) );
            return nodes;
        }

        /** <summary>Return a List of subtrees matching pattern.</summary> */
        public virtual IList Find( object t, string pattern )
        {
            IList subtrees = new List<object>();
            // Create a TreePattern from the pattern
            TreePatternLexer tokenizer = new TreePatternLexer( pattern );
            TreePatternParser parser =
                new TreePatternParser( tokenizer, this, new TreePatternTreeAdaptor() );
            TreePattern tpattern = (TreePattern)parser.Pattern();
            // don't allow invalid patterns
            if ( tpattern == null ||
                 tpattern.IsNil ||
                 tpattern.GetType() == typeof( WildcardTreePattern ) )
            {
                return null;
            }
            int rootTokenType = tpattern.Type;
            Visit( t, rootTokenType, new FindTreeWizardContextVisitor( this, tpattern, subtrees ) );
            return subtrees;
        }

        public virtual object FindFirst( object t, int ttype )
        {
            return null;
        }

        public virtual object FindFirst( object t, string pattern )
        {
            return null;
        }

        /** <summary>
         *  Visit every ttype node in t, invoking the visitor.  This is a quicker
         *  version of the general visit(t, pattern) method.  The labels arg
         *  of the visitor action method is never set (it's null) since using
         *  a token type rather than a pattern doesn't let us set a label.
         *  </summary>
         */
        public virtual void Visit( object t, int ttype, IContextVisitor visitor )
        {
            _Visit( t, null, 0, ttype, visitor );
        }

        public virtual void Visit( object t, int ttype, System.Action<object> action )
        {
            Visit( t, ttype, new ActionVisitor( action ) );
        }

        /** <summary>Do the recursive work for visit</summary> */
        protected virtual void _Visit( object t, object parent, int childIndex, int ttype, IContextVisitor visitor )
        {
            if ( t == null )
            {
                return;
            }
            if ( adaptor.GetType( t ) == ttype )
            {
                visitor.Visit( t, parent, childIndex, null );
            }
            int n = adaptor.GetChildCount( t );
            for ( int i = 0; i < n; i++ )
            {
                object child = adaptor.GetChild( t, i );
                _Visit( child, t, i, ttype, visitor );
            }
        }

        class VisitTreeWizardContextVisitor : TreeWizard.IContextVisitor
        {
            TreeWizard _outer;
            IContextVisitor _visitor;
            IDictionary<string, object> _labels;
            TreePattern _tpattern;

            public VisitTreeWizardContextVisitor( TreeWizard outer, IContextVisitor visitor, IDictionary<string, object> labels, TreePattern tpattern )
            {
                _outer = outer;
                _visitor = visitor;
                _labels = labels;
                _tpattern = tpattern;
            }

            public void Visit( object t, object parent, int childIndex, IDictionary<string, object> unusedlabels )
            {
                // the unusedlabels arg is null as visit on token type doesn't set.
                _labels.Clear();
                if ( _outer._Parse( t, _tpattern, _labels ) )
                {
                    _visitor.Visit( t, parent, childIndex, _labels );
                }
            }
        }

        /** <summary>
         *  For all subtrees that match the pattern, execute the visit action.
         *  The implementation uses the root node of the pattern in combination
         *  with visit(t, ttype, visitor) so nil-rooted patterns are not allowed.
         *  Patterns with wildcard roots are also not allowed.
         *  </summary>
         */
        public virtual void Visit( object t, string pattern, IContextVisitor visitor )
        {
            // Create a TreePattern from the pattern
            TreePatternLexer tokenizer = new TreePatternLexer( pattern );
            TreePatternParser parser =
                new TreePatternParser( tokenizer, this, new TreePatternTreeAdaptor() );
            TreePattern tpattern = (TreePattern)parser.Pattern();
            // don't allow invalid patterns
            if ( tpattern == null ||
                 tpattern.IsNil ||
                 tpattern.GetType() == typeof( WildcardTreePattern ) )
            {
                return;
            }
            IDictionary<string, object> labels = new Dictionary<string, object>(); // reused for each _parse
            int rootTokenType = tpattern.Type;
            Visit( t, rootTokenType, new VisitTreeWizardContextVisitor( this, visitor, labels, tpattern ) );
        }

        /** <summary>
         *  Given a pattern like (ASSIGN %lhs:ID %rhs:.) with optional labels
         *  on the various nodes and '.' (dot) as the node/subtree wildcard,
         *  return true if the pattern matches and fill the labels Map with
         *  the labels pointing at the appropriate nodes.  Return false if
         *  the pattern is malformed or the tree does not match.
         *  </summary>
         *
         *  <remarks>
         *  If a node specifies a text arg in pattern, then that must match
         *  for that node in t.
         *
         *  TODO: what's a better way to indicate bad pattern? Exceptions are a hassle 
         *  </remarks>
         */
        public virtual bool Parse( object t, string pattern, IDictionary<string, object> labels )
        {
            TreePatternLexer tokenizer = new TreePatternLexer( pattern );
            TreePatternParser parser =
                new TreePatternParser( tokenizer, this, new TreePatternTreeAdaptor() );
            TreePattern tpattern = (TreePattern)parser.Pattern();
            /*
            System.out.println("t="+((Tree)t).toStringTree());
            System.out.println("scant="+tpattern.toStringTree());
            */
            bool matched = _Parse( t, tpattern, labels );
            return matched;
        }

        public virtual bool Parse( object t, string pattern )
        {
            return Parse( t, pattern, null );
        }

        /** <summary>
         *  Do the work for parse. Check to see if the t2 pattern fits the
         *  structure and token types in t1.  Check text if the pattern has
         *  text arguments on nodes.  Fill labels map with pointers to nodes
         *  in tree matched against nodes in pattern with labels.
         *  </summary>
         */
        protected virtual bool _Parse( object t1, TreePattern tpattern, IDictionary<string, object> labels )
        {
            // make sure both are non-null
            if ( t1 == null || tpattern == null )
            {
                return false;
            }
            // check roots (wildcard matches anything)
            if ( tpattern.GetType() != typeof( WildcardTreePattern ) )
            {
                if ( adaptor.GetType( t1 ) != tpattern.Type )
                {
                    return false;
                }
                // if pattern has text, check node text
                if ( tpattern.hasTextArg && !adaptor.GetText( t1 ).Equals( tpattern.Text ) )
                {
                    return false;
                }
            }
            if ( tpattern.label != null && labels != null )
            {
                // map label in pattern to node in t1
                labels[tpattern.label] = t1;
            }
            // check children
            int n1 = adaptor.GetChildCount( t1 );
            int n2 = tpattern.ChildCount;
            if ( n1 != n2 )
            {
                return false;
            }
            for ( int i = 0; i < n1; i++ )
            {
                object child1 = adaptor.GetChild( t1, i );
                TreePattern child2 = (TreePattern)tpattern.GetChild( i );
                if ( !_Parse( child1, child2, labels ) )
                {
                    return false;
                }
            }
            return true;
        }

        /** <summary>
         *  Create a tree or node from the indicated tree pattern that closely
         *  follows ANTLR tree grammar tree element syntax:
         *
         *      (root child1 ... child2).
         *  </summary>
         *
         *  <remarks>
         *  You can also just pass in a node: ID
         * 
         *  Any node can have a text argument: ID[foo]
         *  (notice there are no quotes around foo--it's clear it's a string).
         *
         *  nil is a special name meaning "give me a nil node".  Useful for
         *  making lists: (nil A B C) is a list of A B C.
         *  </remarks>
         */
        public virtual object Create( string pattern )
        {
            TreePatternLexer tokenizer = new TreePatternLexer( pattern );
            TreePatternParser parser = new TreePatternParser( tokenizer, this, adaptor );
            object t = parser.Pattern();
            return t;
        }

        /** <summary>
         *  Compare t1 and t2; return true if token types/text, structure match exactly.
         *  The trees are examined in their entirety so that (A B) does not match
         *  (A B C) nor (A (B C)). 
         *  </summary>
         *
         *  <remarks>
         *  TODO: allow them to pass in a comparator
         *  TODO: have a version that is nonstatic so it can use instance adaptor
         *
         *  I cannot rely on the tree node's equals() implementation as I make
         *  no constraints at all on the node types nor interface etc... 
         *  </remarks>
         */
        public static bool Equals( object t1, object t2, ITreeAdaptor adaptor )
        {
            return _Equals( t1, t2, adaptor );
        }

        /** <summary>
         *  Compare type, structure, and text of two trees, assuming adaptor in
         *  this instance of a TreeWizard.
         *  </summary>
         */
        public virtual new bool Equals( object t1, object t2 )
        {
            return _Equals( t1, t2, adaptor );
        }

        protected static bool _Equals( object t1, object t2, ITreeAdaptor adaptor )
        {
            // make sure both are non-null
            if ( t1 == null || t2 == null )
            {
                return false;
            }
            // check roots
            if ( adaptor.GetType( t1 ) != adaptor.GetType( t2 ) )
            {
                return false;
            }
            if ( !adaptor.GetText( t1 ).Equals( adaptor.GetText( t2 ) ) )
            {
                return false;
            }
            // check children
            int n1 = adaptor.GetChildCount( t1 );
            int n2 = adaptor.GetChildCount( t2 );
            if ( n1 != n2 )
            {
                return false;
            }
            for ( int i = 0; i < n1; i++ )
            {
                object child1 = adaptor.GetChild( t1, i );
                object child2 = adaptor.GetChild( t2, i );
                if ( !_Equals( child1, child2, adaptor ) )
                {
                    return false;
                }
            }
            return true;
        }

#if BUILD_INDEXES
        // TODO: next stuff taken from CommonTreeNodeStream

        /** <summary>
         *  Given a node, add this to the reverse index tokenTypeToStreamIndexesMap.
         *  You can override this method to alter how indexing occurs.  The
         *  default is to create a
         *
         *    Map&lt;Integer token type,ArrayList&lt;Integer stream index&gt;&gt;
         *  </summary>
         *
         *  <remarks>
         *  This data structure allows you to find all nodes with type INT in order.
         *
         *  If you really need to find a node of type, say, FUNC quickly then perhaps
         *
         *    Map&lt;Integertoken type,Map&lt;Object tree node,Integer stream index&gt;&gt;
         *
         *  would be better for you.  The interior maps map a tree node to
         *  the index so you don't have to search linearly for a specific node.
         *
         *  If you change this method, you will likely need to change
         *  getNodeIndex(), which extracts information.
         *  </remarks>
         */
        protected void fillReverseIndex( object node, int streamIndex )
        {
            //System.out.println("revIndex "+node+"@"+streamIndex);
            if ( tokenTypesToReverseIndex == null )
            {
                return; // no indexing if this is empty (nothing of interest)
            }
            if ( tokenTypeToStreamIndexesMap == null )
            {
                tokenTypeToStreamIndexesMap = new Dictionary<int, IList<int>>(); // first indexing op
            }
            int tokenType = adaptor.getType( node );
            if ( !( tokenTypesToReverseIndex == INDEX_ALL ||
                   tokenTypesToReverseIndex.Contains( tokenType ) ) )
            {
                return; // tokenType not of interest
            }
            IList<int> indexes;

            if ( !tokenTypeToStreamIndexesMap.TryGetValue( tokenType, out indexes ) || indexes == null )
            {
                indexes = new List<int>(); // no list yet for this token type
                indexes.Add( streamIndex ); // not there yet, add
                tokenTypeToStreamIndexesMap[tokenType] = indexes;
            }
            else
            {
                if ( !indexes.Contains( streamIndex ) )
                {
                    indexes.Add( streamIndex ); // not there yet, add
                }
            }
        }

        /** <summary>
         *  Track the indicated token type in the reverse index.  Call this
         *  repeatedly for each type or use variant with Set argument to
         *  set all at once.
         *  </summary>
         *
         *  <param name="tokenType" />
         */
        public void reverseIndex( int tokenType )
        {
            if ( tokenTypesToReverseIndex == null )
            {
                tokenTypesToReverseIndex = new HashSet<int>();
            }
            else if ( tokenTypesToReverseIndex == INDEX_ALL )
            {
                return;
            }
            tokenTypesToReverseIndex.add( tokenType );
        }

        /** <summary>
         *  Track the indicated token types in the reverse index. Set
         *  to INDEX_ALL to track all token types.
         *  </summary>
         */
        public void reverseIndex( HashSet<int> tokenTypes )
        {
            tokenTypesToReverseIndex = tokenTypes;
        }

        /** <summary>
         *  Given a node pointer, return its index into the node stream.
         *  This is not its Token stream index.  If there is no reverse map
         *  from node to stream index or the map does not contain entries
         *  for node's token type, a linear search of entire stream is used.
         *  </summary>
         *
         *  <remarks>
         *  Return -1 if exact node pointer not in stream.
         *  </remarks>
         */
        public int getNodeIndex( object node )
        {
            //System.out.println("get "+node);
            if ( tokenTypeToStreamIndexesMap == null )
            {
                return getNodeIndexLinearly( node );
            }
            int tokenType = adaptor.getType( node );
            IList<int> indexes;
            if ( !tokenTypeToStreamIndexesMap.TryGetValue( tokenType, out indexes ) || indexes == null )
            {
                //System.out.println("found linearly; stream index = "+getNodeIndexLinearly(node));
                return getNodeIndexLinearly( node );
            }
            for ( int i = 0; i < indexes.size(); i++ )
            {
                int streamIndex = indexes[i];
                object n = get( streamIndex );
                if ( n == node )
                {
                    //System.out.println("found in index; stream index = "+streamIndexI);
                    return streamIndex; // found it!
                }
            }
            return -1;
        }
#endif

    }
}
