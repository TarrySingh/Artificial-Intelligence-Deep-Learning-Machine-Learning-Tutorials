/** A buffered stream of tree nodes.  Nodes can be from a tree of ANY kind.
 *
 *  This node stream sucks all nodes out of the tree specified in
 *  the constructor during construction and makes pointers into
 *  the tree using an array of Object pointers. The stream necessarily
 *  includes pointers to DOWN and UP and EOF nodes.
 *
 *  This stream knows how to mark/release for backtracking.
 *
 *  This stream is most suitable for tree interpreters that need to
 *  jump around a lot or for tree parsers requiring speed (at cost of memory).
 *  There is some duplicated functionality here with UnBufferedTreeNodeStream
 *  but just in bookkeeping, not tree walking etc...
 *
 *  @see UnBufferedTreeNodeStream
 */
org.antlr.runtime.tree.CommonTreeNodeStream = function(adaptor,
                                                    tree,
                                                    initialBufferSize)
{
    if (arguments.length===1) {
        tree = adaptor;
        adaptor = new org.antlr.runtime.tree.CommonTreeAdaptor();
    }
    if (arguments.length <= 2) {
        initialBufferSize =
            org.antlr.runtime.tree.CommonTreeNodeStream.DEFAULT_INITIAL_BUFFER_SIZE;
    }

    /** Reuse same DOWN, UP navigation nodes unless this is true */
    this.uniqueNavigationNodes = false;

    /** The index into the nodes list of the current node (next node
     *  to consume).  If -1, nodes array not filled yet.
     */
    this.p = -1;

    var Token = org.antlr.runtime.Token;
    this.root = tree;
    this.adaptor = adaptor;
    this.nodes = []; //new ArrayList(initialBufferSize);
    this.down = this.adaptor.create(Token.DOWN, "DOWN");
    this.up = this.adaptor.create(Token.UP, "UP");
    this.eof = this.adaptor.create(Token.EOF, "EOF");
};

org.antlr.lang.augmentObject(org.antlr.runtime.tree.CommonTreeNodeStream, {
    DEFAULT_INITIAL_BUFFER_SIZE: 100,
    INITIAL_CALL_STACK_SIZE: 10
});

org.antlr.lang.extend(org.antlr.runtime.tree.CommonTreeNodeStream,
                  org.antlr.runtime.tree.TreeNodeStream, 
{
    StreamIterator: function() {
        var i = 0,
            nodes = this.nodes,
            eof = this.eof;

        return {
            hasNext: function() {
                return i<nodes.length;
            },

            next: function() {
                var current = i;
                i++;
                if ( current < nodes.length ) {
                    return nodes[current];
                }
                return eof;
            },

            remove: function() {
                throw new Error("cannot remove nodes from stream");
            }
        };
    },

    /** Walk tree with depth-first-search and fill nodes buffer.
     *  Don't do DOWN, UP nodes if its a list (t is isNil).
     */
    fillBuffer: function(t) {
        var reset_p = false;
        if (org.antlr.lang.isUndefined(t)) {
            t = this.root;
            reset_p = true;
        }

        var nil = this.adaptor.isNil(t);
        if ( !nil ) {
            this.nodes.push(t); // add this node
        }
        // add DOWN node if t has children
        var n = this.adaptor.getChildCount(t);
        if ( !nil && n>0 ) {
            this.addNavigationNode(org.antlr.runtime.Token.DOWN);
        }
        // and now add all its children
        var c, child;
        for (c=0; c<n; c++) {
            child = this.adaptor.getChild(t,c);
            this.fillBuffer(child);
        }
        // add UP node if t has children
        if ( !nil && n>0 ) {
            this.addNavigationNode(org.antlr.runtime.Token.UP);
        }

        if (reset_p) {
            this.p = 0; // buffer of nodes intialized now
        }
    },

    getNodeIndex: function(node) {
        if ( this.p==-1 ) {
            this.fillBuffer();
        }
        var i, t;
        for (i=0; i<this.nodes.length; i++) {
            t = this.nodes[i];
            if ( t===node ) {
                return i;
            }
        }
        return -1;
    },

    /** As we flatten the tree, we use UP, DOWN nodes to represent
     *  the tree structure.  When debugging we need unique nodes
     *  so instantiate new ones when uniqueNavigationNodes is true.
     */
    addNavigationNode: function(ttype) {
        var navNode = null;
        if ( ttype===org.antlr.runtime.Token.DOWN ) {
            if ( this.hasUniqueNavigationNodes() ) {
                navNode = this.adaptor.create(org.antlr.runtime.Token.DOWN, "DOWN");
            }
            else {
                navNode = this.down;
            }
        }
        else {
            if ( this.hasUniqueNavigationNodes() ) {
                navNode = this.adaptor.create(org.antlr.runtime.Token.UP, "UP");
            }
            else {
                navNode = this.up;
            }
        }
        this.nodes.push(navNode);
    },

    get: function(i) {
        if ( this.p===-1 ) {
            this.fillBuffer();
        }
        return this.nodes[i];
    },

    LT: function(k) {
        if ( this.p===-1 ) {
            this.fillBuffer();
        }
        if ( k===0 ) {
            return null;
        }
        if ( k<0 ) {
            return this.LB(-1*k);
        }
        if ( (this.p+k-1) >= this.nodes.length ) {
            return this.eof;
        }
        return this.nodes[this.p+k-1];
    },

    getCurrentSymbol: function() { return this.LT(1); },

    /** Look backwards k nodes */
    LB: function(k) {
        if ( k===0 ) {
            return null;
        }
        if ( (this.p-k)<0 ) {
            return null;
        }
        return this.nodes[this.p-k];
    },

    getTreeSource: function() {
        return this.root;
    },

    getSourceName: function() {
        return this.getTokenStream().getSourceName();
    },

    getTokenStream: function() {
        return this.tokens;
    },

    setTokenStream: function(tokens) {
        this.tokens = tokens;
    },

    getTreeAdaptor: function() {
        return this.adaptor;
    },

    setTreeAdaptor: function(adaptor) {
        this.adaptor = adaptor;
    },

    hasUniqueNavigationNodes: function() {
        return this.uniqueNavigationNodes;
    },

    setUniqueNavigationNodes: function(uniqueNavigationNodes) {
        this.uniqueNavigationNodes = uniqueNavigationNodes;
    },

    consume: function() {
        if ( this.p===-1 ) {
            this.fillBuffer();
        }
        this.p++;
    },

    LA: function(i) {
        return this.adaptor.getType(this.LT(i));
    },

    mark: function() {
        if ( this.p===-1 ) {
            this.fillBuffer();
        }
        this.lastMarker = this.index();
        return this.lastMarker;
    },

    release: function(marker) {
        // no resources to release
    },

    index: function() {
        return this.p;
    },

    rewind: function(marker) {
        if (!org.antlr.lang.isNumber(marker)) {
            marker = this.lastMarker;
        }
        this.seek(marker);
    },

    seek: function(index) {
        if ( this.p===-1 ) {
            this.fillBuffer();
        }
        this.p = index;
    },

    /** Make stream jump to a new location, saving old location.
     *  Switch back with pop().
     */
    push: function(index) {
        if ( !this.calls ) {
            this.calls = [];
        }
        this.calls.push(this.p); // save current index
        this.seek(index);
    },

    /** Seek back to previous index saved during last push() call.
     *  Return top of stack (return index).
     */
    pop: function() {
        var ret = this.calls.pop();
        this.seek(ret);
        return ret;
    },

    reset: function() {
        this.p = 0;
        this.lastMarker = 0;
        if (this.calls) {
            this.calls = [];
        }
    },

    size: function() {
        if ( this.p===-1 ) {
            this.fillBuffer();
        }
        return this.nodes.length;
    },

    iterator: function() {
        if ( this.p===-1 ) {
            this.fillBuffer();
        }
        return this.StreamIterator();
    },

    replaceChildren: function(parent, startChildIndex, stopChildIndex, t) {
        if ( parent ) {
            this.adaptor.replaceChildren(parent, startChildIndex, stopChildIndex, t);
        }
    },

    /** Debugging */
    toTokenString: function(start, stop) {
        if ( this.p===-1 ) {
            this.fillBuffer();
        }
        var buf='', i, t;
        for (i = start; i < this.nodes.length && i <= stop; i++) {
            t = this.nodes[i];
            buf += " "+this.adaptor.getToken(t);
        }
        return buf;
    },

    /** Used for testing, just return the token type stream */
    toString: function(start, stop) {
        var buf = "",
            text,
            t,
            i;
        if (arguments.length===0) {
            if ( this.p===-1 ) {
                this.fillBuffer();
            }
            for (i = 0; i < this.nodes.length; i++) {
                t = this.nodes[i];
                buf += " ";
                buf += this.adaptor.getType(t);
            }
            return buf;
        } else {
            if ( !org.antlr.lang.isNumber(start) || !org.antlr.lang.isNumber(stop) ) {
                return null;
            }
            if ( this.p===-1 ) {
                this.fillBuffer();
            }
            //System.out.println("stop: "+stop);
            if ( start instanceof org.antlr.runtime.tree.CommonTree ) {
                //System.out.print("toString: "+((CommonTree)start).getToken()+", ");
            } else {
                //System.out.println(start);
            }
            if ( stop instanceof org.antlr.runtime.tree.CommonTree ) {
                //System.out.println(((CommonTree)stop).getToken());
            } else {
                //System.out.println(stop);
            }
            // if we have the token stream, use that to dump text in order
            var beginTokenIndex,
                endTokenIndex;
            if ( this.tokens ) {
                beginTokenIndex = this.adaptor.getTokenStartIndex(start);
                endTokenIndex = this.adaptor.getTokenStopIndex(stop);
                // if it's a tree, use start/stop index from start node
                // else use token range from start/stop nodes
                if ( this.adaptor.getType(stop)===org.antlr.runtime.Token.UP ) {
                    endTokenIndex = this.adaptor.getTokenStopIndex(start);
                }
                else if ( this.adaptor.getType(stop)==org.antlr.runtime.Token.EOF )
                {
                    endTokenIndex = this.size()-2; // don't use EOF
                }
                return this.tokens.toString(beginTokenIndex, endTokenIndex);
            }
            // walk nodes looking for start
            t = null;
            i = 0;
            for (; i < this.nodes.length; i++) {
                t = this.nodes[i];
                if ( t===start ) {
                    break;
                }
            }
            // now walk until we see stop, filling string buffer with text
            buf = text = "";
            t = this.nodes[i];
            while ( t!==stop ) {
                text = this.adaptor.getText(t);
                if ( !org.antlr.lang.isString(text) ) {
                    text = " "+this.adaptor.getType(t).toString();
                }
                buf += text;
                i++;
                t = nodes[i];
            }
            // include stop node too
            text = this.adaptor.getText(stop);
            if ( !org.antlr.lang.isString(text) ) {
                text = " "+this.adaptor.getType(stop).toString();
            }
            buf += text;
            return buf;
        }
    }
});
