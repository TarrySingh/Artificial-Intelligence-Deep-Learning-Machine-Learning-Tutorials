/* Useful for dumping out the input stream after doing some
 *  augmentation or other manipulations.
 *
 *  You can insert stuff, replace, and delete chunks.  Note that the
 *  operations are done lazily--only if you convert the buffer to a
 *  String.  This is very efficient because you are not moving data around
 *  all the time.  As the buffer of tokens is converted to strings, the
 *  toString() method(s) check to see if there is an operation at the
 *  current index.  If so, the operation is done and then normal String
 *  rendering continues on the buffer.  This is like having multiple Turing
 *  machine instruction streams (programs) operating on a single input tape. :)
 *
 *  Since the operations are done lazily at toString-time, operations do not
 *  screw up the token index values.  That is, an insert operation at token
 *  index i does not change the index values for tokens i+1..n-1.
 *
 *  Because operations never actually alter the buffer, you may always get
 *  the original token stream back without undoing anything.  Since
 *  the instructions are queued up, you can easily simulate transactions and
 *  roll back any changes if there is an error just by removing instructions.
 *  For example,
 *
 *   CharStream input = new ANTLRFileStream("input");
 *   TLexer lex = new TLexer(input);
 *   TokenRewriteStream tokens = new TokenRewriteStream(lex);
 *   T parser = new T(tokens);
 *   parser.startRule();
 *
 *      Then in the rules, you can execute
 *      Token t,u;
 *      ...
 *      input.insertAfter(t, "text to put after t");}
 *         input.insertAfter(u, "text after u");}
 *         System.out.println(tokens.toString());
 *
 *  Actually, you have to cast the 'input' to a TokenRewriteStream. :(
 *
 *  You can also have multiple "instruction streams" and get multiple
 *  rewrites from a single pass over the input.  Just name the instruction
 *  streams and use that name again when printing the buffer.  This could be
 *  useful for generating a C file and also its header file--all from the
 *  same buffer:
 *
 *      tokens.insertAfter("pass1", t, "text to put after t");}
 *         tokens.insertAfter("pass2", u, "text after u");}
 *         System.out.println(tokens.toString("pass1"));
 *         System.out.println(tokens.toString("pass2"));
 *
 *  If you don't use named rewrite streams, a "default" stream is used as
 *  the first example shows.
 */

org.antlr.runtime.TokenRewriteStream = function() {
    var sup = org.antlr.runtime.TokenRewriteStream.superclass;

    /** You may have multiple, named streams of rewrite operations.
     *  I'm calling these things "programs."
     *  Maps String (name) -> rewrite (List)
     */
    this.programs = null;

    /** Map String (program name) -> Integer index */
    this.lastRewriteTokenIndexes = null;


    if (arguments.length===0) {
        this.init();
    } else {
        sup.constructor.apply(this, arguments);
        this.init();
    }
};

(function(){
var trs = org.antlr.runtime.TokenRewriteStream;

org.antlr.lang.augmentObject(trs, {
    DEFAULT_PROGRAM_NAME: "default",
    PROGRAM_INIT_SIZE: 100,
    MIN_TOKEN_INDEX: 0
});

//
// Define the rewrite operation hierarchy
//

trs.RewriteOperation = function(index, text) {
    this.index = index;
    this.text = text;
};

/** Execute the rewrite operation by possibly adding to the buffer.
 *  Return the index of the next token to operate on.
 */
trs.RewriteOperation.prototype = {
    execute: function(buf) {
        return this.index;
    },
    toString: function() {
        /*String opName = getClass().getName();
        int $index = opName.indexOf('$');
        opName = opName.substring($index+1, opName.length());
        return opName+"@"+index+'"'+text+'"';*/
        return this.text;
    }
};

trs.InsertBeforeOp = function(index, text) {
    trs.InsertBeforeOp.superclass.constructor.call(this, index, text);
};
org.antlr.lang.extend(trs.InsertBeforeOp, trs.RewriteOperation, {
    execute: function(buf) {
        buf.push(this.text);
        return this.index;
    }
});

/** I'm going to try replacing range from x..y with (y-x)+1 ReplaceOp
 *  instructions.
 */
trs.ReplaceOp = function(from, to, text) {
    trs.ReplaceOp.superclass.constructor.call(this, from, text); 
    this.lastIndex = to;
};
org.antlr.lang.extend(trs.ReplaceOp, trs.RewriteOperation, {
    execute: function(buf) {
        if (org.antlr.lang.isValue(this.text)) {
            buf.push(this.text);
        }
        return this.lastIndex+1;
    }
});

trs.DeleteOp = function(from, to) {
    trs.DeleteOp.superclass.constructor.call(this, from, to); 
};
org.antlr.lang.extend(trs.DeleteOp, trs.ReplaceOp);

org.antlr.lang.extend(trs, org.antlr.runtime.CommonTokenStream, {
    init: function() {
        this.programs = {};
        this.programs[trs.DEFAULT_PROGRAM_NAME] = [];
        this.lastRewriteTokenIndexes = {};
    },

    /** Rollback the instruction stream for a program so that
     *  the indicated instruction (via instructionIndex) is no
     *  longer in the stream.  UNTESTED!
     */
    rollback: function() {
        var programName,
            instructionIndex;

        if (arguments.length===1) {
            programName = trs.DEFAULT_PROGRAM_NAME;
            instructionIndex = arguments[0];
        } else if (arguments.length===2) {
            programName = arguments[0];
            instructionIndex = arguments[1];
        }
        var is = this.programs[programName];
        if (is) {
            programs[programName] = is.slice(trs.MIN_TOKEN_INDEX, this.instructionIndex);
        }
    },

    /** Reset the program so that no instructions exist */
    deleteProgram: function(programName) {
        programName = programName || trs.DEFAULT_PROGRAM_NAME;
        this.rollback(programName, trs.MIN_TOKEN_INDEX);
    },

    /** Add an instruction to the rewrite instruction list ordered by
     *  the instruction number (use a binary search for efficiency).
     *  The list is ordered so that toString() can be done efficiently.
     *
     *  When there are multiple instructions at the same index, the instructions
     *  must be ordered to ensure proper behavior.  For example, a delete at
     *  index i must kill any replace operation at i.  Insert-before operations
     *  must come before any replace / delete instructions.  If there are
     *  multiple insert instructions for a single index, they are done in
     *  reverse insertion order so that "insert foo" then "insert bar" yields
     *  "foobar" in front rather than "barfoo".  This is convenient because
     *  I can insert new InsertOp instructions at the index returned by
     *  the binary search.  A ReplaceOp kills any previous replace op.  Since
     *  delete is the same as replace with null text, i can check for
     *  ReplaceOp and cover DeleteOp at same time. :)
     */
    addToSortedRewriteList: function() {
        var programName,
            op;
        if (arguments.length===1) {
            programName = trs.DEFAULT_PROGRAM_NAME;
            op = arguments[0];
        } else if (arguments.length===2) {
            programName = arguments[0];
            op = arguments[1];
        }

        var rewrites = this.getProgram(programName);
        var len, pos, searchOp, replaced, prevOp, i;
        for (pos=0, len=rewrites.length; pos<len; pos++) {
            searchOp = rewrites[pos];
            if (searchOp.index===op.index) {
                // now pos is the index in rewrites of first op with op.index

                // an instruction operating already on that index was found;
                // make this one happen after all the others
                if (op instanceof trs.ReplaceOp) {
                    replaced = false;
                    // look for an existing replace
                    for (i=pos; i<rewrites.length; i++) {
                        prevOp = rewrites[pos];
                        if (prevOp.index!==op.index) {
                            break;
                        }
                        if (prevOp instanceof trs.ReplaceOp) {
                            rewrites[pos] = op; // replace old with new
                            replaced=true;
                            break;
                        }
                        // keep going; must be an insert
                    }
                    if ( !replaced ) {
                        // add replace op to the end of all the inserts
                        rewrites.splice(i, 0, op);
                    }
                } else {
                    // inserts are added in front of existing inserts
                    rewrites.splice(pos, 0, op);
                }
                break;
            } else if (searchOp.index > op.index) {
                rewrites.splice(pos, 0, op);
                break;
            }
        }
        if (pos===len) {
            rewrites.push(op);
        }
    },

    insertAfter: function() {
        var index, programName, text;
        if (arguments.length===2) {
            programName = trs.DEFAULT_PROGRAM_NAME;
            index = arguments[0];
            text = arguments[1];
        } else if (arguments.length===3) {
            programName = arguments[0];
            index = arguments[1];
            text = arguments[2];
        }

        if (index instanceof org.antlr.runtime.Token) {
            // index is a Token, grab it's stream index
            index = index.index; // that's ugly
        }

        // insert after is the same as insert before the next index
        this.insertBefore(programName, index+1, text);
    },

    insertBefore: function() {
        var index, programName, text;
        if (arguments.length===2) {
            programName = trs.DEFAULT_PROGRAM_NAME;
            index = arguments[0];
            text = arguments[1];
        } else if (arguments.length===3) {
            programName = arguments[0];
            index = arguments[1];
            text = arguments[2];
        }

        if (index instanceof org.antlr.runtime.Token) {
            // index is a Token, grab it's stream index
            index = index.index; // that's ugly
        }

        this.addToSortedRewriteList(
                programName,
                new trs.InsertBeforeOp(index,text)
                );
    },

    replace: function() {
        var programName, first, last, text;
        if (arguments.length===2) {
            programName = trs.DEFAULT_PROGRAM_NAME;
            first = arguments[0];
            last = arguments[0];
            text = arguments[1];
        } else if (arguments.length===3) {
            programName = trs.DEFAULT_PROGRAM_NAME;
            first = arguments[0];
            last = arguments[1];
            text = arguments[2];
        } if (arguments.length===4) {
            programName = arguments[0];
            first = arguments[1];
            last = arguments[2];
            text = arguments[3];
        } 

        if (first instanceof org.antlr.runtime.Token) {
            first = first.index;
        }

        if (last instanceof org.antlr.runtime.Token) {
            last = last.index; // that's ugly
        }

        if ( first > last || last<0 || first<0 ) {
            return;
        }
        this.addToSortedRewriteList(
                programName,
                new trs.ReplaceOp(first, last, text));
    },

    // !!! API Break: delete is a JS keyword, so using remove instead.
    remove: function() {
        // convert arguments to a real array
        var args=[], i=arguments.length-1;
        while (i>=0) {
            args[i] = arguments[i];
            i--;
        }

        args.push("");
        this.replace.apply(this, args);
    },

    getLastRewriteTokenIndex: function(programName) {
        programName = programName || trs.DEFAULT_PROGRAM_NAME;
        return this.lastRewriteTokenIndexes[programName] || -1;
    },

    setLastRewriteTokenIndex: function(programName, i) {
        this.lastRewriteTokenIndexes[programName] = i;
    },

    getProgram: function(name) {
        var is = this.programs[name];
        if ( !is ) {
            is = this.initializeProgram(name);
        }
        return is;
    },

    initializeProgram: function(name) {
        var is = [];
        this.programs[name] = is;
        return is;
    },

    toOriginalString: function(start, end) {
        if (!org.antlr.lang.isNumber(start)) {
            start = trs.MIN_TOKEN_INDEX;
        }
        if (!org.antlr.lang.isNumber(end)) {
            end = this.size()-1;
        }

        var buf = [], i;
        for (i=start; i>=trs.MIN_TOKEN_INDEX && i<=end && i<this.tokens.length; i++) {
            buf.push(this.get(i).getText());
        }
        return buf.join("");
    },

    toString: function() {
        var programName, start, end;
        if (arguments.length===0) {
            programName = trs.DEFAULT_PROGRAM_NAME;
            start = trs.MIN_TOKEN_INDEX;
            end = this.size() - 1;
        } else if (arguments.length===1) {
            programName = arguments[0];
            start = trs.MIN_TOKEN_INDEX;
            end = this.size() - 1;
        } else if (arguments.length===2) {
            programName = trs.DEFAULT_PROGRAM_NAME;
            start = arguments[0];
            end = arguments[1];
        }

        var rewrites = this.programs[programName];
        if ( !rewrites || rewrites.length===0 ) {
            return this.toOriginalString(start,end);
        }

        /// Index of first rewrite we have not done
        var rewriteOpIndex = 0,
            tokenCursor=start,
            buf = [],
            op;
        while ( tokenCursor>=trs.MIN_TOKEN_INDEX &&
                tokenCursor<=end &&
                tokenCursor<this.tokens.length )
        {
            // execute instructions associated with this token index
            if ( rewriteOpIndex<rewrites.length ) {
                op = rewrites[rewriteOpIndex];

                // skip all ops at lower index
                while (op.index<tokenCursor && rewriteOpIndex<rewrites.length) {
                    rewriteOpIndex++;
                    if ( rewriteOpIndex<rewrites.length ) {
                        op = rewrites[rewriteOpIndex];
                    }
                }

                // while we have ops for this token index, exec them
                while (tokenCursor===op.index && rewriteOpIndex<rewrites.length) {
                    //System.out.println("execute "+op+" at instruction "+rewriteOpIndex);
                    tokenCursor = op.execute(buf);
                    //System.out.println("after execute tokenCursor = "+tokenCursor);
                    rewriteOpIndex++;
                    if ( rewriteOpIndex<rewrites.length ) {
                        op = rewrites[rewriteOpIndex];
                    }
                }
            }
            // dump the token at this index
            if ( tokenCursor<=end ) {
                buf.push(this.get(tokenCursor).getText());
                tokenCursor++;
            }
        }
        // now see if there are operations (append) beyond last token index
        var opi;
        for (opi=rewriteOpIndex; opi<rewrites.length; opi++) {
            op = rewrites[opi];
            if ( op.index>=this.size() ) {
                op.execute(buf); // must be insertions if after last token
            }
        }

        return buf.join("");
    },

    toDebugString: function(start, end) {
        if (!org.antlr.lang.isNumber(start)) {
            start = trs.MIN_TOKEN_INDEX;
        }
        if (!org.antlr.lang.isNumber(end)) {
            end = this.size()-1;
        }

        var buf = [],
            i;
        for (i=start; i>=trs.MIN_TOKEN_INDEX && i<=end && i<this.tokens.length; i++) {
            buf.push(this.get(i));
        }
        return buf.join("");
    }
});

})();
