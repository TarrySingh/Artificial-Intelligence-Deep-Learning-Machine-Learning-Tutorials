/** This is identical to the ParserRuleReturnScope except that
 *  the start property is a tree nodes not Token object
 *  when you are parsing trees.  To be generic the tree node types
 *  have to be Object.
 */
org.antlr.runtime.tree.TreeRuleReturnScope = function(){};

org.antlr.lang.extend(org.antlr.runtime.tree.TreeRuleReturnScope,
                      org.antlr.runtime.RuleReturnScope,
{
    getStart: function() { return this.start; }
});
