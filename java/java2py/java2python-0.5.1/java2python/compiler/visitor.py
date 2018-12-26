#!/usr/bin/env python
# -*- coding: utf-8 -*-
# java2python.compiler.visitor -> Base classes for walking ASTs.
#
# This module defines classes that accept nodes during AST walking.  These
# classes are the primary source of the language translation semantics as
# implemented by java2python.
#
# These classes implement the node handling behavior of the block classes built
# at runtime.  These classes use their factory callable more often than their
# template counterparts; during walking, the typical behavior is to either define
# the specific Python source, or to defer it to another block, or both.


from functools import reduce, partial
from itertools import ifilter, ifilterfalse, izip, tee
from logging import debug, warn
from re import compile as recompile, sub as resub

from java2python.lang import tokens
from java2python.lib import FS


class Memo(object):
    """ Memo -> AST walking luggage. """

    def __init__(self):
        self.comments, self.last = set(), 0


class Base(object):
    """ Base ->  Parent class for AST visitors. """

    commentSubs = map(recompile, ['^\s*/(\*)+', '(\*)+/\s*$', '^\s*//'])

    def accept(self, node, memo):
        """ Accept a node, possibly creating a child visitor. """
        tokType = tokens.map.get(node.token.type)
        missing = lambda node, memo:self
        call = getattr(self, 'accept{0}'.format(tokens.title(tokType)), missing)
        if call is missing:
            debug('no visitor accept method for %s', tokType)
        return call(node, memo)

    def insertComments(self, tmpl, tree, index, memo):
        """ Add comments to the template from tokens in the tree. """
        prefix = self.config.last('commentPrefix', '# ')
        cache, parser, comTypes = memo.comments, tree.parser, tokens.commentTypes
        comNew = lambda t:t.type in comTypes and (t.index not in cache)

        for tok in ifilter(comNew, parser.input.tokens[memo.last:index]):
            cache.add(tok.index)

            # loop over parents until we find the top expression
            base = tmpl
            while base:
                if base and base.parent and base.parent.isExpression:
                    base = base.parent
                else:
                    break

            if hasattr(base, 'tail') and tok.line==parser.input.tokens[index].line:
                base.tail += prefix if not base.tail.startswith(prefix) else ''
                base.tail += ''.join(self.stripComment(tok.text))
            else:
                for line in self.stripComment(tok.text):
                    self.factory.comment(left=prefix, right=line, parent=self)
        memo.last = index

    def stripComment(self, text):
        """ Regex substitutions for comments; removes comment characters. """
        subText = lambda value, regex:resub(regex, '', value)
        for text in ifilter(unicode.strip, text.split('\n')):
            yield reduce(subText, self.commentSubs, text)

    def walk(self, tree, memo=None):
        """ Depth-first visiting of the given AST. """
        if not tree:
            return
        memo = Memo() if memo is None else memo
        comIns = self.insertComments
        comIns(self, tree, tree.tokenStartIndex, memo)
        visitor = self.accept(tree, memo)
        if visitor:
            for child in tree.children:
                visitor.walk(child, memo)
                comIns(visitor, child, child.tokenStopIndex, memo)
        comIns(self, tree, tree.tokenStopIndex, memo)
        if tree.isJavaSource:
            comIns(self, tree, len(tree.parser.input.tokens), memo)
        # fixme: we're calling the mutators far too frequently instead
        # of only once per object after its walk is finished.
        for handler in self.configHandlers('PostWalk', suffix='Mutators'):
            handler(self)

    def zipWalk(self, nodes, visitors, memo):
        """ Walk the given nodes zipped with the given visitors. """
        for node, visitor in izip(nodes, visitors):
            visitor.walk(node, memo)

    def nodeTypeToString(self, node):
        """ Returns the TYPE or QUALIFIED_TYPE_IDENT of the given node. """
        alt = self.altIdent
        ntype = node.firstChildOfType(tokens.TYPE)
        nnext = ntype.children[0]
        if nnext.type == tokens.QUALIFIED_TYPE_IDENT:
            names = [alt(t.text) for t in nnext.childrenOfType(tokens.IDENT)]
            stype = '.'.join(names)
        else:
            stype = nnext.text
        return alt(stype)


class TypeAcceptor(object):
    """ TypeAcceptor -> shared visitor method(s) for type declarations. """

    def makeAcceptType(ft):
        """ Creates an accept function for the given factory type. """
        def acceptType(self, node, memo):
            """ Creates and returns a new template for a type. """
            try:
                name = node.firstChildOfType(tokens.IDENT).text
            except (AttributeError, ):
                return
            self.variables.append(name)
            return getattr(self.factory, ft)(name=name, parent=self)
        return acceptType

    acceptAt = makeAcceptType('at')
    acceptClass = makeAcceptType('klass')
    acceptEnum = makeAcceptType('enum')
    acceptInterface = makeAcceptType('interface')


class Module(TypeAcceptor, Base):
    """ Module -> accepts AST branches for module-level objects. """

    def makeAcceptHandledDecl(part):
        """ Creates an accept function for a decl to be processed by a handler. """
        def acceptDecl(self, node, memo):
            """ Processes a decl by creating a new template expression. """
            expr = self.factory.expr()
            expr.walk(node.firstChild(), memo)
            handler = self.configHandler(part)
            if handler:
                handler(self, expr)
        return acceptDecl

    acceptImport = makeAcceptHandledDecl('ImportDeclaration')
    acceptPackage = makeAcceptHandledDecl('PackageDeclaration')


class ModifiersAcceptor(object):
    """ ModifiersAcceptor -> shared behavior of classes and methods. """

    def acceptModifierList(self, node, memo):
        """ Accept and process class and method modifiers. """
        isAnno = lambda token:token.type==tokens.AT
        for ano in ifilter(isAnno, node.children):
            self.nodesToAnnos(ano, memo)
        for mod in ifilterfalse(isAnno, node.children):
            self.nodesToModifiers(mod, node)
        return self

    def nodesToAnnos(self, branch, memo):
        """ Convert the annotations in the given branch to a decorator. """
        name = branch.firstChildOfType(tokens.IDENT).text
        init = branch.firstChildOfType(tokens.ANNOTATION_INIT_BLOCK)
        if not init:
            deco = self.factory.expr(left=name, fs='@{left}()')
        else:
            defKey = init.firstChildOfType(tokens.ANNOTATION_INIT_DEFAULT_KEY)
            if defKey:
                deco = self.factory.expr(left=name, fs='@{left}({right})')
                deco.right = right = self.factory.expr(parent=deco)
                right.walk(defKey.firstChild(), memo)
            else:
                deco = self.factory.expr(left=name, fs='@{left}({right})')
                arg = deco.right = self.factory.expr(parent=deco)
                keys = init.firstChildOfType(tokens.ANNOTATION_INIT_KEY_LIST)
                for child in keys.children:
                    fs, expr = child.text + '={right}', child.firstChild()
                    fs += (', ' if child is not keys.children[-1] else '')
                    arg.left = self.factory.expr(fs=fs, parent=arg)
                    arg.left.walk(expr, memo)
                    arg.right = arg = self.factory.expr(parent=arg)
            self.decorators.append(deco)

    def nodesToModifiers(self, branch, root):
        """ Convert the modifiers in the given branch to template modifiers. """
        if root.parentType in tokens.methodTypes:
            self.modifiers.extend(n.text for n in root.children)
            if self.isStatic and self.parameters:
                self.parameters[0]['name'] = 'cls'
        self.modifiers.append(branch.text)


class VarAcceptor(object):
    """ Mixin for blocks that accept handle var declarations. """

    def acceptVarDeclaration(self, node, memo):
        """ Creates a new expression for a variable declaration. """
        varDecls = node.firstChildOfType(tokens.VAR_DECLARATOR_LIST)
        for varDecl in varDecls.childrenOfType(tokens.VAR_DECLARATOR):
            ident = varDecl.firstChildOfType(tokens.IDENT)
            self.variables.append(ident.text)

            identExp = self.factory.expr(left=ident.text, parent=self)
            identExp.type = self.nodeTypeToString(node)
            if node.firstChildOfType(tokens.MODIFIER_LIST):
                identExp.modifiers = [child.text for child in node.firstChildOfType(tokens.MODIFIER_LIST).children]

            declExp = varDecl.firstChildOfType(tokens.EXPR)
            assgnExp = identExp.pushRight(' = ')

            declArr = varDecl.firstChildOfType(tokens.ARRAY_INITIALIZER)
            if declExp:
                assgnExp.walk(declExp, memo)
            elif declArr:
                assgnExp.right = exp = self.factory.expr(fs='['+FS.lr+']', parent=identExp)
                children = list(declArr.childrenOfType(tokens.EXPR))
                for child in children:
                    fs = FS.lr if child is children[-1] else FS.lr + ', '
                    exp.left = self.factory.expr(fs=fs, parent=identExp)
                    exp.left.walk(child, memo)
                    exp.right = exp = self.factory.expr(parent=identExp)
            else:
                if node.firstChildOfType(tokens.TYPE).firstChildOfType(tokens.ARRAY_DECLARATOR_LIST):
                    val = assgnExp.pushRight('[]')
                else:
                    val = assgnExp.pushRight('{0}()'.format(identExp.type))
        return self


class Class(VarAcceptor, TypeAcceptor, ModifiersAcceptor, Base):
    """ Class -> accepts AST branches for class-level objects. """

    def nodeIdentsToBases(self, node, memo):
        """ Turns node idents into template bases. """
        idents = node.findChildrenOfType(tokens.IDENT)
        self.bases.extend(n.text for n in idents)

    acceptExtendsClause = nodeIdentsToBases
    acceptImplementsClause = nodeIdentsToBases

    def acceptAt(self, node, memo):
        """ Accept and ignore an annotation declaration. """
        # this overrides the TypeAcceptor implementation and
        # ignores AT tokens; they're sent within the class modifier
        # list and we have no use for them here.

    def acceptConstructorDecl(self, node, memo):
        """ Accept and process a constructor declaration. """
        method = self.factory.method(name='__init__', type=self.name, parent=self)
        superCalls = node.findChildrenOfType(tokens.SUPER_CONSTRUCTOR_CALL)
        if not any(superCalls) and any(self.bases):
            # from the java tutorial:
            # Note: If a constructor does not explicitly invoke a
            # superclass constructor, the Java compiler automatically
            # inserts a call to the no-argument constructor of the
            # superclass.
            fs = 'super(' + FS.r + ', self).__init__()'
            self.factory.expr(fs=fs, right=self.name, parent=method)
        return method

    def acceptFunctionMethodDecl(self, node, memo):
        """ Accept and process a typed method declaration. """
        ident = node.firstChildOfType(tokens.IDENT)
        type = node.firstChildOfType(tokens.TYPE).children[0].text
        mods = node.firstChildOfType(tokens.MODIFIER_LIST)
        self.variables.append(ident.text)
        return self.factory.method(name=ident.text, type=type, parent=self)

    def acceptVoidMethodDecl(self, node, memo):
        """ Accept and process a void method declaration. """
        ident = node.firstChildOfType(tokens.IDENT)
        mods = node.firstChildOfType(tokens.MODIFIER_LIST)
        self.variables.append(ident.text)
        return self.factory.method(name=ident.text, type='void', parent=self)


class Annotation(Class):
    """ Annotation -> accepts AST branches for Java annotations. """

    def acceptAnnotationTopLevelScope(self, node, memo):
        """ Accept and process an annotation scope. """
        # We're processing the entire annotation top level scope here
        # so as to easily find all of the default values and construct
        # the various python statements in one pass.
        args = []
        for child in node.children:
            if child.type == tokens.ANNOTATION_METHOD_DECL:
                mods, type, ident = child.children[0:3]
                type, name = type.children[0].text, ident.text
                meth = self.factory.method(parent=self, name=name, type=type)
                meth.factory.expr(fs='return self._{left}', parent=meth, left=name)
                default = child.children[3] if len(child.children) > 3 else None
                args.append((name, type, default))
            elif child.type == tokens.VAR_DECLARATION:
                self.acceptVarDeclaration(child, memo)
            else:
                self.walk(child, memo)
        self.addCall(args, memo)
        self.addInit(args, memo)
        self.children.sort(lambda a, b:-1 if a.name == '__call__' else 0)
        self.children.sort(lambda a, b:-1 if a.name == '__init__' else 0)

    def addInit(self, args, memo):
        """ Make an __init__ function in this annotation class. """
        meth = self.factory.method(parent=self, name='__init__')
        factory = partial(meth.factory.expr, fs='self._{left} = {right}', parent=meth)
        for name, type, default in args:
            if default is not None:
                expr = self.factory.expr()
                expr.walk(default, memo)
            else:
                expr = None
            meth.parameters.append(self.makeParam(name, type, default=expr))
            factory(left=name, right=name)

    def addCall(self, args, memo):
        """ Make a __call__ function in this class (so it's a decorator). """
        meth = self.factory.method(parent=self, name='__call__')
        meth.parameters.append(self.makeParam('obj', 'object'))
        factory = partial(self.factory.expr, parent=meth)
        factory(fs='setattr(obj, self.__class__.__name__, self)')
        factory(fs='return obj')


class Enum(Class):
    """ Enum -> accepts AST branches for Java enums. """

    def acceptEnumTopLevelScope(self, node, memo):
        """ Accept and process an enum scope """
        idents = node.childrenOfType(tokens.IDENT)
        factory = self.factory.expr
        handler = self.configHandler('Value')
        setFs = lambda v:'{0}.{1} = {2}'.format(self.name, v, self.name)
        for index, ident in enumerate(idents):
            args = list(ident.findChildrenOfType(tokens.ARGUMENT_LIST))
            if args:
                call = factory(left=setFs(ident), parent=self.parent)
                call.right = arg = factory(fs='('+FS.lr+')')
                argl = ident.firstChildOfType(tokens.ARGUMENT_LIST)
                exprs = list(argl.findChildrenOfType(tokens.EXPR))
                for expr in exprs:
                    fs = FS.r + ('' if expr is exprs[-1] else ', ')
                    arg.left = factory(fs=fs)
                    arg.left.walk(expr, memo)
                    arg.right = arg = factory()
            else:
                expr = factory(fs=ident.text+' = '+FS.r, parent=self)
                expr.pushRight(handler(self, index, ident.text))
        return self


class Interface(Class):
    """ Interface -> accepts AST branches for Java interfaces. """


class MethodContent(Base):
    """ MethodContent -> accepts trees for blocks within methods. """

    def acceptAssert(self, node, memo):
        """ Accept and process an assert statement. """
        assertStat = self.factory.statement('assert', fs=FS.lsr, parent=self)
        assertStat.expr.walk(node.firstChild(), memo)

    def acceptBreak(self, node, memo):
        """ Accept and process a break statement. """
        # possible parents of a break statement:  switch, while, do, for
        # we want to skip inserting a break statement if we're inside a switch.
        insert, ok_types = True, [tokens.WHILE, tokens.DO, tokens.FOR]
        for parent in node.parents():
            if parent.type == tokens.SWITCH:
                insert = False
                break
            if parent.type in ok_types:
                break
        if insert:
            if len(node.children):
                warn('Detected unhandled break statement with label; generated code incorrect.')
            breakStat = self.factory.statement('break', parent=self)

    def acceptCatch(self, node, memo):
        """ Accept and process a catch statement. """
        decl = node.firstChildOfType(tokens.FORMAL_PARAM_STD_DECL)
        dtype = decl.firstChildOfType(tokens.TYPE)
        tnames = dtype.findChildrenOfType(tokens.IDENT)
        cname = '.'.join(n.text for n in tnames)
        cvar = decl.firstChildOfType(tokens.IDENT)
        block = node.firstChildOfType(tokens.BLOCK_SCOPE)
        if not block.children:
            self.factory.expr(left='pass', parent=self)
        self.expr.fs = FS.lsrc
        self.expr.right = self.factory.expr(fs=FS.l+' as '+FS.r, left=cname, right=cvar)
        self.walk(block, memo)


    def acceptContinue(self, node, memo):
        """ Accept and process a continue statement. """
        contStat = self.factory.statement('continue', fs=FS.lsr, parent=self)
        if len(node.children):
            warn('Detected unhandled continue statement with label; generated code incorrect.')

    def acceptDo(self, node, memo):
        """ Accept and process a do-while block. """
        # DO - BLOCK_SCOPE - PARENTESIZED_EXPR
        blkNode, parNode = node.children
        whileStat = self.factory.statement('while', fs=FS.lsrc, parent=self)
        whileStat.expr.right = 'True'
        whileStat.walk(blkNode, memo)
        fs = FS.l+ ' ' + 'not ({right}):'
        ifStat = self.factory.statement('if', fs=fs, parent=whileStat)
        ifStat.expr.walk(parNode, memo)
        breakStat = self.factory.statement('break', parent=ifStat)

    goodExprParents = (
        tokens.BLOCK_SCOPE,
        tokens.CASE,
        tokens.DEFAULT,
        tokens.FOR_EACH,
        tokens.METHOD_CALL,
        tokens.ARGUMENT_LIST,
    )

    def acceptExpr(self, node, memo):
        """ Creates a new expression. """
        # this works but isn't precise
        if node.parentType in self.goodExprParents:
            return self.factory.expr(parent=self)

    def acceptFor(self, node, memo):
        """ Accept and process a 'for' statement. """
        self.walk(node.firstChildOfType(tokens.FOR_INIT), memo)
        whileStat = self.factory.statement('while', fs=FS.lsrc, parent=self)
        cond = node.firstChildOfType(tokens.FOR_CONDITION)
        if not cond.children:
            whileStat.expr.right = 'True'
        else:
            whileStat.expr.walk(cond, memo)
        whileBlock = self.factory.methodContent(parent=self)
        if not node.firstChildOfType(tokens.BLOCK_SCOPE).children:
            self.factory.expr(left='pass', parent=whileBlock)
        else:
            whileBlock.walk(node.firstChildOfType(tokens.BLOCK_SCOPE), memo)
        updateStat = self.factory.expr(parent=whileBlock)
        updateStat.walk(node.firstChildOfType(tokens.FOR_UPDATE), memo)

    def acceptForEach(self, node, memo):
        """ Accept and process a 'for each' style statement. """
        forEach = self.factory.statement('for', fs=FS.lsrc, parent=self)
        identExpr = forEach.expr.right = self.factory.expr(fs=FS.l+' in '+FS.r)
        identExpr.walk(node.firstChildOfType(tokens.IDENT), memo)
        inExpr = identExpr.right = self.factory.expr()
        inExpr.walk(node.firstChildOfType(tokens.EXPR), memo)
        forBlock = self.factory.methodContent(parent=self)
        forBlock.walk(node.children[4], memo)

    def acceptIf(self, node, memo):
        """ Accept and process an if statement. """
        children = node.children
        ifStat = self.factory.statement('if', fs=FS.lsrc, parent=self)
        ifStat.expr.walk(children[0], memo)

        if node.children[1].type == tokens.EXPR:
            ifBlock = self.factory.expr(parent=ifStat)
            ifBlock.walk(node.children[1], memo)
        else:
            ifBlock = self.factory.methodContent(parent=self)
            ifBlock.walk(node.children[1], memo)

        if len(children) == 3:
            nextNode = children[2]
            nextType = nextNode.type

            while nextType == tokens.IF:
                nextStat = self.factory.statement('elif', fs=FS.lsrc, parent=self)
                nextStat.expr.walk(nextNode.children[0], memo)
                if nextNode.children[1].type == tokens.EXPR:
                    nextBlock = self.factory.expr(parent=nextStat)
                else:
                    nextBlock = self.factory.methodContent(parent=self)
                nextBlock.walk(nextNode.children[1], memo)

                try:
                    nextNode = nextNode.children[2]
                except (IndexError, ):
                    nextType = None
                else:
                    nextType = nextNode.type

            if nextType == tokens.EXPR:
                elseStat = self.factory.statement('else', fs=FS.lc, parent=self)
                elseBlock = self.factory.expr(parent=elseStat)
                elseBlock.walk(nextNode, memo)
            elif nextType: # nextType != tokens.BLOCK_SCOPE:
                elseStat = self.factory.statement('else', fs=FS.lc, parent=self)
                if nextNode.children:
                    self.factory.methodContent(parent=self).walk(nextNode, memo)
                else:
                    self.factory.expr(left='pass', parent=elseStat)


    def acceptSwitch(self, node, memo):
        """ Accept and process a switch block. """
        # This implementation needs a lot of work to handle case
        # statements without breaks, out-of-order default labels, etc.
        # Given the current size and complexity, consider growing a
        # Case block type.
        parNode = node.firstChildOfType(tokens.PARENTESIZED_EXPR)
        lblNode = node.firstChildOfType(tokens.SWITCH_BLOCK_LABEL_LIST)
        caseNodes = lblNode.children
        # empty switch statement
        if not len(caseNodes):
            return
        # we have at least one node...
        parExpr = self.factory.expr(parent=self)
        parExpr.walk(parNode, memo)
        eqFs = FS.l + '==' + FS.r
        for caseIdx, caseNode in enumerate(caseNodes):
            isDefault, isFirst = caseNode.type==tokens.DEFAULT, caseIdx==0

            if isFirst:
                caseExpr = self.factory.statement('if', fs=FS.lsrc, parent=self)
            elif not isDefault:
                caseExpr = self.factory.statement('elif', fs=FS.lsrc, parent=self)
            elif isDefault:
                caseExpr = self.factory.statement('else', fs=FS.lc, parent=self)

            if not isDefault:
                right = self.factory.expr(parent=parExpr)
                right.walk(caseNode.firstChildOfType(tokens.EXPR), memo)
                caseExpr.expr.right = self.factory.expr(left=parExpr, right=right, fs=eqFs)
                caseContent = self.factory.methodContent(parent=self)
                for child in caseNode.children[1:]:
                    caseContent.walk(child, memo)
                if not caseNode.children[1:]:
                    self.factory.expr(left='pass', parent=caseContent)
            if isDefault:
                if isFirst:
                    caseExpr.expr.right = 'True'
                caseContent = self.factory.methodContent(parent=self)
                for child in caseNode.children:
                    caseContent.walk(child, memo)
                if not caseNode.children:
                    self.factory.expr(left='pass', parent=caseContent)
        self.children.remove(parExpr)

    def acceptSynchronized(self, node, memo):
        """ Accept and process a synchronized statement (not a modifier). """
        module = self.parents(lambda x:x.isModule).next()
        module.needsSyncHelpers = True
        if node.parent.type == tokens.MODIFIER_LIST:
            # Skip any synchronized modifier
            return
        lockName = self.configHandler('LockFunction', 'Name', 'lock_for_object')
        withFs = '{left} %s({right}):' % lockName
        sync = self.factory.statement('with', fs=withFs, parent=self)
        sync.expr.walk(node.children[0], memo)
        sync.walk(node.children[1], memo)

    def acceptThrow(self, node, memo):
        """ Accept and process a throw statement. """
        throw = self.factory.statement('raise', fs=FS.lsr, parent=self)
        throw.expr.walk(node.children[0], memo)

    def acceptTry(self, node, memo):
        """ Accept and process a try/catch block. """
        children = node.children
        tryNode = children[0]
        if len(children) == 3:
            catchClausesNode, finNode = children[1:3]
        elif len(children) == 2:
            catchClausesNode = finNode = None
            if children[1].type == tokens.CATCH_CLAUSE_LIST:
                catchClausesNode = children[1]
            else:
                finNode = children[1]
        tryStat = self.factory.statement('try', fs=FS.lc, parent=self)
        tryStat.walk(tryNode, memo)
        if catchClausesNode:
            for catchNode in catchClausesNode.children:
                exStat = self.factory.statement('except', fs=FS.lsrc, parent=self)
                exStat.walk(catchNode, memo)
        if finNode:
            finStat = self.factory.statement('finally', fs=FS.lc, parent=self)
            finStat.walk(finNode, memo)

    goodReturnParents = (
        tokens.BLOCK_SCOPE,
    )

    def acceptReturn(self, node, memo):
        """ Creates a new return expression. """
        # again, this works but isn't as precise as it should be
        if node.parentType: # in self.goodReturnParents:
            expr = self.factory.expr(left='return', parent=self)
            if node.children:
                expr.fs, expr.right = FS.lsr, self.factory.expr(parent=expr)
                expr.right.walk(node, memo)
            return expr

    def acceptWhile(self, node, memo):
        """ Accept and process a while block. """
        # WHILE - PARENTESIZED_EXPR - BLOCK_SCOPE
        parNode, blkNode = node.children
        whileStat = self.factory.statement('while', fs=FS.lsrc, parent=self)
        whileStat.expr.walk(parNode, memo)
        if not blkNode.children:
            self.factory.expr(left='pass', parent=whileStat)
        else:
            whileStat.walk(blkNode, memo)


class Method(VarAcceptor, ModifiersAcceptor, MethodContent):
    """ Method -> accepts AST branches for method-level objects. """

    def acceptFormalParamStdDecl(self, node, memo):
        """ Accept and process a single parameter declaration. """
        ident = node.firstChildOfType(tokens.IDENT)
        ptype = self.nodeTypeToString(node)
        self.parameters.append(self.makeParam(ident.text, ptype))
        return self

    def acceptFormalParamVarargDecl(self, node, memo):
        """ Accept and process a var arg declaration. """
        ident = node.firstChildOfType(tokens.IDENT)
        param = {'name':'*{0}'.format(ident.text), 'type':'A'}
        self.parameters.append(param)
        return self


class Expression(Base):
    """ Expression -> accepts trees for expression objects. """

    def nodeTextExpr(self, node, memo):
        """ Assigns node text to the left side of this expression. """
        self.left = node.text

    acceptCharacterLiteral = nodeTextExpr
    acceptStringLiteral = nodeTextExpr
    acceptFloatingPointLiteral = nodeTextExpr
    acceptDecimalLiteral = nodeTextExpr
    acceptHexLiteral = nodeTextExpr
    acceptOctalLiteral = nodeTextExpr
    acceptTrue = nodeTextExpr
    acceptFalse = nodeTextExpr
    acceptNull = nodeTextExpr

    def nodeOpExpr(self, node, memo):
        """ Accept and processes an operator expression. """
        factory = self.factory.expr
        self.fs = FS.l + ' ' + node.text + ' ' + FS.r
        self.left, self.right = visitors = factory(parent=self), factory(parent=self)
        self.zipWalk(node.children, visitors, memo)

    acceptAnd = nodeOpExpr
    acceptAndAssign = nodeOpExpr
    acceptAssign = nodeOpExpr
    acceptDivAssign = nodeOpExpr
    acceptEqual = nodeOpExpr
    acceptGreaterOrEqual = nodeOpExpr
    acceptGreaterThan = nodeOpExpr
    acceptLessOrEqual = nodeOpExpr
    acceptLessThan = nodeOpExpr
    acceptMinusAssign = nodeOpExpr
    acceptMod = nodeOpExpr
    acceptModAssign = nodeOpExpr
    acceptNotEqual = nodeOpExpr
    acceptOr = nodeOpExpr
    acceptOrAssign = nodeOpExpr
    acceptPlusAssign = nodeOpExpr
    acceptShiftLeft = nodeOpExpr
    acceptShiftLeftAssign = nodeOpExpr
    acceptShiftRight = nodeOpExpr
    acceptShiftRightAssign = nodeOpExpr
    acceptStarAssign = nodeOpExpr
    acceptXor = nodeOpExpr
    acceptXorAssign = nodeOpExpr

    def makeNodePreformattedExpr(fs):
        """ Make an accept method for expressions with a predefined format string. """
        def acceptPreformatted(self, node, memo):
            expr = self.factory.expr
            self.fs = fs
            self.left, self.right = vs = expr(parent=self), expr(parent=self)
            self.zipWalk(node.children, vs, memo)
        return acceptPreformatted

    acceptArrayElementAccess = makeNodePreformattedExpr(FS.l + '[' + FS.r + ']')
    acceptDiv = makeNodePreformattedExpr(FS.l + ' / ' + FS.r)
    acceptLogicalAnd = makeNodePreformattedExpr(FS.l + ' and ' + FS.r)
    acceptLogicalNot = makeNodePreformattedExpr('not ' + FS.l)
    acceptLogicalOr = makeNodePreformattedExpr(FS.l + ' or ' + FS.r)
    acceptMinus = makeNodePreformattedExpr(FS.l + ' - ' + FS.r)
    acceptNot = makeNodePreformattedExpr('~' + FS.l)
    acceptPlus = makeNodePreformattedExpr(FS.l + ' + ' + FS.r)
    acceptStar = makeNodePreformattedExpr(FS.l + ' * ' + FS.r)
    acceptUnaryMinus = makeNodePreformattedExpr('-' + FS.l)
    acceptUnaryPlus = makeNodePreformattedExpr('+' + FS.l)

    def acceptCastExpr(self, node, memo):
        """ Accept and process a cast expression. """
        # If the type of casting is a primitive type,
        # then do the cast, else drop it.
        factory = self.factory.expr
        typeTok = node.firstChildOfType(tokens.TYPE)
        typeIdent = typeTok.firstChild()
        typeName = typeIdent.text
        if typeIdent.type == tokens.QUALIFIED_TYPE_IDENT:
            typeName = typeIdent.firstChild().text

        if typeName in tokens.primitiveTypeNames:
            # Cast using the primitive type constructor
            self.fs = typeName + '(' + FS.r + ')'
        else:
            handler = self.configHandler('Cast')
            if handler:
                handler(self, node)
            else:
                warn('No handler to perform cast of non-primitive type %s.', typeName)
        self.left, self.right = visitors = factory(parent=self), factory(parent=self)
        self.zipWalk(node.children, visitors, memo)

    def makeAcceptPrePost(suffix, pre):
        """ Make an accept method for pre- and post- assignment expressions. """
        def acceptPrePost(self, node, memo):
            factory = self.factory.expr
            if node.withinExpr:
                name = node.firstChildOfType(tokens.IDENT).text
                handler = self.configHandler('VariableNaming')
                rename = handler(name)
                block = self.parents(lambda x:x.isMethod).next()
                if pre:
                    left = name
                else:
                    left = rename
                    block.adopt(factory(fs=FS.l+' = '+FS.r, left=rename, right=name))
                self.left = factory(parent=self, fs=FS.l, left=left)
                block.adopt(factory(fs=FS.l + suffix, left=name))
            else:
                self.fs = FS.l + suffix
                self.left, self.right = vs = factory(parent=self), factory(parent=self)
                self.zipWalk(node.children, vs, memo)
        return acceptPrePost

    acceptPostInc = makeAcceptPrePost(' += 1', pre=False)
    acceptPreInc = makeAcceptPrePost(' += 1', pre=True)
    acceptPostDec = makeAcceptPrePost(' -= 1', pre=False)
    acceptPreDec = makeAcceptPrePost(' -= 1', pre=True)

    def acceptBitShiftRight(self, node, memo):
        """ Accept and process a bit shift right expression. """
        factory = self.factory.expr
        self.fs = 'bsr(' + FS.l + ', ' + FS.r + ')'
        self.left, self.right = visitors = factory(parent=self), factory()
        self.zipWalk(node.children, visitors, memo)
        module = self.parents(lambda x:x.isModule).next()
        module.needsBsrFunc = True

    def acceptBitShiftRightAssign(self, node, memo):
        """ Accept and process a bit shift right expression with assignment. """
        factory = self.factory.expr
        self.fs = FS.l + ' = bsr(' + FS.l + ', ' + FS.r + ')'
        self.left, self.right = visitors = factory(parent=self), factory()
        self.zipWalk(node.children, visitors, memo)
        module = self.parents(lambda x:x.isModule).next()
        module.needsBsrFunc = True

    def acceptClassConstructorCall(self, node, memo):
        """ Accept and process a class constructor call. """
        self.acceptMethodCall(node, memo)
        typeIdent = node.firstChildOfType(tokens.QUALIFIED_TYPE_IDENT)
        if typeIdent and typeIdent.children:
            names = [self.altIdent(child.text) for child in typeIdent.children]
            self.left = '.'.join(names)

    def acceptDot(self, node, memo):
        """ Accept and process a dotted expression. """
        expr = self.factory.expr
        self.fs = FS.l + '.' + FS.r
        self.left, self.right = visitors = expr(parent=self), expr()
        self.zipWalk(node.children, visitors, memo)

    def acceptExpr(self, node, memo):
        """ Create a new expression within this one. """
        return self.pushRight()

    def acceptIdent(self, node, memo):
        """ Accept and process an ident expression. """
        self.left = self.altIdent(node.text)

    def acceptInstanceof(self, node, memo):
        """ Accept and process an instanceof expression. """
        self.fs = 'isinstance({right}, ({left}, ))'
        self.right = self.factory.expr(parent=self)
        self.right.walk(node.firstChildOfType(tokens.IDENT), memo)
        self.left = self.factory.expr(parent=self)
        self.left.walk(node.firstChildOfType(tokens.TYPE), memo)

    def acceptMethodCall(self, node, memo):
        """ Accept and process a method call. """
        # NB: this creates one too many expression levels.
        expr = self.factory.expr
        self.fs = FS.l + '(' + FS.r + ')'
        self.left = expr(parent=self)
        self.left.walk(node.firstChild(), memo)
        children = node.firstChildOfType(tokens.ARGUMENT_LIST).children
        self.right = arg = expr(parent=self)
        for child in children:
            fs = FS.r + (', ' if child is not children[-1] else '')
            arg.left = expr(fs=fs, parent=self)
            arg.left.walk(child, memo)
            arg.right = arg = expr(parent=self)

    skipParensParents = (
        tokens.IF,
        tokens.WHILE,
        tokens.SWITCH,
        tokens.SYNCHRONIZED,
        )

    def acceptParentesizedExpr(self, node, memo):
        if node.parent.type not in self.skipParensParents:
            right = self.pushRight()
            right.fs = '(' + FS.lr + ')'
            return right
        return self

    def acceptThisConstructorCall(self, node, memo):
        """ Accept and process a 'this(...)' constructor call. """
        self.acceptMethodCall(node, memo)
        self.left = 'self.__init__'

    def acceptStaticArrayCreator(self, node, memo):
        """ Accept and process a static array expression. """
        self.right = self.factory.expr(fs='[None]*{left}')
        self.right.left = self.factory.expr()
        self.right.left.walk(node.firstChildOfType(tokens.EXPR), memo)

    def acceptSuper(self, node, memo):
        """ Accept and process a super expression. """
        cls = self.parents(lambda c:c.isClass).next()
        self.right = self.factory.expr(fs='super({name}, self)'.format(name=cls.name))

    def acceptSuperConstructorCall(self, node, memo):
        """ Accept and process a super constructor call. """
        cls = self.parents(lambda c:c.isClass).next()
        fs = 'super(' + FS.l + ', self).__init__(' + FS.r + ')'
        self.right = self.factory.expr(fs=fs, left=cls.name)
        return self.right

    def acceptThis(self, node, memo):
        """ Accept and process a 'this' expression. """
        self.pushRight('self')

    def acceptQuestion(self, node, memo):
        """ Accept and process a terinary expression. """
        expr = self.factory.expr
        self.fs = FS.l + ' if ' + FS.r
        self.left = expr(parent=self)
        self.right = expr(fs=FS.l+' else '+FS.r, parent=self)
        self.right.left = expr(parent=self.right)
        self.right.right = expr(parent=self.right)
        visitors = (self.right.left, self.left, self.right.right)
        self.zipWalk(node.children, visitors, memo)

    def acceptVoid(self, node, memo):
        """ Accept and process a the void half of a void.class expression. """
        self.pushRight('None')

    def acceptClass(self, node, memo):
        """ Accept and process a .class expression. """
        self.pushRight('__class__')

    def pushRight(self, value=''):
        """ Creates a new right expression, sets it, and returns it. """
        self.right = self.factory.expr(left=value, parent=self)
        return self.right


class Comment(Expression):
    """ Comment -> implemented for type building in __init__.py. """


class Statement(MethodContent):
    """ Statement -> accept AST branches for statement objects. """
