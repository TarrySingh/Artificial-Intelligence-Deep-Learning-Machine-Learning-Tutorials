#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" j2py -> Java to Python compiler script.

This is all very ordinary.  We import the package bits, open and read
a file, translate it, and write it out.

"""
import sys
from argparse import ArgumentParser, ArgumentTypeError
from collections import defaultdict
from logging import _levelNames as logLevels, exception, warning, info, basicConfig
from os import path, makedirs
from time import time

from java2python.compiler import Module, buildAST, transformAST
from java2python.config import Config
from java2python.lib import escapes


version = '0.5.1'


def logLevel(value):
    """ Returns a valid logging level or raises and exception. """
    msg = 'invalid loglevel: %r'
    try:
        lvl = int(value)
    except (ValueError, ):
        name = value.upper()
        if name not in logLevels:
            raise ArgumentTypeError(msg % value)
        lvl = logLevels[name]
    else:
        if lvl not in logLevels:
            raise ArgumentTypeError(msg % value)
    return lvl


def configFromDir(inname, dirname):
    """ Returns a file name from the given config directory. """
    name = path.join(dirname, path.basename(path.splitext(inname)[0]))
    return '%s.py' % path.abspath(name)


def runMain(options):
    """ Runs our main function with profiling if indicated by options. """
    if options.profile:
        import cProfile, pstats
        prof = cProfile.Profile()
        prof.runcall(runOneOrMany, options)
        stats = pstats.Stats(prof, stream=sys.stderr)
        stats.strip_dirs().sort_stats('cumulative')
        stats.print_stats().print_callers()
        return 0
    else:
        return runOneOrMany(options)

def runOneOrMany(options):
    """ Runs our main transformer with each of the input files. """
    infile, outfile = options.inputfile, options.outputfile

    if infile and not isinstance(infile, file) and path.isdir(infile):
        if outfile and not isinstance(outfile, file) and not path.isdir(outfile):
            warning('Must specify output directory or stdout when using input directory.')
            return 2
        def walker(arg, dirname, files):
            for name in [name for name in files if name.endswith('.java')]:
                fullname = path.join(dirname, name)
                options.inputfile = fullname
                info('opening %s', fullname)
                if outfile and outfile != '-' and not isinstance(outfile, file):
                    full = path.abspath(path.join(outfile, fullname))
                    head, tail = path.split(full)
                    tail = path.splitext(tail)[0] + '.py'
                    if not path.exists(head):
                        makedirs(head)
                    options.outputfile = path.join(head, tail)
                runTransform(options)
        path.walk(infile, walker, None)
        return 0
    else:
        return runTransform(options)


def runTransform(options):
    """ Compile the indicated java source with the given options. """
    timed = defaultdict(time)
    timed['overall']

    filein = fileout = filedefault = '-'
    if options.inputfile and not isinstance(options.inputfile, file):
        filein = options.inputfile
    if options.outputfile and not isinstance(options.outputfile, file):
        fileout = options.outputfile
    elif fileout != filedefault:
        fileout = '%s.py' % (path.splitext(filein)[0])

    configs = options.configs
    if options.configdirs and not isinstance(filein, file):
        for configdir in options.configdirs:
            dirname = configFromDir(filein, configdir)
            if path.exists(dirname):
                configs.insert(0, dirname)
    if options.includedefaults:
        configs.insert(0, 'java2python.config.default')

    try:
        if filein != '-':
            source = open(filein).read()
        else:
            source = sys.stdin.read()
    except (IOError, ), exc:
        code, msg = exc.args[0:2]
        print 'IOError: %s.' % (msg, )
        return code

    timed['comp']
    try:
        tree = buildAST(source)
    except (Exception, ), exc:
        exception('exception while parsing')
        return 1
    timed['comp_finish']

    config = Config(configs)
    timed['xform']
    transformAST(tree, config)
    timed['xform_finish']

    timed['visit']
    module = Module(config)
    module.sourceFilename = path.abspath(filein) if filein != '-' else None
    module.name = path.splitext(path.basename(filein))[0] if filein != '-' else '<stdin>'
    module.walk(tree)
    timed['visit_finish']

    timed['encode']
    source = unicode(module)
    timed['encode_finish']
    timed['overall_finish']

    if options.lexertokens:
        for idx, tok in enumerate(tree.parser.input.tokens):
            print >> sys.stderr, '{0}  {1}'.format(idx, tok)
        print >> sys.stderr

    if options.javaast:
        tree.dump(sys.stderr)
        print >> sys.stderr

    if options.pytree:
        module.dumpRepr(sys.stderr)
        print >> sys.stderr

    if not options.skipsource:
        if fileout == filedefault:
            output = sys.stdout
        else:
            output = open(fileout, 'w')
        module.name = path.splitext(filein)[0] if filein != '-' else '<stdin>'
        print >> output, source

    if not options.skipcompile:
        try:
            compile(source, '<string>', 'exec')
        except (SyntaxError, ), ex:
            warning('Generated source has invalid syntax. %s', ex)
        else:
            info('Generated source has valid syntax.')

    info('Parse:     %.4f seconds', timed['comp_finish'] - timed['comp'])
    info('Visit:     %.4f seconds', timed['visit_finish'] - timed['visit'])
    info('Transform: %.4f seconds', timed['xform_finish'] - timed['xform'])
    info('Encode:    %.4f seconds', timed['encode_finish'] - timed['encode'])
    info('Total:     %.4f seconds', timed['overall_finish'] - timed['overall'])
    return 0


def isWindows():
    """ True if running on Windows. """
    return sys.platform.startswith('win')


def configLogging(loglevel):
    """ Configure the logging package. """
    fmt = '# %(levelname)s %(funcName)s: %(message)s'
    basicConfig(level=loglevel, format=fmt)


def configColors(nocolor):
    """ Configure the color escapes. """
    if isWindows() or nocolor:
        escapes.clear()


def configScript(argv):
    """ Return an options object from the given argument sequence. """
    parser = ArgumentParser(
        description='Translate Java source code to Python.',
        epilog='Refer to https://github.com/natural/java2python for docs and support.'
        )

    add = parser.add_argument
    add(dest='inputfile', nargs='?',
        help='Read from INPUT.  May use - for stdin (default).',
        metavar='INPUT', default=None)
    add(dest='outputfile', nargs='?',
        help='Write to OUTPUT.  May use - for stdout (default).',
        metavar='OUTPUT', default=None)
    add('-c', '--config', dest='configs',
        help='Use CONFIG file or module.  May be repeated.',
        metavar='CONFIG', default=[], action='append')
    add('-d', '--config-dir', dest='configdirs',
        help='Use DIR to match input filename with config filename.',
        metavar='DIR', default=[], action='append')
    add('-f', '--profile', dest='profile',
        help='Profile execution and print results to stderr.',
        default=False, action='store_true')
    add('-j', '--java-ast', dest='javaast',
        help='Print java source AST tree to stderr.',
        default=False, action='store_true')
    add('-k', '--skip-compile', dest='skipcompile',
        help='Skip compile check on translated source.',
        default=False, action='store_true')
    add('-l', '--log-level', dest='loglevel',
        help='Set log level by name or value.',
        default='WARN', type=logLevel)
    add('-n', '--no-defaults', dest='includedefaults',
        help='Ignore default configuration module.',
        default=True, action='store_false')
    add('-p', '--python-tree', dest='pytree',
        help='Print python object tree to stderr.',
        default=False, action='store_true')
    add('-r', '--no-color', dest='nocolor',
        help='Disable color output.' +\
            ('  No effect on Win OS.' if isWindows() else ''),
        default=False, action='store_true')
    add('-s', '--skip-source', dest='skipsource',
        help='Skip writing translated source; useful when printing trees',
        default=False, action='store_true')
    add('-t', '--lexer-tokens', dest='lexertokens',
        help='Print lexer tokens to stderr.',
        default=False, action='store_true')
    add('-v', '--version', action='version', version='%(prog)s ' + version)

    ns = parser.parse_args(argv)
    if ns.inputfile == '-':
        ns.inputfile = sys.stdin
    if ns.outputfile == '-':
        ns.outputfile = sys.stdout

    configColors(ns.nocolor)
    configLogging(ns.loglevel)
    return ns


if __name__ == '__main__':
    sys.exit(runMain(configScript(sys.argv[1:])))
