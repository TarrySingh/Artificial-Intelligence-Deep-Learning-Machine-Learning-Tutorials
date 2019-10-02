## Introduction

This is an introduction.  Just like the title told you it would be.

### What it Does

java2python reads the Java source files you give it and produces somewhat
roughly equivalent Python source code.  It tries to make the same decisions you
would if you were porting the code manually.  It can perform the translation
faster and more accurately than you could (usually).

### Where It's Useful

java2python can help in two situations.  First, if you're doing a one-time port
of a Java project to Python, it can save you a lot of time and effort by
getting you really far really fast.

Second, if you've got a Java project and you'd like to generate a Python port
and keep the port up to date, you'll find that java2python can help
tremendously.  The per-project and per-file configuration system helps out a
lot in this area.

### Where It's Not

Where java2python is not useful is also important.  It won't be useful to you
if you expect your newly translated Python code to run correctly the first
time.  The platforms are too different and this tool is too limited for that to
happen.  Also, you won't find java2python very useful if you expect to convert
Java sources at runtime.  I suppose you could try, but I wouldn't.

### How it Works

java2python first converts the source code you give it into an abstract syntax
tree.  (That's a lie, really.  java2python doesn't do this step, [ANTLR][] does
this step, and [ANTLR][] is a whole lot bigger and cooler than java2python
could ever be.  Obviously, really smart people worked on [ANTLR][] and only one
fairly dim one worked on java2python).

After the syntax tree is constructed, it's walked and its nodes are converted
to their Python equivalents.  When the walking is complete, java2python takes a
few more swipes at it and prints it out.  It's all very boring, like geology or
watching someone learn to play the xylophone.

This is all well and good for most cases where there exists a very similar
Python construct for the given Java construct.  Classes, for example, are
pretty much the same in both languages.  The trouble spots are places where a
construct exists in Java that is not readily available in Python.

Note: yes, of course, we're dealing with Turing Machines and they're
equivalent.  If it works in Java, it can work in Python, and I'm not saying
that it can't.  But what I am saying is that there are chunks of Java source
code that you can't make into nice and neat and obvious Python equivalents.

To get around these trouble spots, java2python takes the approach of trying
make the problem go away.  For example, in Java the `if` statement can contain
an assignment expression:

```java
if (++x == 0) { ... }
```

There isn't a single statement equivalent in Python because assignments are
statements there, not expressions.  So java2python does what it can, presumably
what you would do:

```python
x += 1
if x == 0:
    ...
```

Careful readers will have spotted just how close we came to driving over a
cliff with that `++x` expression.  If the increment had been done on the other
side of the variable, the meaning of the statement would have changed and the
Python code would have been wrong.  Fortunately, I've driven by lots of cliffs
and have been scared by all of them so I thought of this ahead of time and
decided to do something about it:

```java
if (x++ ==0) { ... }
```

will translate to:

```python
mangled_name_for_x = x
x += 1
if mangled_name_for_x == 0:
    ...
```

See what java2python did there?  It tried to do what you would do.  For further
explanation and enumeration see the [translation details][] page.


### Why Bother?

I bothered to write this because [I needed a Java package][1] to run on the CPython
interpreter.  I got tired of porting by hand so I wrote this instead.  And
it's an interesting problem (kind of).


[ANTLR]: http://www.antlr.org
[translation details]: https://github.com/natural/java2python/tree/master/doc/translation.md
[1]:  http://roundrockriver.wordpress.com/2007/02/15/automated-translation-of-java-to-python/
