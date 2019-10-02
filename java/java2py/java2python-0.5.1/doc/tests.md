## Tests

The java2python package includes a [test suite][] for exercising the compiler and
its various translation features.  This doc explains how the tests work, how to
run these suite, and how to add new tests to it.

### How the Test Suite Works

The test suite is a [makefile][] that finds `.java` files in the same directory,
converts each to Python, runs both programs, and then compares their output.  If
the output matches, the test is considered successful.  If not, it's considered
a failure.

### How to Run the Test Suite

The simplest way to run the suite is to run all of it:

```bash
$ cd some_path_to/java2python/test
$ make
```

This will print lines like this:

```
...
[PASS] Class00
[PASS] Class01
[PASS] Class02
...
```

You can also run an individual test like this:

```bash
$ make Class02
[PASS] Class02
```

Notice that there isn't a suffix to the file name; you don't run `make
Class02.java`, just `make Class02`.  If you supply an extension, nothing will
happen and the test won't run.

The `test` directory contains two helper scripts that you can use during
development.  The first is [runjava][], which runs the Java compiler and the
Java VM with the indicated file.  Use it like this:

```bash
$ ./runjava Class01.java
Hello, world.
```

The second script is [runj2py][], which is a handy shortcut for running the
`j2py` script with preset command line arguments for the test configuration.
You run it like this:

```bash
$ ./runj2py Class01.java
#!/usr/bin/env python
""" generated source for module Class01 """
class Class01(object):
...
```

### Adding New Tests

When a new compiler feature is added, or when the translation semantics change,
it's a good idea to add one or more tests to the test suite.  Follow this
general outline:

1.  Create a Java source file that exhibits the language feature in question.

2.  Name the Java source file `FeatureNN` where `NN` is the next number in
sequence for `Feature`, e.g., `Class14.java`.

3.  In your Java source, write one or more values to stdout with
`System.out.println`.

4.  Check the comparison via `make FeatureNN`.  If the test passes, it might
indicate the new feature is working correctly.

[test suite]: https://github.com/natural/java2python/tree/master/test/
[makefile]: https://github.com/natural/java2python/blob/master/test/Makefile
[runjava]: https://github.com/natural/java2python/blob/master/test/runjava
[runj2py]: https://github.com/natural/java2python/blob/master/test/runj2py
