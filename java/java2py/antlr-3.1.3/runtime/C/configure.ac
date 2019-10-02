#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_INIT(libantlr3c, 3.1.3, jimi@idle.ws)
AC_PREREQ(2.60)
AC_COPYRIGHT([
 (The "BSD licence")
 Copyright (c) 2005-2009 Jim Idle, Temporal Wave LLC
 http://www.temporal-wave.com
 http://www.linkedin.com/in/jimidle
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

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
])
AC_CONFIG_SRCDIR(src/antlr3parser.c)


AC_ARG_ENABLE([debuginfo],
		[AS_HELP_STRING([--enable-debuginfo], [Compiles debug info into the library (default --disable-debuginfo)])],
		[], [enable_debuginfo=no])

AC_ARG_ENABLE([64bit],
		[AS_HELP_STRING([--enable-64bit], [Turns on flags that produce 64 bit object code if any are required (default --disable-64bit)])],
		[], [enable_64bit=no])

AC_ARG_ENABLE([abiflags],
		[AS_HELP_STRING([--disable-abiflags], [Does not add ABI flags -m32 or -m64 for older versions of gcc, such as itanium 3.x (default --enable-abiflags=yes)])],
		[], [enable_abiflags=yes])

AC_ARG_ENABLE([antlrdebug],
		[AS_HELP_STRING([--disable-antlrdebug], [Turns off default flags that include the antlr debugger in the runtime. Specify to remove debugger and the socket dependancies (default --enable-antlrdebug)])],
		[], [enable_antlrdebug=yes])

AM_INIT_AUTOMAKE(foreign)
AC_LANG(C)
AC_PROG_CC([xlc aCC gcc cc])
AM_MAINTAINER_MODE
AM_PROG_LIBTOOL

AC_CANONICAL_BUILD
AC_CANONICAL_HOST

OBJECT_MODE=
# Checks for programs.
AC_MSG_CHECKING([compiler flags required for compiling ANTLR with $CC C compiler on host $host])
WARNFLAGS=
case $CC in
xlc*)
 CPPCMNTFLAGS="-qcpluscmt"
 if test x"$enable_64bit" = xyes; then
  CCFLAGS64="-q64 -Wl,-b64"
  OBJECT_MODE="64"
 else
  OBJECT_MODE="32"
 fi
 OPTIMFLAGS="-O2"
 if test x"$enable_debuginfo" = xyes; then
  DEBUGFLAGS="-g"
 fi
 ;;

aCC*)
 CPPCMNTFLAGS=
 if test x"$enable_64bit" = xyes; then
  CCFLAGS64="+DD64"
 fi
 OPTIMFLAGS="-O"
 if test $DEBUGINF = 1; then
  DEBUGFLAGS="-g"
 fi
 ;;

gcc*)
 CPPCMNTFLAGS=
 if test x"$enable_64bit" = xyes; then
  GCCABIFLAGS="-m64"
 else
  GCCABIFLAGS="-m32"
 fi
 if test x"$enable_abiflags" = xyes; then
  CCFLAGS64=$GCCABIFLAGS
 fi
 OPTIMFLAGS="-O2"
 if test x"$enable_debuginfo" = xyes; then
  DEBUGFLAGS="-g"
 fi
 WARNFLAGS=-Wall
 ;;

*)

case $host in
sparc*-*solaris*)
 CPPCMNTFLAGS=
 if test x"$enable_64bit" = xyes; then
  CCFLAGS64="-fast -xtarget=ultra4 -m64 -xarch=sparcvis"
 fi
 OPTIMFLAGS="-O"
 if test x"$enable_debuginfo" = xyes; then
  DEBUGFLAGS='-g'
 fi
 ;;

*)
 CPPCMNTFLAGS=
 CCFLAGS64=
 OPTIMFLAGS="-O"
 if test x"$enable_debuginfo" = xyes; then
  DEBUGFLAGS='-g'
 fi
 ;;
esac

 ;;
esac

CFLAGS="$CCFLAGS64 $CPPCMNTFLAGS $OPTIMFLAGS $DEBUGFLAGS $WARNFLAGS"
AC_MSG_RESULT([$CFLAGS])
AC_SUBST([OBJECT_MODE])

AS_IF([test "x$enable_antlrdebug" = xno], [AC_DEFINE([ANTLR3_NODEBUGGER], [1], [Define if ANTLR debugger not required])], [])
AS_IF([test x"$enable_64bit" = xyes], [AC_DEFINE([ANTLR3_USE_64BIT], [1], [Define if 64 bit mode required])], [])

AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET

# Checks for libraries.

# Checks for header files.
AC_INCLUDES_DEFAULT()
AC_HEADER_RESOLV
AC_CHECK_HEADERS([sys/malloc.h malloc.h], [], [],
[[#ifdef HAVE_SYS_MALLOC_H
#include <sys/malloc.h>
#endif
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
]])
AC_CHECK_HEADERS([stdarg.h], [], [],
[[#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
]])

AC_CHECK_HEADERS([sys/stat.h], [], [],
[[#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
]])

AC_CHECK_HEADERS([ctype.h], [], [],
[[#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif
]])

AC_CHECK_HEADERS([netinet/tcp.h], [], [],
[[#ifdef HAVE_NETINET_TCP_H
#include <netinet/tcp.h>
#endif
]])

AC_CHECK_HEADERS([sys/socket.h socket.h], [], [],
[[#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_SOCKET_H
#include <socket.h>
#endif
]])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_TYPE_SIZE_T
AC_TYPE_INT8_T
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INTPTR_T
AC_TYPE_UINT8_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINTPTR_T
AC_C_INLINE


# Checks for library functions.
AC_CHECK_FUNCS([memmove memset strdup accept])

AC_CONFIG_HEADERS(antlr3config.h)
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
